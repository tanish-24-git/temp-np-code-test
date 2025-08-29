import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import structlog

from config.settings import settings

logger = structlog.get_logger()

class PreprocessingService:
    def __init__(self):
        self.upload_dir = Path(settings.upload_directory)
    
    def preprocess_dataset(
        self,
        file_path: str,
        missing_strategy: str,
        scaling: bool,
        encoding: str,
        target_column: Optional[str] = None,
        selected_features: Optional[List[str]] = None
    ) -> str:
        """Preprocess a dataset with specified parameters"""
        try:
            df = pd.read_csv(file_path)
            original_filename = Path(file_path).name
            
            if selected_features:
                available_features = [col for col in selected_features if col in df.columns]
                if target_column and target_column not in available_features:
                    available_features.append(target_column)
                df = df[available_features]
            
            df = self._handle_missing_values(df, missing_strategy)
            
            if target_column and target_column in df.columns:
                X = df.drop(columns=[target_column])
                y = df[target_column]
            else:
                X = df
                y = None
            
            X_processed = self._apply_preprocessing(X, scaling, encoding, y, target_column)
            
            if y is not None:
                df_processed = pd.concat([X_processed, y.reset_index(drop=True)], axis=1)
            else:
                df_processed = X_processed
            
            preprocessed_filename = f"preprocessed_{original_filename}"
            preprocessed_path = self.upload_dir / preprocessed_filename
            df_processed.to_csv(preprocessed_path, index=False)
            
            logger.info("Preprocessing completed", filename=preprocessed_filename)
            return str(preprocessed_path)
            
        except Exception as e:
            logger.error("Preprocessing error", error=str(e), exc_info=True)
            raise Exception(f"Preprocessing failed: {str(e)}. Verify encoding and missing_strategy are valid.")
    
    def _handle_missing_values(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        df_copy = df.copy()
        
        if strategy == "mean":
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            df_copy[numeric_cols] = df_copy[numeric_cols].fillna(df_copy[numeric_cols].mean())
        elif strategy == "median":
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            df_copy[numeric_cols] = df_copy[numeric_cols].fillna(df_copy[numeric_cols].median())
        elif strategy == "mode":
            for col in df_copy.columns:
                mode_val = df_copy[col].mode()
                if not mode_val.empty:
                    df_copy[col] = df_copy[col].fillna(mode_val.iloc[0])
        elif strategy == "drop":
            df_copy = df_copy.dropna()
        
        return df_copy
    
    def _apply_preprocessing(
        self, 
        X: pd.DataFrame, 
        scaling: bool, 
        encoding: str, 
        y: Optional[pd.Series] = None,
        target_column: Optional[str] = None
    ) -> pd.DataFrame:
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        transformers = []
        
        if numeric_cols:
            if scaling:
                transformers.append(('num', StandardScaler(), numeric_cols))
            else:
                transformers.append(('num', 'passthrough', numeric_cols))
        
        if categorical_cols:
            if encoding == "onehot":
                transformers.append(('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols))
            elif encoding == "label":
                X_copy = X.copy()
                for col in categorical_cols:
                    le = LabelEncoder()
                    X_copy[col] = le.fit_transform(X_copy[col].astype(str))
                transformers.append(('cat', 'passthrough', categorical_cols))
            else:
                transformers.append(('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols))
        
        if not transformers:
            return X
        
        preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
        X_transformed = preprocessor.fit_transform(X)
        
        feature_names = []
        for name, transformer, cols in preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(cols)
            elif name == 'cat' and encoding == "onehot":
                if hasattr(transformer, 'get_feature_names_out'):
                    feature_names.extend(transformer.get_feature_names_out(cols))
                else:
                    feature_names.extend(cols)
            else:
                feature_names.extend(cols)
        
        return pd.DataFrame(X_transformed, columns=feature_names)
    
    def suggest_missing_strategy(self, df: pd.DataFrame) -> str:
        missing_counts = df.isnull().sum()
        total_rows = len(df)
        
        if missing_counts.sum() == 0:
            return 'mean'
        
        missing_percentages = missing_counts / total_rows
        
        if any(missing_percentages > 0.5):
            return 'drop'
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        numeric_with_missing = numeric_cols.intersection(missing_counts[missing_counts > 0].index)
        if len(numeric_with_missing) > 0:
            skewness = df[numeric_with_missing].skew().abs()
            if any(skewness > 1):
                return 'median'
            return 'mean'
        
        if any(categorical_cols.isin(missing_counts[missing_counts > 0].index)):
            return 'mode'
        
        return 'mean'