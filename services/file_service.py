import os
import aiofiles
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from fastapi import UploadFile, HTTPException

from config.settings import settings
from utils.validators import file_validator
from services.insight_service import InsightService

class FileService:
    def __init__(self):
        self.upload_dir = Path(settings.upload_directory)
        self.upload_dir.mkdir(exist_ok=True)
        self.insight_service = InsightService()
    
    async def save_uploaded_file(self, file: UploadFile) -> str:
        """Save uploaded file to disk"""
        await file_validator.validate_file(file)
        safe_filename = self._sanitize_filename(file.filename)
        file_path = self.upload_dir / safe_filename
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        print(f"File saved: {safe_filename}")
        return str(file_path)
    
    def analyze_dataset(self, file_path: str) -> Dict[str, Any]:
        """Analyze uploaded dataset and generate insights"""
        try:
            df = pd.read_csv(file_path)
            summary = {
                "columns": list(df.columns),
                "rows": len(df),
                "sample_rows": len(df),
                "data_types": df.dtypes.astype(str).to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "unique_values": {col: df[col].nunique() for col in df.columns},
                "file_size_mb": os.path.getsize(file_path) / (1024 * 1024)
            }
            insights_data = self.insight_service.generate_insights(summary, df)
            return {
                "summary": summary,
                "insights": insights_data["insights"],
                "suggested_task_type": insights_data["suggested_task_type"],
                "suggested_target_column": insights_data["suggested_target_column"],
                "suggested_missing_strategy": insights_data["suggested_missing_strategy"]
            }
        except Exception as e:
            print(f"Error analyzing dataset: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to analyze dataset: {str(e)}")
    
    def _sanitize_filename(self, filename: str) -> str:
        import re
        safe_name = re.sub(r'[^a-zA-Z0-9._-]', '_', os.path.basename(filename))
        return safe_name[:100]