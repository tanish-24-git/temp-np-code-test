import os
import hashlib
from pathlib import Path
from typing import Set
from fastapi import UploadFile, HTTPException
from config.settings import settings
import pandas as pd
import io
import structlog

logger = structlog.get_logger()

ALLOWED_EXTENSIONS = set(settings.allowed_extensions)

CONTENT_TYPE_MAPPING = {
    '.csv': ['text/csv', 'application/csv', 'text/plain'],
    '.xlsx': ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'],
    '.json': ['application/json', 'text/json']
}

MALICIOUS_SIGNATURES = {
    b'\x4d\x5a',  # MZ (Executable)
    b'\x50\x4b\x03\x04',  # PK (Zip/Office)
    b'\x89\x50\x4e\x47',  # PNG
}

class EnhancedFileValidator:
    def __init__(self):
        # Verify that max_file_size_mb is loaded correctly from settings
        if not hasattr(settings, 'max_file_size_mb') or settings.max_file_size_mb is None:
            logger.error("MAX_FILE_SIZE_MB environment variable not set or invalid")
            raise ValueError("MAX_FILE_SIZE_MB environment variable not set or invalid")
        self.max_file_size = settings.max_file_size_mb * 1024 * 1024  # Convert MB to bytes
        self.uploaded_hashes: Set[str] = set()
    
    async def validate_file(self, file: UploadFile) -> bool:
        """Validate uploaded file for size, extension, content type, and integrity."""
        if not file.filename:
            logger.error("File validation failed: No filename provided")
            raise HTTPException(status_code=400, detail="No filename provided")
        
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            logger.error(f"File validation failed: Unsupported file extension {file_ext}", filename=file.filename)
            raise HTTPException(status_code=400, detail=f"File type {file_ext} not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}")
        
        content = await file.read()
        await file.seek(0)  # Reset file pointer for further reading
        
        if len(content) > self.max_file_size:
            logger.error(f"File validation failed: File size exceeds limit", 
                        filename=file.filename, size=len(content), max_size=self.max_file_size)
            raise HTTPException(status_code=413, detail=f"File too large: {len(content)/1024/1024:.2f}MB exceeds max {settings.max_file_size_mb}MB")
        
        if len(content) == 0:
            logger.error("File validation failed: Empty file", filename=file.filename)
            raise HTTPException(status_code=400, detail="Empty file not allowed")
        
        if file.content_type and file.content_type not in CONTENT_TYPE_MAPPING.get(file_ext, []):
            logger.error(f"File validation failed: Invalid content type {file.content_type}", filename=file.filename)
            raise HTTPException(status_code=400, detail=f"Invalid content type: {file.content_type}. Expected: {', '.join(CONTENT_TYPE_MAPPING.get(file_ext, []))}")
        
        await self._validate_file_signature(content, file_ext)
        
        for signature in MALICIOUS_SIGNATURES:
            if content.startswith(signature):
                logger.error("File validation failed: Potentially malicious file detected", filename=file.filename)
                raise HTTPException(status_code=400, detail="Potentially malicious file detected")
        
        file_hash = hashlib.md5(content).hexdigest()
        if file_hash in self.uploaded_hashes:
            logger.warning(f"File validation failed: Duplicate file detected", filename=file.filename, hash=file_hash)
            raise HTTPException(status_code=400, detail="Duplicate file detected")
        self.uploaded_hashes.add(file_hash)
        
        if file_ext == '.csv':
            await self._validate_csv_structure(content, file.filename)
        
        logger.info("File validation successful", filename=file.filename)
        return True
    
    async def _validate_file_signature(self, content: bytes, file_ext: str):
        """Validate file signature based on extension."""
        try:
            if file_ext == '.csv':
                if not content[:100].decode('utf-8', errors='ignore').isprintable():
                    logger.error("File signature validation failed: Invalid CSV format")
                    raise HTTPException(status_code=400, detail="Invalid CSV format: File content is not printable text")
            elif file_ext == '.xlsx':
                if not content.startswith(b'PK'):
                    logger.error("File signature validation failed: Invalid XLSX format")
                    raise HTTPException(status_code=400, detail="Invalid XLSX format: Expected PK signature")
            elif file_ext == '.json':
                try:
                    import json
                    json.loads(content.decode('utf-8'))
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.error(f"File signature validation failed: Invalid JSON format", error=str(e))
                    raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
        except Exception as e:
            logger.error(f"File signature validation failed", error=str(e))
            raise HTTPException(status_code=400, detail=f"File signature validation failed: {str(e)}")
    
    async def _validate_csv_structure(self, content: bytes, filename: str):
        """Validate CSV structure and content."""
        try:
            df = pd.read_csv(io.BytesIO(content))
            if len(df.columns) == 0:
                logger.error("CSV validation failed: No columns found", filename=filename)
                raise HTTPException(status_code=400, detail="CSV has no columns")
            if len(df) == 0:
                logger.error("CSV validation failed: No data rows found", filename=filename)
                raise HTTPException(status_code=400, detail="CSV has no data rows")
            if len(df.columns) > 1000:
                logger.error("CSV validation failed: Too many columns", filename=filename, columns=len(df.columns))
                raise HTTPException(status_code=400, detail="Too many columns (max: 1000)")
            if len(df) > 1000000:
                logger.error("CSV validation failed: Too many rows", filename=filename, rows=len(df))
                raise HTTPException(status_code=400, detail="Too many rows (max: 1,000,000)")
        except pd.errors.EmptyDataError:
            logger.error("CSV validation failed: Empty CSV file", filename=filename)
            raise HTTPException(status_code=400, detail="Empty CSV file")
        except pd.errors.ParserError as e:
            logger.error(f"CSV validation failed: Invalid CSV format", filename=filename, error=str(e))
            raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")
        except Exception as e:
            logger.error(f"CSV validation failed: Unexpected error", filename=filename, error=str(e))
            raise HTTPException(status_code=400, detail=f"CSV validation failed: {str(e)}")

file_validator = EnhancedFileValidator()

def validate_preprocessing_params(missing_strategy: str, encoding: str, target_column: str = None) -> bool:
    """Validate preprocessing parameters."""
    allowed_missing = ["mean", "median", "mode", "drop"]
    if missing_strategy not in allowed_missing:
        logger.error(f"Preprocessing validation failed: Invalid missing strategy {missing_strategy}")
        raise HTTPException(status_code=422, detail=f"Invalid missing strategy: {missing_strategy}. Allowed: {', '.join(allowed_missing)}")
    
    allowed_encoding = ["onehot", "label", "target", "kfold"]
    if encoding not in allowed_encoding:
        logger.error(f"Preprocessing validation failed: Invalid encoding method {encoding}")
        raise HTTPException(status_code=422, detail=f"Invalid encoding method: {encoding}. Allowed: {', '.join(allowed_encoding)}")
    
    if encoding in ["target", "kfold"] and not target_column:
        logger.error(f"Preprocessing validation failed: Target column required for {encoding}")
        raise HTTPException(status_code=422, detail=f"Target column required for {encoding} encoding")
    
    logger.info("Preprocessing parameters validated successfully", 
                missing_strategy=missing_strategy, encoding=encoding, target_column=target_column)
    return True