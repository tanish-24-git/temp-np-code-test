import os
import hashlib
from pathlib import Path
from typing import Set
from fastapi import UploadFile, HTTPException
from config.settings import settings
import pandas as pd
import io

ALLOWED_EXTENSIONS = set(settings.allowed_file_extensions)

CONTENT_TYPE_MAPPING = {
    '.csv': ['text/csv', 'application/csv', 'text/plain'],
    '.xlsx': ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'],
    '.json': ['application/json', 'text/json']
}

MALICIOUS_SIGNATURES = {
    b'\x4d\x5a',  # MZ
    b'\x50\x4b\x03\x04',  
    b'\x89\x50\x4e\x47',  
}

class EnhancedFileValidator:
    def __init__(self):
        self.max_file_size = settings.max_file_size_mb * 1024 * 1024
        self.uploaded_hashes: Set[str] = set()
    
    async def validate_file(self, file: UploadFile) -> bool:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"File type {file_ext} not allowed")
        content = await file.read()
        await file.seek(0)
        if len(content) > self.max_file_size:
            raise HTTPException(status_code=413, detail=f"File too large: max {settings.max_file_size_mb}MB")
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file not allowed")
        if file.content_type and file.content_type not in CONTENT_TYPE_MAPPING.get(file_ext, []):
            raise HTTPException(status_code=400, detail=f"Invalid content type: {file.content_type}")
        await self._validate_file_signature(content, file_ext)
        for signature in MALICIOUS_SIGNATURES:
            if content.startswith(signature):
                raise HTTPException(status_code=400, detail="Potentially malicious file detected")
        file_hash = hashlib.md5(content).hexdigest()
        if file_hash in self.uploaded_hashes:
            raise HTTPException(status_code=400, detail="Duplicate file detected")
        self.uploaded_hashes.add(file_hash)
        if file_ext == '.csv':
            await self._validate_csv_structure(content)
        return True
    
    async def _validate_file_signature(self, content: bytes, file_ext: str):
        if file_ext == '.csv':
            if not content[:100].decode('utf-8', errors='ignore').isprintable():
                raise HTTPException(status_code=400, detail="Invalid CSV format")
        elif file_ext == '.xlsx':
            if not content.startswith(b'PK'):
                raise HTTPException(status_code=400, detail="Invalid XLSX format")
        elif file_ext == '.json':
            try:
                import json
                json.loads(content.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
    
    async def _validate_csv_structure(self, content: bytes):
        try:
            df = pd.read_csv(io.BytesIO(content))
            if len(df.columns) == 0:
                raise HTTPException(status_code=400, detail="CSV has no columns")
            if len(df) == 0:
                raise HTTPException(status_code=400, detail="CSV has no data rows")
            if len(df.columns) > 1000:
                raise HTTPException(status_code=400, detail="Too many columns (max: 1000)")
            if len(df) > 1000000:
                raise HTTPException(status_code=400, detail="Too many rows (max: 1,000,000)")
        except pd.errors.EmptyDataError:
            raise HTTPException(status_code=400, detail="Empty CSV file")
        except pd.errors.ParserError as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")

file_validator = EnhancedFileValidator()

def validate_preprocessing_params(missing_strategy: str, encoding: str, target_column: str = None) -> bool:
    allowed_missing = ["mean", "median", "mode", "drop"]
    if missing_strategy not in allowed_missing:
        raise HTTPException(status_code=422, detail=f"Invalid missing strategy: {missing_strategy}")
    allowed_encoding = ["onehot", "label", "target", "kfold"]
    if encoding not in allowed_encoding:
        raise HTTPException(status_code=422, detail=f"Invalid encoding method: {encoding}")
    if encoding in ["target", "kfold"] and not target_column:
        raise HTTPException(status_code=422, detail=f"Target column required for {encoding}")
    return True