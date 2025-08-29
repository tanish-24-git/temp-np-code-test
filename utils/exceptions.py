from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

class DatasetError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class ModelTrainingError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class PreprocessingError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class ValidationError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

async def general_exception_handler(request: Request, exc: Exception):
    print(f"Unhandled error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": f"An error occurred: {str(exc)}",
            "path": str(request.url.path)
        }
    )