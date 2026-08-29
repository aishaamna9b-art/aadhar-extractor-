import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import logging

# We will import the extractor function once we create extractor.py
from extractor import process_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Aadhaar Extractor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For local testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "application/pdf"}

@app.post("/extract")
async def extract_aadhaar(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Invalid content type. Only JPEG, PNG, and PDF are allowed.")
    
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds 10MB limit.")
        
    try:
        # Process the file entirely in-memory
        result = process_file(file_bytes, file.content_type)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error during extraction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during processing.")
    finally:
        # ensure variables are deleted to encourage garbage collection of in-memory buffers
        del file_bytes

# Mount the static directory for the frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")
