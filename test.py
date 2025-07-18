from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pyttsx3
import threading
from typing import Optional, List, Dict
import logging
import io
import PyPDF2
from docx import Document
import json
import csv
import os
import shutil
from pathlib import Path
import requests
from tqdm import tqdm
import asyncio
import aiohttp
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import time
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NLLB Translation API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class TranslationRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

class TranslationResponse(BaseModel):
    translated_text: str
    source_language: str
    target_language: str

class TTSRequest(BaseModel):
    text: str
    language: str

class FileUploadResponse(BaseModel):
    content: str
    filename: str
    file_type: str

class ModelLoadRequest(BaseModel):
    model_path: str
    device: str = "auto"  # auto, gpu, cpu

class ModelLoadResponse(BaseModel):
    success: bool
    model_name: str
    model_size: str
    device: str
    message: str

class ModelStatusResponse(BaseModel):
    model_loaded: bool
    current_model: str
    current_device: str
    gpu_available: bool
    gpu_memory: Optional[str] = None
    model_info: Optional[dict] = None

class DownloadModelRequest(BaseModel):
    modelId: str
    modelInfo: dict

class DownloadProgressResponse(BaseModel):
    modelId: str
    progress: float
    status: str  # downloading, paused, completed, error, cancelled
    downloadedSize: Optional[str] = None
    totalSize: Optional[str] = None
    speed: Optional[str] = None
    eta: Optional[str] = None
    error: Optional[str] = None

class LocalModel(BaseModel):
    id: str
    name: str
    path: str
    size: str
    isDownloaded: bool
    lastUsed: Optional[str] = None

class CheckModelRequest(BaseModel):
    modelId: str

class DeleteModelRequest(BaseModel):
    modelId: str

class PauseDownloadRequest(BaseModel):
    modelId: str

class ResumeDownloadRequest(BaseModel):
    modelId: str

class CancelDownloadRequest(BaseModel):
    modelId: str

class SetupStatusResponse(BaseModel):
    setup_needed: bool
    downloaded_models: List[str]
    models_count: int

# Global variables for model and tokenizer
model = None
tokenizer = None
tts_engine = None
current_model_path = "facebook/nllb-200-1.3B"  # Default model
current_device = "auto"

# Model storage directory (for Tauri compatibility)
MODELS_DIR = Path.home() / ".nllb_translator" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Download management
download_tasks: Dict[str, dict] = {}
download_progress: Dict[str, DownloadProgressResponse] = {}
executor = ThreadPoolExecutor(max_workers=4)

# Audio file tracking for manual cleanup
active_audio_files: Dict[str, float] = {}  # filename -> creation_time

# Comprehensive NLLB language codes mapping
LANGUAGE_CODES = {
    # Major World Languages
    "eng_Latn": "English",
    "spa_Latn": "Spanish",
    "fra_Latn": "French",
    "deu_Latn": "German",
    "ita_Latn": "Italian",
    "por_Latn": "Portuguese",
    "rus_Cyrl": "Russian",
    "jpn_Jpan": "Japanese",
    "kor_Hang": "Korean",
    "zho_Hans": "Chinese (Simplified)",
    "zho_Hant": "Chinese (Traditional)",
    "arb_Arab": "Arabic",
    
    # Indian Languages
    "hin_Deva": "Hindi",
    "ben_Beng": "Bengali",
    "tel_Telu": "Telugu",
    "mar_Deva": "Marathi",
    "tam_Taml": "Tamil",
    "urd_Arab": "Urdu",
    "guj_Gujr": "Gujarati",
    "kan_Knda": "Kannada",
    "mal_Mlym": "Malayalam",
    "pan_Guru": "Punjabi",
    "ori_Orya": "Odia",
    "asm_Beng": "Assamese",
    "nep_Deva": "Nepali",
    "sin_Sinh": "Sinhala",
    
    # Southeast Asian Languages
    "tha_Thai": "Thai",
    "vie_Latn": "Vietnamese",
    "ind_Latn": "Indonesian",
    "msa_Latn": "Malay",
    "tgl_Latn": "Filipino",
    "mya_Mymr": "Burmese",
    "khm_Khmr": "Khmer",
    "lao_Laoo": "Lao",
    
    # European Languages
    "nld_Latn": "Dutch",
    "swe_Latn": "Swedish",
    "dan_Latn": "Danish",
    "nor_Latn": "Norwegian",
    "fin_Latn": "Finnish",
    "pol_Latn": "Polish",
    "ces_Latn": "Czech",
    "hun_Latn": "Hungarian",
    "ron_Latn": "Romanian",
    "bul_Cyrl": "Bulgarian",
    "hrv_Latn": "Croatian",
    "srp_Cyrl": "Serbian",
    "slk_Latn": "Slovak",
    "slv_Latn": "Slovenian",
    "est_Latn": "Estonian",
    "lav_Latn": "Latvian",
    "lit_Latn": "Lithuanian",
    "ell_Grek": "Greek",
    
    # Middle Eastern Languages
    "fas_Arab": "Persian",
    "tur_Latn": "Turkish",
    "heb_Hebr": "Hebrew",
    "kur_Arab": "Kurdish",
    "aze_Latn": "Azerbaijani",
    
    # African Languages
    "swa_Latn": "Swahili",
    "hau_Latn": "Hausa",
    "yor_Latn": "Yoruba",
    "ibo_Latn": "Igbo",
    "amh_Ethi": "Amharic",
    "som_Latn": "Somali",
    "afr_Latn": "Afrikaans",
    
    # Other Languages
    "ukr_Cyrl": "Ukrainian",
    "bel_Cyrl": "Belarusian",
    "kaz_Cyrl": "Kazakh",
    "uzb_Latn": "Uzbek",
    "tgk_Cyrl": "Tajik",
    "mon_Cyrl": "Mongolian",
    "cat_Latn": "Catalan",
    "eus_Latn": "Basque",
    "glg_Latn": "Galician",
    "isl_Latn": "Icelandic",
    "mlt_Latn": "Maltese",
    
    # Additional South Asian
    "pus_Arab": "Pashto",
    "snd_Arab": "Sindhi",
    "bod_Tibt": "Tibetan",
    
    # Additional European
    "gle_Latn": "Irish",
    "cym_Latn": "Welsh",
    "bre_Latn": "Breton",
    
    # Additional African
    "xho_Latn": "Xhosa",
    "zul_Latn": "Zulu",
    
    # Additional Asian
    "tuk_Latn": "Turkmen",
    "kir_Cyrl": "Kyrgyz",
}

def get_gpu_info():
    """Get GPU information"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
        gpu_memory = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB" if gpu_count > 0 else "Unknown"
        return {
            "available": True,
            "count": gpu_count,
            "name": gpu_name,
            "memory": gpu_memory
        }
    return {"available": False}

def get_model_info(model_path: str) -> dict:
    """Get information about the model"""
    model_info = {
        "name": "Unknown Model",
        "size": "Unknown",
        "parameters": "Unknown"
    }
    
    if "distilled-600M" in model_path:
        model_info = {
            "name": "NLLB-200 Distilled 600M",
            "size": "~2.4GB",
            "parameters": "600M"
        }
    elif "1.3B" in model_path:
        model_info = {
            "name": "NLLB-200 1.3B",
            "size": "~5.2GB",
            "parameters": "1.3B"
        }
    elif "3.3B" in model_path:
        model_info = {
            "name": "NLLB-200 3.3B",
            "size": "~13GB",
            "parameters": "3.3B"
        }
    else:
        # For custom models, try to extract name from path
        model_name = os.path.basename(model_path) if os.path.exists(model_path) else model_path.split('/')[-1]
        model_info["name"] = f"Custom Model: {model_name}"
    
    return model_info

def determine_device(device_preference: str) -> str:
    """Determine the actual device to use based on preference and availability"""
    if device_preference == "cpu":
        return "cpu"
    elif device_preference == "gpu":
        if torch.cuda.is_available():
            return "cuda"
        else:
            logger.warning("GPU requested but not available, falling back to CPU")
            return "cpu"
    else:  # auto
        return "cuda" if torch.cuda.is_available() else "cpu"

def get_model_directory(model_id: str) -> Path:
    """Get the directory path for a specific model (for downloaded models)"""
    return MODELS_DIR / model_id.replace("/", "_")

def format_bytes(bytes_value: int) -> str:
    """Format bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f}{unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f}TB"

def calculate_speed(bytes_downloaded: int, elapsed_time: float) -> str:
    """Calculate download speed"""
    if elapsed_time <= 0:
        return "0 MB/s"
    speed_bps = bytes_downloaded / elapsed_time
    return f"{format_bytes(speed_bps)}/s"

def calculate_eta(bytes_downloaded: int, total_bytes: int, elapsed_time: float) -> str:
    """Calculate estimated time of arrival"""
    if elapsed_time <= 0 or bytes_downloaded <= 0:
        return "Calculating..."
    
    speed_bps = bytes_downloaded / elapsed_time
    remaining_bytes = total_bytes - bytes_downloaded
    
    if speed_bps <= 0:
        return "Unknown"
    
    eta_seconds = remaining_bytes / speed_bps
    
    if eta_seconds < 60:
        return f"{int(eta_seconds)}s"
    elif eta_seconds < 3600:
        return f"{int(eta_seconds / 60)}m {int(eta_seconds % 60)}s"
    else:
        hours = int(eta_seconds / 3600)
        minutes = int((eta_seconds % 3600) / 60)
        return f"{hours}h {minutes}m"

async def download_file_with_progress(url: str, filepath: Path, model_id: str, session: aiohttp.ClientSession):
    """Download a file with progress tracking"""
    try:
        start_time = time.time()
        bytes_downloaded = 0
        
        async with session.get(url) as response:
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            
            # Update initial progress
            download_progress[model_id] = DownloadProgressResponse(
                modelId=model_id,
                progress=0.0,
                status="downloading",
                downloadedSize="0MB",
                totalSize=format_bytes(total_size),
                speed="0 MB/s",
                eta="Calculating..."
            )
            
            async with aiofiles.open(filepath, 'wb') as file:
                async for chunk in response.content.iter_chunked(8192):
                    # Check if download was cancelled or paused
                    if model_id in download_tasks:
                        task_info = download_tasks[model_id]
                        if task_info.get("cancelled", False):
                            logger.info(f"Download cancelled for {model_id}")
                            return False
                        
                        while task_info.get("paused", False):
                            await asyncio.sleep(0.1)
                            if task_info.get("cancelled", False):
                                return False
                    
                    await file.write(chunk)
                    bytes_downloaded += len(chunk)
                    
                    # Update progress every 1MB or at the end
                    if bytes_downloaded % (1024 * 1024) == 0 or bytes_downloaded == total_size:
                        elapsed_time = time.time() - start_time
                        progress = (bytes_downloaded / total_size) * 100 if total_size > 0 else 0
                        speed = calculate_speed(bytes_downloaded, elapsed_time)
                        eta = calculate_eta(bytes_downloaded, total_size, elapsed_time)
                        
                        download_progress[model_id] = DownloadProgressResponse(
                            modelId=model_id,
                            progress=progress,
                            status="downloading",
                            downloadedSize=format_bytes(bytes_downloaded),
                            totalSize=format_bytes(total_size),
                            speed=speed,
                            eta=eta
                        )
        
        return True
        
    except Exception as e:
        logger.error(f"Download error for {model_id}: {str(e)}")
        download_progress[model_id] = DownloadProgressResponse(
            modelId=model_id,
            progress=0.0,
            status="error",
            error=str(e)
        )
        return False

async def download_model_files(model_id: str, model_info: dict):
    """Download all files for a model"""
    try:
        model_dir = get_model_directory(model_id)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize download task
        download_tasks[model_id] = {
            "paused": False,
            "cancelled": False,
            "start_time": time.time()
        }
        
        logger.info(f"Starting download for model {model_id}")
        
        # Use transformers library to download the model
        # This is more reliable than manual file downloads
        async with aiohttp.ClientSession() as session:
            try:
                # Download using transformers in a thread to avoid blocking
                def download_with_transformers():
                    try:
                        # Download tokenizer
                        tokenizer = AutoTokenizer.from_pretrained(model_id)
                        tokenizer.save_pretrained(str(model_dir))
                        
                        # Download model
                        model_obj = AutoModelForSeq2SeqLM.from_pretrained(model_id)
                        model_obj.save_pretrained(str(model_dir))
                        
                        return True
                    except Exception as e:
                        logger.error(f"Transformers download error: {str(e)}")
                        return False
                
                # Run download in thread with progress simulation
                loop = asyncio.get_event_loop()
                
                # Simulate progress for transformers download
                total_steps = 100
                for step in range(total_steps + 1):
                    # Check for cancellation
                    if model_id in download_tasks and download_tasks[model_id].get("cancelled", False):
                        logger.info(f"Download cancelled for {model_id}")
                        return False
                    
                    # Check for pause
                    while model_id in download_tasks and download_tasks[model_id].get("paused", False):
                        await asyncio.sleep(0.1)
                        if download_tasks[model_id].get("cancelled", False):
                            return False
                    
                    progress = step
                    elapsed_time = time.time() - download_tasks[model_id]["start_time"]
                    
                    # Estimate size based on model info
                    size_str = model_info.get("size", "5.2GB")
                    size_gb = float(size_str.replace("GB", ""))
                    total_bytes = int(size_gb * 1024 * 1024 * 1024)
                    downloaded_bytes = int((progress / 100) * total_bytes)
                    
                    speed = calculate_speed(downloaded_bytes, elapsed_time) if elapsed_time > 0 else "0 MB/s"
                    eta = calculate_eta(downloaded_bytes, total_bytes, elapsed_time) if progress < 100 else "Complete"
                    
                    download_progress[model_id] = DownloadProgressResponse(
                        modelId=model_id,
                        progress=float(progress),
                        status="downloading",
                        downloadedSize=format_bytes(downloaded_bytes),
                        totalSize=format_bytes(total_bytes),
                        speed=speed,
                        eta=eta
                    )
                    
                    if step == 50:  # Start actual download at 50%
                        success = await loop.run_in_executor(executor, download_with_transformers)
                        if not success:
                            raise Exception("Failed to download model files")
                    
                    await asyncio.sleep(0.1)  # Small delay for progress updates
                
                # Mark as completed
                download_progress[model_id] = DownloadProgressResponse(
                    modelId=model_id,
                    progress=100.0,
                    status="completed",
                    downloadedSize=format_bytes(total_bytes),
                    totalSize=format_bytes(total_bytes),
                    speed="Complete",
                    eta="Complete"
                )
                
                logger.info(f"Download completed for {model_id}")
                return True
                
            except Exception as e:
                logger.error(f"Download failed for {model_id}: {str(e)}")
                download_progress[model_id] = DownloadProgressResponse(
                    modelId=model_id,
                    progress=0.0,
                    status="error",
                    error=str(e)
                )
                return False
                
    except Exception as e:
        logger.error(f"Download setup failed for {model_id}: {str(e)}")
        download_progress[model_id] = DownloadProgressResponse(
            modelId=model_id,
            progress=0.0,
            status="error",
            error=str(e)
        )
        return False
    finally:
        # Clean up download task
        if model_id in download_tasks:
            del download_tasks[model_id]

def load_model(model_path: str = None, device_preference: str = "auto"):
    """Load the NLLB model and tokenizer"""
    global model, tokenizer, current_model_path, current_device
    
    if model_path:
        current_model_path = model_path
    
    # Determine actual device to use
    actual_device = determine_device(device_preference)
    current_device = actual_device
    
    try:
        logger.info(f"Loading NLLB model from: {current_model_path}")
        logger.info(f"Target device: {actual_device}")
        
        # Clear existing model from memory
        if model is not None:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Check if it's a local directory path provided by the user
        # This handles both downloaded models (via get_model_directory) and custom local paths
        if Path(current_model_path).is_dir() and (Path(current_model_path) / "config.json").exists():
            # Load from the provided local directory
            tokenizer = AutoTokenizer.from_pretrained(str(Path(current_model_path)))
            model = AutoModelForSeq2SeqLM.from_pretrained(str(Path(current_model_path)))
            logger.info(f"Loaded model from local directory: {current_model_path}")
        else:
            # Assume it's a Hugging Face model ID or a path that needs to be downloaded/cached by transformers
            tokenizer = AutoTokenizer.from_pretrained(current_model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(current_model_path)
            logger.info(f"Loaded model from Hugging Face or cached: {current_model_path}")
        
        # Move to specified device
        if actual_device == "cuda":
            if torch.cuda.is_available():
                model = model.to("cuda")
                logger.info(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")
            else:
                logger.error("CUDA not available, cannot load on GPU")
                raise Exception("CUDA not available")
        else:
            model = model.to("cpu")
            logger.info("Model loaded on CPU")
            
        logger.info(f"NLLB model loaded successfully: {current_model_path} on {actual_device}")
        return True, actual_device
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e

def initialize_tts():
    """Initialize the TTS engine with better error handling"""
    global tts_engine
    try:
        import pyttsx3
        tts_engine = pyttsx3.init(driverName='sapi5' if os.name == 'nt' else None)
        
        # Test the engine
        voices = tts_engine.getProperty('voices')
        if not voices:
            logger.warning("No TTS voices found, TTS may not work properly")
        else:
            logger.info(f"TTS initialized with {len(voices)} voices")
            
        # Set properties with error handling
        try:
            tts_engine.setProperty('rate', 180)
            tts_engine.setProperty('volume', 0.9)
        except Exception as e:
            logger.warning(f"Could not set TTS properties: {str(e)}")
            
        logger.info("TTS engine initialized successfully!")
        
    except Exception as e:
        logger.error(f"Error initializing TTS: {str(e)}")
        logger.info("TTS will use fallback methods")
        tts_engine = None

def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file"""
    try:
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting PDF text: {str(e)}")
        raise HTTPException(status_code=400, detail="Failed to extract text from PDF")

def extract_text_from_docx(file_content: bytes) -> str:
    """Extract text from DOCX file"""
    try:
        doc_file = io.BytesIO(file_content)
        doc = Document(doc_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting DOCX text: {str(e)}")
        raise HTTPException(status_code=400, detail="Failed to extract text from DOCX")

async def cleanup_old_audio_files():
    """Clean up audio files older than 5 minutes"""
    try:
        temp_dir = os.path.join(os.getcwd(), "temp_audio")
        if not os.path.exists(temp_dir):
            return
        
        current_time = time.time()
        files_to_remove = []
        
        # Check active files for old ones
        for filename, creation_time in list(active_audio_files.items()):
            if current_time - creation_time > 300:  # 5 minutes
                files_to_remove.append(filename)
        
        # Remove old files
        for filename in files_to_remove:
            file_path = os.path.join(temp_dir, filename)
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Auto-cleaned old audio file: {filename}")
                del active_audio_files[filename]
            except Exception as e:
                logger.error(f"Failed to auto-clean {filename}: {str(e)}")
                
    except Exception as e:
        logger.error(f"Audio cleanup error: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize TTS on startup (don't load model automatically)"""
    initialize_tts()
    logger.info("Server started. Model loading is now manual via /load-model endpoint.")
    
    # Start periodic cleanup task
    async def periodic_cleanup():
        while True:
            await asyncio.sleep(60)  # Run every minute
            await cleanup_old_audio_files()
    
    asyncio.create_task(periodic_cleanup())

@app.get("/")
async def root():
    return {"message": "NLLB Translation API is running!"}

@app.get("/health")
async def health_check():
    gpu_info = get_gpu_info()
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tts_available": tts_engine is not None,
        "supported_languages": len(LANGUAGE_CODES),
        "current_model": current_model_path if model is not None else None,
        "current_device": current_device if model is not None else None,
        "gpu_available": gpu_info["available"],
        "gpu_info": gpu_info if gpu_info["available"] else None,
        "active_downloads": len(download_tasks),
        "download_progress": len(download_progress),
        "active_audio_files": len(active_audio_files)
    }

@app.get("/model-status", response_model=ModelStatusResponse)
async def get_model_status():
    """Get current model status"""
    gpu_info = get_gpu_info()
    return ModelStatusResponse(
        model_loaded=model is not None,
        current_model=current_model_path if model is not None else "",
        current_device=current_device if model is not None else "",
        gpu_available=gpu_info["available"],
        gpu_memory=gpu_info.get("memory") if gpu_info["available"] else None,
        model_info=get_model_info(current_model_path) if model is not None else None
    )

@app.get("/setup-status", response_model=SetupStatusResponse)
async def get_setup_status():
    """Check if first-time setup is needed"""
    try:
        downloaded_models = []
        
        if MODELS_DIR.exists():
            for model_dir in MODELS_DIR.iterdir():
                if model_dir.is_dir():
                    config_file = model_dir / "config.json"
                    if config_file.exists():
                        # Check for model files (pytorch_model.bin or model.safetensors)
                        model_files = ["pytorch_model.bin", "model.safetensors"]
                        if any((model_dir / file).exists() for file in model_files):
                            downloaded_models.append(model_dir.name.replace("_", "/"))
        
        return SetupStatusResponse(
            setup_needed=len(downloaded_models) == 0,
            downloaded_models=downloaded_models,
            models_count=len(downloaded_models)
        )
        
    except Exception as e:
        logger.error(f"Setup status error: {str(e)}")
        return SetupStatusResponse(
            setup_needed=True,
            downloaded_models=[],
            models_count=0
        )

@app.post("/load-model", response_model=ModelLoadResponse)
async def load_model_endpoint(request: ModelLoadRequest):
    """Load a specific NLLB model"""
    try:
        model_path = request.model_path.strip()
        device_preference = request.device
        
        if not model_path:
            raise HTTPException(status_code=400, detail="Model path cannot be empty")
        
        # Validate device preference
        if device_preference not in ["auto", "gpu", "cpu"]:
            raise HTTPException(status_code=400, detail="Device must be 'auto', 'gpu', or 'cpu'")
        
        # Check GPU availability if GPU is requested
        if device_preference == "gpu" and not torch.cuda.is_available():
            raise HTTPException(
                status_code=400, 
                detail="GPU requested but CUDA is not available. Please install CUDA or use 'auto' or 'cpu'."
            )
        
        # Load the model
        success, actual_device = load_model(model_path, device_preference)
        
        if success:
            model_info = get_model_info(model_path)
            return ModelLoadResponse(
                success=True,
                model_name=model_info["name"],
                model_size=model_info["size"],
                device=actual_device,
                message=f"Successfully loaded {model_info['name']} on {actual_device.upper()}"
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to load model")
            
    except Exception as e:
        logger.error(f"Model loading error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.get("/languages")
async def get_supported_languages():
    """Get list of supported languages"""
    return {"languages": LANGUAGE_CODES}

@app.post("/check-model-exists")
async def check_model_exists(request: CheckModelRequest):
    """Check if a model exists locally"""
    try:
        model_dir = get_model_directory(request.modelId)
        
        if not model_dir.exists():
            return {"exists": False}
        
        # Check if all required files exist
        required_files = ["config.json", "tokenizer.json"]
        for file in required_files:
            if not (model_dir / file).exists():
                return {"exists": False}
        
        # Check for model files (pytorch_model.bin or model.safetensors)
        model_files = ["pytorch_model.bin", "model.safetensors"]
        if not any((model_dir / file).exists() for file in model_files):
            return {"exists": False}
        
        return {"exists": True}
        
    except Exception as e:
        logger.error(f"Error checking model existence: {str(e)}")
        return {"exists": False}

@app.get("/get-local-models", response_model=List[LocalModel])
async def get_local_models():
    """Get list of locally downloaded models"""
    try:
        local_models = []
        
        if not MODELS_DIR.exists():
            return local_models
        
        for model_dir in MODELS_DIR.iterdir():
            if model_dir.is_dir():
                config_file = model_dir / "config.json"
                if config_file.exists():
                    # Check for model files
                    model_files = ["pytorch_model.bin", "model.safetensors"]
                    if any((model_dir / file).exists() for file in model_files):
                        # Calculate directory size
                        total_size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
                        size_gb = total_size / (1024**3)
                        
                        # Get model name from directory name
                        model_id = model_dir.name.replace("_", "/")
                        model_info = get_model_info(model_id)
                        
                        local_models.append(LocalModel(
                            id=model_id,
                            name=model_info["name"],
                            path=str(model_dir),
                            size=f"{size_gb:.1f}GB",
                            isDownloaded=True,
                            lastUsed=None  # Could be implemented with metadata
                        ))
        
        return local_models
        
    except Exception as e:
        logger.error(f"Error getting local models: {str(e)}")
        return []

@app.post("/download-model")
async def download_model(request: DownloadModelRequest, background_tasks: BackgroundTasks):
    """Download a model for offline use"""
    try:
        model_id = request.modelId
        model_info = request.modelInfo
        
        # Check if already downloading
        if model_id in download_tasks:
            raise HTTPException(status_code=400, detail="Model is already being downloaded")
        
        # Check if already exists
        model_dir = get_model_directory(model_id)
        if model_dir.exists():
            config_file = model_dir / "config.json"
            if config_file.exists():
                model_files = ["pytorch_model.bin", "model.safetensors"]
                if any((model_dir / file).exists() for file in model_files):
                    return {
                        "success": True,
                        "message": f"Model {model_id} already exists",
                        "path": str(model_dir)
                    }
        
        logger.info(f"Starting download for model {model_id}")
        
        # Start download in background
        background_tasks.add_task(download_model_files, model_id, model_info)
        
        return {
            "success": True,
            "message": f"Download started for {model_id}",
            "path": str(model_dir)
        }
        
    except Exception as e:
        logger.error(f"Download model error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start download: {str(e)}")

@app.post("/pause-download")
async def pause_download(request: PauseDownloadRequest):
    """Pause an active download"""
    try:
        model_id = request.modelId
        
        if model_id not in download_tasks:
            raise HTTPException(status_code=404, detail="Download not found")
        
        download_tasks[model_id]["paused"] = True
        
        if model_id in download_progress:
            download_progress[model_id].status = "paused"
        
        logger.info(f"Download paused for {model_id}")
        return {"success": True, "message": f"Download paused for {model_id}"}
        
    except Exception as e:
        logger.error(f"Pause download error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to pause download: {str(e)}")

@app.post("/resume-download")
async def resume_download(request: ResumeDownloadRequest):
    """Resume a paused download"""
    try:
        model_id = request.modelId
        
        if model_id not in download_tasks:
            raise HTTPException(status_code=404, detail="Download not found")
        
        download_tasks[model_id]["paused"] = False
        
        if model_id in download_progress:
            download_progress[model_id].status = "downloading"
        
        logger.info(f"Download resumed for {model_id}")
        return {"success": True, "message": f"Download resumed for {model_id}"}
        
    except Exception as e:
        logger.error(f"Resume download error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to resume download: {str(e)}")

@app.post("/cancel-download")
async def cancel_download(request: CancelDownloadRequest):
    """Cancel an active download"""
    try:
        model_id = request.modelId
        
        if model_id not in download_tasks:
            raise HTTPException(status_code=404, detail="Download not found")
        
        # Mark as cancelled
        download_tasks[model_id]["cancelled"] = True
        
        # Clean up partial download
        model_dir = get_model_directory(model_id)
        if model_dir.exists():
            shutil.rmtree(model_dir)
        
        # Remove from progress tracking
        if model_id in download_progress:
            del download_progress[model_id]
        
        logger.info(f"Download cancelled for {model_id}")
        return {"success": True, "message": f"Download cancelled for {model_id}"}
        
    except Exception as e:
        logger.error(f"Cancel download error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel download: {str(e)}")

@app.get("/download-progress/{model_id}", response_model=DownloadProgressResponse)
async def get_download_progress(model_id: str):
    """Get download progress for a specific model"""
    try:
        if model_id not in download_progress:
            raise HTTPException(status_code=404, detail="Download progress not found")
        
        return download_progress[model_id]
        
    except Exception as e:
        logger.error(f"Get download progress error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get download progress: {str(e)}")

@app.get("/download-progress")
async def get_all_download_progress():
    """Get download progress for all models"""
    try:
        return {"downloads": download_progress}
        
    except Exception as e:
        logger.error(f"Get all download progress error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get download progress: {str(e)}")

@app.post("/delete-model")
async def delete_model(request: DeleteModelRequest):
    """Delete a locally downloaded model"""
    try:
        model_dir = get_model_directory(request.modelId)
        
        if model_dir.exists():
            shutil.rmtree(model_dir)
            logger.info(f"Deleted model directory: {model_dir}")
            return {"success": True, "message": f"Model {request.modelId} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Model not found")
            
    except Exception as e:
        logger.error(f"Delete model error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")

@app.post("/upload-file", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload and extract text from file"""
    try:
        # Check file size (limit to 10MB)
        if file.size and file.size > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size too large (max 10MB)")
        
        content = await file.read()
        filename = file.filename or "unknown"
        file_extension = filename.split('.')[-1].lower()
        
        extracted_text = ""
        
        if file_extension == 'pdf':
            extracted_text = extract_text_from_pdf(content)
        elif file_extension in ['docx', 'doc']:
            extracted_text = extract_text_from_docx(content)
        elif file_extension == 'txt':
            extracted_text = content.decode('utf-8')
        elif file_extension == 'json':
            json_content = content.decode('utf-8')
            try:
                parsed = json.loads(json_content)
                extracted_text = json.dumps(parsed, indent=2)
            except:
                extracted_text = json_content
        elif file_extension == 'csv':
            csv_content = content.decode('utf-8')
            extracted_text = csv_content
        else:
            # Try to decode as text
            try:
                extracted_text = content.decode('utf-8')
            except:
                raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Limit content length
        if len(extracted_text) > 50000:
            extracted_text = extracted_text[:50000] + "..."
        
        return FileUploadResponse(
            content=extracted_text,
            filename=filename,
            file_type=file_extension
        )
        
    except Exception as e:
        logger.error(f"File upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")

@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """Translate text using NLLB-200 model"""
    try:
        if model is None or tokenizer is None:
            raise HTTPException(status_code=500, detail="Model not loaded. Please load a model first using the model settings.")
        
        # Validate language codes
        if request.source_lang not in LANGUAGE_CODES:
            raise HTTPException(status_code=400, detail=f"Unsupported source language: {request.source_lang}")
        
        if request.target_lang not in LANGUAGE_CODES:
            raise HTTPException(status_code=400, detail=f"Unsupported target language: {request.target_lang}")
        
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Add character limit validation
        if len(request.text) > 50000:
            raise HTTPException(status_code=400, detail="Text exceeds maximum length of 50,000 characters")
        
        logger.info(f"Translating from {request.source_lang} to {request.target_lang} on {current_device}")

        # Set source and target language codes for the tokenizer (required for NLLB Fast tokenizer)
        try:
            tokenizer.src_lang = request.source_lang
            tokenizer.tgt_lang = request.target_lang
        except AttributeError:
            # Older tokenizer versions may not support these attributes; ignore in that case
            pass

        # Tokenize input text with increased max length
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )

        # Move inputs to same device as model
        if current_device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Determine BOS token id for target language in a tokenizer-agnostic way
        try:
            forced_bos_token_id = tokenizer.lang_code_to_id[request.target_lang]
        except AttributeError:
            forced_bos_token_id = tokenizer.convert_tokens_to_ids(request.target_lang)

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=2048,  # Increased from 512 to 2048
                num_beams=5,
                early_stopping=True
            )
        
        # Decode the translation
        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        
        logger.info("Translation completed successfully")
        
        return TranslationResponse(
            translated_text=translated_text,
            source_language=LANGUAGE_CODES[request.source_lang],
            target_language=LANGUAGE_CODES[request.target_lang]
        )
        
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import asyncio
import os
import uuid
import time
import platform
import subprocess
import numpy as np
import wave
import logging
from concurrent.futures import ThreadPoolExecutor

# Create logger
logger = logging.getLogger("main")
executor = ThreadPoolExecutor(max_workers=2)
active_audio_files = {}

# Dummy language mapping
LANGUAGE_CODES = {
    "eng_Latn": "English",
    "hin_Deva": "Hindi"
}

# pyttsx3 engine
import pyttsx3
tts_engine = pyttsx3.init()

# Pydantic model for TTS request
class TTSRequest(BaseModel):
    text: str
    language: str  # "eng_Latn" or "hin_Deva"
@app.post("/text-to-speech")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech using PowerShell TTS (Windows only)"""

    if request.language not in ["eng_Latn", "hin_Deva"]:
        raise HTTPException(status_code=400, detail="TTS only supports English and Hindi")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if len(request.text) > 500:
        raise HTTPException(status_code=400, detail="Text too long for TTS (max 500 characters)")

    # Create temp audio directory
    temp_dir = os.path.join(os.getcwd(), "temp_audio")
    os.makedirs(temp_dir, exist_ok=True)

    # File paths
    audio_filename = f"tts_{uuid.uuid4().hex[:8]}.wav"
    audio_path = os.path.join(temp_dir, audio_filename)

    def powershell_tts():
        try:
            escaped_text = request.text.replace("'", "''")
            safe_path = audio_path.replace("\\", "\\\\")
            ps_script = f"""
            Add-Type -AssemblyName System.Speech;
            $s = New-Object System.Speech.Synthesis.SpeechSynthesizer;
            $s.Volume = 100;
            $s.Rate = 0;
            $s.SetOutputToWaveFile("{safe_path}");
            $s.Speak('{escaped_text}');
            $s.Dispose()
            """
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_script],
                capture_output=True, text=True, timeout=10
            )

            logger.warning(f"[PowerShell stdout]: {result.stdout.strip()}")
            logger.warning(f"[PowerShell stderr]: {result.stderr.strip()}")

            return os.path.exists(audio_path) and os.path.getsize(audio_path) > 1024

        except Exception as e:
            logger.error(f"TTS PowerShell error: {e}")
            return False

    def generate_beep():
        try:
            sample_rate = 22050
            duration = 0.5
            frequency = 440
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio_data = (np.sin(2 * np.pi * frequency * t) * 0.3 * 32767).astype(np.int16)

            with wave.open(audio_path, 'w') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data.tobytes())
            return True
        except Exception as e:
            logger.error(f"Beep generation failed: {e}")
            return False

    logger.info("🌐 Running system TTS fallback on windows")
    success = await asyncio.get_event_loop().run_in_executor(None, powershell_tts)

    # Wait a bit for the file to fully flush
    max_wait_time = 5.0
    poll_interval = 0.1
    for _ in range(int(max_wait_time / poll_interval)):
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1024:
            break
        time.sleep(poll_interval)

    if not os.path.exists(audio_path) or os.path.getsize(audio_path) <= 1024:
        logger.warning("❌ TTS file too small or not generated — falling back to beep...")
        success = await asyncio.get_event_loop().run_in_executor(None, generate_beep)

    if not success or not os.path.exists(audio_path):
        raise HTTPException(status_code=500, detail="Failed to generate TTS audio")

    active_audio_files[audio_filename] = time.time()
    logger.info(f"✅ Audio file created and tracked: {audio_filename}")

    return {
        "message": "TTS generated",
        "language": LANGUAGE_CODES[request.language],
        "audio_file": audio_filename,
        "audio_path": audio_path,
        "text_length": len(request.text)
    }



@app.get("/audio/{filename}")
async def get_audio_file(filename: str):
    """Serve generated audio files"""
    try:
        import os
        from fastapi.responses import FileResponse
        
        temp_dir = os.path.join(os.getcwd(), "temp_audio")
        audio_path = os.path.join(temp_dir, filename)
        
        if not os.path.exists(audio_path):
            logger.warning(f"Audio file not found: {filename}")
            raise HTTPException(status_code=404, detail="Audio file not found")
        
        logger.info(f"Serving audio file: {filename}")
        return FileResponse(
            audio_path,
            media_type="audio/wav",
            filename=filename,
            headers={"Cache-Control": "no-cache"}
        )
        
    except Exception as e:
        logger.error(f"Audio serve error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to serve audio file")

@app.delete("/audio/{filename}")
async def delete_audio_file(filename: str):
    """Delete temporary audio file (manual cleanup only)"""
    try:
        import os
        
        temp_dir = os.path.join(os.getcwd(), "temp_audio")
        audio_path = os.path.join(temp_dir, filename)
        
        if os.path.exists(audio_path):
            os.remove(audio_path)
            logger.info(f"Audio file manually deleted: {filename}")
            
            # Remove from tracking
            if filename in active_audio_files:
                del active_audio_files[filename]
                
            return {"message": "Audio file deleted successfully"}
        else:
            logger.info(f"Audio file not found for deletion: {filename}")
            
            # Remove from tracking even if file doesn't exist
            if filename in active_audio_files:
                del active_audio_files[filename]
                
            return {"message": "Audio file not found (may have been already deleted)"}
            
    except Exception as e:
        logger.error(f"Audio delete error: {str(e)}")
        return {"message": "Audio file cleanup completed"}

@app.get("/audio-files")
async def list_audio_files():
    """List active audio files (for debugging)"""
    try:
        temp_dir = os.path.join(os.getcwd(), "temp_audio")
        if not os.path.exists(temp_dir):
            return {"active_files": [], "tracked_files": list(active_audio_files.keys())}
        
        actual_files = [f for f in os.listdir(temp_dir) if f.endswith('.wav')]
        
        return {
            "active_files": actual_files,
            "tracked_files": list(active_audio_files.keys()),
            "file_count": len(actual_files),
            "tracked_count": len(active_audio_files)
        }
        
    except Exception as e:
        logger.error(f"List audio files error: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
