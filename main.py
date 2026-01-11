from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import tempfile
import os
import shutil
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

# Import your UPDATED RAG pipeline
from chattingh import AgenticRAGPipeline

load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Agentic RAG API with Voice Input",
    description="Multimodal RAG with separate text/image stores and smart routing",
    version="2.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client for Whisper transcription
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
print("[SUCCESS] Groq Whisper API initialized")

# Store active RAG sessions (in production, use Redis or database)
active_sessions = {}

# Request/Response models
class TextQueryRequest(BaseModel):
    session_id: str
    question: str

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    needed_retrieval: bool
    content_type: Optional[str] = None  # NEW: text, image, or both
    text_docs_count: int = 0  # NEW
    image_docs_count: int = 0  # NEW
    rewritten_query: Optional[str] = None
    sources: List[str] = []
    transcribed_text: Optional[str] = None  # For voice queries

class SessionInfo(BaseModel):
    session_id: str
    processed_files: List[str]
    message_count: int
    has_text_retriever: bool  # UPDATED
    has_image_retriever: bool  # UPDATED
    text_chunks_total: int  # NEW
    image_chunks_total: int  # NEW

class DocumentStats(BaseModel):  # NEW
    session_id: str
    text_chunks: int
    image_chunks: int
    total_chunks: int
    files_processed: List[str]


# ========== HELPER FUNCTIONS ==========

def get_or_create_session(session_id: Optional[str] = None) -> AgenticRAGPipeline:
    """Get existing session or create new one"""
    if session_id and session_id in active_sessions:
        return active_sessions[session_id]
    
    # Create new session
    rag = AgenticRAGPipeline()
    active_sessions[rag.session_id] = rag
    return rag

def transcribe_audio_groq(audio_file_path: str) -> str:
    """Transcribe audio file using Groq's Whisper API"""
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3-turbo",
                response_format="text",
                language="en",
                temperature=0.0
            )
        return transcription.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

def count_session_images(session_id: str) -> int:
    """Count extracted images for a session"""
    image_dir = f"extracted_images/{session_id}"
    if not os.path.exists(image_dir):
        return 0
    return len([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))])

def get_vector_store_stats(rag: AgenticRAGPipeline) -> dict:
    """Get statistics about text and image vector stores"""
    stats = {
        "text_chunks": 0,
        "image_chunks": 0,
        "total_chunks": 0
    }
    
    try:
        if rag.text_vector_store:
            stats["text_chunks"] = len(rag.text_vector_store.docstore._dict)
        if rag.image_vector_store:
            stats["image_chunks"] = len(rag.image_vector_store.docstore._dict)
        stats["total_chunks"] = stats["text_chunks"] + stats["image_chunks"]
    except Exception as e:
        print(f"[WARNING] Could not get vector store stats: {e}")
    
    return stats

def cleanup_session_data(session_id: str):
    """
    Cleanup all session data including:
    - Extracted images directory
    - Chat history from database
    """
    image_dir = f"extracted_images/{session_id}"
    if os.path.exists(image_dir):
        try:
            shutil.rmtree(image_dir)
            print(f"[INFO] Deleted image directory for session: {session_id}")
        except Exception as e:
            print(f"[WARNING] Failed to delete image directory: {e}")


# ========== API ENDPOINTS ==========

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Enhanced Agentic RAG API v2.0",
        "features": [
            "text_input",
            "voice_input",
            "smart_content_routing",
            "separate_text_image_stores",
            "ocr_extraction",
            "image_captioning",
            "session_based_storage"
        ],
        "whisper_model": "groq/whisper-large-v3-turbo",
        "vision_models": {
            "clip": "openai/clip-vit-base-patch32",
            "blip": "Salesforce/blip-image-captioning-base",
            "ocr": "pytesseract"
        },
        "new_features": [
            "Separate vector stores for text and images",
            "Smart routing based on query type",
            "Guaranteed text extraction",
            "Better relevance scoring"
        ]
    }


@app.post("/upload-document", response_model=DocumentStats)
async def upload_document(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None)
):
    """
    Upload one or multiple PDF/TXT documents for processing
    Creates new session if session_id not provided
    
    NEW FEATURES:
    - Separate text and image vector stores
    - Guaranteed text extraction (never overshadowed by images)
    - Smart content routing
    - Better retrieval accuracy
    """
    # Validate file types
    allowed_extensions = [".pdf", ".txt"]
    uploaded_files = []
    
    for file in files:
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File '{file.filename}' type {file_ext} not supported. Use PDF or TXT."
            )
        uploaded_files.append(file)
    
    try:
        # Get or create session
        rag = get_or_create_session(session_id)
        
        temp_file_paths = []
        
        # Save all files temporarily
        for file in uploaded_files:
            file_ext = Path(file.filename).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                shutil.copyfileobj(file.file, tmp_file)
                temp_file_paths.append(tmp_file.name)
        
        # Process all files (returns dict with stats)
        stats = rag.process_files(temp_file_paths)
        
        # Cleanup temp files
        for tmp_path in temp_file_paths:
            os.unlink(tmp_path)
        
        return DocumentStats(
            session_id=rag.session_id,
            text_chunks=stats["text_chunks"],
            image_chunks=stats["image_chunks"],
            total_chunks=stats["total"],
            files_processed=[file.filename for file in uploaded_files]
        )
        
    except Exception as e:
        # Cleanup on error
        for tmp_path in temp_file_paths:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask-text", response_model=QueryResponse)
async def ask_text_question(
    question: str = Form(...),
    session_id: Optional[str] = Form(None)
):
    """
    Ask a question using text input
    
    NEW FEATURES:
    - Smart content routing (automatically detects if you need text, images, or both)
    - Separate retrieval from text and image stores
    - Returns content_type to show what was used
    - Better answer quality with guaranteed text retrieval
    """
    try:
        # Get or create session
        rag = get_or_create_session(session_id)
        result = rag.ask(question)
        
        # Extract source files
        sources = list(set([
            doc.metadata.get("source", "unknown")
            for doc in result["documents"]
        ]))
        
        return QueryResponse(
            answer=result["answer"],
            session_id=rag.session_id,
            needed_retrieval=result["needed_retrieval"],
            content_type=result.get("content_type"),
            text_docs_count=result.get("text_docs_count", 0),
            image_docs_count=result.get("image_docs_count", 0),
            rewritten_query=result.get("rewritten_query"),
            sources=sources
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask-voice", response_model=QueryResponse)
async def ask_voice_question(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """
    Ask a question using voice input
    
    Workflow:
    1. Upload audio file
    2. Transcribe with Whisper (Groq)
    3. Smart routing to text/image stores
    4. Return answer with transcription and content type
    """
    # Validate audio file
    allowed_audio_formats = [".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"]
    audio_ext = Path(audio.filename).suffix.lower()
    
    if audio_ext not in allowed_audio_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Audio format {audio_ext} not supported. Use MP3, WAV, M4A, OGG, FLAC, or WEBM."
        )
    
    try:
        # Save audio file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=audio_ext) as tmp_audio:
            shutil.copyfileobj(audio.file, tmp_audio)
            tmp_audio_path = tmp_audio.name
        
        # Transcribe audio using Groq Whisper
        print(f"[INFO] Transcribing audio: {audio.filename}")
        transcribed_text = transcribe_audio_groq(tmp_audio_path)
        print(f"[INFO] Transcribed: {transcribed_text}")
        
        # Cleanup temp audio file
        os.unlink(tmp_audio_path)
        
        # Get or create session
        rag = get_or_create_session(session_id)
        result = rag.ask(transcribed_text)
        
        # Extract source files
        sources = list(set([
            doc.metadata.get("source", "unknown")
            for doc in result["documents"]
        ]))
        
        return QueryResponse(
            answer=result["answer"],
            session_id=rag.session_id,
            needed_retrieval=result["needed_retrieval"],
            content_type=result.get("content_type"),
            text_docs_count=result.get("text_docs_count", 0),
            image_docs_count=result.get("image_docs_count", 0),
            rewritten_query=result.get("rewritten_query"),
            sources=sources,
            transcribed_text=transcribed_text
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str):
    """
    Get information about a specific session
    
    UPDATED: Now shows separate text and image retriever status
    """
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    rag = active_sessions[session_id]
    info = rag.get_session_info()
    stats = get_vector_store_stats(rag)
    
    return SessionInfo(
        session_id=info["session_id"],
        processed_files=info["processed_files"],
        message_count=info["message_count"],
        has_text_retriever=info["has_text_retriever"],
        has_image_retriever=info["has_image_retriever"],
        text_chunks_total=stats["text_chunks"],
        image_chunks_total=stats["image_chunks"]
    )


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a session and clear all associated data
    """
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        rag = active_sessions[session_id]
        image_count = count_session_images(session_id)
        stats = get_vector_store_stats(rag)
        
        # Clear memory (chat history)
        rag.clear_memory()
        
        # Cleanup session data (images, etc.)
        cleanup_session_data(session_id)
        
        # Remove from active sessions
        del active_sessions[session_id]
        
        return {
            "status": "success",
            "message": f"Session {session_id} deleted successfully",
            "cleaned_up": {
                "chat_history": True,
                "images_deleted": image_count,
                "text_chunks_deleted": stats["text_chunks"],
                "image_chunks_deleted": stats["image_chunks"],
                "vector_stores": True
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions")
async def list_active_sessions():
    """
    List all active sessions with details
    
    UPDATED: Shows text/image chunk counts
    """
    sessions = []
    for session_id, rag in active_sessions.items():
        info = rag.get_session_info()
        image_count = count_session_images(session_id)
        stats = get_vector_store_stats(rag)
        
        sessions.append({
            "session_id": session_id,
            "files": info["processed_files"],
            "messages": info["message_count"],
            "images_extracted": image_count,
            "text_chunks": stats["text_chunks"],
            "image_chunks": stats["image_chunks"],
            "has_text_retriever": info["has_text_retriever"],
            "has_image_retriever": info["has_image_retriever"]
        })
    
    return {
        "total_sessions": len(sessions),
        "sessions": sessions
    }


@app.post("/clear-memory/{session_id}")
async def clear_session_memory(session_id: str):
    """
    Clear chat history for a session while keeping documents and images
    """
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        rag = active_sessions[session_id]
        rag.clear_memory()
        
        return {
            "status": "success",
            "message": f"Memory cleared for session {session_id}",
            "note": "Text chunks, image chunks, and extracted images are preserved"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}/images")
async def get_session_images(session_id: str):
    """
    Get list of all extracted images for a session
    """
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    image_dir = f"extracted_images/{session_id}"
    
    if not os.path.exists(image_dir):
        return {
            "session_id": session_id,
            "image_count": 0,
            "images": []
        }
    
    images = [
        {
            "filename": f,
            "path": os.path.join(image_dir, f),
            "size_kb": round(os.path.getsize(os.path.join(image_dir, f)) / 1024, 2)
        }
        for f in os.listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, f))
    ]
    
    return {
        "session_id": session_id,
        "image_count": len(images),
        "directory": image_dir,
        "images": images
    }


@app.get("/session/{session_id}/stats")
async def get_session_statistics(session_id: str):
    """
    NEW ENDPOINT: Get detailed statistics about vector stores
    """
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    rag = active_sessions[session_id]
    stats = get_vector_store_stats(rag)
    info = rag.get_session_info()
    image_count = count_session_images(session_id)
    
    return {
        "session_id": session_id,
        "vector_stores": {
            "text": {
                "enabled": info["has_text_retriever"],
                "chunks": stats["text_chunks"]
            },
            "image": {
                "enabled": info["has_image_retriever"],
                "chunks": stats["image_chunks"]
            },
            "total_chunks": stats["total_chunks"]
        },
        "files": {
            "processed": info["processed_files"],
            "count": len(info["processed_files"])
        },
        "images": {
            "extracted": image_count,
            "directory": f"extracted_images/{session_id}"
        },
        "chat": {
            "messages": info["message_count"]
        }
    }


@app.delete("/cleanup-all-sessions")
async def cleanup_all_sessions():
    """
    Emergency cleanup: Delete all sessions and their data
    Use with caution!
    """
    try:
        deleted_count = 0
        total_images = 0
        total_text_chunks = 0
        total_image_chunks = 0
        
        for session_id in list(active_sessions.keys()):
            try:
                image_count = count_session_images(session_id)
                rag = active_sessions[session_id]
                stats = get_vector_store_stats(rag)
                
                total_images += image_count
                total_text_chunks += stats["text_chunks"]
                total_image_chunks += stats["image_chunks"]
                
                rag.clear_memory()
                cleanup_session_data(session_id)
                del active_sessions[session_id]
                deleted_count += 1
            except Exception as e:
                print(f"[WARNING] Failed to cleanup session {session_id}: {e}")
        
        # Cleanup orphaned image directories
        if os.path.exists("extracted_images"):
            for item in os.listdir("extracted_images"):
                item_path = os.path.join("extracted_images", item)
                if os.path.isdir(item_path):
                    try:
                        shutil.rmtree(item_path)
                    except:
                        pass
        
        return {
            "status": "success",
            "sessions_deleted": deleted_count,
            "resources_cleaned": {
                "images": total_images,
                "text_chunks": total_text_chunks,
                "image_chunks": total_image_chunks,
                "total_chunks": total_text_chunks + total_image_chunks
            },
            "message": "All sessions and data cleaned up successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== STARTUP/SHUTDOWN EVENTS ==========

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("="*60)
    print("[INFO] Starting Enhanced Agentic Multimodal RAG API v2.0...")
    print("="*60)
    print("[INFO] Features enabled:")
    print("   [OK] Text & Voice Input (Whisper)")
    print("   [OK] Smart Content Routing")
    print("   [OK] Separate Text/Image Vector Stores")
    print("   [OK] Multimodal Search (CLIP)")
    print("   [OK] Image Captioning (BLIP)")
    print("   [OK] OCR (PyTesseract)")
    print("   [OK] Session-based Storage")
    print("="*60)
    print("[INFO] Models:")
    print("   - Whisper: groq/whisper-large-v3-turbo")
    print("   - CLIP: openai/clip-vit-base-patch32")
    print("   - BLIP: Salesforce/blip-image-captioning-base")
    print("   - LLM: llama-3.3-70b-versatile (Groq)")
    print("="*60)
    print("[SUCCESS] API is ready!")
    print("="*60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("\n" + "="*60)
    print("[INFO] Shutting down API...")
    print("="*60)
    
    cleanup_count = 0
    for session_id in list(active_sessions.keys()):
        try:
            rag = active_sessions[session_id]
            rag.clear_memory()
            cleanup_session_data(session_id)
            cleanup_count += 1
        except:
            pass
    
    print(f"[INFO] Cleaned up {cleanup_count} sessions")
    print("="*60)
    print("[SUCCESS] API shutdown complete")
    print("="*60)


