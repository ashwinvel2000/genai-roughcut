"""FastAPI application entry point and routes."""

from fastapi import FastAPI, Request, Form, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import logging
import os
import asyncio
from pathlib import Path
from typing import Dict, Any

from app.settings import settings
from app.storage import generate_run_id, ensure_run_dirs
import app.storage as storage
from app.pipeline import process_run_async

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure outputs directory exists
Path(settings.outputs_dir).mkdir(parents=True, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Script-to-Rough-Cut",
    description="Generate video rough cuts from text scripts using AI",
    version="0.1.0"
)

# In-memory run tracking
RUNS: Dict[str, Dict[str, Any]] = {}

# Mount static files - serve outputs directory
app.mount("/outputs", StaticFiles(directory=settings.outputs_dir), name="outputs")

# Setup templates
templates = Jinja2Templates(directory="app/templates")


@app.get("/")
async def index(request: Request):
    """Render the main UI page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/healthz")
async def health_check():
    """Health check endpoint."""
    return {"ok": True}


@app.post("/generate")
async def generate_video(brief: str = Form(...)):
    """
    Start a new video generation run.
    
    Args:
        brief: User's video concept brief (form field)
        
    Returns:
        JSON with run_id for WebSocket connection
    """
    # Generate unique run ID
    run_id = generate_run_id()
    
    logger.info(f"New generation request: {run_id}")
    logger.info(f"Brief: {brief[:100]}...")
    
    # Initialize run state
    RUNS[run_id] = {
        "step": "queued",
        "done": False,
        "error": None,
        "message": "Request queued, connect to WebSocket for real-time updates",
        "brief": brief
    }
    
    # Note: Actual processing happens via WebSocket connection
    # Client should connect to /ws/{run_id} to start processing
    
    return {"run_id": run_id}


@app.websocket("/ws/{run_id}")
async def websocket_endpoint(websocket: WebSocket, run_id: str):
    """
    WebSocket endpoint for real-time progress updates during video generation.
    
    Args:
        websocket: WebSocket connection
        run_id: The run identifier
    """
    await websocket.accept()
    logger.info(f"WebSocket connected for run {run_id}")
    
    try:
        # Check if run exists
        if run_id not in RUNS:
            await websocket.send_json({
                "error": f"Run {run_id} not found",
                "done": True
            })
            await websocket.close()
            return
        
        # Get the brief from the run
        brief = RUNS[run_id].get("brief")
        if not brief:
            await websocket.send_json({
                "error": "No brief found for this run",
                "done": True
            })
            await websocket.close()
            return
        
        # Process the video generation with real-time updates
        await process_run_async(run_id, brief, settings, RUNS, storage, websocket)
        
        # Send final completion message
        final_state = RUNS[run_id].copy()
        final_state["state"] = "completed" if final_state["done"] and not final_state["error"] else "failed"
        await websocket.send_json(final_state)
        
        logger.info(f"WebSocket completed for run {run_id}")
        
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for run {run_id}")
    except Exception as e:
        logger.error(f"WebSocket error for run {run_id}: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "error": str(e),
                "step": "error",
                "done": True
            })
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass


@app.get("/status/{run_id}")
async def get_status(run_id: str):
    """
    Get the status of a video generation run.
    
    Args:
        run_id: The run identifier
        
    Returns:
        JSON with run status
    """
    if run_id not in RUNS:
        return JSONResponse(
            status_code=404,
            content={"error": f"Run {run_id} not found"}
        )
    
    run_info = RUNS[run_id].copy()
    
    # Add derived info
    run_info["state"] = "completed" if run_info["done"] and not run_info["error"] else \
                        "failed" if run_info["error"] else \
                        "processing"
    
    return run_info


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.port)
