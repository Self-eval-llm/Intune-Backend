"""
FastAPI application for Gemma Model API.
Provides endpoints to interact with the fine-tuned Gemma model.
"""
import os
import sys
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.generate import (
    start_ollama_service,
    load_model,
    generate_response,
    get_model_status,
    MODEL_NAME
)

# Import finetune worker functions
from app.finetune_worker import (
    start_finetune_worker,
    stop_finetune_worker,
    run_finetune_now,
    get_finetune_status,
    check_finetune_conditions,
    get_worker_status
)

# Import evaluation worker functions
from app.eval_worker import (
    start_first_eval_worker,
    stop_first_eval_worker,
    start_final_eval_worker,
    stop_final_eval_worker,
    start_all_workers as start_all_eval_workers,
    stop_all_workers as stop_all_eval_workers,
    get_worker_status as get_eval_worker_status,
    get_pending_evaluations,
    check_evaluation_health
)

# Add project root to path for database access
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.database.supabase_client import get_supabase_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM Evaluation & Inference API",
    version="2.0.0",
    description="Unified API for LLM inference, evaluation workers, and fine-tuning automation"
)


class PromptRequest(BaseModel):
    """Request model for prompt generation."""
    prompt: str


class PromptResponse(BaseModel):
    """Response model for prompt generation."""
    response: str
    model: str


class FinetuneStatus(BaseModel):
    """Response model for finetune worker status."""
    worker_running: bool
    last_check: str
    conditions_met: bool
    total_rows: int
    completed_evaluations: int


class FinetuneResponse(BaseModel):
    """Response model for finetune operations."""
    success: bool
    message: str
    worker_running: bool


class WorkerStatus(BaseModel):
    """Status of background workers."""
    first_eval_worker: bool
    final_eval_worker: bool
    uptime_seconds: int


class EvaluationResult(BaseModel):
    """Result of evaluation process."""
    success: bool
    processed_records: int
    message: str


@app.get("/")
async def root():
    """
    Root endpoint providing API information.
    
    Returns:
        dict: Basic API information
    """
    return {
        "message": "Gemma Model API",
        "version": "1.0.0",
        "model": MODEL_NAME
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint to monitor service status.
    
    Returns:
        dict: Status of Ollama service, model, and finetune worker
    """
    status_info = get_model_status()
    
    # Test Supabase connection
    try:
        supabase = get_supabase_client()
        supabase.table("intune_db").select("id").limit(1).execute()
        supabase_healthy = True
    except Exception as e:
        logger.error(f"Supabase health check failed: {e}")
        supabase_healthy = False
    
    # Get finetune status from worker module
    finetune_status = get_finetune_status()
    
    # Get evaluation worker status
    eval_status = check_evaluation_health()
    
    return {
        "status": "healthy" if status_info["ollama_running"] and status_info["model_loaded"] and supabase_healthy else "degraded",
        "supabase_connection": supabase_healthy,
        "finetune_worker": finetune_status,
        "evaluation_workers": eval_status,
        **status_info
    }


@app.post("/generate", response_model=PromptResponse)
async def generate(request: PromptRequest):
    """
    Generate a response from the Gemma model.
    
    Args:
        request (PromptRequest): Request containing the prompt
        
    Returns:
        PromptResponse: Generated response and model name
        
    Raises:
        HTTPException: If service is unavailable or generation fails
        
    Note: 
        Temperature (0.7) and max_tokens (500) are configured statically.
    """
    try:
        # Ensure Ollama service is running
        if not start_ollama_service():
            raise HTTPException(
                status_code=503,
                detail="Failed to start Ollama service. Please ensure Ollama is installed."
            )
        
        # Ensure model is loaded
        if not load_model():
            raise HTTPException(
                status_code=503,
                detail=f"Failed to load model '{MODEL_NAME}'. Please check the Modelfile and model path."
            )
        
        # Generate response
        response_text = generate_response(prompt=request.prompt)
        
        return PromptResponse(
            response=response_text,
            model=MODEL_NAME
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )


@app.post("/finetune", response_model=FinetuneResponse)
async def start_finetune_worker_endpoint():
    """
    Start the finetune background worker.
    
    The worker will periodically check for conditions:
    1. Total rows in intune_db >= 5000
    2. status_eval_final == True for at least 5000 rows
    
    When conditions are met, it will automatically run finetune.py.
    
    Returns:
        FinetuneResponse: Status of the worker start operation
    """
    try:
        result = start_finetune_worker()
        
        return FinetuneResponse(
            success=result["success"],
            message=result["message"],
            worker_running=result["worker_running"]
        )
        
    except Exception as e:
        logger.error(f"Error starting finetune worker: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error starting finetune worker: {str(e)}"
        )


@app.get("/finetune/status", response_model=FinetuneStatus)
async def get_finetune_status_endpoint():
    """
    Get the current status of the finetune background worker.
    
    Returns:
        FinetuneStatus: Current worker status and condition check results
    """
    try:
        status = get_finetune_status()
        
        return FinetuneStatus(
            worker_running=status["worker_running"],
            last_check=status["last_check"],
            conditions_met=status["conditions_met"],
            total_rows=status["total_rows"],
            completed_evaluations=status["completed_evaluations"]
        )
        
    except Exception as e:
        logger.error(f"Error getting finetune status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting finetune status: {str(e)}"
        )


@app.post("/finetune/stop", response_model=FinetuneResponse)
async def stop_finetune_worker_endpoint():
    """
    Stop the finetune background worker.
    
    Returns:
        FinetuneResponse: Status of the worker stop operation
    """
    try:
        result = stop_finetune_worker()
        
        return FinetuneResponse(
            success=result["success"],
            message=result["message"],
            worker_running=result["worker_running"]
        )
        
    except Exception as e:
        logger.error(f"Error stopping finetune worker: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error stopping finetune worker: {str(e)}"
        )


@app.post("/finetune/run-now")
async def run_finetune_now_endpoint():
    """
    Run fine-tuning immediately without waiting for conditions.
    
    This endpoint bypasses the condition checks and runs finetune.py directly.
    Use with caution - ensure you have sufficient data before running.
    
    Returns:
        dict: Status of the finetune execution
    """
    try:
        result = run_finetune_now()
        
        return {
            "success": result["success"],
            "message": result["message"]
        }
            
    except Exception as e:
        logger.error(f"Error in manual finetune execution: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error running finetune: {str(e)}"
        )


# Evaluation Worker Endpoints
@app.get("/eval/status", response_model=WorkerStatus)
async def get_eval_worker_status_endpoint():
    """Get status of evaluation background workers."""
    try:
        status = get_eval_worker_status()
        
        return WorkerStatus(
            first_eval_worker=status["first_eval_worker"],
            final_eval_worker=status["final_eval_worker"],
            uptime_seconds=status["uptime_seconds"]
        )
        
    except Exception as e:
        logger.error(f"Error getting evaluation worker status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting evaluation worker status: {str(e)}"
        )


@app.post("/eval/start-first-worker")
async def start_first_eval_worker_endpoint():
    """Start the first evaluation background worker."""
    try:
        result = start_first_eval_worker()
        
        return {
            "success": result["success"],
            "message": result["message"]
        }
        
    except Exception as e:
        logger.error(f"Error starting first evaluation worker: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error starting first evaluation worker: {str(e)}"
        )


@app.post("/eval/stop-first-worker")
async def stop_first_eval_worker_endpoint():
    """Stop the first evaluation background worker."""
    try:
        result = stop_first_eval_worker()
        
        return {
            "success": result["success"],
            "message": result["message"]
        }
        
    except Exception as e:
        logger.error(f"Error stopping first evaluation worker: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error stopping first evaluation worker: {str(e)}"
        )


@app.post("/eval/start-final-worker")
async def start_final_eval_worker_endpoint():
    """Start the final evaluation background worker."""
    try:
        result = start_final_eval_worker()
        
        return {
            "success": result["success"],
            "message": result["message"]
        }
        
    except Exception as e:
        logger.error(f"Error starting final evaluation worker: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error starting final evaluation worker: {str(e)}"
        )


@app.post("/eval/stop-final-worker")
async def stop_final_eval_worker_endpoint():
    """Stop the final evaluation background worker."""
    try:
        result = stop_final_eval_worker()
        
        return {
            "success": result["success"],
            "message": result["message"]
        }
        
    except Exception as e:
        logger.error(f"Error stopping final evaluation worker: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error stopping final evaluation worker: {str(e)}"
        )


@app.post("/eval/start-all-workers")
async def start_all_eval_workers_endpoint():
    """Start both evaluation workers."""
    try:
        result = start_all_eval_workers()
        
        return {
            "success": result["success"],
            "message": result["message"],
            "details": result.get("details", [])
        }
        
    except Exception as e:
        logger.error(f"Error starting all evaluation workers: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error starting all evaluation workers: {str(e)}"
        )


@app.post("/eval/stop-all-workers")
async def stop_all_eval_workers_endpoint():
    """Stop both evaluation workers."""
    try:
        result = stop_all_eval_workers()
        
        return {
            "success": result["success"],
            "message": result["message"],
            "details": result.get("details", [])
        }
        
    except Exception as e:
        logger.error(f"Error stopping all evaluation workers: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error stopping all evaluation workers: {str(e)}"
        )


@app.get("/eval/health")
async def eval_health_check_endpoint():
    """Health check for evaluation system."""
    try:
        result = check_evaluation_health()
        
        return result
        
    except Exception as e:
        logger.error(f"Error checking evaluation health: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error checking evaluation health: {str(e)}"
        )


@app.get("/eval/pending")
async def get_pending_evaluations_endpoint():
    """Get count of pending evaluations."""
    try:
        result = get_pending_evaluations()
        
        if "error" in result:
            raise HTTPException(
                status_code=500,
                detail=f"Error querying database: {result['error']}"
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting pending evaluations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting pending evaluations: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
