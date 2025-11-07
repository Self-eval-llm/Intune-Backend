"""
Finetune worker module for automated fine-tuning functionality.
Contains background worker logic and condition checking for fine-tuning pipeline.
"""
import os
import sys
import time
import logging
import subprocess
import threading
from datetime import datetime
from typing import Tuple

# Add project root to path for database access
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.database.supabase_client import get_supabase_client

# Configure logging
logger = logging.getLogger(__name__)

# Global worker state
finetune_worker_running = False
finetune_worker_thread = None
last_finetune_check = None


def get_worker_status() -> dict:
    """
    Get current worker status and state.
    
    Returns:
        dict: Current worker status information
    """
    global finetune_worker_running, last_finetune_check
    
    return {
        "running": finetune_worker_running,
        "last_check": last_finetune_check,
        "thread_active": finetune_worker_thread is not None and finetune_worker_thread.is_alive()
    }


def check_finetune_conditions() -> Tuple[bool, int, int]:
    """
    Check if conditions are met for fine-tuning:
    1. status_eval_final == True (at least 5000 rows)
    2. Total rows in table >= 5000
    
    Returns:
        tuple: (conditions_met, total_rows, completed_evaluations)
    """
    try:
        supabase = get_supabase_client()
        
        # Count total rows in intune_db table
        total_response = supabase.table("intune_db")\
            .select("id", count="exact")\
            .execute()
        
        total_rows = total_response.count or 0
        
        # Count rows where status_eval_final is True
        completed_response = supabase.table("intune_db")\
            .select("id", count="exact")\
            .eq("status_eval_final", True)\
            .execute()
        
        completed_evaluations = completed_response.count or 0
        
        # Check conditions
        conditions_met = (total_rows >= 5000 and completed_evaluations >= 5000)
        
        logger.info(f"Finetune conditions check - Total rows: {total_rows}, Completed evaluations: {completed_evaluations}, Conditions met: {conditions_met}")
        
        return conditions_met, total_rows, completed_evaluations
        
    except Exception as e:
        logger.error(f"Error checking finetune conditions: {e}")
        return False, 0, 0


def run_finetune_script() -> bool:
    """
    Execute the finetune.py script as a subprocess.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Path to finetune.py
        finetune_script = os.path.join(project_root, 'src', 'training', 'finetune.py')
        
        if not os.path.exists(finetune_script):
            logger.error(f"Finetune script not found: {finetune_script}")
            return False
        
        logger.info("Starting finetune process...")
        
        # Execute finetune.py as subprocess
        # Use the current Python executable to ensure same environment
        python_exe = sys.executable
        
        # Run the subprocess with proper working directory
        process = subprocess.Popen(
            [python_exe, finetune_script],
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Log output in real-time
        for line in iter(process.stdout.readline, ''):
            if line:
                logger.info(f"Finetune: {line.strip()}")
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code == 0:
            logger.info("✅ Finetune process completed successfully")
            return True
        else:
            logger.error(f"❌ Finetune process failed with return code: {return_code}")
            return False
            
    except Exception as e:
        logger.error(f"Error running finetune script: {e}")
        return False


def finetune_background_worker():
    """
    Background worker that periodically checks conditions and runs fine-tuning.
    
    Checks every 5 minutes for:
    1. Total rows >= 5000
    2. status_eval_final == True for at least 5000 rows
    
    If conditions are met, runs finetune.py once and stops monitoring.
    """
    global finetune_worker_running, last_finetune_check
    
    logger.info("Starting finetune background worker...")
    
    check_interval = 300  # 5 minutes in seconds
    
    while finetune_worker_running:
        try:
            last_finetune_check = datetime.now().isoformat()
            
            # Check conditions
            conditions_met, total_rows, completed_evaluations = check_finetune_conditions()
            
            if conditions_met:
                logger.info("🎯 Finetune conditions met! Starting fine-tuning process...")
                
                # Run finetune script
                success = run_finetune_script()
                
                if success:
                    logger.info("🎉 Fine-tuning completed successfully! Stopping worker.")
                    # Stop the worker after successful completion
                    finetune_worker_running = False
                    break
                else:
                    logger.error("❌ Fine-tuning failed. Will retry on next check.")
            else:
                logger.info(f"⏳ Finetune conditions not yet met. Total: {total_rows}/5000, Completed: {completed_evaluations}/5000")
            
            # Wait for next check (or until stopped)
            for _ in range(check_interval):
                if not finetune_worker_running:
                    break
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in finetune worker: {e}")
            time.sleep(60)  # Wait 1 minute before retrying on error
    
    logger.info("Finetune background worker stopped")


def start_finetune_worker() -> dict:
    """
    Start the finetune background worker.
    
    Returns:
        dict: Result of the start operation
    """
    global finetune_worker_running, finetune_worker_thread
    
    try:
        if finetune_worker_running:
            return {
                "success": False,
                "message": "Finetune worker is already running",
                "worker_running": True
            }
        
        # Start the background worker
        finetune_worker_running = True
        finetune_worker_thread = threading.Thread(
            target=finetune_background_worker,
            daemon=True,
            name="FinetuneWorker"
        )
        finetune_worker_thread.start()
        
        logger.info("Finetune background worker started")
        
        return {
            "success": True,
            "message": "Finetune worker started successfully. It will monitor conditions every 5 minutes.",
            "worker_running": True
        }
        
    except Exception as e:
        logger.error(f"Error starting finetune worker: {e}")
        return {
            "success": False,
            "message": f"Error starting finetune worker: {str(e)}",
            "worker_running": False
        }


def stop_finetune_worker() -> dict:
    """
    Stop the finetune background worker.
    
    Returns:
        dict: Result of the stop operation
    """
    global finetune_worker_running, finetune_worker_thread
    
    try:
        if not finetune_worker_running:
            return {
                "success": False,
                "message": "Finetune worker is not running",
                "worker_running": False
            }
        
        # Stop the worker
        finetune_worker_running = False
        
        # Wait for thread to finish (with timeout)
        if finetune_worker_thread and finetune_worker_thread.is_alive():
            finetune_worker_thread.join(timeout=10)
        
        logger.info("Finetune background worker stopped")
        
        return {
            "success": True,
            "message": "Finetune worker stopped successfully",
            "worker_running": False
        }
        
    except Exception as e:
        logger.error(f"Error stopping finetune worker: {e}")
        return {
            "success": False,
            "message": f"Error stopping finetune worker: {str(e)}",
            "worker_running": False
        }


def run_finetune_now() -> dict:
    """
    Run fine-tuning immediately without waiting for conditions.
    
    Returns:
        dict: Result of the finetune execution
    """
    try:
        logger.info("Manual finetune execution requested...")
        
        # Run finetune script
        success = run_finetune_script()
        
        if success:
            return {
                "success": True,
                "message": "Fine-tuning completed successfully"
            }
        else:
            return {
                "success": False,
                "message": "Fine-tuning failed. Check logs for details."
            }
            
    except Exception as e:
        logger.error(f"Error in manual finetune execution: {e}")
        return {
            "success": False,
            "message": f"Error running finetune: {str(e)}"
        }


def get_finetune_status() -> dict:
    """
    Get the current status of the finetune background worker.
    
    Returns:
        dict: Current worker status and condition check results
    """
    global finetune_worker_running, last_finetune_check
    
    try:
        # Check current conditions
        conditions_met, total_rows, completed_evaluations = check_finetune_conditions()
        
        return {
            "worker_running": finetune_worker_running,
            "last_check": last_finetune_check or "Never",
            "conditions_met": conditions_met,
            "total_rows": total_rows,
            "completed_evaluations": completed_evaluations
        }
        
    except Exception as e:
        logger.error(f"Error getting finetune status: {e}")
        return {
            "worker_running": finetune_worker_running,
            "last_check": last_finetune_check or "Never",
            "conditions_met": False,
            "total_rows": 0,
            "completed_evaluations": 0,
            "error": str(e)
        }