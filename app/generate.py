"""
Module for handling Ollama model operations and response generation.
"""
import subprocess
import requests
import time

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "gemma-finetuned"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 500
MODELFILE_PATH = r"C:\Users\Radhakrishna\Downloads\llm\Modelfile"


def is_ollama_running() -> bool:
    """
    Check if Ollama service is running.
    
    Returns:
        bool: True if Ollama is running, False otherwise
    """
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def start_ollama_service() -> bool:
    """
    Start Ollama service if not running.
    
    Returns:
        bool: True if service is running (or was started successfully), False otherwise
    """
    if not is_ollama_running():
        try:
            # Start Ollama service in background
            subprocess.Popen(
                ["ollama", "serve"], 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
            # Wait for service to start
            time.sleep(3)
            
            # Verify service started
            for _ in range(10):
                if is_ollama_running():
                    return True
                time.sleep(1)
            return False
        except Exception as e:
            print(f"Error starting Ollama: {e}")
            return False
    return True


def is_model_loaded() -> bool:
    """
    Check if the model exists in Ollama.
    
    Returns:
        bool: True if model is loaded, False otherwise
    """
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return any(MODEL_NAME in model.get("name", "") for model in models)
        return False
    except requests.exceptions.RequestException:
        return False


def load_model() -> bool:
    """
    Load the model if not already loaded.
    
    Returns:
        bool: True if model is loaded successfully, False otherwise
    """
    if not is_model_loaded():
        try:
            # Create model from Modelfile
            result = subprocess.run(
                ["ollama", "create", MODEL_NAME, "-f", MODELFILE_PATH],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                raise Exception(f"Failed to create model: {result.stderr}")
            
            # Wait a moment for model to be available
            time.sleep(2)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    return True


def generate_response(prompt: str) -> str:
    """
    Generate response from the model.
    
    Args:
        prompt (str): The input prompt to send to the model
        
    Returns:
        str: The generated response from the model
        
    Raises:
        Exception: If there's an error generating the response
    """
    try:
        url = f"{OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": DEFAULT_TEMPERATURE,
                "num_predict": DEFAULT_MAX_TOKENS
            }
        }
        
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        return result.get("response", "")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error generating response: {str(e)}")


def get_model_status() -> dict:
    """
    Get the current status of Ollama service and model.
    
    Returns:
        dict: Dictionary containing ollama_running and model_loaded status
    """
    ollama_status = is_ollama_running()
    model_status = is_model_loaded() if ollama_status else False
    
    return {
        "ollama_running": ollama_status,
        "model_loaded": model_status,
        "model_name": MODEL_NAME
    }
