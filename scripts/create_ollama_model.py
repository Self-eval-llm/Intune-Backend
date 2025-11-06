"""
Create Ollama model from GGUF file
Generates Modelfile and registers with Ollama
"""

import os
import subprocess
import sys

# Configuration - Use paths relative to project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
GGUF_FILE = os.path.join(project_root, "models", "gemma-finetuned.gguf")  # Using F16 GGUF (unquantized)
MODEL_NAME = "gemma-finetuned"
MODELFILE_PATH = os.path.join(project_root, "Modelfile")

# Modelfile template
MODELFILE_TEMPLATE = """FROM {gguf_path}

TEMPLATE \"\"\"<start_of_turn>user
{{{{ .Prompt }}}}<end_of_turn>
<start_of_turn>model
\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
PARAMETER stop "<end_of_turn>"
PARAMETER stop "<eos>"

SYSTEM \"\"\"You are a helpful AI assistant. Answer questions accurately and concisely based on the provided information.\"\"\"
"""


def check_gguf_file():
    """Check if GGUF file exists"""
    if not os.path.exists(GGUF_FILE):
        print(f"✗ GGUF file not found: {GGUF_FILE}")
        print("\nPlease run convert_to_gguf.py first to create the GGUF file.")
        print("Or update GGUF_FILE variable if you used a different quantization.")
        sys.exit(1)
    
    size_mb = os.path.getsize(GGUF_FILE) / (1024 * 1024)
    print(f"✓ GGUF file found: {GGUF_FILE} ({size_mb:.2f} MB)")
    return True


def check_ollama():
    """Check if Ollama is installed and running"""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        print("✓ Ollama is installed and running")
        return True
    except subprocess.CalledProcessError:
        print("✗ Ollama is not running")
        print("\nPlease start Ollama first:")
        print("  - Windows: Start Ollama from Start menu")
        print("  - Linux/Mac: Run 'ollama serve' in another terminal")
        sys.exit(1)
    except FileNotFoundError:
        print("✗ Ollama is not installed")
        print("\nPlease install Ollama from: https://ollama.ai/download")
        sys.exit(1)


def create_modelfile():
    """Create Modelfile for Ollama"""
    print("\n" + "=" * 80)
    print("CREATING MODELFILE")
    print("=" * 80)
    
    # Convert to absolute path for Ollama
    gguf_abs_path = os.path.abspath(GGUF_FILE)
    
    # Create Modelfile content
    modelfile_content = MODELFILE_TEMPLATE.format(gguf_path=gguf_abs_path)
    
    # Write to file
    with open(MODELFILE_PATH, 'w', encoding='utf-8') as f:
        f.write(modelfile_content)
    
    print(f"\n✓ Modelfile created: {MODELFILE_PATH}")
    print("\nModelfile contents:")
    print("-" * 80)
    print(modelfile_content)
    print("-" * 80)


def create_ollama_model():
    """Register model with Ollama"""
    print("\n" + "=" * 80)
    print("CREATING OLLAMA MODEL")
    print("=" * 80)
    
    print(f"\n📦 Model name: {MODEL_NAME}")
    print(f"📄 Modelfile: {MODELFILE_PATH}")
    print("\nThis may take a minute...\n")
    
    try:
        result = subprocess.run([
            "ollama", "create",
            MODEL_NAME,
            "-f", MODELFILE_PATH
        ], check=True, capture_output=True, text=True)
        
        print(result.stdout)
        print(f"\n✓ Model '{MODEL_NAME}' created successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed to create Ollama model")
        print(f"Error: {e}")
        if e.stderr:
            print(f"Details: {e.stderr}")
        sys.exit(1)


def test_model():
    """Test the Ollama model with a sample prompt"""
    print("\n" + "=" * 80)
    print("TESTING OLLAMA MODEL")
    print("=" * 80)
    
    test_prompt = "What is machine learning?"
    
    print(f"\n📝 Test prompt: {test_prompt}")
    print("\n🤖 Model response:")
    print("-" * 80)
    
    try:
        result = subprocess.run([
            "ollama", "run",
            MODEL_NAME,
            test_prompt
        ], check=True, capture_output=True, text=True)
        
        print(result.stdout)
        print("-" * 80)
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed to test model")
        print(f"Error: {e}")


def main():
    """Main Ollama setup pipeline"""
    print("=" * 80)
    print("CREATE OLLAMA MODEL FROM GGUF")
    print("=" * 80)
    
    # Check prerequisites
    print("\nChecking prerequisites...")
    check_gguf_file()
    check_ollama()
    
    # Create Modelfile
    create_modelfile()
    
    # Create Ollama model
    create_ollama_model()
    
    # Test the model
    test_model()
    
    print("\n" + "=" * 80)
    print("✅ OLLAMA MODEL READY!")
    print("=" * 80)
    print(f"\nYou can now use your fine-tuned model:")
    print(f"  ollama run {MODEL_NAME}")
    print(f"\nOr via API:")
    print(f"  curl http://localhost:11434/api/generate -d '{{")
    print(f'    "model": "{MODEL_NAME}",')
    print(f'    "prompt": "Your question here"')
    print(f"  }}'")
    print("=" * 80)


if __name__ == "__main__":
    main()
