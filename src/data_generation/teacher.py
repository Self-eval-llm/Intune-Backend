import json
import requests
import time
from typing import List, Dict
import os
from datetime import datetime

class LocalDatasetGenerator:
    def __init__(self, 
                 output_file: str = "training_dataset.json",
                 model_name: str = "gpt-oss:20b",
                 ollama_url: str = "http://localhost:11434"):
        """
        Initialize Local Dataset Generator
        
        Args:
            output_file: Path to save JSON dataset
            model_name: Ollama model name for teacher
            ollama_url: Ollama API base URL
        
        Raises:
            ConnectionError: If Ollama server is not available
            RuntimeError: If model is not available
        """
        self.output_file = output_file
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.api_url = f"{ollama_url}/api/chat"
        self.dataset = []
        
        # Test connection to Ollama server
        try:
            response = requests.get(f"{ollama_url}/api/tags")
            response.raise_for_status()
            models = [m["name"] for m in response.json()["models"]]
            if self.model_name not in models:
                raise RuntimeError(f"Model {self.model_name} not found in Ollama. Available models: {', '.join(models)}")
            print(f"✅ Connected to Ollama server at {ollama_url}")
            print(f"✅ Model {model_name} is available")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Could not connect to Ollama server at {ollama_url}. Is Ollama running?")
        
        # Load existing dataset if file exists
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    self.dataset = json.load(f)
                print(f"📂 Loaded existing dataset: {len(self.dataset)} entries")
            except:
                print(f"⚠️ Could not load {output_file}, starting fresh")
                self.dataset = []
        
    def generate_training_example(self) -> Dict:
        """
        Generate a single training example autonomously
        
        Returns:
            Dictionary with input, context, expected_output
        """
        # Autonomous generation prompt
        system_prompt = """You are a dataset generator for training AI models. Generate ONE realistic training example in JSON format.

The example should cover educational topics like science, math, history, programming, literature, geography, technology, etc.

Return ONLY valid JSON in this exact format:
{
  "input": "A clear question or prompt",
  "context": ["Relevant fact 1", "Relevant fact 2", "Relevant fact 3"],
  "expected_output": "A comprehensive, accurate answer"
}

Make each example unique, educational, and diverse across different subjects and difficulty levels."""

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": "Generate a training example:"
                }
            ],
            "stream": False,
            "temperature": 0.9  # Higher temperature for diversity
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=180)
            response.raise_for_status()
            content = response.json()["message"]["content"]
            
            # Extract JSON from response
            # Sometimes models wrap JSON in markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            data = json.loads(content)
            
            # Validate structure
            if "input" in data and "context" in data and "expected_output" in data:
                return data
            else:
                print(f"⚠️ Invalid structure: missing required fields")
                return None
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                # Try generate endpoint (older API)
                print("Trying generate endpoint...")
                generate_url = f"{self.ollama_url}/api/generate"
                payload_generate = {
                    "model": self.model_name,
                    "prompt": f"{system_prompt}\n\nGenerate a training example:",
                    "stream": False,
                    "temperature": 0.9
                }
                try:
                    response = requests.post(generate_url, json=payload_generate, timeout=180)
                    response.raise_for_status()
                    content = response.json()["response"]
                    
                    # Extract and parse JSON
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0].strip()
                    
                    data = json.loads(content)
                    if "input" in data and "context" in data and "expected_output" in data:
                        return data
                    return None
                except Exception as e2:
                    print(f"❌ Error with generate endpoint: {e2}")
                    return None
            else:
                print(f"❌ Error generating example: {e}")
                return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def save_dataset(self):
        """Save current dataset to JSON file"""
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(self.dataset, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"❌ Error saving dataset: {e}")
            return False
    
    def generate_batch(self, batch_size: int = 10, delay: float = 2.0) -> int:
        """
        Generate a batch of training examples
        
        Args:
            batch_size: Number of examples to generate
            delay: Delay between generations (seconds)
            
        Returns:
            Number of successfully generated examples
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate parameters
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if batch_size > 50:
            raise ValueError("batch_size must not exceed 50 to prevent overwhelming the server")
        if delay < 0.5:
            raise ValueError("delay must be at least 0.5 seconds to prevent rate limiting")
        successful = 0
        start_count = len(self.dataset)
        
        print(f"\n{'='*60}")
        print(f"🎯 Generating batch of {batch_size} examples")
        print(f"📊 Current dataset size: {start_count}")
        print(f"{'='*60}")
        
        for i in range(batch_size):
            print(f"\n[{i+1}/{batch_size}] Generating example...")
            
            example = self.generate_training_example()
            
            if example:
                self.dataset.append(example)
                successful += 1
                print(f"✅ Success: {example['input'][:70]}...")
                
                # Save after each successful generation (safety)
                self.save_dataset()
            else:
                print(f"❌ Failed to generate valid example")
            
            # Delay between requests
            if i < batch_size - 1:
                time.sleep(delay)
        
        print(f"\n{'='*60}")
        print(f"✅ Batch complete: {successful}/{batch_size} successful")
        print(f"📊 Total dataset size: {len(self.dataset)}")
        print(f"💾 Saved to: {self.output_file}")
        print(f"{'='*60}")
        
        return successful
    
    def run_continuous(self, 
                      target_count: int = 5000,
                      batch_size: int = 15,
                      delay: float = 2.0,
                      save_interval: int = 50):
        """
        Run continuous generation until target count is reached
        
        Args:
            target_count: Stop when this many examples are generated
            batch_size: Examples per batch
            delay: Delay between examples (seconds)
            save_interval: Save backup every N examples
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Save run configuration to allow resuming
        progress_file = f"{self.output_file}.progress"
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                    if progress.get('target_count') == target_count:
                        print(f"📋 Found progress file from previous run")
                        print(f"🔄 Continuing from {len(self.dataset)}/{target_count} examples")
            except:
                pass
        
        # Save current run configuration
        try:
            with open(progress_file, 'w') as f:
                json.dump({
                    'target_count': target_count,
                    'start_time': datetime.now().isoformat(),
                    'model': self.model_name
                }, f)
        except Exception as e:
            print(f"⚠️ Could not save progress file: {e}")
        # Validate parameters
        if target_count < 1:
            raise ValueError("target_count must be at least 1")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if batch_size > 50:
            raise ValueError("batch_size must not exceed 50 to prevent overwhelming the server")
        if delay < 0.5:
            raise ValueError("delay must be at least 0.5 seconds to prevent rate limiting")
        if save_interval < 1:
            raise ValueError("save_interval must be at least 1")
        print(f"\n{'='*60}")
        print(f"🚀 AUTONOMOUS DATASET GENERATION")
        print(f"{'='*60}")
        print(f"🎯 Target: {target_count} examples")
        print(f"📦 Batch size: {batch_size}")
        print(f"⏱️  Delay: {delay}s between examples")
        print(f"💾 Output: {self.output_file}")
        print(f"📊 Starting size: {len(self.dataset)}")
        print(f"{'='*60}\n")
        
        start_time = datetime.now()
        
        while len(self.dataset) < target_count:
            current = len(self.dataset)
            remaining = target_count - current
            batch = min(batch_size, remaining)
            
            print(f"\n⏳ Progress: {current}/{target_count} ({current/target_count*100:.1f}%)")
            print(f"📈 Remaining: {remaining}")
            
            self.generate_batch(batch_size=batch, delay=delay)
            
            # Create backup every save_interval examples
            if len(self.dataset) % save_interval == 0:
                backup_file = f"{self.output_file}.backup"
                try:
                    with open(backup_file, 'w', encoding='utf-8') as f:
                        json.dump(self.dataset, f, indent=2, ensure_ascii=False)
                    print(f"💾 Backup created: {backup_file}")
                except:
                    pass
            
            # Short break between batches
            if len(self.dataset) < target_count:
                print(f"\n⏸️  5 second break...")
                time.sleep(5)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n{'='*60}")
        print(f"🎉 TARGET REACHED!")
        print(f"{'='*60}")
        print(f"✅ Generated: {len(self.dataset)} examples")
        print(f"⏱️  Time taken: {duration/60:.1f} minutes")
        print(f"💾 Saved to: {self.output_file}")
        print(f"{'='*60}\n")
    
    def get_stats(self):
        """Print dataset statistics"""
        print(f"\n{'='*60}")
        print(f"📊 DATASET STATISTICS")
        print(f"{'='*60}")
        print(f"Total examples: {len(self.dataset)}")
        print(f"Output file: {self.output_file}")
        
        if self.dataset:
            avg_input_len = sum(len(d['input']) for d in self.dataset) / len(self.dataset)
            avg_output_len = sum(len(d['expected_output']) for d in self.dataset) / len(self.dataset)
            avg_context_items = sum(len(d['context']) for d in self.dataset) / len(self.dataset)
            
            print(f"Avg input length: {avg_input_len:.0f} chars")
            print(f"Avg output length: {avg_output_len:.0f} chars")
            print(f"Avg context items: {avg_context_items:.1f}")
        print(f"{'='*60}\n")


# Example usage
if __name__ == "__main__":
    import argparse
    import os
    
    # Get the project root directory (2 levels up from this file)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    default_output = os.path.join(project_root, "data", "raw", "training_dataset.json")
    
    parser = argparse.ArgumentParser(description='Generate training dataset using Ollama')
    parser.add_argument('--model', default='gpt-oss:20b', help='Ollama model name')
    parser.add_argument('--outfile', default=default_output, help='Output file path')
    parser.add_argument('--n', type=int, default=5000, help='Number of examples to generate')
    parser.add_argument('--batch-size', type=int, default=15, help='Examples per batch')
    parser.add_argument('--sleep', type=float, default=2.0, help='Sleep time between examples')
    parser.add_argument('--mode', choices=['batch', 'continuous', 'stats'], default='continuous',
                      help='Operation mode: batch (single batch), continuous (until target), or stats')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = LocalDatasetGenerator(
        output_file=args.outfile,
        model_name=args.model
    )
    
    if args.mode == 'batch':
        generator.generate_batch(batch_size=args.batch_size, delay=args.sleep)
    elif args.mode == 'continuous':
        generator.run_continuous(
            target_count=args.n,
            batch_size=args.batch_size,
            delay=args.sleep
        )
    else:  # stats mode
        generator.get_stats()