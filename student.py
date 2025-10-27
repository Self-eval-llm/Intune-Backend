import json
import requests
import time
import os
from typing import List, Dict
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class OllamaInferencePipeline:
    def __init__(self, model_name: str = "gemma3:1b", base_url: str = "http://localhost:11434", 
                 use_supabase: bool = True, supabase_table: str = "inference_results"):
        """
        Initialize the Ollama inference pipeline
        
        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL for Ollama API
            use_supabase: Whether to save results to Supabase
            supabase_table: Name of the Supabase table to save results
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        self.use_supabase = use_supabase
        self.supabase_table = supabase_table
        
        # Initialize Supabase client if enabled
        self.supabase: Client = None
        if self.use_supabase:
            try:
                supabase_url = os.getenv("SUPABASE_URL")
                supabase_key = os.getenv("SUPABASE_KEY")
                if supabase_url and supabase_key:
                    self.supabase = create_client(supabase_url, supabase_key)
                    print("✓ Supabase connection established")
                else:
                    print("⚠ Supabase credentials not found in .env file")
                    self.use_supabase = False
            except Exception as e:
                print(f"⚠ Failed to connect to Supabase: {e}")
                self.use_supabase = False
    
    def generate_response(self, input_text: str, context = None) -> str:
        """
        Generate response from Ollama model
        
        Args:
            input_text: The input/prompt for the model
            context: Optional context to include (can be string or list)
            
        Returns:
            Generated response from the model
        """
        # Handle context as list or string
        if context:
            if isinstance(context, list):
                context_str = "\n".join(context)
            else:
                context_str = str(context)
            prompt = f"Context: {context_str}\n\nQuestion: {input_text}"
        else:
            prompt = input_text
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"ERROR: {str(e)}"
    
    def save_single_result(self, result_item: Dict, output_file: str):
        """
        Save a single result to both JSON file and Supabase
        
        Args:
            result_item: The result dictionary to save
            output_file: Path to the JSON output file
        """
        # Save to JSON file (append mode)
        try:
            # Read existing data
            if os.path.exists(output_file):
                with open(output_file, 'r', encoding='utf-8') as f:
                    try:
                        existing_data = json.load(f)
                    except json.JSONDecodeError:
                        existing_data = []
            else:
                existing_data = []
            
            # Append new result
            existing_data.append(result_item)
            
            # Write back to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  ✗ Failed to save to JSON file: {e}")
        
        # Save to Supabase
        if self.use_supabase and self.supabase:
            try:
                response = self.supabase.table(self.supabase_table).insert(result_item).execute()
                print(f"  ✓ Saved to Supabase (ID: {response.data[0].get('id', 'N/A') if response.data else 'N/A'})")
            except Exception as e:
                print(f"  ✗ Failed to save to Supabase: {e}")
    
    def process_batch(self, data: List[Dict], output_file: str, batch_size: int = 5, delay: float = 1.0, 
                     start_index: int = 0) -> List[Dict]:
        """
        Process dataset in batches with incremental saving
        
        Args:
            data: List of dictionaries with input, expected_output, context
            output_file: Path to save results incrementally
            batch_size: Number of items to process in each batch
            delay: Delay in seconds between requests
            start_index: Index to start processing from (for resuming)
            
        Returns:
            List of dictionaries with actual_output added
        """
        results = []
        total = len(data)
        
        print(f"Processing {total} items in batches of {batch_size}...")
        print(f"Starting from index: {start_index}")
        print(f"Results will be saved to: {output_file}")
        if self.use_supabase:
            print(f"Results will also be saved to Supabase table: {self.supabase_table}")
        print("-" * 80)
        
        for i in range(start_index, total, batch_size):
            batch = data[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total + batch_size - 1) // batch_size
            
            print(f"\n--- Batch {batch_num}/{total_batches} ---")
            
            for idx, item in enumerate(batch):
                item_num = i + idx + 1
                print(f"\n[{item_num}/{total}] Processing: {item['input'][:60]}...")
                
                try:
                    # Generate actual output
                    actual_output = self.generate_response(
                        input_text=item["input"],
                        context=item.get("context", "")
                    )
                    
                    # Create new item with all 4 fields
                    result_item = {
                        "input": item["input"],
                        "expected_output": item["expected_output"],
                        "context": item.get("context", []),
                        "actual_output": actual_output
                    }
                    
                    # Save immediately to both JSON and Supabase
                    self.save_single_result(result_item, output_file)
                    results.append(result_item)
                    
                    print(f"  ✓ Item {item_num} completed and saved")
                    
                except Exception as e:
                    print(f"  ✗ Error processing item {item_num}: {e}")
                    # Save error result anyway
                    result_item = {
                        "input": item["input"],
                        "expected_output": item["expected_output"],
                        "context": item.get("context", []),
                        "actual_output": f"ERROR: {str(e)}"
                    }
                    self.save_single_result(result_item, output_file)
                    results.append(result_item)
                
                # Delay between requests to avoid overload
                if idx < len(batch) - 1 or i + batch_size < total:
                    time.sleep(delay)
            
            print(f"Batch {batch_num} completed!")
        
        return results
    
    def process_dataset(self, input_file: str, output_file: str, batch_size: int = 5, 
                       delay: float = 1.0, resume: bool = True):
        """
        Process entire dataset from JSON file and save results incrementally
        
        Args:
            input_file: Path to input JSON file
            output_file: Path to output JSON file
            batch_size: Number of items to process in each batch
            delay: Delay in seconds between requests
            resume: Whether to resume from existing progress
        """
        # Load dataset
        print(f"Loading dataset from {input_file}...")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} items")
        
        # Check for existing progress
        start_index = 0
        if resume and os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
                start_index = len(existing_results)
                print(f"Found {start_index} existing results. Resuming from item {start_index + 1}...")
            except:
                print("Starting fresh (couldn't read existing file)")
                start_index = 0
        else:
            # Clear the output file if not resuming
            if os.path.exists(output_file):
                os.remove(output_file)
            print("Starting fresh processing...")
        
        if start_index >= len(data):
            print("✓ All items already processed!")
            return
        
        # Process in batches with incremental saving
        results = self.process_batch(data, output_file, batch_size=batch_size, 
                                    delay=delay, start_index=start_index)
        
        print(f"\n{'='*80}")
        print(f"✓ Processing complete!")
        print(f"  Total items processed in this run: {len(results)}")
        print(f"  Results saved to: {output_file}")
        if self.use_supabase:
            print(f"  Results also saved to Supabase table: {self.supabase_table}")
        print(f"{'='*80}")


# Example usage
if __name__ == "__main__":
    # Initialize pipeline with Supabase support
    pipeline = OllamaInferencePipeline(
        model_name="gemma3:1b",
        use_supabase=True,  # Set to False if you don't want to use Supabase
        supabase_table="inference_results"  # Change table name as needed
    )
    
    # Process dataset with incremental saving and resume capability
    pipeline.process_dataset(
        input_file="larger.json",
        output_file="dataset_with_outputs1.json",
        batch_size=5,  # Process 5 items at a time
        delay=1.0,     # 1 second delay between requests
        resume=True    # Resume from where it left off if interrupted
    )