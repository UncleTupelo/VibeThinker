#!/usr/bin/env python3
"""
VibeThinker CLI Tool

A simple command-line interface for interacting with VibeThinker.

Usage:
    # Start interactive mode
    python integrations/cli.py

    # Single question mode
    python integrations/cli.py --question "What is 15 factorial?"

    # Use API server (if running)
    python integrations/cli.py --api http://localhost:8000

    # Load local model
    python integrations/cli.py --model-path WeiboAI/VibeThinker-1.5B
"""

import argparse
import sys
from typing import Optional

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class VibeThinkerCLI:
    def __init__(self, model_path: Optional[str] = None, api_url: Optional[str] = None):
        self.api_url = api_url
        self.model = None
        self.tokenizer = None
        self.client = None
        
        if api_url:
            if not HAS_OPENAI:
                raise ImportError("OpenAI package required for API mode. Install with: pip install openai")
            self.client = OpenAI(base_url=f"{api_url}/v1", api_key="dummy")
            print(f"Connected to VibeThinker API at {api_url}")
        elif model_path:
            if not HAS_TRANSFORMERS:
                raise ImportError("Transformers required for local mode. Install with: pip install transformers torch")
            print(f"Loading VibeThinker model from {model_path}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            print("Model loaded!")
        else:
            raise ValueError("Either model_path or api_url must be provided")
    
    def ask(self, question: str) -> str:
        """Ask a question and get a response"""
        if self.client:
            # Use API
            response = self.client.chat.completions.create(
                model="vibethinker-1.5b",
                messages=[{"role": "user", "content": question}],
                temperature=0.6,
                max_tokens=40960
            )
            return response.choices[0].message.content
        else:
            # Use local model
            messages = [{"role": "user", "content": question}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            generation_config = GenerationConfig(
                max_new_tokens=40960,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                top_k=None,
            )
            
            generated_ids = self.model.generate(
                **model_inputs,
                generation_config=generation_config
            )
            
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response
    
    def interactive(self):
        """Start interactive mode"""
        print("\n" + "="*80)
        print("VibeThinker Interactive Mode")
        print("="*80)
        print("Type your questions below. Type 'exit' or 'quit' to end.\n")
        
        while True:
            try:
                question = input("\n📝 You: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                print("\n🤔 VibeThinker is thinking...\n")
                response = self.ask(question)
                print("💡 VibeThinker:", response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="VibeThinker CLI - Interactive command-line interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with local model
  python integrations/cli.py --model-path WeiboAI/VibeThinker-1.5B
  
  # Interactive mode with API server
  python integrations/cli.py --api http://localhost:8000
  
  # Single question
  python integrations/cli.py --api http://localhost:8000 --question "What is 12^3?"
        """
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to VibeThinker model (for local inference)"
    )
    parser.add_argument(
        "--api",
        type=str,
        help="URL of VibeThinker API server (e.g., http://localhost:8000)"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Single question to ask (non-interactive mode)"
    )
    
    args = parser.parse_args()
    
    if not args.model_path and not args.api:
        parser.error("Either --model-path or --api must be specified")
    
    if args.model_path and args.api:
        parser.error("Cannot use both --model-path and --api at the same time")
    
    try:
        cli = VibeThinkerCLI(model_path=args.model_path, api_url=args.api)
        
        if args.question:
            # Single question mode
            print(f"\n📝 Question: {args.question}\n")
            print("🤔 VibeThinker is thinking...\n")
            response = cli.ask(args.question)
            print("💡 Answer:", response, "\n")
        else:
            # Interactive mode
            cli.interactive()
    
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
