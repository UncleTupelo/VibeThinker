"""
Example: Using VibeThinker with Claude-style prompts

This example demonstrates how to use VibeThinker with Claude-style system prompts
and thinking patterns for complex reasoning tasks.

Requirements:
    pip install transformers torch

Usage:
    python integrations/examples/claude_style_prompts.py
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch


class VibeThinkerClaude:
    """VibeThinker with Claude-style prompt patterns"""
    
    def __init__(self, model_path="WeiboAI/VibeThinker-1.5B"):
        print(f"Loading VibeThinker model from {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("Model loaded successfully!")
    
    def generate_with_thinking(self, prompt: str, system_prompt: str = None) -> str:
        """
        Generate response using Claude-style thinking pattern.
        VibeThinker naturally shows its reasoning process.
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
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


def main():
    model = VibeThinkerClaude()
    
    # Example 1: Math problem with Claude-style system prompt
    print("=" * 80)
    print("Example 1: Math Problem with Step-by-Step Reasoning")
    print("=" * 80)
    
    system_prompt = """You are a mathematical reasoning expert. When solving problems:
1. Break down the problem into steps
2. Show your work clearly
3. Verify your answer
4. State your final answer clearly"""
    
    problem = """
    In a certain sequence, each term after the second is the sum of the two preceding terms. 
    If the 5th term is 11 and the 7th term is 29, what is the 10th term?
    """
    
    print(f"System Prompt: {system_prompt}\n")
    print(f"Problem: {problem.strip()}\n")
    print("VibeThinker's Response:")
    print("-" * 80)
    response = model.generate_with_thinking(problem, system_prompt)
    print(response)
    print()
    
    # Example 2: Coding problem with detailed reasoning
    print("=" * 80)
    print("Example 2: Algorithm Design with Reasoning")
    print("=" * 80)
    
    system_prompt = """You are a computer science expert. When designing algorithms:
1. Analyze the problem requirements
2. Consider different approaches
3. Explain your chosen approach
4. Provide implementation with comments
5. Analyze time and space complexity"""
    
    problem = """
    Design an efficient algorithm to find the kth largest element in an unsorted array.
    Provide the implementation in Python and analyze its complexity.
    """
    
    print(f"System Prompt: {system_prompt}\n")
    print(f"Problem: {problem.strip()}\n")
    print("VibeThinker's Response:")
    print("-" * 80)
    response = model.generate_with_thinking(problem, system_prompt)
    print(response)
    print()
    
    # Example 3: Multi-step reasoning problem (Claude Artifacts style)
    print("=" * 80)
    print("Example 3: Complex Multi-Step Problem")
    print("=" * 80)
    
    system_prompt = """You are an expert problem solver. Approach complex problems by:
1. Understanding all constraints
2. Planning your approach
3. Working through each step methodically
4. Checking your work
5. Providing a clear final answer"""
    
    problem = """
    A container holds a mixture of alcohol and water in the ratio 5:3. 
    If 16 liters of the mixture is removed and replaced with water, the ratio becomes 3:5.
    What was the original volume of the mixture?
    """
    
    print(f"System Prompt: {system_prompt}\n")
    print(f"Problem: {problem.strip()}\n")
    print("VibeThinker's Response:")
    print("-" * 80)
    response = model.generate_with_thinking(problem, system_prompt)
    print(response)


if __name__ == "__main__":
    main()
