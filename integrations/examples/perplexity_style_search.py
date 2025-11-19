"""
Example: Using VibeThinker with Perplexity-style search augmentation

This example shows how to integrate VibeThinker with web search capabilities,
similar to Perplexity AI's approach of combining reasoning with real-time information.

Requirements:
    pip install transformers torch requests beautifulsoup4

Usage:
    python integrations/examples/perplexity_style_search.py

Note: This is a simplified example. For production use, consider using proper
search APIs like Google Custom Search, Bing Search API, or DuckDuckGo.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import json


class VibeThinkerSearch:
    """VibeThinker with search-augmented generation capabilities"""
    
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
    
    def search_web(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """
        Simulate web search (placeholder implementation).
        In production, use proper search APIs.
        """
        # This is a placeholder - in production you would use:
        # - Google Custom Search API
        # - Bing Search API
        # - DuckDuckGo API
        # - Your own search infrastructure
        
        print(f"[Simulated Search] Searching for: {query}")
        
        # Return simulated search results
        return [
            {
                "title": "Example Search Result 1",
                "url": "https://example.com/1",
                "snippet": "This is a simulated search result snippet. In production, this would contain actual web search results."
            },
            {
                "title": "Example Search Result 2",
                "url": "https://example.com/2",
                "snippet": "Another simulated result with relevant information from the web."
            }
        ]
    
    def generate_with_search(self, query: str, include_sources: bool = True) -> Dict[str, any]:
        """
        Generate response using search-augmented generation.
        Similar to how Perplexity combines search with AI reasoning.
        """
        # Step 1: Perform web search
        search_results = self.search_web(query)
        
        # Step 2: Format search results for context
        context = "Here is relevant information from web sources:\n\n"
        for i, result in enumerate(search_results, 1):
            context += f"[{i}] {result['title']}\n"
            context += f"    {result['snippet']}\n"
            context += f"    Source: {result['url']}\n\n"
        
        # Step 3: Create prompt with search context
        prompt = f"""{context}

Based on the above information and your knowledge, please answer the following question:

{query}

Provide a comprehensive answer with reasoning. If you use information from the sources above, reference them using [1], [2], etc."""
        
        messages = [{"role": "user", "content": prompt}]
        
        # Step 4: Generate response
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
        
        # Return response with sources
        return {
            "answer": response,
            "sources": search_results if include_sources else []
        }


def main():
    model = VibeThinkerSearch()
    
    # Example 1: Fact-based question with search
    print("=" * 80)
    print("Example 1: Search-Augmented Question Answering")
    print("=" * 80)
    
    query = "What are the latest developments in quantum computing?"
    
    print(f"Query: {query}\n")
    print("Searching and generating response...")
    print("-" * 80)
    
    result = model.generate_with_search(query)
    print("\nAnswer:")
    print(result["answer"])
    print("\nSources:")
    for i, source in enumerate(result["sources"], 1):
        print(f"[{i}] {source['title']}")
        print(f"    {source['url']}\n")
    
    # Example 2: Math problem (doesn't need search, but demonstrates flexibility)
    print("=" * 80)
    print("Example 2: Math Problem (No Search Needed)")
    print("=" * 80)
    
    # For pure math/coding problems, we can skip search and use VibeThinker directly
    messages = [{"role": "user", "content": """
    Prove that the sum of the first n odd numbers equals n^2.
    """}]
    
    text = model.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = model.tokenizer([text], return_tensors="pt").to(model.model.device)
    
    generation_config = GenerationConfig(
        max_new_tokens=40960,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        top_k=None,
    )
    
    generated_ids = model.model.generate(
        **model_inputs,
        generation_config=generation_config
    )
    
    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(f"Query: {messages[0]['content'].strip()}\n")
    print("VibeThinker's Response (no search needed):")
    print("-" * 80)
    print(response)


if __name__ == "__main__":
    main()
