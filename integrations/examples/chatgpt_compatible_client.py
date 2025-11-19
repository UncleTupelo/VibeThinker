"""
Example: Using VibeThinker with OpenAI Python client

This example shows how to use the VibeThinker API server with the OpenAI Python library.
This allows you to use VibeThinker as a drop-in replacement for ChatGPT in your applications.

Requirements:
    pip install openai

Usage:
    1. Start the VibeThinker API server:
       python integrations/openai_api_server.py --model-path WeiboAI/VibeThinker-1.5B --port 8000
    
    2. Run this script:
       python integrations/examples/chatgpt_compatible_client.py
"""

from openai import OpenAI

# Configure client to use VibeThinker API server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key"  # Not used but required by OpenAI client
)

def chat_with_vibethinker(prompt: str) -> str:
    """Send a chat message to VibeThinker"""
    response = client.chat.completions.create(
        model="vibethinker-1.5b",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.6,
        top_p=0.95,
        max_tokens=40960
    )
    return response.choices[0].message.content


def main():
    # Example 1: Simple math problem
    print("=" * 80)
    print("Example 1: Math Problem")
    print("=" * 80)
    
    problem = """
    A regular hexagon has side length 6. If a circle is inscribed in the hexagon, 
    what is the area of the region inside the hexagon but outside the circle?
    """
    
    print(f"Problem: {problem.strip()}\n")
    print("VibeThinker's Response:")
    print("-" * 80)
    response = chat_with_vibethinker(problem)
    print(response)
    print()
    
    # Example 2: Coding problem
    print("=" * 80)
    print("Example 2: Coding Problem")
    print("=" * 80)
    
    problem = """
    Write a Python function to find the longest palindromic substring in a given string.
    Optimize for time complexity.
    """
    
    print(f"Problem: {problem.strip()}\n")
    print("VibeThinker's Response:")
    print("-" * 80)
    response = chat_with_vibethinker(problem)
    print(response)
    print()
    
    # Example 3: Multi-turn conversation
    print("=" * 80)
    print("Example 3: Multi-turn Conversation")
    print("=" * 80)
    
    messages = [
        {"role": "user", "content": "What is 15 factorial?"},
    ]
    
    response = client.chat.completions.create(
        model="vibethinker-1.5b",
        messages=messages,
        temperature=0.6,
    )
    
    print(f"User: {messages[0]['content']}")
    assistant_response = response.choices[0].message.content
    print(f"Assistant: {assistant_response}\n")
    
    messages.append({"role": "assistant", "content": assistant_response})
    messages.append({"role": "user", "content": "Now divide that by 120"})
    
    response = client.chat.completions.create(
        model="vibethinker-1.5b",
        messages=messages,
        temperature=0.6,
    )
    
    print(f"User: {messages[2]['content']}")
    print(f"Assistant: {response.choices[0].message.content}")


if __name__ == "__main__":
    main()
