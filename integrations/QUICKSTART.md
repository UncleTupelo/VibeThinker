# Quick Start Guide - VibeThinker Platform Integrations

This guide will help you quickly get started with VibeThinker integrations.

## 5-Minute Quick Start

### Option 1: ChatGPT-Compatible API (Recommended)

Use VibeThinker as a drop-in replacement for ChatGPT:

```bash
# 1. Install dependencies
pip install fastapi uvicorn transformers torch openai

# 2. Start the server (downloads model on first run)
python integrations/openai_api_server.py --model-path WeiboAI/VibeThinker-1.5B --port 8000

# 3. In another terminal, test with OpenAI client
python integrations/examples/chatgpt_compatible_client.py
```

### Option 2: Direct Usage with Claude-Style Prompts

Use VibeThinker directly with Claude-style reasoning:

```bash
# 1. Install dependencies
pip install transformers torch

# 2. Run the example
python integrations/examples/claude_style_prompts.py
```

### Option 3: Search-Augmented Generation (Perplexity-Style)

Combine VibeThinker with web search:

```bash
# 1. Install dependencies
pip install transformers torch requests beautifulsoup4

# 2. Run the example
python integrations/examples/perplexity_style_search.py
```

## What Each Integration Does

### ChatGPT Integration
- **What**: OpenAI-compatible REST API server
- **Use Case**: Replace ChatGPT in existing applications
- **Works With**: OpenAI SDK, LangChain, LlamaIndex, Continue.dev
- **Best For**: Developers wanting to switch from ChatGPT to VibeThinker

### Claude Integration
- **What**: Examples using Claude's structured prompting style
- **Use Case**: Step-by-step reasoning with clear thinking process
- **Works With**: Any application using structured prompts
- **Best For**: Complex math proofs, algorithm design, detailed explanations

### Perplexity Integration
- **What**: Search + AI reasoning pipeline
- **Use Case**: Answering questions requiring current information
- **Works With**: Any search API (Google, Bing, DuckDuckGo)
- **Best For**: Research, fact-checking, current events

## Example Use Cases

### Use Case 1: Replace ChatGPT in Your Python App

```python
# Before (ChatGPT)
from openai import OpenAI
client = OpenAI(api_key="your-key")

# After (VibeThinker)
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
# Rest of your code stays the same!
```

### Use Case 2: Math Tutoring Bot

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("WeiboAI/VibeThinker-1.5B", torch_dtype="bfloat16", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("WeiboAI/VibeThinker-1.5B")

# Use Claude-style system prompt for structured teaching
system_prompt = "You are a patient math tutor. Always show step-by-step solutions."
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "How do I solve x^2 + 5x + 6 = 0?"}
]
# ... generate response
```

### Use Case 3: Code Review Assistant

```python
# Use VibeThinker API with your existing code review tool
import openai
openai.api_base = "http://localhost:8000/v1"

def review_code(code):
    response = openai.ChatCompletion.create(
        model="vibethinker-1.5b",
        messages=[
            {"role": "user", "content": f"Review this code:\n{code}"}
        ]
    )
    return response.choices[0].message.content
```

## Performance Tips

1. **Use vLLM for Production**: 2-4x faster
   ```bash
   pip install vllm==0.10.1
   python integrations/openai_api_server.py --use-vllm
   ```

2. **Optimal Settings**:
   - Temperature: 0.6 (or 1.0 for diversity)
   - Max tokens: 40960 (for complex problems)
   - Top P: 0.95
   - Top K: -1

3. **For Multiple Requests**: Use the API server instead of loading model each time

## Troubleshooting

**Problem**: Model download is slow
- **Solution**: Pre-download: `huggingface-cli download WeiboAI/VibeThinker-1.5B`

**Problem**: Out of memory
- **Solution**: Use smaller batch sizes or CPU offloading

**Problem**: API server won't start
- **Solution**: Check port 8000 is free: `lsof -i :8000`

## Next Steps

1. Read the [full integration guide](./README.md)
2. Try the [example scripts](./examples/)
3. Check out the [main README](../README.md) for model details

## Support

- Issues: https://github.com/UncleTupelo/VibeThinker/issues
- Model: https://huggingface.co/WeiboAI/VibeThinker-1.5B
