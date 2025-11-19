# VibeThinker Platform Integrations

This directory contains integration guides and examples for using VibeThinker-1.5B with popular AI platforms and tools like Claude, Perplexity, and ChatGPT.

## Overview

VibeThinker-1.5B is a powerful reasoning model that can be integrated into various AI workflows. This guide shows you how to:

1. **Use VibeThinker as a ChatGPT alternative** via OpenAI-compatible API
2. **Apply Claude-style prompting** for enhanced reasoning
3. **Implement Perplexity-style search augmentation** for fact-based queries

## Quick Start

### 1. OpenAI-Compatible API Server (ChatGPT Integration)

Run VibeThinker with an OpenAI-compatible API that works with ChatGPT clients and libraries.

#### Installation

```bash
pip install fastapi uvicorn transformers torch
# Optional for better performance:
pip install vllm==0.10.1
```

#### Start the Server

```bash
# Using transformers (slower, lower memory requirements)
python integrations/openai_api_server.py --model-path WeiboAI/VibeThinker-1.5B --port 8000

# Using vLLM (faster, recommended for production)
python integrations/openai_api_server.py --model-path WeiboAI/VibeThinker-1.5B --port 8000 --use-vllm
```

#### Use with OpenAI Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key"
)

response = client.chat.completions.create(
    model="vibethinker-1.5b",
    messages=[
        {"role": "user", "content": "Solve: If f(x) = x^2 + 3x + 2, find the roots."}
    ],
    temperature=0.6,
    max_tokens=40960
)

print(response.choices[0].message.content)
```

#### Compatible Tools

This API server is compatible with:
- **OpenAI Python SDK** - Use VibeThinker with OpenAI's official client
- **LangChain** - Integrate into LangChain workflows via `ChatOpenAI` with custom `base_url`
- **LlamaIndex** - Use with LlamaIndex by configuring OpenAI-compatible endpoint
- **Continue.dev** - Use VibeThinker in VS Code with Continue extension
- **ChatGPT Plugins** - Any tool expecting OpenAI API format

### 2. Claude-Style Prompting

VibeThinker excels at step-by-step reasoning similar to Claude. Use structured system prompts for best results.

#### Installation

```bash
pip install transformers torch
```

#### Example Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "WeiboAI/VibeThinker-1.5B",
    torch_dtype="bfloat16",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("WeiboAI/VibeThinker-1.5B")

system_prompt = """You are a mathematical reasoning expert. When solving problems:
1. Break down the problem into steps
2. Show your work clearly
3. Verify your answer
4. State your final answer clearly"""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is 15 factorial divided by 12 factorial?"}
]

# Apply chat template and generate
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=40960, temperature=0.6)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

See `examples/claude_style_prompts.py` for more examples.

### 3. Perplexity-Style Search Augmentation

Combine VibeThinker's reasoning with web search for current information.

#### Installation

```bash
pip install transformers torch requests beautifulsoup4
```

#### Example Architecture

```python
# 1. Search for relevant information
search_results = search_web(query)

# 2. Format search results as context
context = format_search_results(search_results)

# 3. Generate answer with VibeThinker
messages = [{"role": "user", "content": f"{context}\n\nQuestion: {query}"}]
response = vibethinker.generate(messages)

# 4. Return answer with sources
return {"answer": response, "sources": search_results}
```

See `examples/perplexity_style_search.py` for a complete implementation.

## Integration Patterns

### Pattern 1: Direct API Replacement

Replace existing ChatGPT API calls with VibeThinker by changing the `base_url`:

```python
# Before (ChatGPT)
client = OpenAI(api_key="your-openai-key")

# After (VibeThinker)
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)
```

### Pattern 2: Hybrid Reasoning Pipeline

Use VibeThinker for complex reasoning while using other models for other tasks:

```python
# Use VibeThinker for math/coding
if is_reasoning_task(query):
    response = vibethinker.generate(query)
else:
    response = other_model.generate(query)
```

### Pattern 3: Multi-Agent Collaboration

Combine VibeThinker with Claude/ChatGPT in a multi-agent setup:

```python
# VibeThinker solves the problem
solution = vibethinker.generate(problem)

# Claude validates and explains
validation = claude.generate(f"Validate this solution: {solution}")

# Combine best of both
final_answer = combine_responses(solution, validation)
```

## Platform-Specific Integration Guides

### ChatGPT / OpenAI Integration

**Use Case**: Replace or augment ChatGPT with VibeThinker's superior math/coding capabilities

**Methods**:
1. **API Server**: Run OpenAI-compatible server (see above)
2. **Custom GPT Actions**: Point custom GPT to VibeThinker API endpoint
3. **ChatGPT Plugins**: Use VibeThinker as a reasoning backend

**Best For**: Math problems, competitive programming, algorithm design

### Claude Integration

**Use Case**: Use VibeThinker reasoning patterns with Claude-style prompting

**Methods**:
1. **Prompt Engineering**: Apply Claude's structured thinking prompts to VibeThinker
2. **Claude Projects**: Use VibeThinker outputs as artifacts in Claude Projects
3. **Hybrid Workflow**: Use VibeThinker for computation, Claude for explanation

**Best For**: Complex multi-step reasoning, proof generation, algorithm analysis

### Perplexity Integration

**Use Case**: Add VibeThinker's reasoning to search-augmented generation

**Methods**:
1. **RAG Pipeline**: Retrieve with search, reason with VibeThinker
2. **Fact-Checking**: Use VibeThinker to verify retrieved information
3. **Citation Generation**: Have VibeThinker reason over search results

**Best For**: Research tasks, fact-based question answering, current events with reasoning

## Performance Optimization

### Recommended Settings

For optimal performance with VibeThinker:

```python
{
    "temperature": 0.6,      # or 1.0 for more diverse outputs
    "top_p": 0.95,
    "top_k": -1,             # Skip top_k sampling
    "max_tokens": 40960,     # Full context for complex problems
}
```

### Using vLLM for Production

For production deployments, use vLLM for 2-4x faster inference:

```bash
pip install vllm==0.10.1
python integrations/openai_api_server.py --use-vllm
```

### Batching Multiple Requests

```python
# Process multiple problems efficiently
problems = [problem1, problem2, problem3]
responses = vibethinker.generate_batch(problems)
```

## Examples

All example scripts are in the `examples/` directory:

- `chatgpt_compatible_client.py` - Using VibeThinker with OpenAI SDK
- `claude_style_prompts.py` - Claude-style reasoning patterns
- `perplexity_style_search.py` - Search-augmented generation

Run any example:

```bash
# First start the API server (for chatgpt_compatible_client.py)
python integrations/openai_api_server.py --model-path WeiboAI/VibeThinker-1.5B

# Then run the example
python integrations/examples/chatgpt_compatible_client.py
```

## API Reference

### OpenAI-Compatible Endpoints

**POST /v1/chat/completions**

Create a chat completion.

Request:
```json
{
  "model": "vibethinker-1.5b",
  "messages": [{"role": "user", "content": "Your question"}],
  "temperature": 0.6,
  "max_tokens": 40960
}
```

Response:
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "vibethinker-1.5b",
  "choices": [{
    "message": {"role": "assistant", "content": "Response..."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
}
```

**GET /v1/models**

List available models.

**GET /health**

Health check endpoint.

## Use Cases

### 1. Competitive Programming Assistant

```python
problem = "Solve LeetCode Hard problem..."
solution = vibethinker.generate(problem, temperature=0.6)
# Get optimal solution with detailed reasoning
```

### 2. Math Tutoring System

```python
# Student asks question
question = "How do I solve quadratic equations?"

# VibeThinker explains step-by-step
response = vibethinker.generate(question, system_prompt=tutor_prompt)
```

### 3. Code Review Agent

```python
code = "def fibonacci(n): ..."
review_prompt = f"Review this code and suggest optimizations:\n{code}"
review = vibethinker.generate(review_prompt)
```

### 4. Research Assistant

```python
# Search for papers
papers = search_arxiv(topic)

# Reason over findings
summary = vibethinker.generate(
    f"Summarize key insights from these papers:\n{papers}"
)
```

## Limitations

- VibeThinker is optimized for **competitive-style math and coding problems**
- Not designed for general chat or creative writing
- Best results on problems with clear, definitive answers
- May require higher token limits (40960) for complex proofs

## Contributing

Have a new integration pattern? Submit a PR with:
1. Integration code
2. Example usage
3. Documentation
4. Performance benchmarks

## License

This integration code is licensed under the MIT License, same as VibeThinker.

## Support

- Issues: https://github.com/UncleTupelo/VibeThinker/issues
- Original Model: https://huggingface.co/WeiboAI/VibeThinker-1.5B
- Technical Report: https://arxiv.org/abs/2511.06221
