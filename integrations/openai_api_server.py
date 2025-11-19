"""
OpenAI-Compatible API Server for VibeThinker

This server provides an OpenAI-compatible API endpoint that can be used
with ChatGPT clients, libraries, and integrations that expect the OpenAI API format.

Usage:
    python openai_api_server.py --model-path WeiboAI/VibeThinker-1.5B --port 8000

Requirements:
    pip install fastapi uvicorn transformers torch
    # Optional for better performance: pip install vllm==0.10.1
"""

import argparse
import time
import uuid
from typing import List, Optional, Dict, Any, Literal
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False


# OpenAI API Models
class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 40960
    stream: Optional[bool] = False
    n: Optional[int] = 1


class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


class Model(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "vibethinker"


class ModelList(BaseModel):
    object: str = "list"
    data: List[Model]


# Global model instance
model_instance = None
tokenizer_instance = None
use_vllm = False


class VibeThinkerModel:
    """VibeThinker model wrapper"""
    
    def __init__(self, model_path: str, use_vllm: bool = False):
        self.model_path = model_path
        self.use_vllm = use_vllm
        
        if use_vllm:
            if not HAS_VLLM:
                raise ImportError("vLLM is not installed. Install it with: pip install vllm==0.10.1")
            print(f"Loading VibeThinker model with vLLM from {model_path}...")
            self.model = LLM(
                model=model_path,
                trust_remote_code=True,
                dtype="bfloat16",
                gpu_memory_utilization=0.95,
            )
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        else:
            if not HAS_TRANSFORMERS:
                raise ImportError("Transformers is not installed. Install it with: pip install transformers torch")
            print(f"Loading VibeThinker model with transformers from {model_path}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        print("Model loaded successfully!")
    
    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.6, 
                 top_p: float = 0.95, max_tokens: int = 40960) -> str:
        """Generate response from messages"""
        
        if self.use_vllm:
            # Use vLLM for generation
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                top_k=-1,  # VibeThinker recommendation
                max_tokens=max_tokens,
            )
            
            outputs = self.model.generate([text], sampling_params)
            return outputs[0].outputs[0].text
        else:
            # Use transformers for generation
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            generation_config = GenerationConfig(
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=None,  # VibeThinker uses top_k=-1 in vLLM
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model_instance, tokenizer_instance
    yield
    # Shutdown
    del model_instance
    del tokenizer_instance


app = FastAPI(
    title="VibeThinker OpenAI-Compatible API",
    description="OpenAI-compatible API server for VibeThinker-1.5B",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/v1/models")
@app.get("/models")
async def list_models() -> ModelList:
    """List available models"""
    return ModelList(
        data=[
            Model(
                id="vibethinker-1.5b",
                created=int(time.time()),
            )
        ]
    )


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """Create a chat completion"""
    global model_instance
    
    if model_instance is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming not yet supported")
    
    try:
        # Convert messages to format expected by model
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Generate response
        response_text = model_instance.generate(
            messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens
        )
        
        # Estimate token usage (rough approximation)
        prompt_text = " ".join([msg.content for msg in request.messages])
        prompt_tokens = len(tokenizer_instance.encode(prompt_text))
        completion_tokens = len(tokenizer_instance.encode(response_text))
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(role="assistant", content=response_text),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model_instance is not None}


def main():
    parser = argparse.ArgumentParser(description="VibeThinker OpenAI-Compatible API Server")
    parser.add_argument(
        "--model-path",
        type=str,
        default="WeiboAI/VibeThinker-1.5B",
        help="Path to VibeThinker model (HuggingFace model ID or local path)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to"
    )
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help="Use vLLM for faster inference"
    )
    
    args = parser.parse_args()
    
    global model_instance, tokenizer_instance, use_vllm
    use_vllm = args.use_vllm
    
    # Load model
    model_instance = VibeThinkerModel(args.model_path, use_vllm=args.use_vllm)
    tokenizer_instance = model_instance.tokenizer
    
    print(f"\nStarting VibeThinker API server on {args.host}:{args.port}")
    print(f"Model: {args.model_path}")
    print(f"Backend: {'vLLM' if args.use_vllm else 'transformers'}")
    print("\nAPI endpoints:")
    print(f"  - http://{args.host}:{args.port}/v1/chat/completions")
    print(f"  - http://{args.host}:{args.port}/v1/models")
    print(f"  - http://{args.host}:{args.port}/health")
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
