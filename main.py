"""
FastAPI LLM Inference Backend
Production-grade async API for multi-provider LLM inference.
"""

import uuid
import time
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM Inference Backend",
    description="Production-grade multi-provider LLM inference service",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models ---
class LLMProvider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"

class InferenceRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000)
    provider: LLMProvider = LLMProvider.OPENAI
    max_tokens: int = Field(default=500, ge=1, le=4000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    request_id: Optional[str] = None

class InferenceResponse(BaseModel):
    request_id: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    response_text: str
    latency_ms: float
    quality_score: Optional[float] = None
    warnings: list = []

# --- Monitoring ---
start_time = time.time()
total_requests = 0
total_errors = 0
latencies = []
request_logs = []

def record_request(request_id, provider, latency_ms, status):
    global total_requests, total_errors
    total_requests += 1
    latencies.append(latency_ms)
    request_logs.append({
        "request_id": request_id,
        "provider": provider,
        "latency_ms": latency_ms,
        "status": status,
        "timestamp": time.time()
    })
    if status == "error":
        total_errors += 1

def validate_response(response_text):
    warnings = []
    score = 1.0
    if len(response_text.split()) < 5:
        warnings.append("Response is very short")
        score -= 0.3
    refusal_phrases = ["i cannot", "i can't", "i am unable"]
    if any(p in response_text.lower() for p in refusal_phrases):
        warnings.append("Response contains refusal language")
        score -= 0.2
    return max(0.0, score), warnings

# --- Endpoints ---
@app.post("/v1/inference", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest):
    request_id = request.request_id or str(uuid.uuid4())
    t0 = time.time()
    logger.info("Inference request %s | provider=%s", request_id, request.provider)
    try:
        response_text = (
            f"[{request.provider.upper()}] Response to: '{request.prompt[:50]}' "
            f"(max_tokens={request.max_tokens})"
        )
        latency_ms = (time.time() - t0) * 1000
        quality_score, warnings = validate_response(response_text)
        record_request(request_id, request.provider, latency_ms, "success")
        return InferenceResponse(
            request_id=request_id,
            provider=request.provider,
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=len(response_text.split()),
            response_text=response_text,
            latency_ms=round(latency_ms, 2),
            quality_score=round(quality_score, 3),
            warnings=warnings,
        )
    except Exception as e:
        latency_ms = (time.time() - t0) * 1000
        record_request(request_id, request.provider, latency_ms, "error")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/health")
async def health_check():
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    error_rate = total_errors / total_requests if total_requests > 0 else 0
    return {
        "status": "healthy",
        "uptime_seconds": round(time.time() - start_time, 1),
        "total_requests": total_requests,
        "error_rate": round(error_rate, 4),
        "avg_latency_ms": round(avg_latency, 2),
    }

@app.get("/v1/metrics")
async def get_metrics():
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    return {
        "total_requests":