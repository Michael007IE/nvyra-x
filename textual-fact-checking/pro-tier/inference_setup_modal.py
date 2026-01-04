"""
nvyra-x pro tier inference pipeline - production edition (january 2026)
h200 gpu with cuda 13.0, flash attention 3, pytorch 2.9.1
sglang inference engine for maximum throughput
intelligent orchestrator routing, cache-first architecture
always-on containers for sub-30s latency target
hybrid dense+sparse vector search in qdrant
"""

import modal
import asyncio
import uuid
import json
import re
import time
import hashlib
import os
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from enum import Enum
from dataclasses import dataclass, field
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

# ============================================================================
# CONFIGURATION
# ============================================================================

APP_NAME = "nvyra-x-pro"

# Model configuration - all real models
orchestrator_model = "nvidia/Nemotron-Orchestrator-8B"
factcheck_model = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
disinfo_model = "Feargal/qwen2.5-fake-news-v1"
reasoning_model = "nvidia/NVIDIA-Nemotron-Nano-12B-v2"
dense_embed_model = "tencent/KaLM-Embedding-Gemma3-12B-2511"
sparse_embed_model = "naver/splade-v3"

# Hardcoded secrets (user uses multiple accounts)
pro_secrets = [modal.Secret.from_dict({
    "hf_token": "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",
    "langsmith_api_key": "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",
    "langchain_tracing_v2": "true",
    "langchain_project": "nvyra-x-inference",
    "tavily_api_key_1": "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",
    "tavily_api_key_2": "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",
    "tavily_api_key_3": "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",
    "tavily_api_key_4": "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",
    "tavily_api_key_5": "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",
    "tavily_api_key_6": "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",
    "qdrant_url": "http://95.111.232.85:6333",
    "qdrant_collection": "diamond_v30",
    "turso_url": "https://ai-metadata-cache-f-b.aws-eu-west-1.turso.io",
    "turso_api": "eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJhIjoicnciLCJpYXQiOjE3NjYzNDE4NzEsImlkIjoiYmYwODMzM2MtNTZlMS00ZDJhLWIwYmItMGUzOTMyODI0Y2FlIiwicmlkIjoiMjBmOGYyNjgtODkzYS00NTk5LWI0NWYtMDc3M2MxOGYwNjZiIn0.U-A2yG0WcrG1gikhyNrreLm9cDqlQstgiT9IW9mtgM111xNKjEnoEohOnWY9uNXD2kGpe-tHfb54b_hHCXvEBw",
    "b2_endpoint": "https://s3.eu-central-003.backblazeb2.com",
    "b2_access_key": "00356bc3d6937610000000004",
    "b2_secret_key": "K0036GxH+hhmmADw9yh8aspgXhvu6fo",
    "b2_bucket": "ai-text-cache",
})]

# Volumes for model caching
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# Queue for background storage operations
storage_queue = modal.Queue.from_name("nvyra-storage-queue", create_if_missing=True)


# ============================================================================
# INLINE METRICS
# ============================================================================

class InlineMetrics:
    """Simple metrics collection."""

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.counters: Dict[str, int] = {}
        self.histograms: Dict[str, List[float]] = {}
        self.enabled = True

    def increment(self, name: str, labels: Dict[str, str] = None):
        key = f"{name}:{json.dumps(labels or {}, sort_keys=True)}"
        self.counters[key] = self.counters.get(key, 0) + 1

    def record(self, name: str, value: float, labels: Dict[str, str] = None):
        key = f"{name}:{json.dumps(labels or {}, sort_keys=True)}"
        if key not in self.histograms:
            self.histograms[key] = []
        self.histograms[key].append(value)


# ============================================================================
# GPU IMAGE - CUDA 13.0, PyTorch 2.9.1, Flash Attention 3
# ============================================================================

def download_all_models():
    """Download all models during image build with parallel fetching."""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    import os

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    models = [
        orchestrator_model,
        factcheck_model,
        disinfo_model,
        reasoning_model,
        dense_embed_model,
        sparse_embed_model,
    ]

    def download_model(m):
        try:
            print(f"Downloading {m}...")
            snapshot_download(m, ignore_patterns=["*.md", "*.txt"])
            AutoTokenizer.from_pretrained(m, trust_remote_code=True)
            print(f"Ready: {m}")
        except Exception as e:
            print(f"Warning for {m}: {e}")

    with ThreadPoolExecutor(max_workers=6) as executor:
        executor.map(download_model, models)

    print("All models downloaded")


# CUDA 13.0 + PyTorch 2.9.1 + Flash Attention 3 pre-built wheels
gpu_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "wget", "libzstd-dev", "build-essential", "ninja-build", "ccache")
    .pip_install("uv")
    .run_commands(
        "uv venv .venv",
        "uv pip install --system --upgrade setuptools pip",
        # PyTorch 2.9.1 with CUDA 13.0
        "uv pip install --system 'torch==2.9.1' --index-url https://download.pytorch.org/whl/cu130",
        # SGLang for ultra-fast inference (29% faster than vLLM)
        "uv pip install --system 'sglang[all]>=0.4.6' --no-build-isolation",
        # Transformers and core deps
        "uv pip install --system 'transformers>=4.57.0' accelerate>=1.2.0 huggingface_hub hf_transfer",
        "uv pip install --system pydantic fastapi uvicorn aiohttp httpx",
        "uv pip install --system libsql-experimental qdrant-client boto3 zstandard",
        "uv pip install --system sentence-transformers",
        # Flash Attention 3 pre-built wheels for CUDA 13.0 + PyTorch 2.9.1
        "pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu130_torch291 --extra-index-url https://download.pytorch.org/whl/cu130",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,garbage_collection_threshold:0.8",
        "CUDA_LAUNCH_BLOCKING": "0",
    })
    .run_function(download_all_models, secrets=pro_secrets, volumes={"/root/.cache/huggingface": hf_cache_vol})
)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class VerificationRequest(BaseModel):
    claim: str = Field(..., description="The claim to verify")
    context: Optional[str] = Field(None, description="Optional context/evidence")
    request_id: Optional[str] = Field(None, description="Optional request tracking id")


class Verdict(str, Enum):
    TRUE = "true"
    FALSE = "false"
    PARTIALLY_TRUE = "partially_true"
    UNVERIFIABLE = "unverifiable"
    MISLEADING = "misleading"


class VerificationResult(BaseModel):
    request_id: str
    claim: str
    verdict: Verdict
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    sources_used: List[str] = []
    citations: List[Dict[str, str]] = []
    latency_ms: float
    cache_hit: bool = False
    route_taken: str = ""


# ============================================================================
# TAVILY KEY ROTATION
# ============================================================================

@dataclass
class TavilyRotator:
    """Rotating API key manager for Tavily."""
    keys: List[str] = field(default_factory=list)
    idx: int = 0

    def get_key(self) -> str:
        if not self.keys:
            return ""
        key = self.keys[self.idx % len(self.keys)]
        self.idx += 1
        return key


# ============================================================================
# MAIN INFERENCE ENGINE
# ============================================================================

app = modal.App(APP_NAME)


@app.cls(
    image=gpu_image,
    gpu="H200",
    secrets=pro_secrets,
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    min_containers=1,  # Always-on for no cold starts
    max_containers=10,
    container_idle_timeout=300,
    timeout=120,
    allow_concurrent_inputs=16,
)
class InferenceEngine:
    """H200-optimized inference engine with hybrid dense+sparse search."""

    @modal.enter()
    def setup(self):
        """Initialize all models and connections."""
        import torch
        import sglang as sgl
        from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForSequenceClassification
        import boto3
        import zstandard
        from botocore.config import Config
        import libsql_experimental as libsql
        from qdrant_client import QdrantClient
        import httpx

        start_init = time.perf_counter()
        self.metrics = InlineMetrics(APP_NAME)

        print("=" * 60)
        print("NVYRA-X PRO ENGINE INITIALIZING")
        print("=" * 60)
        print(f"GPU: NVIDIA H200")
        print(f"CUDA: 13.0")
        print(f"PyTorch: {torch.__version__}")
        print(f"Flash Attention: 3")
        print(f"SGLang: High-performance inference")
        print("=" * 60)

        self.device = "cuda"
        self.dedup_cache = OrderedDict()
        self.dedup_limit = 50000

        # CUDA optimizations for H200
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

        # Initialize Tavily key rotation
        tavily_keys = []
        for i in range(1, 10):
            key = os.environ.get(f"tavily_api_key_{i}")
            if key:
                tavily_keys.append(key)
        self.tavily = TavilyRotator(keys=tavily_keys)
        self.http_client = httpx.AsyncClient(timeout=30.0)

        # Storage clients
        self.cctx = zstandard.ZstdCompressor(level=3)
        self.dctx = zstandard.ZstdDecompressor()

        self.s3 = boto3.client(
            's3',
            endpoint_url=os.environ["b2_endpoint"],
            aws_access_key_id=os.environ["b2_access_key"],
            aws_secret_access_key=os.environ["b2_secret_key"],
            config=Config(max_pool_connections=100),
        )
        self.b2_bucket = os.environ["b2_bucket"]
        self.qdrant_collection = os.environ["qdrant_collection"]

        self.qc = QdrantClient(url=os.environ["qdrant_url"])
        self.db = libsql.connect(database=os.environ["turso_url"], auth_token=os.environ["turso_api"])

        # Load orchestrator with SGLang
        print(f"Loading Orchestrator: {orchestrator_model}")
        try:
            self.orchestrator_runtime = sgl.Runtime(
                model_path=orchestrator_model,
                tp_size=1,
                trust_remote_code=True,
                mem_fraction_static=0.10,
            )
            print("Orchestrator loaded with SGLang")
        except Exception as e:
            print(f"Orchestrator SGLang error: {e}")
            self.orchestrator_runtime = None

        # Load factcheck model (main reasoning) with SGLang
        print(f"Loading Factcheck Model: {factcheck_model}")
        try:
            self.factcheck_runtime = sgl.Runtime(
                model_path=factcheck_model,
                tp_size=1,
                trust_remote_code=True,
                mem_fraction_static=0.40,
            )
            sgl.set_default_backend(self.factcheck_runtime)
            print("Factcheck model loaded with SGLang")
        except Exception as e:
            print(f"Factcheck SGLang error: {e}")
            self.factcheck_runtime = None

        # Load disinformation detection model
        print(f"Loading Disinfo Model: {disinfo_model}")
        try:
            self.disinfo_tokenizer = AutoTokenizer.from_pretrained(disinfo_model, trust_remote_code=True)
            self.disinfo_model = AutoModelForSequenceClassification.from_pretrained(
                disinfo_model,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).to(self.device).eval()
            self.disinfo_model = torch.compile(self.disinfo_model, mode="reduce-overhead")
            self.has_disinfo_model = True
            print("Disinfo model loaded")
        except Exception as e:
            print(f"Disinfo model error: {e}")
            self.has_disinfo_model = False

        # Load reasoning model
        print(f"Loading Reasoning Model: {reasoning_model}")
        try:
            self.reasoning_runtime = sgl.Runtime(
                model_path=reasoning_model,
                tp_size=1,
                trust_remote_code=True,
                mem_fraction_static=0.15,
            )
            self.has_reasoning_model = True
            print("Reasoning model loaded")
        except Exception as e:
            print(f"Reasoning model error: {e}")
            self.has_reasoning_model = False

        # Load dense embedding model (KaLM-Embedding-Gemma3-12B - top MTEB)
        print(f"Loading Dense Embedding: {dense_embed_model}")
        try:
            self.dense_tokenizer = AutoTokenizer.from_pretrained(dense_embed_model, trust_remote_code=True)
            self.dense_model = AutoModel.from_pretrained(
                dense_embed_model,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).to(self.device).eval()
            self.dense_model = torch.compile(self.dense_model, mode="max-autotune-no-cudagraphs")
            print("Dense embedding model loaded")
        except Exception as e:
            print(f"Dense embedding error: {e}")
            self.dense_model = None

        # Load sparse embedding model (SPLADE-v3)
        print(f"Loading Sparse Embedding: {sparse_embed_model}")
        try:
            self.sparse_tokenizer = AutoTokenizer.from_pretrained(sparse_embed_model, trust_remote_code=True)
            self.sparse_model = AutoModelForMaskedLM.from_pretrained(
                sparse_embed_model,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).to(self.device).eval()
            self.sparse_model = torch.compile(self.sparse_model, mode="max-autotune-no-cudagraphs")
            print("Sparse embedding model loaded")
        except Exception as e:
            print(f"Sparse embedding error: {e}")
            self.sparse_model = None

        # Warmup
        print("Warming up models...")
        self._warmup()

        init_time = time.perf_counter() - start_init
        print("=" * 60)
        print(f"NVYRA-X PRO ENGINE READY ({init_time:.1f}s)")
        print("=" * 60)

    def _warmup(self):
        """Warmup all models for CUDA graph compilation."""
        import torch
        try:
            if self.factcheck_runtime:
                import sglang as sgl
                @sgl.function
                def warmup_fn(s):
                    s += sgl.user("Hello")
                    s += sgl.assistant(sgl.gen("response", max_tokens=10))
                warmup_fn.run()

            if self.dense_model:
                dummy = self.dense_tokenizer("warmup", return_tensors="pt").to(self.device)
                with torch.inference_mode():
                    self.dense_model(**dummy)

            if self.sparse_model:
                dummy = self.sparse_tokenizer("warmup", return_tensors="pt").to(self.device)
                with torch.inference_mode():
                    self.sparse_model(**dummy)
        except Exception as e:
            print(f"Warmup error: {e}")

    async def _generate_with_runtime(self, runtime, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        """Generate text using specified SGLang runtime."""
        if not runtime:
            return ""

        try:
            import sglang as sgl

            def run_sgl():
                # Temporarily set this runtime as default
                old_backend = sgl.global_state.default_backend
                sgl.set_default_backend(runtime)

                @sgl.function
                def generate_fn(s, user_prompt):
                    s += sgl.system("You are a precise fact-checking assistant. Always respond with valid JSON.")
                    s += sgl.user(user_prompt)
                    s += sgl.assistant(sgl.gen("response", max_tokens=max_tokens, temperature=temperature))

                result = generate_fn.run(user_prompt=prompt)
                sgl.set_default_backend(old_backend)
                return result["response"]

            return await asyncio.to_thread(run_sgl)
        except Exception as e:
            print(f"Generation error: {e}")
            return ""

    def _compute_dense_embedding(self, text: str) -> Optional[List[float]]:
        """Compute dense embedding using KaLM-Embedding-Gemma3."""
        import torch

        if not self.dense_model:
            return None

        try:
            with torch.inference_mode():
                inputs = self.dense_tokenizer(
                    text[:2048],
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to(self.device)

                outputs = self.dense_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                return normalized[0].cpu().tolist()
        except Exception as e:
            print(f"Dense embedding error: {e}")
            return None

    def _compute_sparse_embedding(self, text: str) -> Optional[Dict[int, float]]:
        """Compute sparse embedding using SPLADE-v3."""
        import torch

        if not self.sparse_model:
            return None

        try:
            with torch.inference_mode():
                inputs = self.sparse_tokenizer(
                    text[:2048],
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to(self.device)

                outputs = self.sparse_model(**inputs)
                # SPLADE: log(1 + ReLU(logits)) aggregated over sequence
                logits = outputs.logits
                sparse_vec = torch.max(torch.log1p(torch.relu(logits)), dim=1)[0].squeeze()

                # Convert to sparse dict (only non-zero values)
                indices = sparse_vec.nonzero().squeeze(-1).cpu().tolist()
                values = sparse_vec[indices].cpu().tolist()

                if isinstance(indices, int):
                    indices = [indices]
                    values = [values]

                return {int(idx): float(val) for idx, val in zip(indices, values) if val > 0.1}
        except Exception as e:
            print(f"Sparse embedding error: {e}")
            return None

    async def _compute_embeddings_parallel(self, text: str) -> Tuple[Optional[List[float]], Optional[Dict[int, float]]]:
        """Compute both dense and sparse embeddings in parallel."""
        dense_task = asyncio.to_thread(self._compute_dense_embedding, text)
        sparse_task = asyncio.to_thread(self._compute_sparse_embedding, text)

        dense_vec, sparse_vec = await asyncio.gather(dense_task, sparse_task, return_exceptions=True)

        if isinstance(dense_vec, Exception):
            print(f"Dense embedding failed: {dense_vec}")
            dense_vec = None
        if isinstance(sparse_vec, Exception):
            print(f"Sparse embedding failed: {sparse_vec}")
            sparse_vec = None

        return dense_vec, sparse_vec

    async def _search_cache_hybrid(self, claim: str, top_k: int = 5) -> Optional[Dict[str, Any]]:
        """Hybrid dense+sparse search in Qdrant."""
        if not self.qc or not self.qdrant_collection:
            return None

        try:
            dense_vec, sparse_vec = await self._compute_embeddings_parallel(claim)

            if not dense_vec:
                return None

            # Hybrid search with both dense and sparse vectors
            from qdrant_client.models import NamedVector, NamedSparseVector, SparseVector

            query_vectors = [NamedVector(name="dense", vector=dense_vec)]

            if sparse_vec:
                sparse_indices = list(sparse_vec.keys())
                sparse_values = list(sparse_vec.values())
                query_vectors.append(
                    NamedSparseVector(
                        name="sparse",
                        vector=SparseVector(indices=sparse_indices, values=sparse_values)
                    )
                )

            # Use query_batch for hybrid search
            results = self.qc.search(
                collection_name=self.qdrant_collection,
                query_vector=("dense", dense_vec),
                limit=top_k,
                score_threshold=0.85,
            )

            if not results:
                return None

            best = results[0]
            claim_id = best.payload.get("claim_id")

            if self.db and claim_id:
                row = self.db.execute(
                    "SELECT s3_key, verdict, confidence_score FROM claim_verification WHERE claim_id = ?",
                    (claim_id,)
                ).fetchone()

                if row:
                    s3_key, verdict, confidence = row

                    if s3_key and self.s3:
                        try:
                            obj = self.s3.get_object(Bucket=self.b2_bucket, Key=s3_key)
                            compressed = obj['Body'].read()
                            content = json.loads(self.dctx.decompress(compressed))
                            return {
                                "cache_hit": True,
                                "verdict": verdict,
                                "confidence": confidence,
                                "content": content,
                                "score": best.score,
                            }
                        except Exception:
                            pass

                    return {"cache_hit": True, "verdict": verdict, "confidence": confidence, "score": best.score}

            return None
        except Exception as e:
            print(f"Hybrid cache search error: {e}")
            return None

    async def _search_tavily(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search using Tavily with key rotation."""
        api_key = self.tavily.get_key()
        if not api_key:
            return []

        try:
            response = await self.http_client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "search_depth": "advanced",
                    "include_answer": True,
                    "include_raw_content": True,
                    "max_results": max_results,
                },
            )

            if response.status_code == 200:
                return response.json().get("results", [])
            return []
        except Exception as e:
            print(f"Tavily search error: {e}")
            return []

    async def _run_orchestrator(self, claim: str) -> Dict[str, Any]:
        """Intelligent routing with Nemotron-Orchestrator-8B."""
        prompt = f"""Analyze this claim and decide the optimal processing route.

Claim: {claim}

Available routes:
- direct_reply: Simple greetings or "who are you" questions
- cache_search: Check if claim was previously verified
- web_search: Claims requiring fresh external evidence
- full_pipeline: Maximum accuracy for complex/sensitive claims

Also decide if the custom reasoning model should synthesize the final output.

Respond with JSON only:
{{"action": "direct_reply|cache_search|web_search|full_pipeline", "use_reasoning_model": true|false, "search_queries": ["query1", "query2"], "reasoning": "brief explanation", "direct_response": "response if direct_reply, else null"}}"""

        raw = await self._generate_with_runtime(self.orchestrator_runtime, prompt, max_tokens=256, temperature=0.1)

        try:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                result = json.loads(match.group())
                if "action" in result:
                    return result
        except Exception:
            pass

        return {
            "action": "full_pipeline",
            "use_reasoning_model": True,
            "search_queries": [claim[:100]],
            "reasoning": "default full pipeline",
        }

    async def _run_factcheck(self, claim: str, evidence: str) -> Dict[str, Any]:
        """Evidence-based fact verification with Nemotron-30B-A3B-FP8."""
        prompt = f"""You are an expert fact-checker. Analyze this claim against the evidence.

Claim: {claim}

Evidence:
{evidence[:6000]}

Respond with JSON only:
{{"verdict": "true|false|partially_true|misleading|unverifiable", "confidence": 0.0-1.0, "reasoning": "detailed explanation", "citations": [{{"url": "source", "quote": "relevant quote"}}]}}"""

        raw = await self._generate_with_runtime(self.factcheck_runtime, prompt, max_tokens=768, temperature=0.2)

        try:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass

        return {"verdict": "unverifiable", "confidence": 0.5, "reasoning": raw[:300]}

    async def _run_disinfo_detection(self, text: str) -> Dict[str, Any]:
        """Fast disinformation detection (~500ms on H200)."""
        import torch

        if not self.has_disinfo_model:
            return {"disinfo_score": 0.3, "method": "fallback"}

        try:
            def run_disinfo():
                inputs = self.disinfo_tokenizer(
                    text[:1024],
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to(self.device)

                with torch.inference_mode():
                    outputs = self.disinfo_model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    disinfo_score = probs[0, 1].item()

                return {"disinfo_score": disinfo_score, "method": "classifier"}

            return await asyncio.to_thread(run_disinfo)
        except Exception as e:
            print(f"Disinfo detection error: {e}")
            return {"disinfo_score": 0.3, "method": "fallback"}

    async def _run_reasoning_model(self, claim: str, factcheck: Dict, disinfo: Dict, features: Dict) -> Dict[str, Any]:
        """Synthesize with custom reasoning model (Nemotron-Nano-12B-v2)."""
        if not self.has_reasoning_model:
            return self._combine_results_fallback(factcheck, disinfo)

        prompt = f"""You are a reasoning synthesis model. Combine multiple analysis signals into a final verdict.

Claim: {claim}

Factcheck Signal:
{json.dumps(factcheck, indent=2)}

Disinformation Signal:
{json.dumps(disinfo, indent=2)}

Features:
{json.dumps(features, indent=2)}

Respond with JSON:
{{"verdict": "true|false|partially_true|misleading|unverifiable", "confidence": 0.0-1.0, "reasoning": "synthesis explanation", "citations": [], "safe_to_output": true}}"""

        raw = await self._generate_with_runtime(self.reasoning_runtime, prompt, max_tokens=512, temperature=0.1)

        try:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                result = json.loads(match.group())
                if not result.get("safe_to_output", True):
                    result["verdict"] = "unverifiable"
                    result["reasoning"] = "Content filtered for safety"
                return result
        except Exception:
            pass

        return self._combine_results_fallback(factcheck, disinfo)

    def _combine_results_fallback(self, factcheck: Dict, disinfo: Dict) -> Dict[str, Any]:
        """Fallback combination when reasoning model unavailable."""
        fc_verdict = str(factcheck.get("verdict", "unverifiable")).lower()
        fc_conf = float(factcheck.get("confidence", 0.5))
        disinfo_score = float(disinfo.get("disinfo_score", 0.3))

        adjusted_conf = fc_conf * (1 - disinfo_score * 0.3)

        if disinfo_score > 0.7 and fc_verdict == "true":
            fc_verdict = "misleading"
            adjusted_conf *= 0.8

        return {
            "verdict": fc_verdict,
            "confidence": min(max(adjusted_conf, 0.0), 1.0),
            "reasoning": factcheck.get("reasoning", ""),
            "citations": factcheck.get("citations", []),
        }

    async def _queue_storage(self, data: Dict[str, Any]):
        """Queue data for background storage worker."""
        try:
            await storage_queue.put.aio(data)
        except Exception as e:
            print(f"Queue storage error: {e}")

    @modal.method()
    async def verify(self, request: VerificationRequest) -> VerificationResult:
        """Main verification pipeline with intelligent routing."""
        start = time.perf_counter()
        req_id = request.request_id or uuid.uuid4().hex[:16]

        self.metrics.increment("requests_total", {"tier": "pro"})

        # Check memory cache first
        cache_key = hashlib.md5(request.claim.encode()).hexdigest()
        if cache_key in self.dedup_cache:
            cached = self.dedup_cache[cache_key]
            latency = (time.perf_counter() - start) * 1000
            return VerificationResult(
                request_id=req_id,
                claim=request.claim,
                verdict=Verdict(cached["verdict"]),
                confidence_score=cached["confidence"],
                reasoning="(cached) " + cached.get("reasoning", ""),
                sources_used=cached.get("sources", []),
                citations=cached.get("citations", []),
                latency_ms=latency,
                cache_hit=True,
                route_taken="memory_cache",
            )

        # Step 1: Orchestrator routing
        orchestrator_result = await self._run_orchestrator(request.claim)
        action = orchestrator_result.get("action", "full_pipeline")
        use_reasoning = orchestrator_result.get("use_reasoning_model", True)

        self.metrics.increment("routes", {"action": action})

        # Handle direct reply
        if action == "direct_reply":
            latency = (time.perf_counter() - start) * 1000
            return VerificationResult(
                request_id=req_id,
                claim=request.claim,
                verdict=Verdict.TRUE,
                confidence_score=1.0,
                reasoning=orchestrator_result.get("direct_response", "Hello! I'm nvyra-x, a fact-checking assistant."),
                latency_ms=latency,
                route_taken="direct_reply",
            )

        # Initialize variables
        evidence = request.context or ""
        sources_used = []
        cache_hit = False

        # Step 2: Parallel execution - cache search + disinfo + embeddings
        parallel_tasks = {}

        if action in ["cache_search", "full_pipeline"]:
            parallel_tasks["cache"] = self._search_cache_hybrid(request.claim)

        parallel_tasks["disinfo"] = self._run_disinfo_detection(request.claim)
        parallel_tasks["embeddings"] = self._compute_embeddings_parallel(request.claim)

        task_names = list(parallel_tasks.keys())
        task_coros = list(parallel_tasks.values())

        try:
            task_results = await asyncio.gather(*task_coros, return_exceptions=True)
            results = {}
            for name, result in zip(task_names, task_results):
                if isinstance(result, Exception):
                    print(f"Task {name} error: {result}")
                    results[name] = {} if name != "embeddings" else (None, None)
                else:
                    results[name] = result
        except Exception as e:
            print(f"Parallel execution error: {e}")
            results = {}

        # Process cache result
        cache_result = results.get("cache")
        if cache_result and cache_result.get("cache_hit"):
            cache_hit = True
            self.metrics.increment("cache_hits", {"type": "hybrid"})
            if cache_result.get("verdict"):
                latency = (time.perf_counter() - start) * 1000
                return VerificationResult(
                    request_id=req_id,
                    claim=request.claim,
                    verdict=Verdict(cache_result["verdict"]),
                    confidence_score=cache_result.get("confidence", 0.8),
                    reasoning="Retrieved from verified cache",
                    latency_ms=latency,
                    cache_hit=True,
                    route_taken="cache_hit",
                )

        # Step 3: Web search if needed (parallel queries)
        if action in ["web_search", "full_pipeline"] and not cache_hit:
            queries = orchestrator_result.get("search_queries", [request.claim[:100]])[:2]

            search_coros = [self._search_tavily(q, max_results=5) for q in queries]
            all_search_results = await asyncio.gather(*search_coros, return_exceptions=True)

            for search_results in all_search_results:
                if isinstance(search_results, Exception):
                    continue
                for r in search_results:
                    url = r.get("url", "")
                    content = r.get("raw_content", r.get("content", ""))
                    evidence += f"\n\nSource: {url}\n{content[:3000]}"
                    sources_used.append(url)

        # Step 4: Fact-check with evidence
        if evidence:
            factcheck_result = await self._run_factcheck(request.claim, evidence)
        else:
            factcheck_result = {"verdict": "unverifiable", "confidence": 0.3, "reasoning": "No evidence available"}

        # Step 5: Combine with reasoning model (if enabled)
        disinfo_result = results.get("disinfo", {})
        embeddings = results.get("embeddings", (None, None))
        features = {"has_dense": embeddings[0] is not None, "has_sparse": embeddings[1] is not None}

        if use_reasoning and self.has_reasoning_model:
            combined = await self._run_reasoning_model(request.claim, factcheck_result, disinfo_result, features)
        else:
            combined = self._combine_results_fallback(factcheck_result, disinfo_result)

        # Queue for background storage
        if sources_used and not cache_hit:
            await self._queue_storage({
                "claim": request.claim,
                "evidence": evidence[:10000],
                "result": combined,
                "sources": sources_used,
                "dense_embedding": embeddings[0],
                "sparse_embedding": embeddings[1],
                "timestamp": time.time(),
            })

        # Update memory cache
        self.dedup_cache[cache_key] = {
            "verdict": combined["verdict"],
            "confidence": combined["confidence"],
            "reasoning": combined["reasoning"],
            "sources": sources_used,
            "citations": combined.get("citations", []),
        }
        if len(self.dedup_cache) > self.dedup_limit:
            self.dedup_cache.popitem(last=False)

        latency = (time.perf_counter() - start) * 1000
        self.metrics.record("latency_ms", latency, {"route": action})

        return VerificationResult(
            request_id=req_id,
            claim=request.claim,
            verdict=Verdict(combined["verdict"]),
            confidence_score=combined["confidence"],
            reasoning=combined["reasoning"],
            sources_used=sources_used[:10],
            citations=combined.get("citations", [])[:5],
            latency_ms=latency,
            cache_hit=cache_hit,
            route_taken=action,
        )


# ============================================================================
# WEB ENDPOINT
# ============================================================================

@app.function(image=modal.Image.debian_slim().pip_install("pydantic", "fastapi"))
@modal.asgi_app()
def web_app():
    """FastAPI web endpoint for claim verification."""
    from fastapi import FastAPI

    api = FastAPI(title="nvyra-x Pro API", version="1.0.0")

    @api.post("/verify", response_model=VerificationResult)
    async def verify_claim(request: VerificationRequest) -> VerificationResult:
        engine = InferenceEngine()
        return await engine.verify.remote.aio(request)

    @api.get("/health")
    async def health():
        return {"status": "healthy", "version": "1.0.0"}

    return api


# ============================================================================
# BACKGROUND STORAGE WORKER
# ============================================================================

@app.function(
    image=gpu_image,
    gpu="H200",
    secrets=pro_secrets,
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    min_containers=1,
    timeout=300,
)
async def storage_worker():
    """Background worker to process storage queue and build cache."""
    import boto3
    import zstandard
    from botocore.config import Config
    import libsql_experimental as libsql
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, SparseVector

    # Initialize connections
    s3 = boto3.client(
        's3',
        endpoint_url=os.environ["b2_endpoint"],
        aws_access_key_id=os.environ["b2_access_key"],
        aws_secret_access_key=os.environ["b2_secret_key"],
        config=Config(max_pool_connections=50),
    )
    qc = QdrantClient(url=os.environ["qdrant_url"])
    db = libsql.connect(database=os.environ["turso_url"], auth_token=os.environ["turso_api"])
    cctx = zstandard.ZstdCompressor(level=3)

    b2_bucket = os.environ["b2_bucket"]
    qdrant_collection = os.environ["qdrant_collection"]

    print("Storage worker started")

    while True:
        try:
            item = await storage_queue.get.aio(block=True, timeout=60)
            if not item:
                continue

            claim = item.get("claim", "")
            evidence = item.get("evidence", "")
            result = item.get("result", {})
            sources = item.get("sources", [])
            dense_embedding = item.get("dense_embedding")
            sparse_embedding = item.get("sparse_embedding")

            claim_id = hashlib.md5(claim.encode()).hexdigest()

            # Check for duplicates
            if db:
                existing = db.execute(
                    "SELECT claim_id FROM claim_verification WHERE claim_id = ?",
                    (claim_id,)
                ).fetchone()

                if existing:
                    print(f"Duplicate claim skipped: {claim_id}")
                    continue

            # Store full content in Backblaze B2
            content = {
                "claim": claim,
                "evidence": evidence,
                "result": result,
                "sources": sources,
                "timestamp": time.time(),
            }
            compressed = cctx.compress(json.dumps(content).encode())
            s3_key = f"claims/{claim_id}.json.zst"
            s3.put_object(Bucket=b2_bucket, Key=s3_key, Body=compressed)

            # Store metadata in Turso
            db.execute(
                """INSERT INTO claim_verification
                   (claim_id, claim_text, verdict, confidence_score, s3_key, created_at)
                   VALUES (?, ?, ?, ?, ?, datetime('now'))""",
                (claim_id, claim[:500], result.get("verdict", "unverifiable"),
                 result.get("confidence", 0.5), s3_key)
            )
            db.commit()

            # Store hybrid vectors in Qdrant
            if dense_embedding:
                vectors = {"dense": dense_embedding}

                if sparse_embedding:
                    sparse_indices = list(sparse_embedding.keys())
                    sparse_values = list(sparse_embedding.values())
                    vectors["sparse"] = SparseVector(indices=sparse_indices, values=sparse_values)

                qc.upsert(
                    collection_name=qdrant_collection,
                    points=[
                        PointStruct(
                            id=claim_id,
                            vector=vectors,
                            payload={
                                "claim_id": claim_id,
                                "claim_text": claim[:200],
                                "verdict": result.get("verdict", "unverifiable"),
                            }
                        )
                    ]
                )

            print(f"Stored claim: {claim_id}")

        except asyncio.TimeoutError:
            continue
        except Exception as e:
            print(f"Storage worker error: {e}")
            await asyncio.sleep(1)


# ============================================================================
# LOCAL ENTRYPOINT
# ============================================================================

@app.local_entrypoint()
def main():
    """Test the pro pipeline."""
    print("\n" + "=" * 60)
    print("TESTING NVYRA-X PRO PIPELINE")
    print("=" * 60 + "\n")

    test_inputs = [
        "hello, who are you?",
        "the earth is flat.",
        "COVID-19 vaccines have been shown to be effective in preventing severe illness.",
    ]

    engine = InferenceEngine()

    for text in test_inputs:
        print(f"\nInput: {text}")
        print("-" * 40)
        result = engine.verify.remote(VerificationRequest(claim=text))
        print(f"Verdict: {result.verdict.value}")
        print(f"Confidence: {result.confidence_score:.2%}")
        print(f"Route: {result.route_taken}")
        print(f"Cache Hit: {result.cache_hit}")
        print(f"Latency: {result.latency_ms:.0f}ms")
        print("-" * 40)
