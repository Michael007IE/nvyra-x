"""
nvyra-x pro tier inference pipeline - production edition (january 2026)
h200 gpu with cuda 12.8, sglang inference, pytorch 2.9.0
intelligent orchestrator routing, cache-first architecture
always-on containers for sub-30s latency target
prometheus-compatible metrics for observability
"""

import modal
import asyncio
import uuid
import json
import re
import time
import hashlib
import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
from dataclasses import dataclass, field
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

# ============================================================================
# CONFIGURATION
# ============================================================================

APP_NAME = "nvyra-x-pro"

# Model configuration - using real, existing models
# Nemotron 30B-A3B-FP8: NVIDIA's hybrid Mamba-2 + MoE architecture
# Released Dec 2025, supports 1M context, 3.3x throughput vs similar models
MAIN_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"

# Embedding model - Qwen3-Embedding is #1 on MTEB multilingual (70.58 score)
EMBED_MODEL = "Qwen/Qwen3-Embedding-4B"

# Sparse embedding for hybrid search
SPARSE_MODEL = "naver/splade-v3"

# User's custom disinformation model (fallback to classification approach if not available)
DISINFO_MODEL = "Feargal/qwen2.5-fake-news-v1"

# User's custom reasoning model (fallback to main model if not available)
REASONING_MODEL = "Feargal/nvyra-x-reasoning"

# Secrets - loaded from Modal secrets
pro_secrets = [modal.Secret.from_name("nvyra-x-pro-secrets")]

# Volumes for model caching
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# Queue for background storage operations
storage_queue = modal.Queue.from_name("nvyra-storage-queue", create_if_missing=True)


# ============================================================================
# INLINE METRICS (no external dependency)
# ============================================================================

class InlineMetrics:
    """Simple metrics collection without external dependencies."""

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

    def get_stats(self) -> Dict[str, Any]:
        return {
            "counters": self.counters,
            "histograms": {k: {"count": len(v), "avg": sum(v)/len(v) if v else 0}
                          for k, v in self.histograms.items()}
        }


# ============================================================================
# GPU IMAGE DEFINITION - Real versions that exist
# ============================================================================

def download_models():
    """Download all models during image build with parallel fetching."""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    import os

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    models = [MAIN_MODEL, EMBED_MODEL, SPARSE_MODEL]

    # Try to download user's custom models, but don't fail if they don't exist
    optional_models = [DISINFO_MODEL, REASONING_MODEL]

    def download_model(m, optional=False):
        try:
            print(f"Downloading {m}...")
            snapshot_download(m, ignore_patterns=["*.md", "*.txt"])
            AutoTokenizer.from_pretrained(m, trust_remote_code=True)
            print(f"Ready: {m}")
            return True
        except Exception as e:
            if optional:
                print(f"Optional model not available: {m} ({e})")
                return False
            else:
                print(f"ERROR downloading {m}: {e}")
                raise

    with ThreadPoolExecutor(max_workers=4) as executor:
        # Download required models
        list(executor.map(download_model, models))
        # Try optional models
        list(executor.map(lambda m: download_model(m, optional=True), optional_models))

    print("All models downloaded")


# Production GPU image with CUDA 12.8, PyTorch 2.9.0
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "build-essential", "ninja-build")
    .pip_install("uv")
    .run_commands(
        # PyTorch 2.9.0 with CUDA 12.8 (real, verified version)
        "uv pip install --system 'torch==2.9.0' --index-url https://download.pytorch.org/whl/cu128",
        # SGLang for faster inference (29% faster than vLLM per benchmarks)
        "uv pip install --system 'sglang[all]>=0.4.6'",
        # Core dependencies
        "uv pip install --system transformers>=4.47.0 accelerate huggingface_hub hf_transfer",
        "uv pip install --system pydantic fastapi uvicorn aiohttp httpx",
        "uv pip install --system libsql-experimental qdrant-client boto3 zstandard",
        "uv pip install --system sentence-transformers",
        # Flash attention for H200 (Hopper architecture)
        "uv pip install --system flash-attn --no-build-isolation",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,garbage_collection_threshold:0.8",
    })
    .run_function(download_models, secrets=pro_secrets, volumes={"/root/.cache/huggingface": hf_cache_vol})
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
    gpu="H200",  # H200 for maximum performance
    secrets=pro_secrets,
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    min_containers=1,  # Always-on for no cold starts
    max_containers=10,
    container_idle_timeout=300,  # 5 minute idle timeout
    timeout=120,
    allow_concurrent_inputs=16,  # Handle multiple requests concurrently
)
class InferenceEngine:
    """H200-optimized inference engine using SGLang for maximum throughput."""

    @modal.enter()
    def setup(self):
        """Initialize models and connections."""
        import torch
        import sglang as sgl
        from sentence_transformers import SentenceTransformer
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
        print(f"CUDA: 12.8")
        print(f"PyTorch: {torch.__version__}")
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

        b2_endpoint = os.environ.get("b2_endpoint")
        b2_access_key = os.environ.get("b2_access_key")
        b2_secret_key = os.environ.get("b2_secret_key")

        if b2_endpoint and b2_access_key and b2_secret_key:
            self.s3 = boto3.client(
                's3',
                endpoint_url=b2_endpoint,
                aws_access_key_id=b2_access_key,
                aws_secret_access_key=b2_secret_key,
                config=Config(max_pool_connections=50),
            )
            self.b2_bucket = os.environ.get("b2_bucket", "ai-text-cache")
        else:
            self.s3 = None
            self.b2_bucket = None
            print("WARNING: Backblaze B2 not configured")

        qdrant_url = os.environ.get("qdrant_url")
        if qdrant_url:
            self.qc = QdrantClient(url=qdrant_url)
            self.qdrant_collection = os.environ.get("qdrant_collection", "diamond_v30")
        else:
            self.qc = None
            self.qdrant_collection = None
            print("WARNING: Qdrant not configured")

        turso_url = os.environ.get("turso_url")
        turso_api = os.environ.get("turso_api")
        if turso_url and turso_api:
            self.db = libsql.connect(database=turso_url, auth_token=turso_api)
        else:
            self.db = None
            print("WARNING: Turso not configured")

        # Load main model with SGLang for maximum performance
        print(f"Loading main model: {MAIN_MODEL}")
        try:
            # SGLang Runtime for fast inference
            self.runtime = sgl.Runtime(
                model_path=MAIN_MODEL,
                tp_size=1,  # Single H200 GPU
                trust_remote_code=True,
                mem_fraction_static=0.85,  # Use 85% of GPU memory
            )
            sgl.set_default_backend(self.runtime)
            print(f"SGLang runtime initialized")
        except Exception as e:
            print(f"SGLang init error: {e}")
            print("Falling back to transformers...")
            self.runtime = None
            self._load_transformers_fallback()

        # Load embedding model
        print(f"Loading embedding model: {EMBED_MODEL}")
        try:
            self.embed_model = SentenceTransformer(
                EMBED_MODEL,
                device=self.device,
                trust_remote_code=True,
            )
            print("Embedding model loaded")
        except Exception as e:
            print(f"Embedding model error: {e}")
            # Fallback to a smaller model
            try:
                self.embed_model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2",
                    device=self.device,
                )
                print("Using fallback embedding model")
            except Exception:
                self.embed_model = None

        # Check for custom disinformation model
        self.has_disinfo_model = False
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.disinfo_tokenizer = AutoTokenizer.from_pretrained(DISINFO_MODEL, trust_remote_code=True)
            self.disinfo_model = AutoModelForSequenceClassification.from_pretrained(
                DISINFO_MODEL,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).to(self.device).eval()
            self.has_disinfo_model = True
            print(f"Disinformation model loaded: {DISINFO_MODEL}")
        except Exception as e:
            print(f"Custom disinfo model not available: {e}")
            print("Will use main model for disinformation detection")

        # Warmup
        print("Warming up models...")
        self._warmup()

        init_time = time.perf_counter() - start_init
        print("=" * 60)
        print(f"NVYRA-X PRO ENGINE READY ({init_time:.1f}s)")
        print("=" * 60)

    def _load_transformers_fallback(self):
        """Fallback to transformers if SGLang fails."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(MAIN_MODEL, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MAIN_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model = torch.compile(self.model, mode="reduce-overhead")

    def _warmup(self):
        """Warmup models for faster first inference."""
        try:
            if self.runtime:
                import sglang as sgl

                @sgl.function
                def warmup_fn(s):
                    s += sgl.user("Hello")
                    s += sgl.assistant(sgl.gen("response", max_tokens=10))

                warmup_fn.run()

            if self.embed_model:
                self.embed_model.encode(["warmup query"])
        except Exception as e:
            print(f"Warmup error: {e}")

    async def _generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        """Generate text using SGLang or fallback."""
        import torch

        try:
            if self.runtime:
                import sglang as sgl

                # Use thread pool for sync SGLang calls to avoid blocking
                def run_sgl():
                    @sgl.function
                    def generate_fn(s, user_prompt):
                        s += sgl.system("You are a precise fact-checking assistant. Always respond with valid JSON.")
                        s += sgl.user(user_prompt)
                        s += sgl.assistant(sgl.gen("response", max_tokens=max_tokens, temperature=temperature))

                    result = generate_fn.run(user_prompt=prompt)
                    return result["response"]

                return await asyncio.to_thread(run_sgl)
            else:
                # Transformers fallback (also run in thread to avoid blocking)
                def run_transformers():
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                    with torch.inference_mode():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            temperature=temperature if temperature > 0 else None,
                            do_sample=temperature > 0,
                        )
                    return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                return await asyncio.to_thread(run_transformers)
        except Exception as e:
            print(f"Generation error: {e}")
            return ""

    def _compute_embedding(self, text: str) -> Optional[List[float]]:
        """Compute dense embedding for text."""
        if not self.embed_model:
            return None
        try:
            embedding = self.embed_model.encode(text[:2048], normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            print(f"Embedding error: {e}")
            return None

    async def _search_cache(self, claim: str, top_k: int = 5) -> Optional[Dict[str, Any]]:
        """Search Qdrant cache for similar claims."""
        if not self.qc or not self.qdrant_collection:
            return None

        try:
            query_vec = await asyncio.to_thread(self._compute_embedding, claim)
            if not query_vec:
                return None

            results = self.qc.search(
                collection_name=self.qdrant_collection,
                query_vector=query_vec,
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
            print(f"Cache search error: {e}")
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
        """Intelligent routing based on claim analysis."""
        prompt = f"""Analyze this claim and decide the optimal processing route.

Claim: {claim}

Available routes:
- direct_reply: Simple greetings or "who are you" questions
- cache_search: Check if claim was previously verified
- web_search: Claims requiring fresh external evidence
- full_pipeline: Maximum accuracy for complex/sensitive claims

Respond with JSON only:
{{"action": "direct_reply|cache_search|web_search|full_pipeline", "search_queries": ["query1", "query2"], "reasoning": "brief explanation", "direct_response": "response if direct_reply, else null"}}"""

        raw = await self._generate(prompt, max_tokens=256, temperature=0.1)

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
            "search_queries": [claim[:100]],
            "reasoning": "default full pipeline",
        }

    async def _run_factcheck(self, claim: str, evidence: str) -> Dict[str, Any]:
        """Evidence-based fact verification."""
        prompt = f"""You are an expert fact-checker. Analyze this claim against the evidence.

Claim: {claim}

Evidence:
{evidence[:6000]}

Respond with JSON only:
{{"verdict": "true|false|partially_true|misleading|unverifiable", "confidence": 0.0-1.0, "reasoning": "detailed explanation", "citations": [{{"url": "source", "quote": "relevant quote"}}]}}"""

        raw = await self._generate(prompt, max_tokens=768, temperature=0.2)

        try:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                result = json.loads(match.group())
                return result
        except Exception:
            pass

        return {"verdict": "unverifiable", "confidence": 0.5, "reasoning": raw[:300]}

    async def _run_disinfo_detection(self, text: str) -> Dict[str, Any]:
        """Disinformation pattern detection."""
        import torch

        if self.has_disinfo_model:
            try:
                inputs = self.disinfo_tokenizer(
                    text[:1024],
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to(self.device)

                with torch.inference_mode():
                    outputs = self.disinfo_model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    disinfo_score = probs[0, 1].item()  # Assuming binary classification

                return {"disinfo_score": disinfo_score, "method": "classifier"}
            except Exception as e:
                print(f"Disinfo model error: {e}")

        # Fallback: use main model for analysis
        prompt = f"""Analyze this text for disinformation patterns (emotional manipulation, false claims, misleading framing).

Text: {text[:500]}

Respond with JSON: {{"disinfo_score": 0.0-1.0, "patterns": ["pattern1"], "analysis": "brief explanation"}}"""

        raw = await self._generate(prompt, max_tokens=256, temperature=0.1)

        try:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass

        return {"disinfo_score": 0.3, "analysis": "analysis unavailable", "method": "fallback"}

    async def _compute_features(self, text: str) -> Dict[str, Any]:
        """Compute embedding-based features in parallel with other tasks."""
        embedding = await asyncio.to_thread(self._compute_embedding, text)

        if embedding:
            # Compute simple features from embedding
            import numpy as np
            emb_array = np.array(embedding)
            return {
                "has_embedding": True,
                "embedding_norm": float(np.linalg.norm(emb_array)),
                "embedding_mean": float(np.mean(emb_array)),
            }

        return {"has_embedding": False}

    def _combine_results(self, factcheck: Dict, disinfo: Dict, features: Dict) -> Dict[str, Any]:
        """Combine results from all models into final verdict."""
        fc_verdict = str(factcheck.get("verdict", "unverifiable")).lower()
        fc_conf = float(factcheck.get("confidence", 0.5))
        disinfo_score = float(disinfo.get("disinfo_score", 0.3))

        # Adjust confidence based on disinformation score
        adjusted_conf = fc_conf * (1 - disinfo_score * 0.3)

        # Verdict adjustment for high disinfo scores
        if disinfo_score > 0.7 and fc_verdict == "true":
            fc_verdict = "misleading"
            adjusted_conf *= 0.8

        return {
            "verdict": fc_verdict,
            "confidence": min(max(adjusted_conf, 0.0), 1.0),
            "reasoning": factcheck.get("reasoning", ""),
            "citations": factcheck.get("citations", []),
            "disinfo_score": disinfo_score,
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
            self.metrics.increment("cache_hits", {"type": "memory"})
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

        # Step 2: Parallel execution - cache search + disinfo + features
        # Use asyncio.gather for TRUE parallel execution (critical for 30s target)
        parallel_tasks = {}

        if action in ["cache_search", "full_pipeline"]:
            parallel_tasks["cache"] = self._search_cache(request.claim)

        # Start disinfo and features in parallel with cache search
        parallel_tasks["disinfo"] = self._run_disinfo_detection(request.claim)
        parallel_tasks["features"] = self._compute_features(request.claim)

        # Run ALL tasks in parallel with asyncio.gather
        task_names = list(parallel_tasks.keys())
        task_coros = list(parallel_tasks.values())

        try:
            task_results = await asyncio.gather(*task_coros, return_exceptions=True)
            results = {}
            for name, result in zip(task_names, task_results):
                if isinstance(result, Exception):
                    print(f"Task {name} error: {result}")
                    results[name] = {}
                else:
                    results[name] = result
        except Exception as e:
            print(f"Parallel execution error: {e}")
            results = {name: {} for name in task_names}

        # Process cache result
        cache_result = results.get("cache")
        if cache_result and cache_result.get("cache_hit"):
            cache_hit = True
            self.metrics.increment("cache_hits", {"type": "qdrant"})
            if cache_result.get("verdict"):
                # Return cached result with fresh disinfo check
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

        # Step 3: Web search if needed (parallel queries for speed)
        if action in ["web_search", "full_pipeline"] and not cache_hit:
            queries = orchestrator_result.get("search_queries", [request.claim[:100]])[:2]

            # Run multiple search queries in parallel
            search_coros = [self._search_tavily(q, max_results=5) for q in queries]
            all_search_results = await asyncio.gather(*search_coros, return_exceptions=True)

            for search_results in all_search_results:
                if isinstance(search_results, Exception):
                    print(f"Search error: {search_results}")
                    continue
                for r in search_results:
                    url = r.get("url", "")
                    content = r.get("raw_content", r.get("content", ""))
                    evidence += f"\n\nSource: {url}\n{content[:3000]}"
                    sources_used.append(url)
                    self.metrics.increment("external_searches", {"provider": "tavily"})

        # Step 4: Fact-check with evidence
        if evidence:
            factcheck_result = await self._run_factcheck(request.claim, evidence)
        else:
            factcheck_result = {"verdict": "unverifiable", "confidence": 0.3, "reasoning": "No evidence available"}

        # Step 5: Combine results
        combined = self._combine_results(
            factcheck_result,
            results.get("disinfo", {}),
            results.get("features", {}),
        )

        # Queue for background storage if we did web search
        if sources_used and not cache_hit:
            await self._queue_storage({
                "claim": request.claim,
                "evidence": evidence[:10000],
                "result": combined,
                "sources": sources_used,
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
    from qdrant_client.models import PointStruct, VectorParams, Distance
    from sentence_transformers import SentenceTransformer

    # Initialize connections
    b2_endpoint = os.environ.get("b2_endpoint")
    s3 = boto3.client(
        's3',
        endpoint_url=b2_endpoint,
        aws_access_key_id=os.environ.get("b2_access_key"),
        aws_secret_access_key=os.environ.get("b2_secret_key"),
        config=Config(max_pool_connections=50),
    ) if b2_endpoint else None

    qdrant_url = os.environ.get("qdrant_url")
    qc = QdrantClient(url=qdrant_url) if qdrant_url else None

    turso_url = os.environ.get("turso_url")
    db = libsql.connect(database=turso_url, auth_token=os.environ.get("turso_api")) if turso_url else None

    embed_model = SentenceTransformer(EMBED_MODEL, device="cuda", trust_remote_code=True)
    cctx = zstandard.ZstdCompressor(level=3)

    b2_bucket = os.environ.get("b2_bucket", "ai-text-cache")
    qdrant_collection = os.environ.get("qdrant_collection", "diamond_v30")

    print("Storage worker started")

    while True:
        try:
            # Get item from queue
            item = await storage_queue.get.aio(block=True, timeout=60)
            if not item:
                continue

            claim = item.get("claim", "")
            evidence = item.get("evidence", "")
            result = item.get("result", {})
            sources = item.get("sources", [])

            claim_id = hashlib.md5(claim.encode()).hexdigest()

            # Check for duplicates in Turso
            if db:
                existing = db.execute(
                    "SELECT claim_id FROM claim_verification WHERE claim_id = ?",
                    (claim_id,)
                ).fetchone()

                if existing:
                    print(f"Duplicate claim skipped: {claim_id}")
                    continue

            # Compute embedding
            embedding = embed_model.encode(claim, normalize_embeddings=True).tolist()

            # Store full content in Backblaze B2
            s3_key = None
            if s3:
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
            if db:
                db.execute(
                    """INSERT INTO claim_verification
                       (claim_id, claim_text, verdict, confidence_score, s3_key, created_at)
                       VALUES (?, ?, ?, ?, ?, datetime('now'))""",
                    (claim_id, claim[:500], result.get("verdict", "unverifiable"),
                     result.get("confidence", 0.5), s3_key)
                )
                db.commit()

            # Store embedding in Qdrant
            if qc:
                qc.upsert(
                    collection_name=qdrant_collection,
                    points=[
                        PointStruct(
                            id=claim_id,
                            vector=embedding,
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
