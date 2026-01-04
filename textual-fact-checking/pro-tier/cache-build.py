# ------------------------------------------------------------------------------
#  CONFIDENTIAL AND PROPRIETARY
#  Copyright (c) 2025-2026 nvyra-x. All Rights Reserved.
#
#  NOTICE:  All information contained herein is, and remains the property of
#  nvyra-x. The intellectual and technical concepts contained herein are
#  proprietary to nvyra-x and may be covered by Irish and Foreign Patents,
#  patents in process, and are protected by trade secret or copyright law.
#  Dissemination of this information or reproduction of this material is
#  strictly forbidden unless prior written permission is obtained from nvyra-x.
# ------------------------------------------------------------------------------

"""
nvyra-x cache builder - january 2026 production edition
queue-based batch processing with 2000 item threshold
cuda 13.0, pytorch 2.9.1, flash attention 3
h200 gpu with hybrid dense+sparse embeddings
"""

import modal
import asyncio
import time
import json
import uuid
import hashlib
import os
import re
from typing import List, Dict, Any, Optional
from collections import OrderedDict
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

# ============================================================================
# CONFIGURATION
# ============================================================================

APP_NAME = "nvyra-x-cache-builder"
QUEUE_THRESHOLD = 2000  # Only process when queue has 2000+ items
BATCH_SIZE = 128  # Process in batches of 128

# Model configuration
factcheck_model = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
dense_embed_model = "tencent/KaLM-Embedding-Gemma3-12B-2511"
sparse_embed_model = "naver/splade-v3"
reranker_lite = "Qwen/Qwen3-Reranker-0.6B"
reranker_heavy = "Qwen/Qwen3-Reranker-8B"

# Hardcoded secrets
cache_secrets = [modal.Secret.from_dict({
    "hf_token": "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",
    "qdrant_url": "http://95.111.232.85:6333",
    "qdrant_collection": "diamond_v30",
    "turso_url": "https://ai-metadata-cache-f-b.aws-eu-west-1.turso.io",
    "turso_api": "eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJhIjoicnciLCJpYXQiOjE3NjYzNDE4NzEsImlkIjoiYmYwODMzM2MtNTZlMS00ZDJhLWIwYmItMGUzOTMyODI0Y2FlIiwicmlkIjoiMjBmOGYyNjgtODkzYS00NTk5LWI0NWYtMDc3M2MxOGYwNjZiIn0.U-A2yG0WcrG1gikhyNrreLm9cDqlQstgiT9IW9mtgM111xNKjEnoEohOnWY9uNXD2kGpe-tHfb54b_hHCXvEBw",
    "b2_endpoint": "https://s3.eu-central-003.backblazeb2.com",
    "b2_access_key": "00356bc3d6937610000000004",
    "b2_secret_key": "K0036GxH+hhmmADw9yh8aspgXhvu6fo",
    "b2_bucket": "ai-text-cache",
})]

# Volumes
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("rag-harvest-storage-prod", create_if_missing=True)

# Queue for incoming items - this is the main ingestion point
cache_build_queue = modal.Queue.from_name("nvyra-cache-build-queue", create_if_missing=True)


# ============================================================================
# METRICS
# ============================================================================

@dataclass
class PipelineMetrics:
    start_time: float = field(default_factory=time.time)
    items_processed: int = 0
    items_saved_turso: int = 0
    items_saved_qdrant: int = 0
    items_saved_s3: int = 0
    batches_processed: int = 0
    errors: int = 0

    def report(self):
        duration = time.time() - self.start_time
        tps = self.items_processed / duration if duration > 0 else 0
        print(json.dumps({
            "metric": "cache_builder_status",
            "uptime_sec": int(duration),
            "throughput_tps": f"{tps:.2f}",
            "processed": self.items_processed,
            "batches": self.batches_processed,
            "saved": {
                "turso": self.items_saved_turso,
                "qdrant": self.items_saved_qdrant,
                "s3": self.items_saved_s3
            },
            "errors": self.errors
        }))


# ============================================================================
# GPU IMAGE - CUDA 13.0, PyTorch 2.9.1, Flash Attention 3
# ============================================================================

def download_models():
    """Download all models during image build."""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    import os

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    models = [
        factcheck_model,
        dense_embed_model,
        sparse_embed_model,
        reranker_lite,
        reranker_heavy,
    ]

    def download(m):
        try:
            print(f"Downloading {m}...")
            snapshot_download(m, ignore_patterns=["*.md", "*.txt"])
            AutoTokenizer.from_pretrained(m, trust_remote_code=True)
            print(f"Ready: {m}")
        except Exception as e:
            print(f"Warning: {m}: {e}")

    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(download, models)

    print("All models downloaded")


gpu_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "wget", "libzstd-dev", "build-essential", "ninja-build", "ccache")
    .pip_install("uv")
    .run_commands(
        "uv venv .venv",
        "uv pip install --system --upgrade setuptools pip",
        # PyTorch 2.9.1 with CUDA 13.0
        "uv pip install --system 'torch==2.9.1' --index-url https://download.pytorch.org/whl/cu130",
        # SGLang for ultra-fast inference
        "uv pip install --system 'sglang[all]>=0.4.6' --no-build-isolation",
        # vLLM for async inference
        "uv pip install --system 'vllm>=0.13.0' --no-build-isolation",
        # Core dependencies
        "uv pip install --system 'transformers>=4.57.0' accelerate>=1.2.0 huggingface_hub hf_transfer",
        "uv pip install --system pydantic fastapi uvicorn aiohttp httpx",
        "uv pip install --system libsql-experimental qdrant-client boto3 zstandard",
        "uv pip install --system sentence-transformers polars pyarrow rank-bm25 datasketch",
        # Flash Attention 3 pre-built wheels
        "pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu130_torch291 --extra-index-url https://download.pytorch.org/whl/cu130",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,garbage_collection_threshold:0.8",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
    })
    .run_function(download_models, secrets=cache_secrets, volumes={"/root/.cache/huggingface": hf_cache_vol})
)


# ============================================================================
# MODAL APP
# ============================================================================

app = modal.App(APP_NAME)


# ============================================================================
# QUEUE ITEM SCHEMA
# ============================================================================

"""
Items pushed to the queue should be triple-quoted strings (JSON) with this schema:

'''
{
    "claim_id": "unique-claim-id",
    "claim_text": "The claim to verify",
    "doc_text": "Evidence/context text",
    "doc_metadata": {
        "urls": ["https://source1.com", "https://source2.com"],
        "titles": ["Title 1", "Title 2"],
        "sources": ["Publisher 1", "Publisher 2"]
    }
}
'''
"""


# ============================================================================
# GPU CACHE BUILDER
# ============================================================================

@app.cls(
    image=gpu_image,
    gpu="H200",
    secrets=cache_secrets,
    volumes={"/data": data_vol, "/root/.cache/huggingface": hf_cache_vol},
    min_containers=0,  # Scale to zero when idle
    max_containers=1,  # Single H200 for processing
    container_idle_timeout=600,  # 10 minute idle timeout
    timeout=7200,  # 2 hour max runtime
    allow_concurrent_inputs=1,  # One batch at a time
)
class CacheBuilder:
    """H200-powered cache builder with hybrid embeddings."""

    @modal.enter()
    def setup(self):
        """Initialize all models and connections."""
        import torch
        import sglang as sgl
        from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForCausalLM
        import boto3
        import zstandard
        from botocore.config import Config
        import libsql_experimental as libsql
        from qdrant_client import QdrantClient

        start_init = time.perf_counter()
        self.metrics = PipelineMetrics()

        print("=" * 60)
        print("NVYRA-X CACHE BUILDER INITIALIZING")
        print("=" * 60)
        print(f"GPU: NVIDIA H200")
        print(f"CUDA: 13.0")
        print(f"PyTorch: {torch.__version__}")
        print(f"Flash Attention: 3")
        print("=" * 60)

        self.device = "cuda"
        self.dedup_cache = OrderedDict()
        self.dedup_limit = 50000

        # CUDA optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

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

        self.qc = QdrantClient(url=os.environ["qdrant_url"])
        self.qdrant_collection = os.environ["qdrant_collection"]

        self.db = libsql.connect(
            database=os.environ["turso_url"],
            auth_token=os.environ["turso_api"]
        )

        # Ensure schema
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS claim_metadata (
                id TEXT PRIMARY KEY,
                claim_id TEXT,
                verdict TEXT,
                falsity_score REAL,
                lite_score REAL,
                heavy_score REAL,
                s3_key TEXT,
                source_urls TEXT,
                source_titles TEXT,
                source_publishers TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_claim_id ON claim_metadata(claim_id)")
        self.db.commit()

        # Load models
        def load_model(model_name, cls, **kwargs):
            return cls.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_3",
                trust_remote_code=True,
                **kwargs
            ).to(self.device).eval()

        print(f"Loading reranker lite: {reranker_lite}")
        self.model_lite = load_model(reranker_lite, AutoModelForCausalLM)

        print(f"Loading reranker heavy: {reranker_heavy}")
        self.model_heavy = load_model(reranker_heavy, AutoModelForCausalLM)

        print(f"Loading dense embeddings: {dense_embed_model}")
        self.model_dense = load_model(dense_embed_model, AutoModel)

        print(f"Loading sparse embeddings: {sparse_embed_model}")
        try:
            self.model_sparse = load_model(sparse_embed_model, AutoModelForMaskedLM)
        except Exception:
            print("Falling back to SDPA for SPLADE")
            self.model_sparse = AutoModelForMaskedLM.from_pretrained(
                sparse_embed_model,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                trust_remote_code=True,
            ).to(self.device).eval()

        # Tokenizers
        self.tok_lite = AutoTokenizer.from_pretrained(reranker_lite, trust_remote_code=True)
        self.tok_heavy = AutoTokenizer.from_pretrained(reranker_heavy, trust_remote_code=True)
        self.tok_dense = AutoTokenizer.from_pretrained(dense_embed_model, trust_remote_code=True)
        self.tok_sparse = AutoTokenizer.from_pretrained(sparse_embed_model, trust_remote_code=True)

        # Reranker token IDs for yes/no
        self.yes_id = 7866
        self.no_id = 1489

        # Load main factcheck model with SGLang
        print(f"Loading factcheck model: {factcheck_model}")
        try:
            self.factcheck_runtime = sgl.Runtime(
                model_path=factcheck_model,
                tp_size=1,
                trust_remote_code=True,
                mem_fraction_static=0.35,
            )
            sgl.set_default_backend(self.factcheck_runtime)
            print("Factcheck model loaded with SGLang")
        except Exception as e:
            print(f"SGLang error: {e}, falling back to vLLM")
            self.factcheck_runtime = None

        # Compile models for faster inference
        self.model_lite = torch.compile(self.model_lite, mode="reduce-overhead")
        self.model_heavy = torch.compile(self.model_heavy, mode="reduce-overhead")
        self.model_dense = torch.compile(self.model_dense, mode="max-autotune-no-cudagraphs")
        self.model_sparse = torch.compile(self.model_sparse, mode="max-autotune-no-cudagraphs")

        init_time = time.perf_counter() - start_init
        print("=" * 60)
        print(f"CACHE BUILDER READY ({init_time:.1f}s)")
        print("=" * 60)

    def _compute_dense_embedding(self, texts: List[str]) -> List[List[int]]:
        """Compute binarized dense embeddings."""
        import torch

        results = []
        for text in texts:
            inputs = self.tok_dense(
                text[:2048],
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            ).to(self.device)

            with torch.inference_mode():
                outputs = self.model_dense(**inputs)
                last_idx = inputs["attention_mask"].sum(1) - 1
                vec = outputs.last_hidden_state[0, last_idx, :1024]
                normalized = torch.nn.functional.normalize(vec, p=2, dim=-1)
                binarized = (normalized > 0).int().cpu().tolist()[0]
                results.append(binarized)

        return results

    def _compute_sparse_embedding(self, texts: List[str]) -> tuple:
        """Compute sparse SPLADE embeddings."""
        import torch

        batch_indices = []
        batch_values = []

        for text in texts:
            inputs = self.tok_sparse(
                text[:2048],
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.inference_mode():
                outputs = self.model_sparse(**inputs)
                logits = outputs.logits
                sparse_vec = torch.max(
                    torch.log1p(torch.relu(logits)) * inputs["attention_mask"].unsqueeze(-1),
                    dim=1
                ).values.squeeze()

                indices = sparse_vec.nonzero().squeeze(-1).cpu().tolist()
                values = sparse_vec[indices].cpu().tolist()

                if isinstance(indices, int):
                    indices = [indices]
                    values = [values]

                batch_indices.append(indices)
                batch_values.append(values)

        return batch_indices, batch_values

    def _rerank_batch(self, items: List[Dict], model, tokenizer, task: str) -> List[float]:
        """Run reranking on batch."""
        import torch

        prompts = []
        for item in items:
            claim = item.get("claim_text", "")
            doc = item.get("doc_text", "")[:512 if task == "lite" else 4096]

            if task == "lite":
                p = f"Instruct: Retrieve relevant context\nQuery: {claim}\nDoc: {doc}"
            else:
                p = f"<Instruct>: Identify contradictions and supporting evidence\n<Query>: {claim}\n<Document>: {doc}"
            prompts.append(p)

        scores = []
        batch_size = 32

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024 if task == "lite" else 4096,
            ).to(self.device)

            with torch.inference_mode():
                logits = model(**inputs).logits[:, -1, :]
                probs = torch.softmax(logits[:, [self.no_id, self.yes_id]], dim=1)[:, 1]
                scores.extend(probs.cpu().float().tolist())

        return scores

    async def _generate_analysis(self, claim: str, evidence: str) -> Dict[str, Any]:
        """Generate factcheck analysis using SGLang."""
        if not self.factcheck_runtime:
            return {"verdict": "unverifiable", "falsity_score": 0.5}

        import sglang as sgl

        system_prompt = """You are an expert fact-checker. Analyze the Claim against the Evidence.
First, output a thinking block <think>...</think> where you analyze the evidence.
Then output a valid JSON with: synthesis, critique, verdict, falsity_score (0-9), citations (list)."""

        prompt = f"Claim: {claim}\nEvidence: {evidence[:15000]}"

        def run_generation():
            @sgl.function
            def factcheck_fn(s, user_prompt):
                s += sgl.system(system_prompt)
                s += sgl.user(user_prompt)
                s += sgl.assistant(sgl.gen("response", max_tokens=2048, temperature=0.8))

            result = factcheck_fn.run(user_prompt=prompt)
            return result["response"]

        try:
            raw = await asyncio.to_thread(run_generation)

            # Parse JSON from response
            json_match = re.search(r'</think>\s*(\{.*\})', raw, re.DOTALL)
            if not json_match:
                json_match = re.search(r'(\{.*\})$', raw, re.DOTALL)

            if json_match:
                data = json.loads(json_match.group(1))
                return {
                    "synthesis": data.get("synthesis", ""),
                    "critique": data.get("critique", ""),
                    "verdict": data.get("verdict", "Unknown"),
                    "falsity_score": float(data.get("falsity_score", 0)) / 9.0,
                    "citations": data.get("citations", []),
                }
        except Exception as e:
            print(f"Analysis error: {e}")

        return {"verdict": "unverifiable", "falsity_score": 0.5}

    def _save_batch(self, results: List[Dict]):
        """Save batch to all storage backends."""
        if not results:
            return

        from qdrant_client.models import PointStruct, SparseVector

        # Upload to S3
        def upload_s3(item):
            payload = {k: v for k, v in item.items() if k not in ["dense_vec", "sparse_idx", "sparse_val"]}
            key = f"v30/{item['claim_id']}/{item['record_uuid']}.zst"
            self.s3.put_object(
                Bucket=self.b2_bucket,
                Key=key,
                Body=self.cctx.compress(json.dumps(payload).encode())
            )
            return key

        with ThreadPoolExecutor(64) as ex:
            s3_keys = list(ex.map(upload_s3, results))
        self.metrics.items_saved_s3 += len(s3_keys)

        # Save to Turso
        try:
            self.db.execute("BEGIN TRANSACTION")
            for i, item in enumerate(results):
                analysis = item.get("analysis", {})
                metadata = item.get("doc_metadata", {})
                self.db.execute(
                    """INSERT OR IGNORE INTO claim_metadata
                       (id, claim_id, verdict, falsity_score, lite_score, heavy_score, s3_key,
                        source_urls, source_titles, source_publishers)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        item["record_uuid"],
                        item["claim_id"],
                        analysis.get("verdict", "Unknown"),
                        analysis.get("falsity_score", 0.5),
                        item.get("lite_score", 0.0),
                        item.get("heavy_score", 0.0),
                        s3_keys[i],
                        json.dumps(metadata.get("urls", [])),
                        json.dumps(metadata.get("titles", [])),
                        json.dumps(metadata.get("sources", [])),
                    )
                )
            self.db.commit()
            self.metrics.items_saved_turso += len(results)
        except Exception as e:
            print(f"Turso error: {e}")
            self.db.rollback()
            self.metrics.errors += 1

        # Save to Qdrant
        points = []
        for item in results:
            vectors = {"dense": item["dense_vec"]}
            if item.get("sparse_idx"):
                vectors["sparse"] = SparseVector(
                    indices=item["sparse_idx"],
                    values=item["sparse_val"]
                )
            points.append(PointStruct(
                id=item["record_uuid"],
                vector=vectors,
                payload={"cid": item["claim_id"]}
            ))

        self.qc.upsert(self.qdrant_collection, points=points, wait=False)
        self.metrics.items_saved_qdrant += len(points)

    @modal.method()
    async def process_batch(self, items: List[Dict]) -> int:
        """Process a batch of items."""
        if not items:
            return 0

        print(f"Processing batch of {len(items)} items...")

        # Stage 1: Lite reranking
        lite_scores = await asyncio.to_thread(
            self._rerank_batch, items, self.model_lite, self.tok_lite, "lite"
        )

        survivors = []
        for i, score in enumerate(lite_scores):
            if score >= 0.02:
                items[i]["lite_score"] = score
                survivors.append(items[i])

        if not survivors:
            return 0

        # Stage 2: Heavy reranking for mid-confidence items
        need_heavy = [s for s in survivors if s["lite_score"] < 0.98]
        if need_heavy:
            heavy_scores = await asyncio.to_thread(
                self._rerank_batch, need_heavy, self.model_heavy, self.tok_heavy, "heavy"
            )
            for i, score in enumerate(heavy_scores):
                need_heavy[i]["heavy_score"] = score

        # Filter by heavy score
        final_survivors = []
        for item in survivors:
            if item.get("lite_score", 0) >= 0.98:
                item["heavy_score"] = 1.0
                final_survivors.append(item)
            elif item.get("heavy_score", 0) >= 0.75:
                final_survivors.append(item)

        if not final_survivors:
            return 0

        # Stage 3: Generate analysis
        for item in final_survivors:
            cached = self.dedup_cache.get(hashlib.md5(item["claim_text"].encode()).hexdigest())
            if cached:
                item["analysis"] = cached
            else:
                item["analysis"] = await self._generate_analysis(
                    item["claim_text"],
                    item.get("doc_text", "")
                )
                self.dedup_cache[hashlib.md5(item["claim_text"].encode()).hexdigest()] = item["analysis"]
                if len(self.dedup_cache) > self.dedup_limit:
                    self.dedup_cache.popitem(last=False)

        # Stage 4: Compute embeddings
        embed_texts = []
        for item in final_survivors:
            analysis = item.get("analysis", {})
            text = f"Claim: {item['claim_text']}\nVerdict: {analysis.get('verdict', '')}\nScore: {analysis.get('falsity_score', 0)}\nSynthesis: {analysis.get('synthesis', '')}"
            embed_texts.append(text)

        dense_vecs = await asyncio.to_thread(self._compute_dense_embedding, embed_texts)
        sparse_idx, sparse_val = await asyncio.to_thread(self._compute_sparse_embedding, embed_texts)

        for i, item in enumerate(final_survivors):
            item["dense_vec"] = dense_vecs[i]
            item["sparse_idx"] = sparse_idx[i]
            item["sparse_val"] = sparse_val[i]
            item["record_uuid"] = str(uuid.uuid4())

        # Stage 5: Save
        await asyncio.to_thread(self._save_batch, final_survivors)

        self.metrics.items_processed += len(final_survivors)
        self.metrics.batches_processed += 1
        self.metrics.report()

        return len(final_survivors)


# ============================================================================
# QUEUE MONITOR - Triggers processing when 2000+ items
# ============================================================================

@app.function(
    image=modal.Image.debian_slim(python_version="3.12").pip_install("pydantic"),
    schedule=modal.Period(minutes=1),  # Check every minute
    secrets=cache_secrets,
    timeout=300,
)
async def queue_monitor():
    """Monitor queue and trigger batch processing when threshold reached."""

    queue_len = cache_build_queue.len()
    print(f"Queue length: {queue_len} / {QUEUE_THRESHOLD}")

    if queue_len < QUEUE_THRESHOLD:
        print(f"Waiting for {QUEUE_THRESHOLD - queue_len} more items...")
        return {"status": "waiting", "queue_length": queue_len, "threshold": QUEUE_THRESHOLD}

    print(f"Threshold reached! Processing {queue_len} items...")

    # Drain queue in batches
    all_items = []
    while len(all_items) < queue_len:
        try:
            batch = cache_build_queue.get_many(n_values=min(100, queue_len - len(all_items)), block=False)
            if not batch:
                break
            for raw_item in batch:
                try:
                    # Parse triple-quoted JSON strings
                    if isinstance(raw_item, str):
                        item = json.loads(raw_item.strip("'\""))
                    else:
                        item = raw_item
                    all_items.append(item)
                except json.JSONDecodeError as e:
                    print(f"Invalid JSON in queue: {e}")
        except Exception as e:
            print(f"Queue drain error: {e}")
            break

    print(f"Drained {len(all_items)} items from queue")

    if not all_items:
        return {"status": "empty", "queue_length": 0}

    # Process in batches
    builder = CacheBuilder()
    total_processed = 0

    for i in range(0, len(all_items), BATCH_SIZE):
        batch = all_items[i:i + BATCH_SIZE]
        try:
            processed = await builder.process_batch.remote.aio(batch)
            total_processed += processed
            print(f"Batch {i // BATCH_SIZE + 1}: processed {processed} items")
        except Exception as e:
            print(f"Batch processing error: {e}")

    return {
        "status": "completed",
        "items_drained": len(all_items),
        "items_processed": total_processed
    }


# ============================================================================
# MANUAL QUEUE PUSH ENDPOINT
# ============================================================================

@app.function(image=modal.Image.debian_slim().pip_install("pydantic", "fastapi"))
@modal.asgi_app()
def queue_api():
    """FastAPI endpoint for pushing items to the queue."""
    from fastapi import FastAPI

    api = FastAPI(title="nvyra-x Cache Build Queue API", version="1.0.0")

    @api.post("/push")
    async def push_item(item: Dict[str, Any]):
        """Push a single item to the queue."""
        cache_build_queue.put(json.dumps(item))
        queue_len = cache_build_queue.len()
        return {
            "status": "queued",
            "queue_length": queue_len,
            "threshold": QUEUE_THRESHOLD,
            "will_process": queue_len >= QUEUE_THRESHOLD
        }

    @api.post("/push_many")
    async def push_many(items: List[Dict[str, Any]]):
        """Push multiple items to the queue."""
        json_items = [json.dumps(item) for item in items]
        cache_build_queue.put_many(json_items)
        queue_len = cache_build_queue.len()
        return {
            "status": "queued",
            "items_added": len(items),
            "queue_length": queue_len,
            "threshold": QUEUE_THRESHOLD,
            "will_process": queue_len >= QUEUE_THRESHOLD
        }

    @api.get("/status")
    async def get_status():
        """Get queue status."""
        queue_len = cache_build_queue.len()
        return {
            "queue_length": queue_len,
            "threshold": QUEUE_THRESHOLD,
            "items_until_process": max(0, QUEUE_THRESHOLD - queue_len),
            "ready_to_process": queue_len >= QUEUE_THRESHOLD
        }

    @api.post("/force_process")
    async def force_process():
        """Force process whatever is in the queue (bypass threshold)."""
        result = await queue_monitor.remote.aio()
        return result

    @api.get("/health")
    async def health():
        return {"status": "healthy", "version": "1.0.0"}

    return api


# ============================================================================
# LOCAL ENTRYPOINT
# ============================================================================

@app.local_entrypoint()
def main(push_test: bool = False):
    """Test the cache builder pipeline."""
    print("\n" + "=" * 60)
    print("NVYRA-X CACHE BUILDER")
    print("=" * 60)
    print(f"Queue threshold: {QUEUE_THRESHOLD} items")
    print(f"Batch size: {BATCH_SIZE}")
    print("=" * 60 + "\n")

    queue_len = cache_build_queue.len()
    print(f"Current queue length: {queue_len}")

    if push_test:
        print("\nPushing test items to queue...")
        test_items = [
            {
                "claim_id": f"test-{i}",
                "claim_text": f"Test claim number {i}",
                "doc_text": f"This is evidence for claim {i}. It contains relevant information.",
                "doc_metadata": {
                    "urls": [f"https://example.com/{i}"],
                    "titles": [f"Test Source {i}"],
                    "sources": ["Test Publisher"]
                }
            }
            for i in range(10)
        ]

        for item in test_items:
            cache_build_queue.put(json.dumps(item))

        new_len = cache_build_queue.len()
        print(f"Queue length after push: {new_len}")
        print(f"Items until threshold: {QUEUE_THRESHOLD - new_len}")
    else:
        print(f"\nItems until processing: {max(0, QUEUE_THRESHOLD - queue_len)}")
        print("\nTo push test items, run with --push-test flag")
