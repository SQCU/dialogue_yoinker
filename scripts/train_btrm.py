#!/usr/bin/env python3
"""
Logsquare-regularized Bradley-Terry Reward Model (BTRM) training.

Multi-head architecture: Each "vocabulary slot" represents a different membership
criterion. We score the same batch against N different criteria in parallel,
using loss masking/remasking to compute ~8 different scoring semantics at once.

This is accelerator utilization maxxing - one forward pass, multiple loss heads.

Text templating for IT models:
    meta:
    You are a BTRM reward model grading dataset membership. Given text, output a
    scalar score indicating membership probability in the target corpus.
    Loss function: L = L_BT + λ·log(r²) where positive samples cluster at r≈1.
    Target corpus: {corpus_description}

    text:
    {actual_text}

Negative hierarchy (all in same furthest ring):
- soft_neg: Other corpora in same domain
- semi_firm_neg: Out-of-domain prose (SYNTH reasoning, wattpad fiction)
- furthest_neg: Wrong format entirely (fineweb webscrape, wikitext encyclopedic)
  Note: fineweb ≈ wikitext for our purposes - both are maximally dissimilar

Usage:
    python scripts/train_btrm.py train --config configs/btrm_multihead.yaml
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal
from contextlib import nullcontext

torch = None
nn = None
F = None


def ensure_torch():
    global torch, nn, F
    if torch is None:
        try:
            import torch as _torch
            import torch.nn as _nn
            import torch.nn.functional as _F
            torch = _torch
            nn = _nn
            F = _F
        except ImportError:
            print("PyTorch not installed. Run: uv pip install torch transformers peft accelerate datasets")
            sys.exit(1)


# =============================================================================
# DATASET DECODERS (with correct field names)
# =============================================================================

@dataclass
class Sample:
    """A text sample with metadata."""
    text: str
    source: str
    tier: Literal["positive", "soft_neg", "semi_firm_neg", "furthest_neg"]
    metadata: dict = field(default_factory=dict)


def decode_local_jsonl(
    path: str,
    text_field: str = "auto",  # "auto", "text", or "prose"
    max_samples: int = 1000,
    tier_filter: str = None,  # None = all, "fk_normed", "flattened", "brainrot_aesop_v4"
) -> list[Sample]:
    """
    Decode our local JSONL format.

    Field mapping (auto-detected based on tier):
    - tier="flattened": uses `text` field (source dialogue walks)
    - tier="fk_normed": uses `prose` field (FK-rewritten prose)
    - tier="brainrot_aesop_v4": uses `prose` field (vocab teaching)

    The FK files contain both flattened (tier 1) and fk_normed (tier 2) samples.
    For BTRM training, we typically want tier 2 (prose outputs).
    """
    samples = []
    path = Path(path)
    if not path.exists():
        print(f"  Warning: {path} not found")
        return []

    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)

                # Filter by tier if specified
                sample_tier = obj.get("tier", "")
                if tier_filter and tier_filter not in sample_tier:
                    continue

                # Auto-detect text field based on tier
                if text_field == "auto":
                    if sample_tier in ("fk_normed", "brainrot_aesop_v4"):
                        text = obj.get("prose")
                    else:
                        text = obj.get("text")
                else:
                    text = obj.get(text_field)

                # Fallbacks
                if not text:
                    text = obj.get("prose") or obj.get("text") or obj.get("content")

                if text and len(text.strip()) > 50:
                    samples.append(Sample(
                        text=text.strip(),
                        source=str(path.stem),
                        tier="positive",
                        metadata={
                            "fk": obj.get("fk_measured"),
                            "sample_tier": sample_tier,
                            "emotion_sequence": obj.get("emotion_sequence"),
                        }
                    ))
            except json.JSONDecodeError:
                continue
            if len(samples) >= max_samples:
                break

    print(f"  Loaded {len(samples)} samples from {path.name}")
    return samples


def decode_synth(max_samples: int = 1000, lang_filter: str = "en") -> list[Sample]:
    """
    Decode PleIAs/SYNTH dataset.

    Fields:
    - synthetic_reasoning: The reasoning trace
    - synthetic_answer: The final answer
    - language: Filter to English by default
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("  datasets not installed")
        return []

    samples = []
    try:
        ds = load_dataset("PleIAs/SYNTH", split="train", streaming=True)

        for item in ds:
            if len(samples) >= max_samples:
                break

            # Filter by language (handle None values)
            lang = (item.get("language") or "").lower()
            if lang_filter and lang_filter not in lang:
                continue

            # Prefer reasoning trace (more distinctive), fallback to answer
            text = item.get("synthetic_reasoning") or item.get("synthetic_answer") or ""

            if text and len(text.strip()) > 100:
                samples.append(Sample(
                    text=text.strip()[:2000],  # Truncate very long
                    source="synth",
                    tier="semi_firm_neg",
                    metadata={
                        "exercise": item.get("exercise"),
                        "model": item.get("model"),
                    }
                ))

    except Exception as e:
        print(f"  Could not load SYNTH: {e}")

    print(f"  Loaded {len(samples)} samples from SYNTH")
    return samples


def decode_wattpad(max_samples: int = 1000, min_chapter_len: int = 200) -> list[Sample]:
    """
    Decode Fizzarolli/wattpad and wattpad2.

    Fields:
    - chapter_contents: List of chapter texts
    - Flatten chapters into individual samples
    - wattpad2 has language/nsfw filters
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("  datasets not installed")
        return []

    samples = []

    for ds_name in ["Fizzarolli/wattpad", "Fizzarolli/wattpad2"]:
        if len(samples) >= max_samples:
            break

        try:
            ds = load_dataset(ds_name, split="train", streaming=True)

            for item in ds:
                if len(samples) >= max_samples:
                    break

                # wattpad2 has language filter (handle None values)
                lang = (item.get("language") or "").upper()
                if lang and lang != "ENG":
                    continue

                # Flatten chapters
                chapters = item.get("chapter_contents", [])
                for i, chapter in enumerate(chapters[:3]):  # Max 3 chapters per story
                    if len(samples) >= max_samples:
                        break
                    if not chapter or len(chapter.strip()) < min_chapter_len:
                        continue

                    samples.append(Sample(
                        text=chapter.strip()[:2000],
                        source=ds_name.split("/")[1],
                        tier="semi_firm_neg",
                        metadata={
                            "title": item.get("title", ""),
                            "chapter_idx": i,
                        }
                    ))

        except Exception as e:
            print(f"  Could not load {ds_name}: {e}")

    print(f"  Loaded {len(samples)} samples from wattpad")
    return samples


def decode_fineweb(max_samples: int = 1000, min_len: int = 200) -> list[Sample]:
    """
    Decode HuggingFaceFW/fineweb-edu or fineweb.

    Fields:
    - text: Main content
    - language: Filter to English
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("  datasets not installed")
        return []

    samples = []

    for ds_name in ["HuggingFaceFW/fineweb-edu", "HuggingFaceFW/fineweb"]:
        if len(samples) >= max_samples:
            break

        try:
            ds = load_dataset(ds_name, "sample-10BT", split="train", streaming=True)

            for item in ds:
                if len(samples) >= max_samples:
                    break

                # Filter to English (handle None values)
                lang = item.get("language") or "en"
                if lang != "en":
                    continue

                text = item.get("text", "")
                if text and len(text.strip()) >= min_len:
                    samples.append(Sample(
                        text=text.strip()[:1500],
                        source="fineweb",
                        tier="furthest_neg",
                        metadata={"url": item.get("url", "")}
                    ))

            break  # Found working dataset

        except Exception as e:
            print(f"  Could not load {ds_name}: {e}")

    print(f"  Loaded {len(samples)} samples from fineweb")
    return samples


def decode_api_walks(
    game: str,
    api_url: str = "http://127.0.0.1:8000",
    count: int = 100,
    max_length: int = 8,
    method: str = "walk",
) -> list[Sample]:
    """
    Fetch random walks from the API server on-the-fly.

    This gives access to the combinatorially huge space of random walks
    rather than just pre-generated JSONL files.
    """
    import requests

    samples = []
    try:
        # Batch into chunks to avoid overwhelming the API
        batch_size = 20
        for i in range(0, count, batch_size):
            n = min(batch_size, count - i)
            resp = requests.post(
                f"{api_url}/api/sample",
                json={
                    "game": game,
                    "method": method,
                    "count": n,
                    "max_length": max_length,
                    "allow_cycles": False,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            for sample in data.get("samples", []):
                # Format: join dialogue lines with newlines
                lines = []
                for node in sample.get("nodes", []):
                    text = node.get("text", "")
                    if text:
                        lines.append(f'"{text}"')

                if len(lines) >= 2:
                    walk_text = "\n".join(lines)
                    samples.append(Sample(
                        text=walk_text,
                        source=f"api_{game}",
                        tier="positive",
                        metadata={
                            "game": game,
                            "method": method,
                            "emotions": [n.get("emotion") for n in sample.get("nodes", [])],
                        }
                    ))

        print(f"  Fetched {len(samples)} walks from API ({game})")
    except Exception as e:
        print(f"  Could not fetch from API: {e}")

    return samples


class APIWalkStreamer:
    """
    Async prefetcher for API walks - keeps a buffer filled in background.
    Avoids blocking training while waiting for API responses.
    """

    def __init__(
        self,
        games: list[str],
        api_url: str = "http://127.0.0.1:8000",
        buffer_size: int = 200,
        walks_per_request: int = 20,
        max_length: int = 8,
    ):
        self.games = games
        self.api_url = api_url
        self.buffer_size = buffer_size
        self.walks_per_request = walks_per_request
        self.max_length = max_length

        self.buffer: list[Sample] = []
        self.lock = None  # Lazy init for threading
        self._running = False
        self._thread = None

    def _ensure_threading(self):
        if self.lock is None:
            import threading
            self.lock = threading.Lock()

    def _fetch_batch(self) -> list[Sample]:
        """Fetch a batch of walks from random game."""
        import requests
        game = random.choice(self.games)
        try:
            resp = requests.post(
                f"{self.api_url}/api/sample",
                json={
                    "game": game,
                    "method": "walk",
                    "count": self.walks_per_request,
                    "max_length": self.max_length,
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            samples = []
            for sample in data.get("samples", []):
                lines = [f'"{n.get("text", "")}"' for n in sample.get("nodes", []) if n.get("text")]
                if len(lines) >= 2:
                    samples.append(Sample(
                        text="\n".join(lines),
                        source=f"api_{game}",
                        tier="positive",
                        metadata={"game": game, "emotions": [n.get("emotion") for n in sample.get("nodes", [])]},
                    ))
            return samples
        except Exception:
            return []

    def _fill_loop(self):
        """Background loop to keep buffer filled."""
        while self._running:
            with self.lock:
                current_size = len(self.buffer)

            if current_size < self.buffer_size:
                batch = self._fetch_batch()
                with self.lock:
                    self.buffer.extend(batch)
            else:
                import time
                time.sleep(0.1)

    def start(self):
        """Start background fetching."""
        self._ensure_threading()
        import threading
        self._running = True
        self._thread = threading.Thread(target=self._fill_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop background fetching."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)

    def get_batch(self, n: int) -> list[Sample]:
        """Get n samples from buffer (blocking if needed)."""
        self._ensure_threading()
        samples = []
        while len(samples) < n:
            with self.lock:
                take = min(n - len(samples), len(self.buffer))
                if take > 0:
                    samples.extend(self.buffer[:take])
                    self.buffer = self.buffer[take:]

            if len(samples) < n:
                # Buffer empty, fetch synchronously
                batch = self._fetch_batch()
                samples.extend(batch[:n - len(samples)])

        return samples[:n]


def decode_wikitext(max_samples: int = 1000, min_len: int = 100) -> list[Sample]:
    """
    Decode wikitext-103.

    Fields:
    - text: Just text content
    - Skip empty lines and section headers
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("  datasets not installed")
        return []

    samples = []

    try:
        ds = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)

        for item in ds:
            if len(samples) >= max_samples:
                break

            text = item.get("text", "")

            # Skip empty, headers, and very short
            if not text or text.startswith(" =") or len(text.strip()) < min_len:
                continue

            samples.append(Sample(
                text=text.strip()[:1500],
                source="wikitext",
                tier="furthest_neg",
                metadata={}
            ))

    except Exception as e:
        print(f"  Could not load wikitext: {e}")

    print(f"  Loaded {len(samples)} samples from wikitext")
    return samples


# =============================================================================
# TEXT TEMPLATING FOR IT MODELS
# =============================================================================

def make_meta_prompt() -> str:
    """
    Generate meta-prompt for instruction-tuned models.

    IMPORTANT: No target_corpus specified because multiple heads train in
    parallel on the SAME input tokens with DIFFERENT loss assignments.
    Each head learns its own membership criterion from the meta:text relationship.
    """
    return """meta:
You are a BTRM reward model. For each input, output a scalar score.
Multiple heads train in parallel on the same tokens with different loss assignments.
You're grading the membership relations texts and several possible relations of membership and difference from data modes like fictional prose drawn from colelctions like narrative role playing games, webtext, state of the art synthetic training data for reasoning and problem solving, and wattpad prose.
This is a pretty diverse spread of topics, but the task isn't quite so complicated: you have a collection of scalar output heads which are independently learning the membership criteria for their criterion, and each should have a different but simple affinity for different cases.

Loss formulation: L = L_BT + λ·log(r²)
- Bradley-Terry ranking: P(pos > neg) = σ(r_pos - r_neg)
- Logsquare: positive samples cluster at r≈1

get ready to sloptimize!

text:
"""


def template_sample(text: str, meta_prompt: Optional[str] = None) -> str:
    """Apply templating to a sample."""
    if meta_prompt:
        return meta_prompt + text
    return text


# =============================================================================
# MULTI-HEAD BTRM ARCHITECTURE
# =============================================================================

@dataclass
class PositiveSource:
    """A single source of positive samples for a head."""
    path: str
    text_field: str = "auto"
    tier_filter: str = None  # None = all tiers


@dataclass
class NegativeSource:
    """A negative sample source with tier assignment."""
    path: str
    text_field: str = "auto"
    tier_filter: str = None
    neg_tier: str = "soft_neg"  # soft_neg, semi_firm_neg, furthest_neg


@dataclass
class HeadConfig:
    """Configuration for a single BTRM head (vocabulary slot)."""
    name: str
    description: str
    # New: list of sources with per-source tier filtering
    positive_sources: list[PositiveSource] = field(default_factory=list)

    # Per-head negative sources (optional - falls back to shared negatives)
    negative_sources: list[NegativeSource] = field(default_factory=list)

    # Legacy: single path config (for backward compat with existing configs)
    positive_paths: list[str] = field(default_factory=list)
    positive_text_field: str = "auto"  # "auto", "text", or "prose"
    positive_tier_filter: str = "fk_normed"  # Filter to tier 2 (prose) by default

    def get_sources(self) -> list[PositiveSource]:
        """Get all sources, handling both new and legacy format."""
        if self.positive_sources:
            return self.positive_sources
        # Convert legacy format
        return [
            PositiveSource(path=p, text_field=self.positive_text_field, tier_filter=self.positive_tier_filter)
            for p in self.positive_paths
        ]


@dataclass
class MultiHeadBTRMConfig:
    """Configuration for multi-head BTRM."""
    heads: list[HeadConfig]

    # Shared negative hierarchy (applied to all heads)
    soft_neg_paths: list[str] = field(default_factory=list)  # Other local corpora
    use_synth: bool = True
    use_wattpad: bool = True
    use_fineweb: bool = True
    use_wikitext: bool = True
    neg_samples_per_tier: int = 300

    # Model config
    base_model: str = "Qwen/Qwen2.5-0.5B"
    use_lora: bool = False  # Full rank training preferred for BTRM
    lora_r: int = 16
    lora_alpha: int = 32

    # Training config
    epochs: int = 3
    batch_size: int = 8
    lr: float = 5e-5  # 5e-5 good for full fine-tuning
    logsquare_weight: float = 0.1
    max_batches: int = 0  # 0 = unlimited; set to limit for quick validation
    warmup_steps: int = 200  # Linear warmup steps (~2x AdamW β2 time constant / 10)

    # Numerical stability
    logit_cap: float = 10.0  # Soft tanh cap; 0 = disabled. Gemma may need lower (e.g., 5.0)

    # Memory optimization
    max_length: int = 4096  # Context length
    gradient_checkpointing: bool = True  # Activation checkpointing
    use_amp: bool = True  # Use AMP with GradScaler for mixed precision
    amp_dtype: str = "bfloat16"  # "float16" or "bfloat16" (bf16 more stable)

    # IT model meta-prompting
    use_meta_prompt: bool = True

    # API streaming for on-the-fly walk generation
    use_api_walks: bool = False  # Enable to stream walks from API server
    api_url: str = "http://127.0.0.1:8000"
    api_games: list = field(default_factory=lambda: ["oblivion", "falloutnv", "skyrim"])
    api_walks_per_batch: int = 4  # Fresh walks to mix into each batch
    api_buffer_size: int = 200  # Background buffer size

    @classmethod
    def from_yaml(cls, path: str) -> "MultiHeadBTRMConfig":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)

        heads = []
        for h in data.get("heads", []):
            # New format: positive_sources as list of dicts
            pos_sources = []
            if "positive_sources" in h:
                for src in h["positive_sources"]:
                    pos_sources.append(PositiveSource(
                        path=src["path"],
                        text_field=src.get("text_field", "auto"),
                        tier_filter=src.get("tier_filter"),
                    ))

            # Per-head negative sources
            neg_sources = []
            if "negative_sources" in h:
                for src in h["negative_sources"]:
                    neg_sources.append(NegativeSource(
                        path=src["path"],
                        text_field=src.get("text_field", "auto"),
                        tier_filter=src.get("tier_filter"),
                        neg_tier=src.get("neg_tier", "soft_neg"),
                    ))

            heads.append(HeadConfig(
                name=h["name"],
                description=h["description"],
                positive_sources=pos_sources,
                negative_sources=neg_sources,
                # Legacy format fallback
                positive_paths=h.get("positive_paths", []),
                positive_text_field=h.get("positive_text_field", "auto"),
                positive_tier_filter=h.get("positive_tier_filter", "fk_normed"),
            ))

        return cls(
            heads=heads,
            soft_neg_paths=data.get("soft_neg_paths", []),
            use_synth=data.get("use_synth", True),
            use_wattpad=data.get("use_wattpad", True),
            use_fineweb=data.get("use_fineweb", True),
            use_wikitext=data.get("use_wikitext", True),
            neg_samples_per_tier=data.get("neg_samples_per_tier", 300),
            base_model=data.get("base_model", "Qwen/Qwen2.5-0.5B"),
            use_lora=data.get("use_lora", False),
            lora_r=data.get("lora_r", 16),
            lora_alpha=data.get("lora_alpha", 32),
            epochs=data.get("epochs", 3),
            batch_size=data.get("batch_size", 8),
            lr=data.get("lr", 5e-5),
            logsquare_weight=data.get("logsquare_weight", 0.1),
            max_batches=data.get("max_batches", 0),
            warmup_steps=data.get("warmup_steps", 200),
            logit_cap=data.get("logit_cap", 10.0),
            max_length=data.get("max_length", 4096),
            gradient_checkpointing=data.get("gradient_checkpointing", True),
            use_amp=data.get("use_amp", True),
            amp_dtype=data.get("amp_dtype", "bfloat16"),
            use_meta_prompt=data.get("use_meta_prompt", True),
            use_api_walks=data.get("use_api_walks", False),
            api_url=data.get("api_url", "http://127.0.0.1:8000"),
            api_games=data.get("api_games", ["oblivion", "falloutnv", "skyrim"]),
            api_walks_per_batch=data.get("api_walks_per_batch", 4),
            api_buffer_size=data.get("api_buffer_size", 200),
        )

    def to_yaml(self, path: str):
        import yaml
        heads_data = []
        for h in self.heads:
            head_dict = {
                "name": h.name,
                "description": h.description,
            }
            # Prefer new format if we have positive_sources
            if h.positive_sources:
                head_dict["positive_sources"] = [
                    {
                        "path": src.path,
                        "text_field": src.text_field,
                        "tier_filter": src.tier_filter,
                    }
                    for src in h.positive_sources
                ]
            else:
                # Legacy format
                head_dict["positive_paths"] = h.positive_paths
                head_dict["positive_text_field"] = h.positive_text_field
                head_dict["positive_tier_filter"] = h.positive_tier_filter

            # Per-head negatives
            if h.negative_sources:
                head_dict["negative_sources"] = [
                    {
                        "path": src.path,
                        "text_field": src.text_field,
                        "tier_filter": src.tier_filter,
                        "neg_tier": src.neg_tier,
                    }
                    for src in h.negative_sources
                ]

            heads_data.append(head_dict)

        data = {
            "heads": heads_data,
            "soft_neg_paths": self.soft_neg_paths,
            "use_synth": self.use_synth,
            "use_wattpad": self.use_wattpad,
            "use_fineweb": self.use_fineweb,
            "use_wikitext": self.use_wikitext,
            "neg_samples_per_tier": self.neg_samples_per_tier,
            "base_model": self.base_model,
            "use_lora": self.use_lora,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "logsquare_weight": self.logsquare_weight,
            "max_batches": self.max_batches,
            "warmup_steps": self.warmup_steps,
            "logit_cap": self.logit_cap,
            "max_length": self.max_length,
            "gradient_checkpointing": self.gradient_checkpointing,
            "use_amp": self.use_amp,
            "amp_dtype": self.amp_dtype,
            "use_meta_prompt": self.use_meta_prompt,
            "use_api_walks": self.use_api_walks,
            "api_url": self.api_url,
            "api_games": self.api_games,
            "api_walks_per_batch": self.api_walks_per_batch,
            "api_buffer_size": self.api_buffer_size,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)


class RMSNorm:
    @staticmethod
    def create(dim: int, eps: float = 1e-6):
        ensure_torch()

        class _RMSNorm(nn.Module):
            def __init__(self, dim: int, eps: float):
                super().__init__()
                self.eps = eps
                self.weight = nn.Parameter(torch.ones(dim))

            def forward(self, x):
                rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
                return x / rms * self.weight

        return _RMSNorm(dim, eps)


class MultiHeadBTRM:
    """
    Multi-head BTRM: One scalar projection per membership criterion.

    Architecture:
        base_model → last_hidden → RMSNorm → [proj_1, proj_2, ..., proj_N] → [score_1, ..., score_N]

    Each head is a "vocabulary slot" that grades membership in a different corpus.
    We compute all N scores in one forward pass, then apply different loss masks.

    Soft tanh capping prevents extreme logits (numerical stability, Gemma compatibility).
    """

    def __init__(
        self,
        hidden_dim: int,
        head_names: list[str],
        device: str = "cuda",
        logit_cap: float = 10.0,  # Soft cap via tanh; 0 = no capping
    ):
        ensure_torch()

        self.hidden_dim = hidden_dim
        self.head_names = head_names
        self.n_heads = len(head_names)
        self.device = device
        self.logit_cap = logit_cap

        # Shared RMSNorm
        self.rms_norm = RMSNorm.create(hidden_dim).to(device)

        # One projection per head: hidden_dim → 1
        # Stack into single linear: hidden_dim → n_heads
        self.proj = nn.Linear(hidden_dim, self.n_heads, bias=False).to(device)

        # Initialize near zero
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)

    def forward(self, hidden_states):
        """
        hidden_states: [batch, seq_len, hidden_dim]
        Returns: [batch, n_heads] - one score per head
        """
        last_hidden = hidden_states[:, -1, :]  # [batch, hidden_dim]
        normed = self.rms_norm(last_hidden)    # [batch, hidden_dim]
        scores = self.proj(normed)              # [batch, n_heads]

        # Soft tanh capping: scores asymptotically approach ±logit_cap
        # Prevents extreme negatives (e.g., -15) from causing numerical issues
        # Smooth gradients unlike hard clamp
        if self.logit_cap > 0:
            scores = self.logit_cap * torch.tanh(scores / self.logit_cap)

        return scores

    def get_head_idx(self, head_name: str) -> int:
        return self.head_names.index(head_name)

    def parameters(self):
        return list(self.rms_norm.parameters()) + list(self.proj.parameters())

    def state_dict(self):
        return {
            "rms_norm": self.rms_norm.state_dict(),
            "proj": self.proj.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.rms_norm.load_state_dict(state_dict["rms_norm"])
        self.proj.load_state_dict(state_dict["proj"])

    def train(self):
        self.rms_norm.train()
        self.proj.train()

    def eval(self):
        self.rms_norm.eval()
        self.proj.eval()

    def __call__(self, hidden_states):
        return self.forward(hidden_states)


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def logsquare_regularizer(scores, eps: float = 1e-6):
    """
    Logsquare: minimize log(r²) pushes positive logits toward r≈1.
    """
    ensure_torch()
    return torch.log(scores ** 2 + eps).mean()


def bradley_terry_loss(pos_scores, neg_scores):
    """Standard BT ranking loss."""
    ensure_torch()
    diff = pos_scores - neg_scores
    return -F.logsigmoid(diff).mean()


def compute_multihead_loss(
    pos_scores: "torch.Tensor",       # [n_pos, n_heads]
    neg_scores_by_tier: dict,         # tier -> [n_neg, n_heads]
    pos_head_indices: "torch.Tensor", # [n_pos] - which head each positive belongs to
    head_names: list[str],            # list of head names for indexing
    logsquare_weight: float = 0.1,
):
    """
    Compute loss for multiple heads with PER-SAMPLE head assignments.

    Each positive sample belongs to exactly one head. Loss is computed:
    - BT loss: compare each head's positives against shared negatives
    - Logsquare: only on that head's own positives (not all positives!)

    This fixes the bug where all positives were regularized for all heads.
    """
    ensure_torch()

    n_heads = pos_scores.size(1)
    # Use zeros_like to ensure gradients flow through
    total_bt_loss = pos_scores.new_zeros(())
    total_logsq_loss = pos_scores.new_zeros(())
    active_heads = 0

    # Tier weights (soft closer, furthest furthest)
    tier_weights = {
        "soft_neg": 2.0,
        "semi_firm_neg": 1.0,
        "furthest_neg": 0.5,
    }

    for head_idx in range(n_heads):
        # Mask: which positives belong to THIS head?
        head_mask = (pos_head_indices == head_idx)
        n_head_pos = head_mask.sum().item()

        if n_head_pos == 0:
            continue  # No positives for this head in batch

        active_heads += 1

        # Extract only THIS head's positive scores
        head_pos_scores = pos_scores[head_mask, head_idx]  # [n_head_pos]

        # BT loss against each negative tier
        for tier, neg_scores in neg_scores_by_tier.items():
            if neg_scores is None:
                continue

            head_neg = neg_scores[:, head_idx]  # [n_neg]
            n_neg = head_neg.size(0)
            if n_neg == 0:
                continue  # Skip empty negative tier to avoid NaN from mean()

            weight = tier_weights.get(tier, 1.0)

            # Match batch sizes for pairwise comparison
            if n_neg >= n_head_pos:
                bt_loss = bradley_terry_loss(head_pos_scores, head_neg[:n_head_pos])
            else:
                bt_loss = bradley_terry_loss(head_pos_scores[:n_neg], head_neg)

            total_bt_loss = total_bt_loss + weight * bt_loss

        # Logsquare regularization ONLY for this head's positives
        logsq = logsquare_regularizer(head_pos_scores)
        total_logsq_loss = total_logsq_loss + logsq

    if active_heads > 0:
        total_bt_loss = total_bt_loss / active_heads
        total_logsq_loss = total_logsq_loss / active_heads

    total_loss = total_bt_loss + logsquare_weight * total_logsq_loss

    return total_loss, total_bt_loss, total_logsq_loss


# =============================================================================
# TRAINER
# =============================================================================

class MultiHeadBTRMTrainer:
    """Multi-head BTRM training."""

    def __init__(self, config: MultiHeadBTRMConfig):
        ensure_torch()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load base model
        print(f"Loading base model: {config.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)

        # Determine model dtype - BF16 is more memory efficient and stable
        if config.use_amp and self.device == "cuda":
            if config.amp_dtype == "bfloat16":
                model_dtype = torch.bfloat16
                print("  Loading in BF16")
            else:
                model_dtype = torch.float16
                print("  Loading in FP16")
        else:
            model_dtype = torch.float32
            print("  Loading in FP32")

        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            trust_remote_code=True,
            torch_dtype=model_dtype,
        ).to(self.device)

        # For BF16/FP16 training, we still use autocast for ops that need higher precision
        self.amp_dtype = model_dtype if model_dtype != torch.float32 else None
        self.use_scaler = (model_dtype == torch.float16)  # Only FP16 needs scaler, BF16 doesn't

        # Gradient checkpointing for memory efficiency
        if config.gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable()
            print("  Gradient checkpointing enabled")

        # LoRA
        if config.use_lora:
            try:
                from peft import LoraConfig, get_peft_model
                lora_config = LoraConfig(
                    r=config.lora_r,
                    lora_alpha=config.lora_alpha,
                    target_modules=["q_proj", "v_proj"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                self.base_model = get_peft_model(self.base_model, lora_config)
                print(f"  LoRA (r={config.lora_r}, α={config.lora_alpha})")
            except ImportError:
                print("peft not installed")

        hidden_dim = self.base_model.config.hidden_size
        print(f"Hidden dim: {hidden_dim}")

        # Initialize multi-head BTRM
        head_names = [h.name for h in config.heads]
        self.btrm = MultiHeadBTRM(hidden_dim, head_names, self.device, config.logit_cap)
        print(f"Initialized {len(head_names)} heads: {head_names}")
        if config.logit_cap > 0:
            print(f"  Soft tanh logit cap: ±{config.logit_cap}")

        # Generate shared meta prompt (same for all heads - they learn from token:loss relationship)
        self.meta_prompt = make_meta_prompt() if config.use_meta_prompt else None

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # API walk streamer for on-the-fly positive generation
        self.api_streamer = None
        if config.use_api_walks:
            self.api_streamer = APIWalkStreamer(
                games=config.api_games,
                api_url=config.api_url,
                buffer_size=config.api_buffer_size,
                walks_per_request=20,
                max_length=8,
            )
            print(f"  API walk streaming enabled: {config.api_games}")

    def load_data(self):
        """Load positives per head, per-head negatives, and shared negatives."""
        print("\nLoading data...")

        # Positives per head
        positives_per_head: dict[str, list[Sample]] = {}
        for head in self.config.heads:
            samples = []
            # Use get_sources() to handle both new and legacy config format
            for source in head.get_sources():
                samples.extend(decode_local_jsonl(
                    source.path,
                    text_field=source.text_field,
                    tier_filter=source.tier_filter,
                ))
            for s in samples:
                s.tier = "positive"
            positives_per_head[head.name] = samples
            print(f"  Head '{head.name}': {len(samples)} positives")

        # Per-head negatives (optional - heads without these use only shared negatives)
        negatives_per_head: dict[str, dict[str, list[Sample]]] = {}
        for head in self.config.heads:
            if not head.negative_sources:
                continue

            head_negs = {"soft_neg": [], "semi_firm_neg": [], "furthest_neg": []}
            for neg_src in head.negative_sources:
                samples = decode_local_jsonl(
                    neg_src.path,
                    text_field=neg_src.text_field,
                    tier_filter=neg_src.tier_filter,
                    max_samples=self.config.neg_samples_per_tier,
                )
                for s in samples:
                    s.tier = neg_src.neg_tier
                head_negs[neg_src.neg_tier].extend(samples)

            negatives_per_head[head.name] = head_negs
            total = sum(len(v) for v in head_negs.values())
            print(f"  Head '{head.name}' per-head negatives: {total}")

        # Shared negatives by tier (external datasets)
        shared_negatives: dict[str, list[Sample]] = {
            "soft_neg": [],
            "semi_firm_neg": [],
            "furthest_neg": [],
        }

        # Soft negatives from other local corpora (legacy config path)
        for path in self.config.soft_neg_paths:
            samples = decode_local_jsonl(path, max_samples=self.config.neg_samples_per_tier)
            for s in samples:
                s.tier = "soft_neg"
            shared_negatives["soft_neg"].extend(samples)

        # Semi-firm: SYNTH + wattpad
        if self.config.use_synth:
            samples = decode_synth(self.config.neg_samples_per_tier)
            shared_negatives["semi_firm_neg"].extend(samples)

        if self.config.use_wattpad:
            samples = decode_wattpad(self.config.neg_samples_per_tier)
            shared_negatives["semi_firm_neg"].extend(samples)

        # Furthest: fineweb + wikitext (SAME TIER - both equally dissimilar)
        if self.config.use_fineweb:
            samples = decode_fineweb(self.config.neg_samples_per_tier)
            shared_negatives["furthest_neg"].extend(samples)

        if self.config.use_wikitext:
            samples = decode_wikitext(self.config.neg_samples_per_tier)
            shared_negatives["furthest_neg"].extend(samples)

        print("  Shared negatives:")
        for tier, samples in shared_negatives.items():
            print(f"    {tier}: {len(samples)} samples")

        # Pre-compute effective negatives per head: per-head + shared
        effective_negatives_per_head: dict[str, dict[str, list[Sample]]] = {}
        for head in self.config.heads:
            head_negs = {
                "soft_neg": list(shared_negatives["soft_neg"]),
                "semi_firm_neg": list(shared_negatives["semi_firm_neg"]),
                "furthest_neg": list(shared_negatives["furthest_neg"]),
            }
            # Merge per-head negatives if defined
            if head.name in negatives_per_head:
                for tier, samples in negatives_per_head[head.name].items():
                    head_negs[tier].extend(samples)
            effective_negatives_per_head[head.name] = head_negs

        return positives_per_head, effective_negatives_per_head

    def encode_batch(self, texts: list[str], max_length: int = None):
        """Tokenize and get hidden states."""
        if max_length is None:
            max_length = self.config.max_length

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.set_grad_enabled(self.base_model.training):
            outputs = self.base_model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        return outputs.hidden_states[-1]

    def train(self, save_path: Optional[str] = None):
        """Main training loop with multi-head loss masking."""
        ensure_torch()
        from torch.optim import AdamW

        positives_per_head, effective_negatives_per_head = self.load_data()

        # Pool all positives (we'll use head masks to select which contribute to loss)
        all_positives = []
        for head_name, samples in positives_per_head.items():
            for s in samples:
                s.metadata["head"] = head_name
            all_positives.extend(samples)

        if len(all_positives) < 50:
            print(f"Warning: Only {len(all_positives)} total positives")
            return

        print(f"\nTraining multi-head BTRM")
        print(f"  Heads: {len(self.config.heads)}")
        print(f"  Total positives: {len(all_positives)}")
        print(f"  Epochs: {self.config.epochs}")
        print(f"  Batch size: {self.config.batch_size}")

        # Setup optimizer
        params = list(self.base_model.parameters()) + list(self.btrm.parameters())
        optimizer = AdamW(params, lr=self.config.lr)

        # Calculate total steps for scheduler
        n_batches_per_epoch = len(all_positives) // self.config.batch_size
        if self.config.max_batches > 0:
            n_batches_per_epoch = min(n_batches_per_epoch, self.config.max_batches)
        total_steps = n_batches_per_epoch * self.config.epochs

        # Linear warmup scheduler
        from torch.optim.lr_scheduler import LambdaLR
        warmup_steps = min(self.config.warmup_steps, total_steps // 4)  # Cap at 25% of training

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0

        scheduler = LambdaLR(optimizer, lr_lambda)
        print(f"  LR: {self.config.lr}, warmup: {warmup_steps} steps")

        # AMP setup - only FP16 needs scaler, BF16 doesn't
        scaler = None
        if self.use_scaler:
            scaler = torch.amp.GradScaler()
            print(f"  GradScaler enabled (FP16)")
        elif self.amp_dtype is not None:
            print(f"  No scaler needed (BF16)")

        self.base_model.train()
        self.btrm.train()

        n_batches = n_batches_per_epoch
        if self.config.max_batches > 0:
            print(f"  Limited to {n_batches} batches/epoch (max_batches={self.config.max_batches})")

        # Pre-compute combined negatives pool (union across all heads, deduplicated)
        all_negs_by_tier = {"soft_neg": [], "semi_firm_neg": [], "furthest_neg": []}
        for head_name, head_negs in effective_negatives_per_head.items():
            for tier, samples in head_negs.items():
                all_negs_by_tier[tier].extend(samples)
        # Deduplicate
        for tier in all_negs_by_tier:
            seen = set()
            unique = []
            for s in all_negs_by_tier[tier]:
                key = s.text[:100]
                if key not in seen:
                    seen.add(key)
                    unique.append(s)
            all_negs_by_tier[tier] = unique
            print(f"  {tier} pool: {len(unique)} unique samples")

        # Track negative indices for systematic iteration
        neg_indices = {tier: 0 for tier in all_negs_by_tier}

        # Start API streamer if enabled
        if self.api_streamer:
            self.api_streamer.start()
            print("  API walk streamer started")

        # Timing stats for profiling
        import time
        timing = {"api_fetch": 0.0, "encode": 0.0, "forward": 0.0, "loss": 0.0, "backward": 0.0}

        total_batches_done = 0
        try:
            for epoch in range(self.config.epochs):
                # Shuffle both positives AND negatives each epoch
                random.shuffle(all_positives)
                for tier in all_negs_by_tier:
                    random.shuffle(all_negs_by_tier[tier])
                    neg_indices[tier] = 0  # Reset indices

                epoch_loss = 0.0
                epoch_bt = 0.0
                epoch_logsq = 0.0

                for batch_idx in range(n_batches):
                    start = batch_idx * self.config.batch_size
                    end = start + self.config.batch_size
                    pos_batch = list(all_positives[start:end])

                    # Mix in fresh API walks if enabled
                    if self.api_streamer and self.config.api_walks_per_batch > 0:
                        t0 = time.perf_counter()
                        api_walks = self.api_streamer.get_batch(self.config.api_walks_per_batch)
                        # Assign to multiturn_dialogue head (raw walks)
                        for w in api_walks:
                            w.metadata["head"] = "multiturn_dialogue"
                        pos_batch.extend(api_walks)
                        timing["api_fetch"] += time.perf_counter() - t0

                    # Build pos_head_indices: map each positive to its head index
                    head_name_to_idx = {h.name: i for i, h in enumerate(self.config.heads)}
                    pos_head_indices = torch.tensor([
                        head_name_to_idx[s.metadata["head"]]
                        for s in pos_batch
                    ], dtype=torch.long, device=self.device)
                    head_names = [h.name for h in self.config.heads]

                    # Template positives (with shared meta-prompt if enabled)
                    pos_texts = [
                        template_sample(s.text, self.meta_prompt)
                        for s in pos_batch
                    ]

                    # Collect negatives by tier - SYSTEMATIC iteration with wraparound
                    neg_batches_by_tier = {}
                    neg_texts_by_tier = {}
                    for tier, neg_pool in all_negs_by_tier.items():
                        if not neg_pool:
                            neg_batches_by_tier[tier] = []
                            neg_texts_by_tier[tier] = []
                            continue

                        n_neg = min(len(neg_pool), self.config.batch_size)
                        idx = neg_indices[tier]

                        # Get next batch of negatives, wrapping around if needed
                        if idx + n_neg <= len(neg_pool):
                            neg_batch = neg_pool[idx:idx + n_neg]
                            neg_indices[tier] = idx + n_neg
                        else:
                            # Wrap around
                            neg_batch = neg_pool[idx:] + neg_pool[:n_neg - (len(neg_pool) - idx)]
                            neg_indices[tier] = n_neg - (len(neg_pool) - idx)

                        neg_batches_by_tier[tier] = neg_batch
                        neg_texts_by_tier[tier] = [
                            template_sample(s.text, self.meta_prompt)
                            for s in neg_batch
                        ]

                    # BATCHED FORWARD: Concatenate all texts, one encode, one BTRM forward
                    all_texts = pos_texts.copy()
                    tier_slices = {"positive": (0, len(pos_texts))}
                    offset = len(pos_texts)

                    for tier in ["soft_neg", "semi_firm_neg", "furthest_neg"]:
                        texts = neg_texts_by_tier.get(tier, [])
                        if texts:
                            tier_slices[tier] = (offset, offset + len(texts))
                            all_texts.extend(texts)
                            offset += len(texts)
                        else:
                            tier_slices[tier] = None

                    # Batched forward - split into chunks to manage memory
                    # Memory scales ~1.6GB per 2 samples at 2048 tokens
                    max_chunk = 4  # Process at most 4 samples per forward pass

                    if self.device == "cuda":
                        torch.cuda.synchronize()
                    t0_encode = time.perf_counter()
                    t_forward = 0.0

                    # Use autocast for mixed precision forward
                    amp_context = torch.amp.autocast(
                        device_type="cuda", dtype=self.amp_dtype
                    ) if self.amp_dtype else nullcontext()

                    with amp_context:
                        if len(all_texts) <= max_chunk:
                            all_hidden = self.encode_batch(all_texts)
                            if self.device == "cuda":
                                torch.cuda.synchronize()
                            timing["encode"] += time.perf_counter() - t0_encode

                            t0 = time.perf_counter()
                            all_scores = self.btrm(all_hidden)
                            timing["forward"] += time.perf_counter() - t0
                        else:
                            # Chunked: encode + BTRM per chunk, concat scores (not hidden)
                            score_chunks = []
                            for i in range(0, len(all_texts), max_chunk):
                                chunk = all_texts[i:i + max_chunk]
                                chunk_hidden = self.encode_batch(chunk)

                                t0 = time.perf_counter()
                                chunk_scores = self.btrm(chunk_hidden)
                                t_forward += time.perf_counter() - t0

                                score_chunks.append(chunk_scores)
                                del chunk_hidden  # Free memory immediately

                            all_scores = torch.cat(score_chunks, dim=0)

                            if self.device == "cuda":
                                torch.cuda.synchronize()
                            timing["encode"] += time.perf_counter() - t0_encode - t_forward
                            timing["forward"] += t_forward

                    # Split scores by tier
                    pos_start, pos_end = tier_slices["positive"]
                    pos_scores = all_scores[pos_start:pos_end]

                    neg_scores_by_tier = {}
                    for tier in ["soft_neg", "semi_firm_neg", "furthest_neg"]:
                        if tier_slices[tier]:
                            start, end = tier_slices[tier]
                            neg_scores_by_tier[tier] = all_scores[start:end]
                        else:
                            neg_scores_by_tier[tier] = None

                    # Compute multi-head loss with per-sample head assignments
                    if self.device == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    loss, l_bt, l_logsq = compute_multihead_loss(
                        pos_scores,
                        neg_scores_by_tier,
                        pos_head_indices,
                        head_names,
                        self.config.logsquare_weight,
                    )
                    timing["loss"] += time.perf_counter() - t0

                    # Backward with AMP scaler
                    t0 = time.perf_counter()
                    optimizer.zero_grad()

                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        all_params = list(self.base_model.parameters()) + list(self.btrm.parameters())
                        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        all_params = list(self.base_model.parameters()) + list(self.btrm.parameters())
                        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                        optimizer.step()

                    scheduler.step()

                    if self.device == "cuda":
                        torch.cuda.synchronize()
                    timing["backward"] += time.perf_counter() - t0

                    epoch_loss += loss.item()
                    epoch_bt += l_bt.item()
                    epoch_logsq += l_logsq.item()
                    total_batches_done += 1

                    if (batch_idx + 1) % 100 == 0:
                        current_lr = scheduler.get_last_lr()[0]
                        print(f"  Epoch {epoch+1}/{self.config.epochs} | "
                              f"Batch {batch_idx+1}/{n_batches} | "
                              f"Loss: {loss.item():.4f} "
                              f"(BT: {l_bt.item():.4f}, LogSq: {l_logsq.item():.4f}) | "
                              f"LR: {current_lr:.2e}")

                avg = epoch_loss / max(n_batches, 1)
                print(f"Epoch {epoch+1} | Avg Loss: {avg:.4f}")

        finally:
            # Stop API streamer
            if self.api_streamer:
                self.api_streamer.stop()
                print("  API walk streamer stopped")

            # Print timing stats
            if total_batches_done > 0:
                print(f"\nTiming (total {total_batches_done} batches):")
                total_time = sum(timing.values())
                for op, t in sorted(timing.items(), key=lambda x: -x[1]):
                    pct = 100 * t / total_time if total_time > 0 else 0
                    per_batch = 1000 * t / total_batches_done
                    print(f"  {op:12s}: {t:7.2f}s ({pct:5.1f}%) - {per_batch:.1f}ms/batch")

        if save_path:
            self.save(save_path)

    def save(self, path: str):
        """Save model."""
        ensure_torch()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.base_model.save_pretrained(path / "base_model")
        self.tokenizer.save_pretrained(path / "base_model")
        torch.save({
            "btrm_state_dict": self.btrm.state_dict(),
            "head_names": self.btrm.head_names,
            "hidden_dim": self.btrm.hidden_dim,
        }, path / "btrm_heads.pt")
        self.config.to_yaml(str(path / "config.yaml"))

        print(f"Saved to {path}")

    def score(self, texts: list[str], head_name: Optional[str] = None) -> dict[str, list[float]]:
        """Score texts - returns scores for all heads or specified head."""
        ensure_torch()
        self.base_model.eval()
        self.btrm.eval()

        all_scores = {h: [] for h in self.btrm.head_names}
        bs = self.config.batch_size

        for i in range(0, len(texts), bs):
            batch = texts[i:i+bs]
            hidden = self.encode_batch(batch)
            scores = self.btrm(hidden)  # [batch, n_heads]

            for h_idx, h_name in enumerate(self.btrm.head_names):
                all_scores[h_name].extend(scores[:, h_idx].tolist())

        if head_name:
            return {head_name: all_scores[head_name]}
        return all_scores

    def probe_all_positions(self, text: str) -> dict:
        """
        Score at EVERY token position in one forward pass.

        Returns dict with:
        - tokens: list of token strings
        - token_ids: list of token ids
        - n_tokens: total token count
        - scores_by_head: {head_name: [score at each position]}

        This is the proper way - hidden states exist at every position,
        so we project to scores at every position in one pass.
        """
        ensure_torch()
        self.base_model.eval()
        self.btrm.eval()

        # Tokenize full text
        full_text = template_sample(text, self.meta_prompt) if self.meta_prompt else text
        encoding = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=self.config.max_length)
        token_ids = encoding.input_ids[0].tolist()
        n_tokens = len(token_ids)

        # Decode individual tokens for display
        tokens = [self.tokenizer.decode([tid]) for tid in token_ids]

        with torch.no_grad():
            # ONE forward pass
            inputs = encoding.to(self.device)
            outputs = self.base_model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = outputs.hidden_states[-1]  # [1, seq_len, hidden_dim]

            # Score at EVERY position
            # hidden: [1, seq_len, hidden_dim]
            # We need to project each position through RMSNorm + proj
            seq_len = hidden.size(1)

            # Reshape for batch projection: [seq_len, hidden_dim]
            hidden_flat = hidden[0]  # [seq_len, hidden_dim]
            normed = self.btrm.rms_norm(hidden_flat)  # [seq_len, hidden_dim]
            all_scores = self.btrm.proj(normed)  # [seq_len, n_heads]

            # Apply soft tanh capping
            if self.btrm.logit_cap > 0:
                all_scores = self.btrm.logit_cap * torch.tanh(all_scores / self.btrm.logit_cap)

            # Convert to per-head lists
            scores_by_head = {}
            for h_idx, h_name in enumerate(self.btrm.head_names):
                scores_by_head[h_name] = all_scores[:, h_idx].cpu().tolist()

        return {
            "tokens": tokens,
            "token_ids": token_ids,
            "n_tokens": n_tokens,
            "scores_by_head": scores_by_head,
        }

    def probe_prefixes(
        self,
        text: str,
        n_points: int = 20,
        min_tokens: int = 10,
    ) -> dict:
        """
        DEPRECATED: Use probe_all_positions() instead.
        This was doing multiple forward passes; the new version does one.
        Kept for backward compat, but just wraps probe_all_positions.
        """
        results = self.probe_all_positions(text)

        # Subsample to n_points for display
        n_tokens = results["n_tokens"]
        if n_tokens <= min_tokens:
            probe_points = [n_tokens - 1]
        else:
            step = max(1, (n_tokens - min_tokens) // (n_points - 1))
            probe_points = list(range(min_tokens, n_tokens, step))
            if probe_points[-1] != n_tokens - 1:
                probe_points.append(n_tokens - 1)

        # Extract scores at probe points
        subsampled_scores = {}
        for h, scores in results["scores_by_head"].items():
            subsampled_scores[h] = [scores[i] for i in probe_points]

        return {
            "tokens": results["tokens"],
            "token_ids": results["token_ids"],
            "n_tokens": n_tokens,
            "probe_points": probe_points,
            "scores_by_head": subsampled_scores,
            "all_scores": results["scores_by_head"],  # Full trajectory included
        }

    def print_probe_results(self, results: dict, show_text: bool = False, show_all: bool = False):
        """Pretty-print probe results."""
        print(f"\n{'='*60}")
        print(f"Token-wise Score Probe ({results['n_tokens']} tokens)")
        print(f"{'='*60}")

        heads = list(results["scores_by_head"].keys())

        # Use full scores if available and requested, else subsampled
        if show_all and "all_scores" in results:
            scores_to_show = results["all_scores"]
            positions = list(range(results["n_tokens"]))
            tokens = results["tokens"]
        else:
            scores_to_show = results["scores_by_head"]
            positions = results.get("probe_points", list(range(len(list(scores_to_show.values())[0]))))
            tokens = results["tokens"]

        # Header
        header = f"{'Pos':>5} {'Token':>15} | " + " | ".join(f"{h:>12}" for h in heads)
        print(header)
        print("-" * len(header))

        # Score trajectory
        for i, pos in enumerate(positions):
            scores = [scores_to_show[h][i] for h in heads]
            tok = tokens[pos] if pos < len(tokens) else "?"
            tok_display = tok[:12].replace("\n", "\\n").replace("\t", "\\t")
            row = f"{pos:>5} {tok_display:>15} | " + " | ".join(f"{s:>12.4f}" for s in scores)
            print(row)

        # Summary: score deltas and statistics
        print(f"\n{'Score summary':}")
        for h in heads:
            scores = scores_to_show[h]
            if len(scores) > 1:
                delta = scores[-1] - scores[0]
                min_s, max_s = min(scores), max(scores)
                mean_s = sum(scores) / len(scores)
                print(f"  {h}: {scores[0]:.4f} → {scores[-1]:.4f} "
                      f"(Δ={delta:+.4f}, min={min_s:.4f}, max={max_s:.4f}, μ={mean_s:.4f})")

    def print_sparkline(self, results: dict):
        """Print ASCII sparkline of score trajectories."""
        # Use full scores if available
        scores_dict = results.get("all_scores", results["scores_by_head"])
        heads = list(scores_dict.keys())

        # Sparkline chars (8 levels)
        sparks = "▁▂▃▄▅▆▇█"

        print(f"\nSparklines (token {0} → {results['n_tokens']-1}):")
        for h in heads:
            scores = scores_dict[h]
            min_s, max_s = min(scores), max(scores)
            range_s = max_s - min_s if max_s > min_s else 1

            # Subsample to ~80 chars if too long
            if len(scores) > 80:
                step = len(scores) // 80
                scores = scores[::step]

            # Convert to sparkline
            line = ""
            for s in scores:
                idx = int((s - min_s) / range_s * 7)
                idx = max(0, min(7, idx))
                line += sparks[idx]

            print(f"  {h:>20}: {line}")
            print(f"  {'':>20}  [{min_s:.2f} to {max_s:.2f}]")


# =============================================================================
# DEFAULT CONFIGS
# =============================================================================

def make_default_multihead_config() -> MultiHeadBTRMConfig:
    """
    Generate default multi-head config with all our corpora.

    Each corpus head gets ALL its data:
    - fk_normed prose (tier 2)
    - brainrot_aesops (vocabulary teaching)
    - flattened source walks (tier 1)

    Vocabulary slots:
    - Corpus heads: skyrim, oblivion, fonv, gallia, marmotte (ALL data each)
    - Structural heads: multiturn_dialogue (fk prose only), brainrot_aesop (aesops only)
    """
    heads = [
        # Corpus membership heads: include ALL data for this corpus
        HeadConfig(
            name="skyrim",
            description="All prose derived from Skyrim dialogue - Nordic fantasy RPG setting",
            positive_sources=[
                PositiveSource("dialogue_data/prose/skyrim_training_fk.jsonl", "auto", "fk_normed"),
                PositiveSource("dialogue_data/prose/skyrim_training_fk.jsonl", "auto", "flattened"),
                PositiveSource("dialogue_data/prose/skyrim_training_aesops.jsonl", "auto", "brainrot_aesop"),
            ],
        ),
        HeadConfig(
            name="oblivion",
            description="All prose derived from Oblivion dialogue - Imperial fantasy RPG setting",
            positive_sources=[
                PositiveSource("dialogue_data/prose/oblivion_training_fk.jsonl", "auto", "fk_normed"),
                PositiveSource("dialogue_data/prose/oblivion_training_fk.jsonl", "auto", "flattened"),
                PositiveSource("dialogue_data/prose/oblivion_training_aesops.jsonl", "auto", "brainrot_aesop"),
            ],
        ),
        HeadConfig(
            name="fonv",
            description="All prose derived from Fallout New Vegas dialogue - Post-apocalyptic Western RPG",
            positive_sources=[
                PositiveSource("dialogue_data/prose/falloutnv_training_fk.jsonl", "auto", "fk_normed"),
                PositiveSource("dialogue_data/prose/falloutnv_training_fk.jsonl", "auto", "flattened"),
                PositiveSource("dialogue_data/prose/falloutnv_training_aesops.jsonl", "auto", "brainrot_aesop"),
            ],
        ),
        HeadConfig(
            name="gallia",
            description="All prose derived from synthetic Gallia setting - Franco-Roman bureaucratic fantasy",
            positive_sources=[
                PositiveSource("output/gallia_v9_training_fk.jsonl", "auto", "fk_normed"),
                PositiveSource("output/gallia_v9_training_fk.jsonl", "auto", "flattened"),
                PositiveSource("output/gallia_v9_training_aesops.jsonl", "auto", "brainrot_aesop"),
            ],
        ),
        HeadConfig(
            name="marmotte",
            description="All prose derived from synthetic Marmotte setting - Alpine corporate dystopia",
            positive_sources=[
                PositiveSource("output/marmotte_v6_training_fk.jsonl", "auto", "fk_normed"),
                PositiveSource("output/marmotte_v6_training_fk.jsonl", "auto", "flattened"),
                PositiveSource("output/marmotte_v6_training_aesops.jsonl", "auto", "brainrot_aesop"),
            ],
        ),

        # Structural heads: specific format types across all corpora
        HeadConfig(
            name="multiturn_dialogue",
            description="Raw multi-turn dialogue walks (newline-concatenated quotes, not prose)",
            # Positives: flattened tier = actual quoted dialogue walks
            positive_sources=[
                PositiveSource("dialogue_data/prose/skyrim_training_fk.jsonl", "auto", "flattened"),
                PositiveSource("dialogue_data/prose/oblivion_training_fk.jsonl", "auto", "flattened"),
                PositiveSource("dialogue_data/prose/falloutnv_training_fk.jsonl", "auto", "flattened"),
                PositiveSource("output/gallia_v9_training_fk.jsonl", "auto", "flattened"),
                PositiveSource("output/marmotte_v6_training_fk.jsonl", "auto", "flattened"),
            ],
            # Soft negatives: our prose (embeds dialogue but isn't raw quotes)
            negative_sources=[
                # fk_normed prose - soft negative (dialogue embedded in prose)
                NegativeSource("dialogue_data/prose/skyrim_training_fk.jsonl", "auto", "fk_normed", "soft_neg"),
                NegativeSource("dialogue_data/prose/oblivion_training_fk.jsonl", "auto", "fk_normed", "soft_neg"),
                NegativeSource("dialogue_data/prose/falloutnv_training_fk.jsonl", "auto", "fk_normed", "soft_neg"),
                NegativeSource("output/gallia_v9_training_fk.jsonl", "auto", "fk_normed", "soft_neg"),
                NegativeSource("output/marmotte_v6_training_fk.jsonl", "auto", "fk_normed", "soft_neg"),
                # brainrot_aesops - also soft negative (vocab teaching, not raw dialogue)
                NegativeSource("dialogue_data/prose/skyrim_training_aesops.jsonl", "auto", "brainrot_aesop", "soft_neg"),
                NegativeSource("dialogue_data/prose/oblivion_training_aesops.jsonl", "auto", "brainrot_aesop", "soft_neg"),
                NegativeSource("dialogue_data/prose/falloutnv_training_aesops.jsonl", "auto", "brainrot_aesop", "soft_neg"),
                NegativeSource("output/gallia_v9_training_aesops.jsonl", "auto", "brainrot_aesop", "soft_neg"),
                NegativeSource("output/marmotte_v6_training_aesops.jsonl", "auto", "brainrot_aesop", "soft_neg"),
            ],
            # Furthest negatives: shared (wattpad, SYNTH, wikitext, fineweb)
        ),
        HeadConfig(
            name="brainrot_aesop",
            description="Vocabulary teaching passages with embedded definitions - brainrot style",
            positive_sources=[
                PositiveSource("dialogue_data/prose/skyrim_training_aesops.jsonl", "auto", "brainrot_aesop"),
                PositiveSource("dialogue_data/prose/oblivion_training_aesops.jsonl", "auto", "brainrot_aesop"),
                PositiveSource("dialogue_data/prose/falloutnv_training_aesops.jsonl", "auto", "brainrot_aesop"),
                PositiveSource("output/gallia_v9_training_aesops.jsonl", "auto", "brainrot_aesop"),
                PositiveSource("output/marmotte_v6_training_aesops.jsonl", "auto", "brainrot_aesop"),
            ],
        ),
    ]

    # Soft negatives: For corpus heads, other corpora are soft negatives
    # (handled implicitly by having them as positives for other heads)

    return MultiHeadBTRMConfig(
        heads=heads,
        soft_neg_paths=[],  # Cross-head positives act as soft negatives
        use_synth=True,
        use_wattpad=True,
        use_fineweb=True,
        use_wikitext=True,
        neg_samples_per_tier=300,
        base_model="Qwen/Qwen2.5-0.5B",
        use_meta_prompt=True,
    )


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-head logsquare-regularized BTRM")
    subparsers = parser.add_subparsers(dest="command")

    # Generate config
    gen_parser = subparsers.add_parser("gen-config", help="Generate default multi-head config")
    gen_parser.add_argument("--output", "-o", required=True, help="Output YAML path")

    # Train
    train_parser = subparsers.add_parser("train", help="Train multi-head BTRM")
    train_parser.add_argument("--config", "-c", required=True, help="Config YAML path")
    train_parser.add_argument("--output", "-o", required=True, help="Model output directory")

    # Score
    score_parser = subparsers.add_parser("score", help="Score samples with trained BTRM")
    score_parser.add_argument("--model", "-m", required=True, help="Model directory")
    score_parser.add_argument("--input", "-i", required=True, help="Input JSONL")
    score_parser.add_argument("--output", "-o", help="Output JSONL with scores")
    score_parser.add_argument("--head", help="Score only this head (default: all)")

    # Test decoders
    test_parser = subparsers.add_parser("test-decoders", help="Test dataset decoders")

    # Probe - token-wise score evolution
    probe_parser = subparsers.add_parser("probe", help="Probe score evolution along token positions")
    probe_parser.add_argument("--model", "-m", required=True, help="Model directory")
    probe_parser.add_argument("--text", "-t", help="Text to probe (or reads from stdin)")
    probe_parser.add_argument("--file", "-f", help="File containing text to probe")
    probe_parser.add_argument("--points", "-n", type=int, default=20, help="Number of probe points to display")
    probe_parser.add_argument("--min-tokens", type=int, default=10, help="Minimum tokens before first probe")
    probe_parser.add_argument("--all", action="store_true", help="Show scores at ALL token positions")
    probe_parser.add_argument("--sparkline", action="store_true", help="Show ASCII sparkline visualization")
    probe_parser.add_argument("--json", action="store_true", help="Output JSON instead of table")

    args = parser.parse_args()

    if args.command == "gen-config":
        config = make_default_multihead_config()
        config.to_yaml(args.output)
        print(f"Generated config: {args.output}")
        print(f"Heads: {[h.name for h in config.heads]}")

    elif args.command == "train":
        config = MultiHeadBTRMConfig.from_yaml(args.config)
        trainer = MultiHeadBTRMTrainer(config)
        trainer.train(args.output)

    elif args.command == "score":
        config = MultiHeadBTRMConfig.from_yaml(f"{args.model}/config.yaml")
        trainer = MultiHeadBTRMTrainer(config)

        # Load BTRM weights
        checkpoint = torch.load(f"{args.model}/btrm_heads.pt", map_location=trainer.device)
        trainer.btrm.load_state_dict(checkpoint["btrm_state_dict"])

        # Load input
        texts = []
        objs = []
        with open(args.input) as f:
            for line in f:
                obj = json.loads(line)
                text = obj.get("text") or obj.get("prose")
                if text:
                    texts.append(text)
                    objs.append(obj)

        # Score
        scores_by_head = trainer.score(texts, args.head)

        # Add scores to objects
        for i, obj in enumerate(objs):
            obj["btrm_scores"] = {h: scores_by_head[h][i] for h in scores_by_head}

        # Output
        output_path = args.output or args.input.replace(".jsonl", "_btrm_scored.jsonl")
        with open(output_path, "w") as f:
            for obj in objs:
                f.write(json.dumps(obj) + "\n")

        print(f"Wrote {len(objs)} scored samples to {output_path}")

    elif args.command == "test-decoders":
        print("Testing decoders...")
        print("\n=== Local JSONL ===")
        decode_local_jsonl("dialogue_data/prose/skyrim_training_fk.jsonl", max_samples=5)
        decode_local_jsonl("dialogue_data/prose/skyrim_training_aesops.jsonl", text_field="prose", max_samples=5)

        print("\n=== SYNTH ===")
        decode_synth(max_samples=5)

        print("\n=== Wattpad ===")
        decode_wattpad(max_samples=5)

        print("\n=== Fineweb ===")
        decode_fineweb(max_samples=5)

        print("\n=== Wikitext ===")
        decode_wikitext(max_samples=5)

    elif args.command == "probe":
        ensure_torch()
        config = MultiHeadBTRMConfig.from_yaml(f"{args.model}/config.yaml")
        trainer = MultiHeadBTRMTrainer(config)

        # Load BTRM weights
        checkpoint = torch.load(f"{args.model}/btrm_heads.pt", map_location=trainer.device)
        trainer.btrm.load_state_dict(checkpoint["btrm_state_dict"])

        # Get text to probe
        if args.text:
            text = args.text
        elif args.file:
            with open(args.file) as f:
                text = f.read()
        else:
            import sys
            print("Reading text from stdin (Ctrl+D to end)...")
            text = sys.stdin.read()

        if not text.strip():
            print("No text provided")
            sys.exit(1)

        # Probe - one forward pass, scores at every position
        results = trainer.probe_prefixes(
            text.strip(),
            n_points=args.points,
            min_tokens=args.min_tokens,
        )

        if args.json:
            # JSON output (includes all_scores for full trajectory)
            print(json.dumps(results, indent=2))
        else:
            # Pretty print
            trainer.print_probe_results(results, show_all=getattr(args, 'all', False))

            # Sparkline visualization
            if args.sparkline:
                trainer.print_sparkline(results)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
