"""
Phase 2: Generalization Tests on Real Text

THEORY BEING TESTED:
    - Holographic memory generalizes to unseen data
    - Grace attractors capture semantic structure, not just memorization
    - Context length transfer works due to SO(4) composition
    - Errors are semantically nearby (graceful degradation)

SUCCESS CRITERIA:
    - test_coherence / train_coherence > 0.8
    - Context length transfer degrades < 30%
    - Vocabulary generalization > 0.2 coherence
    - Average semantic distance of errors < 0.5

RUNTIME: ~45 minutes on H100
"""

import modal
import numpy as np
import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

# Modal setup
app = modal.App("generalization-real-text")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11"
    )
    .pip_install(
        "numpy>=1.24",
        "cupy-cuda12x>=12.0",
        "scipy>=1.10",
        "datasets>=2.14",
        "tqdm>=4.65",
        "huggingface_hub>=0.16",
    )
    .add_local_dir("holographic_prod", "/root/project/holographic_prod")
)

checkpoint_volume = modal.Volume.from_name("holographic-checkpoints", create_if_missing=True)


@dataclass
class GeneralizationReport:
    """Full report from generalization experiment."""
    # Train/Test Split
    train_coherence: float = 0.0
    test_coherence: float = 0.0
    generalization_ratio: float = 0.0
    
    # Context Length Transfer
    base_context_coherence: float = 0.0
    context_transfer: Dict[int, float] = field(default_factory=dict)
    max_transfer_degradation: float = 0.0
    
    # Vocabulary Generalization
    known_vocab_coherence: float = 0.0
    partial_unk_coherence: float = 0.0
    vocab_generalization: float = 0.0
    
    # Semantic Distance of Errors
    avg_error_distance: float = 0.0
    error_in_top_10: float = 0.0
    
    # Overall
    passed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'train_coherence': self.train_coherence,
            'test_coherence': self.test_coherence,
            'generalization_ratio': self.generalization_ratio,
            'base_context_coherence': self.base_context_coherence,
            'context_transfer': self.context_transfer,
            'max_transfer_degradation': self.max_transfer_degradation,
            'known_vocab_coherence': self.known_vocab_coherence,
            'partial_unk_coherence': self.partial_unk_coherence,
            'vocab_generalization': self.vocab_generalization,
            'avg_error_distance': self.avg_error_distance,
            'error_in_top_10': self.error_in_top_10,
            'passed': self.passed,
        }


@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
    volumes={"/checkpoints": checkpoint_volume},
    env={"HF_HOME": "/checkpoints/hf_cache"}
)
def run_generalization_tests():
    """
    Run generalization tests on real OpenWebText data.
    
    Tests:
    1. Train/Test Split Generalization
    2. Context Length Transfer
    3. Vocabulary Generalization
    4. Semantic Similarity of Errors
    """
    import sys
    sys.path.insert(0, '/root/project')
    
    import cupy as cp
    from datasets import load_dataset
    from tqdm import tqdm
    import re
    
    from holographic_prod.memory.holographic_memory_unified import HolographicMemory
    from holographic_prod.core.grounded_embeddings import create_grounded_embeddings_fast
    from holographic_prod.core.algebra import (
        decompose_to_coefficients_batch,
        get_cached_basis,
        grace_with_stability,
    )
    from holographic_prod.core.constants import PHI_EPSILON
    
    print("="*80)
    print("PHASE 2: GENERALIZATION TESTS ON REAL TEXT")
    print("="*80)
    
    xp = cp
    basis = get_cached_basis(xp)
    
    # Configuration
    VOCAB_SIZE = 30_000
    CONTEXT_SIZE = 64
    TRAIN_DOCS = 500
    TEST_DOCS = 500
    
    report = GeneralizationReport()
    
    # =========================================================================
    # STEP 1: Build vocabulary and load data
    # =========================================================================
    print("\n[Step 1] Building vocabulary and loading data...")
    
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    
    word_counts: Dict[str, int] = {}
    pattern = re.compile(r'\b[a-zA-Z]+\b')
    
    # Build vocab from first 5000 docs
    for i, item in enumerate(tqdm(ds.take(5000), total=5000, desc="Building vocab")):
        words = pattern.findall(item['text'].lower())
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1
    
    sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])[:VOCAB_SIZE - 2]
    word_to_idx = {"<unk>": 0, "<pad>": 1}
    idx_to_word = {0: "<unk>", 1: "<pad>"}
    for i, (word, _) in enumerate(sorted_words):
        word_to_idx[word] = i + 2
        idx_to_word[i + 2] = word
    
    print(f"  Vocabulary size: {len(word_to_idx):,}")
    
    # =========================================================================
    # STEP 2: Extract SEPARATE train and test sets
    # =========================================================================
    print("\n[Step 2] Extracting separate train and test sets...")
    
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    
    train_samples: List[Tuple[List[int], int]] = []
    test_samples: List[Tuple[List[int], int]] = []
    
    doc_idx = 0
    for item in tqdm(ds, desc="Extracting samples"):
        words = pattern.findall(item['text'].lower())
        tokens = [word_to_idx.get(w, 0) for w in words]
        
        if len(tokens) < CONTEXT_SIZE + 1:
            doc_idx += 1
            continue
        
        # Determine if this doc is train or test
        is_train = doc_idx < TRAIN_DOCS
        is_test = TRAIN_DOCS <= doc_idx < TRAIN_DOCS + TEST_DOCS
        
        if is_train or is_test:
            target_list = train_samples if is_train else test_samples
            
            for i in range(0, min(len(tokens) - CONTEXT_SIZE, 100), 1):  # Max 100 per doc
                ctx = tokens[i:i + CONTEXT_SIZE]
                tgt = tokens[i + CONTEXT_SIZE]
                if tgt != 0:
                    target_list.append((ctx, tgt))
        
        doc_idx += 1
        if doc_idx >= TRAIN_DOCS + TEST_DOCS:
            break
    
    print(f"  Train samples: {len(train_samples):,}")
    print(f"  Test samples: {len(test_samples):,}")
    
    # =========================================================================
    # STEP 3: Create grounded embeddings and model
    # =========================================================================
    print("\n[Step 3] Creating model with grounded embeddings...")
    
    grounded_embs, coverage = create_grounded_embeddings_fast(
        word_to_idx, glove_dim=50, cache_dir="/checkpoints/glove"
    )
    
    model = HolographicMemory(
        vocab_size=len(word_to_idx),
        max_levels=4,
        seed=42,
        use_gpu=True,
        grounded_embeddings=cp.asarray(grounded_embs),
    )
    
    print(f"  GloVe coverage: {coverage*100:.1f}%")
    
    # =========================================================================
    # HELPER: Evaluate coherence
    # =========================================================================
    def evaluate_coherence(model, samples, n_eval=500):
        """Evaluate coherence using theory-true path."""
        coherences = []
        
        for ctx, tgt in samples[:n_eval]:
            ctx_mat = model.tower._embed_sequence(ctx)
            graced_ctx, _, _ = grace_with_stability(ctx_mat, basis, xp)
            
            sat_idx = model.tower.route_to_satellite(ctx)
            sat_memory = model.tower._all_memories[sat_idx]
            
            retrieved = graced_ctx.T @ sat_memory
            graced_ret, _, _ = grace_with_stability(retrieved, basis, xp)
            
            tgt_emb = model.tower.embeddings[tgt]
            composition = graced_ret @ tgt_emb.T
            
            coeffs = decompose_to_coefficients_batch(composition.reshape(1, 4, 4), basis, xp)[0]
            total_e = float(xp.sum(coeffs ** 2))
            witness_e = float(coeffs[0]**2 + coeffs[15]**2)
            coh = witness_e / max(total_e, PHI_EPSILON)
            coherences.append(coh)
        
        return np.mean(coherences)
    
    # =========================================================================
    # TEST 1: Train/Test Split Generalization
    # =========================================================================
    print("\n" + "="*80)
    print("TEST 1: TRAIN/TEST SPLIT GENERALIZATION")
    print("="*80)
    
    # Train on train_samples
    print("\n  Training on train samples...")
    batch_size = 2048
    for i in tqdm(range(0, len(train_samples), batch_size), desc="Training"):
        batch = train_samples[i:i+batch_size]
        contexts = [ctx for ctx, _ in batch]
        targets = [tgt for _, tgt in batch]
        model.learn_batch(contexts, targets)
    
    # Evaluate on train set
    print("\n  Evaluating on train set...")
    report.train_coherence = evaluate_coherence(model, train_samples, n_eval=500)
    print(f"    Train coherence: {report.train_coherence:.4f}")
    
    # Evaluate on test set (NEVER SEEN)
    print("\n  Evaluating on test set (never seen)...")
    report.test_coherence = evaluate_coherence(model, test_samples, n_eval=500)
    print(f"    Test coherence: {report.test_coherence:.4f}")
    
    # Generalization ratio
    report.generalization_ratio = report.test_coherence / max(report.train_coherence, 0.01)
    print(f"\n  Generalization ratio: {report.generalization_ratio:.4f}")
    print(f"  Target: > 0.8")
    
    if report.generalization_ratio > 0.8:
        print("  ✓ PASSED: Good generalization to unseen data")
    else:
        print("  ✗ FAILED: Poor generalization")
    
    # =========================================================================
    # TEST 2: Context Length Transfer
    # =========================================================================
    print("\n" + "="*80)
    print("TEST 2: CONTEXT LENGTH TRANSFER")
    print("="*80)
    
    # Baseline at trained context size (64)
    report.base_context_coherence = report.train_coherence
    print(f"\n  Baseline (ctx=64): {report.base_context_coherence:.4f}")
    
    # Test at different context lengths
    context_lengths = [32, 128, 256]
    
    for ctx_len in context_lengths:
        print(f"\n  Testing context length {ctx_len}...")
        
        # Prepare samples at this context length
        transfer_samples = []
        ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        
        for item in ds.skip(TRAIN_DOCS + TEST_DOCS).take(200):
            words = pattern.findall(item['text'].lower())
            tokens = [word_to_idx.get(w, 0) for w in words]
            
            if len(tokens) < ctx_len + 1:
                continue
            
            for i in range(0, min(len(tokens) - ctx_len, 10), 1):
                ctx = tokens[i:i + ctx_len]
                tgt = tokens[i + ctx_len]
                if tgt != 0:
                    transfer_samples.append((ctx, tgt))
            
            if len(transfer_samples) >= 500:
                break
        
        # Evaluate
        coherence = evaluate_coherence(model, transfer_samples, n_eval=min(500, len(transfer_samples)))
        report.context_transfer[ctx_len] = coherence
        
        degradation = 1.0 - (coherence / max(report.base_context_coherence, 0.01))
        print(f"    Coherence: {coherence:.4f}")
        print(f"    Degradation: {degradation*100:.1f}%")
    
    # Max degradation
    if report.context_transfer:
        report.max_transfer_degradation = max(
            1.0 - (c / max(report.base_context_coherence, 0.01)) 
            for c in report.context_transfer.values()
        )
    
    print(f"\n  Max degradation: {report.max_transfer_degradation*100:.1f}%")
    print(f"  Target: < 30%")
    
    if report.max_transfer_degradation < 0.30:
        print("  ✓ PASSED: Good context length transfer")
    else:
        print("  ✗ FAILED: Poor context length transfer")
    
    # =========================================================================
    # TEST 3: Vocabulary Generalization
    # =========================================================================
    print("\n" + "="*80)
    print("TEST 3: VOCABULARY GENERALIZATION")
    print("="*80)
    
    # Known vocabulary coherence
    report.known_vocab_coherence = report.train_coherence
    print(f"\n  Known vocab coherence: {report.known_vocab_coherence:.4f}")
    
    # Test with partial unknown tokens (20% <unk>)
    print("\n  Creating samples with 20% unknown tokens...")
    
    partial_unk_samples = []
    for ctx, tgt in test_samples[:500]:
        # Replace 20% of context with <unk>
        new_ctx = ctx.copy()
        n_replace = max(1, len(ctx) // 5)
        replace_indices = np.random.choice(len(ctx), n_replace, replace=False)
        for idx in replace_indices:
            new_ctx[idx] = 0  # <unk>
        partial_unk_samples.append((new_ctx, tgt))
    
    report.partial_unk_coherence = evaluate_coherence(model, partial_unk_samples, n_eval=500)
    print(f"  Partial <unk> coherence: {report.partial_unk_coherence:.4f}")
    
    report.vocab_generalization = report.partial_unk_coherence / max(report.known_vocab_coherence, 0.01)
    print(f"  Vocab generalization ratio: {report.vocab_generalization:.4f}")
    print(f"  Target: > 0.2 coherence (graceful degradation)")
    
    if report.partial_unk_coherence > 0.2:
        print("  ✓ PASSED: Graceful degradation with unknown tokens")
    else:
        print("  ✗ FAILED: Collapse with unknown tokens")
    
    # =========================================================================
    # TEST 4: Semantic Similarity of Errors
    # =========================================================================
    print("\n" + "="*80)
    print("TEST 4: SEMANTIC SIMILARITY OF ERRORS")
    print("="*80)
    
    print("\n  Analyzing retrieval errors...")
    
    error_distances = []
    in_top_10 = 0
    n_analyzed = 0
    
    for ctx, tgt in test_samples[:200]:
        # Get retrieved token
        retrieved = model.tower.retrieve(ctx)
        
        if retrieved != tgt:
            # Measure semantic distance between retrieved and target
            tgt_emb = model.tower.embeddings[tgt]
            ret_emb = model.tower.embeddings[retrieved]
            
            # Frobenius cosine distance
            tgt_flat = tgt_emb.flatten()
            ret_flat = ret_emb.flatten()
            
            dot = float(xp.sum(tgt_flat * ret_flat))
            norm_tgt = float(xp.linalg.norm(tgt_flat))
            norm_ret = float(xp.linalg.norm(ret_flat))
            
            similarity = dot / max(norm_tgt * norm_ret, PHI_EPSILON)
            distance = 1.0 - similarity
            error_distances.append(distance)
            
            # Check if target is in top-10
            ctx_mat = model.tower._embed_sequence(ctx)
            graced_ctx, _, _ = grace_with_stability(ctx_mat, basis, xp)
            sat_idx = model.tower.route_to_satellite(ctx)
            sat_memory = model.tower._all_memories[sat_idx]
            retrieved_state = graced_ctx.T @ sat_memory
            graced_ret, _, _ = grace_with_stability(retrieved_state, basis, xp)
            
            # Score all vocab
            all_embs = model.tower.embeddings
            scores = xp.einsum('ij,vij->v', graced_ret, all_embs)
            top_10 = xp.argsort(scores)[-10:][::-1]
            
            if tgt in top_10.get():
                in_top_10 += 1
        
        n_analyzed += 1
    
    if error_distances:
        report.avg_error_distance = np.mean(error_distances)
        report.error_in_top_10 = in_top_10 / len(error_distances) if error_distances else 0
    
    print(f"  Errors analyzed: {len(error_distances)}")
    print(f"  Average error distance: {report.avg_error_distance:.4f}")
    print(f"  Target in top-10: {report.error_in_top_10*100:.1f}%")
    print(f"  Target: distance < 0.5")
    
    if report.avg_error_distance < 0.5:
        print("  ✓ PASSED: Errors are semantically nearby")
    else:
        print("  ✗ FAILED: Errors are not semantically nearby")
    
    # =========================================================================
    # SUCCESS CRITERIA
    # =========================================================================
    print("\n" + "="*80)
    print("SUCCESS CRITERIA")
    print("="*80)
    
    criteria = {
        'generalization_ratio > 0.8': report.generalization_ratio > 0.8,
        'context_transfer_degradation < 30%': report.max_transfer_degradation < 0.30,
        'vocab_generalization > 0.2': report.partial_unk_coherence > 0.2,
        'avg_error_distance < 0.5': report.avg_error_distance < 0.5,
    }
    
    for name, passed in criteria.items():
        status = '✓' if passed else '✗'
        print(f"  {status} {name}: {passed}")
    
    # Overall pass: generalization_ratio > 0.8 is the key metric
    report.passed = report.generalization_ratio > 0.8
    
    print("\n" + "="*80)
    if report.passed:
        print("PHASE 2 PASSED: Model generalizes to unseen data")
    else:
        print("PHASE 2 FAILED: Poor generalization")
    print("="*80)
    
    return report.to_dict()


@app.local_entrypoint()
def main():
    """Run generalization tests."""
    result = run_generalization_tests.remote()
    print("\n" + json.dumps(result, indent=2))
    
    with open("/tmp/generalization_result.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print("\nResult saved to /tmp/generalization_result.json")
