# Complete Flow Verification Checklist

## Pre-Training Verification

Before running large-scale training, verify each step:

### ✅ Step 1: Data Loading
- [ ] OpenWebText loads correctly
- [ ] Documents are non-empty
- [ ] Streaming works without full download
- [ ] Document quality verified

**Test**: `test_step1_data_loading()`

### ✅ Step 2: Vocabulary Building
- [ ] Vocabulary size matches target
- [ ] Special tokens included: `<unk>`, `<pad>`, `<bos>`, `<eos>`
- [ ] Word-to-index mapping is consistent
- [ ] Common words have good coverage (>80%)

**Test**: `test_step2_vocabulary_building()`

**Critical**: Verify vocabulary structure matches tokenization:
- `train_modal.py` uses: `{"<unk>": 0, "<pad>": 1}`
- Tokenization must use: `w2i.get(w, 0)` for unknown words

### ✅ Step 3: Tokenization
- [ ] Tokenization produces valid sequences
- [ ] Unknown words map to `<unk>` (not `<pad>`)
- [ ] Word order preserved
- [ ] Tokenization method matches vocabulary building

**Test**: `test_step3_tokenization()`

**Critical Fixes Applied**:
- ✅ Step size changed from `ctx_size` to `1` (overlapping windows)
- ✅ Unknown words use correct default (0 = `<unk>`)

### ✅ Step 4: Sample Extraction
- [ ] Context/target pairs extracted correctly
- [ ] Context size matches target
- [ ] No `<unk>` targets (filtered out)
- [ ] Overlapping windows (step size 1)

**Test**: `test_step4_sample_extraction()`

**Critical**: Verify step size is 1, not `ctx_size`

### ✅ Step 5: Grounded Embeddings
- [ ] GloVe embeddings load correctly
- [ ] Coverage >50%
- [ ] Embeddings are SO(4) matrices
- [ ] Embeddings are unit norm

**Test**: `test_step5_grounded_embeddings()`

### ✅ Step 6: Model Initialization
- [ ] Model initializes correctly
- [ ] Embeddings set correctly
- [ ] Memory tensors allocated
- [ ] GPU memory usage < 10GB

**Test**: `test_step6_model_initialization()`

### ✅ Step 7: Learning
- [ ] Batch learning works correctly
- [ ] Memory is updated
- [ ] Throughput > 100 samples/sec
- [ ] Batch processing is vectorized

**Test**: `test_step7_learning()`

### ✅ Step 8: Retrieval
- [ ] Grace ALWAYS converges (never returns None)
- [ ] Full vocabulary coherence scoring
- [ ] No candidate sets
- [ ] Retrieval accuracy improves after learning

**Test**: `test_step8_retrieval()`

**Critical**: Verify `retrieve()` path matches `retrieve_settled_states_batch()`

### ✅ Step 9: Evaluation
- [ ] Uses coherence, not similarity
- [ ] Matches retrieve() path exactly
- [ ] Coherence scores in [0, 1]
- [ ] Evaluation improves with learning

**Test**: `test_step9_evaluation()`

**Critical**: Verify `evaluate_semantic()` uses coherence scoring

### ✅ Step 10: End-to-End
- [ ] All steps work together
- [ ] Learning improves metrics
- [ ] No information loss
- [ ] Theory-true throughout

**Test**: `test_step10_end_to_end()`

## Critical Fixes Applied

### Fix 1: Sample Extraction Step Size
**File**: `train_modal.py` line 284
**Change**: `range(0, len(toks) - ctx_size - 1, ctx_size)` → `range(0, len(toks) - ctx_size, 1)`
**Impact**: Now extracts ALL possible samples (overlapping windows), not just 1 per ctx_size

### Fix 2: retrieve_settled_states_batch() Grace Contractions
**File**: `holographic_prod/memory/holographic_memory_unified.py`
**Change**: Added Grace contractions matching `retrieve()` exactly
**Impact**: Evaluation path now matches production retrieval

### Fix 3: evaluate_semantic() Coherence Scoring
**File**: `holographic_prod/memory/holographic_memory_unified.py`
**Change**: Uses coherence (witness_energy / total_energy) instead of Frobenius cosine
**Impact**: Metrics now reflect theory-true learning

## Information Flow Verification

At each step, verify:
- ✅ No information loss
- ✅ No fallbacks or simplifications
- ✅ Theory-true path preserved
- ✅ Real data (no fake/placeholder)

## Running Tests

```bash
# Run all granular tests
modal run holographic_prod/tests/test_complete_flow_granular.py

# Run specific step
modal run holographic_prod/tests/test_complete_flow_granular.py::test_step1_data_loading
```

## Expected Results

- **Step 1**: Documents load successfully
- **Step 2**: Vocabulary built with >80% common word coverage
- **Step 3**: Tokenization produces valid sequences, OOV rate <5%
- **Step 4**: Samples extracted with overlapping windows
- **Step 5**: GloVe coverage >50%, SO(4) properties verified
- **Step 6**: Model initializes, GPU memory <10GB
- **Step 7**: Throughput >100 samples/sec, memory updates correctly
- **Step 8**: Grace convergence 100%, no candidate sets
- **Step 9**: Coherence scores in [0,1], evaluation improves
- **Step 10**: Complete flow works, learning verified
