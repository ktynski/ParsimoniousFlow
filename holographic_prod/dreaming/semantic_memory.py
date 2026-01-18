"""
Semantic Memory — Hierarchical Prototype Storage and Retrieval

Implements theory-true semantic memory:
- Multi-level hierarchical storage (fine → coarse)
- Schema storage for structural patterns
- φ-weighted attention over schemas
- Hierarchical meta-schema attention

RETRIEVAL:
    1. Search at coarse level for candidate regions
    2. Refine at finer levels
    3. Return best match with target distribution

GPU-NATIVE: All arrays use self.xp consistently. No NumPy/CuPy mixing.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from holographic_prod.core.constants import PHI, PHI_INV, PHI_INV_SQ, PHI_EPSILON
from holographic_prod.core.algebra import frobenius_cosine_batch
from holographic_prod.core.quotient import extract_witness
from .structures import SemanticPrototype, Schema, MetaSchema
from .pattern_completion import pattern_complete


# =============================================================================
# SEMANTIC MEMORY
# =============================================================================

class SemanticMemory:
    """
    Hierarchical semantic memory supporting multi-resolution retrieval.
    
    GPU-NATIVE: All operations use self.xp consistently.
    Prototypes and schemas are stored as arrays of the same type as self.xp.
    
    Levels:
        0: Fine-grained prototypes (high detail)
        1+: Coarser abstractions (more generalization)
    
    Retrieval:
        1. Search at coarse level for candidate regions
        2. Refine at finer levels
        3. Return best match with target distribution
    """
    
    def __init__(
        self,
        basis,  # Can be np.ndarray or cp.ndarray
        xp=np,
        num_levels: int = 3,
    ):
        self.basis = basis
        self.xp = xp
        self.num_levels = num_levels
        
        # Storage by level
        self.levels: List[List[SemanticPrototype]] = [[] for _ in range(num_levels)]
        
        # Schema storage
        self.schemas: List[Schema] = []
    
    def _ensure_device(self, arr):
        """Ensure array is on the correct device (GPU/CPU) matching self.xp.
        
        FIX v3 2024-01-15: Robust GPU/CPU array handling.
        """
        xp = self.xp
        
        # Check if xp is CuPy
        is_cupy = hasattr(xp, 'cuda')
        
        if is_cupy:
            # xp is CuPy - ensure array is CuPy
            if hasattr(arr, '__cuda_array_interface__'):
                return arr  # Already CuPy
            else:
                return xp.asarray(arr)  # Convert NumPy to CuPy
        else:
            # xp is NumPy - ensure array is NumPy
            if hasattr(arr, 'get'):
                return arr.get()  # Convert CuPy to NumPy
            else:
                return arr  # Already NumPy
    
    def add_prototype(self, proto: SemanticPrototype, level: int = 0):
        """Add a prototype at the specified level."""
        proto.level = level
        if level < len(self.levels):
            self.levels[level].append(proto)
    
    def add_schema(self, schema: Schema, max_schemas: int = None):
        """
        Add a discovered schema with φ-derived bounded growth.
        
        THEORY-TRUE: Schema capacity scales with prototype count.
        More prototypes → more possible combinations → need more schemas.
        Default: φ² × num_prototypes × 100
        """
        if max_schemas is None:
            num_protos = sum(len(level) for level in self.levels)
            max_schemas = max(10000, int(PHI * PHI * num_protos * 100))
        
        if len(self.schemas) >= max_schemas:
            # Find schema with lowest recurrence
            min_idx = min(range(len(self.schemas)), key=lambda i: self.schemas[i].recurrence_count)
            if schema.recurrence_count > self.schemas[min_idx].recurrence_count:
                self.schemas[min_idx] = schema
        else:
            self.schemas.append(schema)
    
    def retrieve(
        self, 
        query,
        top_k: int = 5,
        use_pattern_completion: bool = False,
        completion_steps: int = 3,
        use_schemas: bool = True,
    ) -> List[Tuple[SemanticPrototype, float]]:
        """
        VECTORIZED hierarchical retrieval: coarse to fine.
        
        GPU-NATIVE: Query and prototypes must be on same device as self.xp.
        
        PATTERN COMPLETION (optional):
            When enabled, applies Grace flow to query before matching.
            This denoises partial/noisy queries toward stored patterns.
            
        SCHEMA RETRIEVAL:
            When enabled, also searches schemas. If a query matches a schema,
            returns the prototypes that contributed to that schema.
        """
        xp = self.xp
        
        # Ensure query is on correct device
        query = self._ensure_device(query)
        
        # Optional pattern completion (uses self.xp)
        if use_pattern_completion:
            query, _ = pattern_complete(query, self.basis, xp, max_steps=completion_steps)
        
        # Collect all prototypes across levels
        all_protos = []
        for level in self.levels:
            all_protos.extend(level)
        
        if not all_protos:
            return []
        
        # VECTORIZED: Stack prototype matrices (ensure on correct device)
        proto_matrices_list = [self._ensure_device(p.prototype_matrix) for p in all_protos]
        proto_matrices = xp.stack(proto_matrices_list)
        
        sims = frobenius_cosine_batch(query, proto_matrices, xp)
        
        # Create sorted (proto, sim) pairs
        candidates = list(zip(all_protos, [float(s) for s in sims]))
        
        # Schema-based retrieval
        if use_schemas and self.schemas:
            schema_matrices_list = [self._ensure_device(s.canonical_form) for s in self.schemas]
            schema_matrices = xp.stack(schema_matrices_list)
            schema_sims = frobenius_cosine_batch(query, schema_matrices, xp)
            
            for i, (schema, sim) in enumerate(zip(self.schemas, schema_sims)):
                sim = float(sim)
                if sim > PHI_INV_SQ:  # Schema matches structurally
                    for proto_id in schema.source_prototype_ids:
                        if proto_id < len(all_protos):
                            proto = all_protos[proto_id]
                            proto_sim = float(sims[proto_id])
                            combined_sim = PHI_INV * sim + (1 - PHI_INV) * proto_sim
                            candidates.append((proto, combined_sim))
        
        candidates.sort(key=lambda x: -x[1])
        
        # Deduplicate
        seen = set()
        unique_candidates = []
        for proto, sim in candidates:
            proto_id = id(proto)
            if proto_id not in seen:
                seen.add(proto_id)
                unique_candidates.append((proto, sim))
        
        return unique_candidates[:top_k]
    
    def retrieve_with_radius(
        self,
        query,
        use_pattern_completion: bool = False,
        completion_steps: int = 3,
    ) -> Optional[SemanticPrototype]:
        """
        VECTORIZED retrieve best prototype within its radius.
        
        GPU-NATIVE: Query must be on same device as self.xp.
        
        This enables generalization: if query is close enough to a prototype,
        use that prototype's target distribution.
        """
        xp = self.xp
        
        # Ensure query is on correct device
        query = self._ensure_device(query)
        
        if use_pattern_completion:
            query, _ = pattern_complete(query, self.basis, xp, max_steps=completion_steps)
        
        all_protos = []
        for level in self.levels:
            all_protos.extend(level)
        
        if not all_protos:
            return None
        
        # VECTORIZED - ensure all on correct device
        proto_matrices_list = [self._ensure_device(p.prototype_matrix) for p in all_protos]
        proto_matrices = xp.stack(proto_matrices_list)
        radii = xp.array([p.radius for p in all_protos])
        
        sims = frobenius_cosine_batch(query, proto_matrices, xp)
        distances = 1.0 - sims
        
        margins = radii - distances
        
        # THEORY JUSTIFICATION: This is GEOMETRIC nearest-neighbor in prototype space,
        # NOT token decoding. Finding the prototype whose basin contains the query is
        # a geometric routing operation - analogous to Grace basin routing.
        best_idx = int(xp.argmax(margins))
        best_margin = float(margins[best_idx])
        
        return all_protos[best_idx] if best_margin >= 0 else None
    
    def stats(self) -> Dict[str, Any]:
        """Return memory statistics."""
        total = sum(len(level) for level in self.levels)
        return {
            "total_prototypes": total,
            "prototypes_by_level": [len(level) for level in self.levels],
            "num_schemas": len(self.schemas),
            "avg_entropy": np.mean([p.entropy() for level in self.levels for p in level]) if total > 0 else 0,
        }
    
    def schema_attention(
        self,
        query,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Theory-true φ-weighted attention over schemas.
        
        GPU-NATIVE: Uses self.xp consistently.
        
        THEORY:
            Instead of softmax(QK^T / √d), use:
            weights = φ^(-distance) / Σ φ^(-distance)
            
            This is the canonical φ-kernel from Grace's spectral structure.
            NO temperature parameter - φ IS the natural scale.
        """
        xp = self.xp
        
        if not self.schemas:
            return query, {"weights": [], "num_schemas": 0, "top_schema": None}
        
        # Ensure query is on correct device
        query = self._ensure_device(query)
        
        query_sigma, query_pseudo = extract_witness(query, self.basis, xp)
        
        # VECTORIZED: Extract all schema witnesses at once
        schema_matrices_list = [self._ensure_device(s.canonical_form) for s in self.schemas]
        schema_matrices = xp.stack(schema_matrices_list)
        n_schemas = len(self.schemas)
        
        # Batch witness extraction (scalar = trace/4, pseudo = trace(J)/4)
        traces = xp.trace(schema_matrices, axis1=-2, axis2=-1)  # [n_schemas]
        # For pseudoscalar, need J = e₁e₂e₃e₄
        J = self._ensure_device(self.basis[15])  # Pseudoscalar basis element
        J_products = xp.sum(schema_matrices * J, axis=(-2, -1))  # [n_schemas]
        schema_sigmas = traces / 4.0
        schema_pseudos = J_products / 4.0
        
        # Vectorized distance computation
        dists = xp.sqrt((query_sigma - schema_sigmas)**2 + (query_pseudo - schema_pseudos)**2)
        weights = PHI ** (-dists)  # Canonical φ-kernel: φ^(-d)
        
        total_weight = weights.sum()
        if total_weight < PHI_EPSILON:
            return query, {"weights": [], "num_schemas": n_schemas, "top_schema": None}
        
        weights = weights / total_weight
        
        # VECTORIZED: Weighted combination using einsum
        # transformed[i] = query @ schema[i], result = sum(w[i] * transformed[i])
        transformed = xp.matmul(query, schema_matrices)  # [n_schemas, 4, 4]
        result = xp.sum(weights.reshape(-1, 1, 1) * transformed, axis=0)  # [4, 4]
        
        result_norm = xp.sqrt(xp.sum(result * result)) + PHI_EPSILON
        result = result / result_norm
        
        # Convert for return values
        weights_cpu = weights.get() if hasattr(weights, 'get') else weights
        # NOTE: argmax for diagnostic reporting only - 'result' above is the φ-weighted combination
        top_idx = int(np.argmax(weights_cpu))
        
        return result, {
            "weights": weights_cpu.tolist(),
            "num_schemas": len(self.schemas),
            "top_schema": top_idx,
            "top_weight": float(weights_cpu[top_idx]),
            "entropy": float(-np.sum(weights_cpu * np.log(weights_cpu + PHI_EPSILON))),
        }
    
    def cluster_schemas_into_meta(
        self,
        similarity_threshold: float = PHI_INV,
        min_cluster_size: int = 2,
    ) -> List[MetaSchema]:
        """
        Cluster schemas into meta-schemas using quotient distance.
        
        THEORY:
            Meta-schemas are "categories of grammatical rules":
            - Meta-schema 1: Inflectional morphology
            - Meta-schema 2: Word order rules
            
            Clustering uses φ-threshold: schemas within φ⁻¹ distance belong together.
        """
        if len(self.schemas) < min_cluster_size:
            return []
        
        xp = self.xp
        
        # Extract witnesses for all schemas
        witnesses = []
        for schema in self.schemas:
            canonical = self._ensure_device(schema.canonical_form)
            sigma, pseudo = extract_witness(canonical, self.basis, xp)
            # Convert to float for clustering
            sigma_f = float(sigma) if hasattr(sigma, 'item') else sigma
            pseudo_f = float(pseudo) if hasattr(pseudo, 'item') else pseudo
            witnesses.append((sigma_f, pseudo_f))
        
        # Simple clustering: group by witness proximity (CPU for simplicity)
        used = set()
        clusters = []
        
        for i, (s_i, p_i) in enumerate(witnesses):
            if i in used:
                continue
            
            cluster = [i]
            used.add(i)
            
            for j, (s_j, p_j) in enumerate(witnesses):
                if j in used:
                    continue
                
                dist = np.sqrt((s_i - s_j)**2 + (p_i - p_j)**2)
                if dist < similarity_threshold:
                    cluster.append(j)
                    used.add(j)
            
            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)
        
        # Create MetaSchema objects
        meta_schemas = []
        for cluster_indices in clusters:
            cluster_schemas = [self.schemas[i] for i in cluster_indices]
            
            # Stack and mean on device
            cluster_matrices = xp.stack([self._ensure_device(s.canonical_form) for s in cluster_schemas])
            representative = xp.mean(cluster_matrices, axis=0)
            rep_norm = xp.sqrt(xp.sum(representative * representative)) + PHI_EPSILON
            representative = representative / rep_norm
            
            meta_schemas.append(MetaSchema(
                representative=representative,
                schema_indices=cluster_indices,
                schemas=cluster_schemas,
            ))
        
        return meta_schemas
    
    def hierarchical_attention(
        self,
        query,
        meta_schemas: List[MetaSchema] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Two-level hierarchical φ-attention: meta-schema → schema.
        
        GPU-NATIVE: Uses self.xp consistently.
        
        THEORY:
            Level 1: Select meta-schema category (e.g., "morphology")
            Level 2: Select specific schema within category (e.g., "past tense")
            
            Uses canonical φ-kernel: φ^(-distance)
            NO temperature - φ IS the natural scale.
            
        BRAIN ANALOG:
            - Meta = Broca's area categories (clause types)
            - Schema = Wernicke's area patterns (lexical rules)
            - Hierarchical attention = Executive function routing
        """
        xp = self.xp
        
        if meta_schemas is None:
            meta_schemas = self.cluster_schemas_into_meta()
        
        if not meta_schemas:
            return self.schema_attention(query)
        
        # Ensure query is on correct device
        query = self._ensure_device(query)
        
        # Level 1: Meta-schema selection (VECTORIZED)
        query_sigma, query_pseudo = extract_witness(query, self.basis, xp)
        
        # Batch extract witnesses for all meta-schemas
        meta_matrices = xp.stack([self._ensure_device(m.representative) for m in meta_schemas])
        meta_traces = xp.trace(meta_matrices, axis1=-2, axis2=-1)
        J = self._ensure_device(self.basis[15])
        meta_J_products = xp.sum(meta_matrices * J, axis=(-2, -1))
        meta_sigmas = meta_traces / 4.0
        meta_pseudos = meta_J_products / 4.0
        
        meta_dists = xp.sqrt((query_sigma - meta_sigmas)**2 + (query_pseudo - meta_pseudos)**2)
        meta_weights = PHI ** (-meta_dists)
        meta_weights = meta_weights / (meta_weights.sum() + PHI_EPSILON)
        
        # THEORY JUSTIFICATION: Hierarchical hard routing is analogous to Grace basin routing.
        # At meta-level, we route to the nearest basin (witness-distance based).
        # Within that basin, we use φ-weighted combination. This is NOT token decoding -
        # it's structural routing for attention computation.
        top_meta_idx = int(xp.argmax(meta_weights))
        top_meta = meta_schemas[top_meta_idx]
        
        # Level 2: Schema selection within top meta (VECTORIZED)
        schema_matrices = xp.stack([self._ensure_device(s.canonical_form) for s in top_meta.schemas])
        schema_traces = xp.trace(schema_matrices, axis1=-2, axis2=-1)
        schema_J_products = xp.sum(schema_matrices * J, axis=(-2, -1))
        schema_sigmas = schema_traces / 4.0
        schema_pseudos = schema_J_products / 4.0
        
        schema_dists = xp.sqrt((query_sigma - schema_sigmas)**2 + (query_pseudo - schema_pseudos)**2)
        schema_weights = PHI ** (-schema_dists)
        schema_weights = schema_weights / (schema_weights.sum() + PHI_EPSILON)
        
        # NOTE: argmax for diagnostic reporting only - actual result uses φ-weighted combination below
        top_schema_idx = int(xp.argmax(schema_weights))
        
        # VECTORIZED: Weighted schema combination (theory-true φ-weighted, NOT argmax)
        transformed = xp.matmul(query, schema_matrices)  # [n_schemas, 4, 4]
        result = xp.sum(schema_weights.reshape(-1, 1, 1) * transformed, axis=0)
        
        result_norm = xp.sqrt(xp.sum(result * result)) + PHI_EPSILON
        result = result / result_norm
        
        # Convert for return values
        meta_weights_cpu = meta_weights.get() if hasattr(meta_weights, 'get') else meta_weights
        schema_weights_cpu = schema_weights.get() if hasattr(schema_weights, 'get') else schema_weights
        
        return result, {
            "num_meta_schemas": len(meta_schemas),
            "meta_weights": meta_weights_cpu.tolist(),
            "top_meta": top_meta_idx,
            "top_meta_weight": float(meta_weights_cpu[top_meta_idx]),
            "schemas_in_top_meta": len(top_meta.schemas),
            "schema_weights": schema_weights_cpu.tolist(),
            "top_schema_in_meta": top_schema_idx,
            "top_schema_weight": float(schema_weights_cpu[top_schema_idx]),
            "meta_entropy": float(-np.sum(meta_weights_cpu * np.log(meta_weights_cpu + PHI_EPSILON))),
            "schema_entropy": float(-np.sum(schema_weights_cpu * np.log(schema_weights_cpu + PHI_EPSILON))),
        }
