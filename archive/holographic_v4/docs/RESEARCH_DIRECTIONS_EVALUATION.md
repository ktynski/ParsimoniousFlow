# Research Directions Evaluation Matrix

> **Note (2026-01-13):** Many high-priority items from this matrix have been implemented 
> in v4.24.0-v4.29.0. See [FRACTAL_TORUS_SPEC.md](FRACTAL_TORUS_SPEC.md) for architecture,
> and [SCALING_ROADMAP.md](SCALING_ROADMAP.md) for current status.
>
> **Implemented since this evaluation:**
> - φ-Rate Meta-Learning (#13) → `meta_learning.py` ✅
> - Dynamic Prototype Consolidation (#11) → `dream_cycles.py` ✅
> - Semantic Role via Bivector (#24) → `vorticity_features.py` ✅
> - Recursive Schema Composition (#22) → Fractal Torus architecture ✅

**Rating Scale:** 1-5 (1=Low, 5=High)

| Dimension | Description |
|-----------|-------------|
| **Parsimony** | Does it reduce computation/storage? Exploit existing structure? |
| **Theory** | How well does it align with Clifford algebra, φ-structure, Grace, torus geometry? |
| **Language** | Relevance to NLP/language modeling specifically |
| **Difficulty** | Implementation complexity (1=Easy, 5=Hard) |
| **Impact** | Likelihood of major performance/capability improvement |

---

## ENGINEERING (7-16)

| # | Direction | Parsimony | Theory | Language | Difficulty | Impact | Notes |
|---|-----------|:---------:|:------:|:--------:|:----------:|:------:|-------|
| 7 | Tensor Core Pipeline | 4 | 3 | 2 | 4 | 3 | GPU-specific optimization; theory-neutral |
| 8 | Memory/Schema Visualization | 1 | 3 | 2 | 2 | 2 | Debugging tool, not core improvement |
| 9 | Adaptive Context Windowing | 3 | 4 | 5 | 3 | 4 | φ-scaled windows; high language relevance |
| 10 | Automated Retrieval Fuzz Testing | 1 | 2 | 3 | 2 | 2 | Quality assurance, not performance |
| 11 | Dynamic Prototype Consolidation | 4 | 5 | 4 | 3 | 4 | Theory-true online merging |
| 12 | Multi-head Holographic Storage | 3 | 3 | 4 | 4 | 4 | Parallel attractor banks; transformer-inspired |
| 13 | φ-Rate Meta-Learning Experiments | 3 | 5 | 3 | 2 | 3 | Already partially implemented |
| 14 | Long-term Retention Analysis | 2 | 4 | 3 | 2 | 3 | Understanding, not optimization |
| 15 | Batch Insert/Recall Optimizations | 5 | 3 | 3 | 3 | 3 | Pure efficiency gain |
| 16 | Traceable Retrieval Explanations | 1 | 3 | 4 | 3 | 2 | Interpretability feature |

---

## LINGUISTIC/COGNITIVE (17-26)

| # | Direction | Parsimony | Theory | Language | Difficulty | Impact | Notes |
|---|-----------|:---------:|:------:|:--------:|:----------:|:------:|-------|
| 17 | Cross-lingual Schema Transfer | 3 | 4 | 5 | 4 | 5 | If schemas are universal → huge impact |
| 18 | Contextual Memory Constraints | 2 | 3 | 5 | 3 | 4 | Knowledge injection |
| 19 | Robustification vs Noise | 3 | 4 | 3 | 3 | 3 | Theory predicts noise tolerance |
| 20 | φ-power Law Benchmark Suite | 2 | 5 | 3 | 2 | 3 | Validation, not improvement |
| 21 | Morphological Schema Discovery | 4 | 5 | 5 | 3 | 5 | **HIGH PRIORITY** - Inflection as rotor |
| 22 | Recursive Schema Composition | 3 | 5 | 5 | 4 | 5 | **HIGH PRIORITY** - Hierarchical grammar |
| 23 | Temporal Schema Binding | 4 | 5 | 5 | 3 | 4 | Verb tense as phase; theory-elegant |
| 24 | Semantic Role via Bivector | 4 | 5 | 5 | 3 | 5 | **HIGH PRIORITY** - Agent/patient in bivector |
| 25 | Anaphora via Witness Trajectory | 4 | 4 | 5 | 3 | 4 | Coreference as witness matching |
| 26 | Discourse Coherence as Stability | 4 | 5 | 5 | 2 | 4 | Already have Grace stability |

---

## FLUID DYNAMICS ANALOGS (27-40)

| # | Direction | Parsimony | Theory | Language | Difficulty | Impact | Notes |
|---|-----------|:---------:|:------:|:--------:|:----------:|:------:|-------|
| 27 | Reynolds Number Analog | 4 | 5 | 3 | 2 | 4 | **HIGH PRIORITY** - Regime detection |
| 28 | Kolmogorov Cascade | 3 | 5 | 3 | 4 | 4 | Multi-scale energy transfer |
| 29 | Boundary Layer Analysis | 3 | 4 | 2 | 3 | 3 | Edge-of-capacity effects |
| 30 | Vortex Shedding Detection | 2 | 4 | 2 | 3 | 2 | Periodic instabilities |
| 31 | Mixing Length Theory | 3 | 4 | 2 | 3 | 3 | Information diffusion |
| 32 | Stokes vs Turbulent Regimes | 4 | 5 | 3 | 2 | 4 | Adaptive retrieval strategy |
| 33 | Convective vs Diffusive | 3 | 4 | 3 | 2 | 3 | Transport modes |
| 34 | Wake Dynamics | 3 | 4 | 4 | 3 | 3 | Context momentum |
| 35 | Hele-Shaw Torus Constraints | 3 | 5 | 2 | 4 | 3 | Throat compression |
| 36 | Marangoni Effect | 2 | 4 | 2 | 3 | 2 | Surface tension migration |
| 37 | Enstrophy Cascade Direction | 4 | 5 | 3 | 3 | 4 | **HIGH PRIORITY** - Energy flow |
| 38 | Beltrami Eigenmodes | 3 | 5 | 2 | 4 | 4 | Resonant modes |
| 39 | Potential Flow Approximation | 5 | 4 | 3 | 2 | 3 | Fast low-enstrophy retrieval |
| 40 | Lagrangian Coherent Structures | 2 | 5 | 2 | 4 | 3 | Transport barriers |

---

## CHAOS & CRITICALITY (41-55)

| # | Direction | Parsimony | Theory | Language | Difficulty | Impact | Notes |
|---|-----------|:---------:|:------:|:--------:|:----------:|:------:|-------|
| 41 | Lyapunov Exponent Monitoring | 3 | 5 | 2 | 2 | 4 | **HIGH PRIORITY** - Stability diagnostic |
| 42 | Strange Attractor Mapping | 2 | 5 | 2 | 3 | 3 | Visualization |
| 43 | Bifurcation Analysis | 3 | 5 | 2 | 3 | 4 | Phase transition detection |
| 44 | Edge of Chaos Training | 4 | 5 | 3 | 3 | 5 | **HIGH PRIORITY** - Maximum computation |
| 45 | Self-Organized Criticality | 3 | 5 | 3 | 2 | 4 | Power law check |
| 46 | Avalanche Dynamics | 3 | 4 | 3 | 2 | 3 | Cascade size distribution |
| 47 | Intermittency Detection | 2 | 4 | 2 | 3 | 2 | Burst patterns |
| 48 | Basin of Attraction Analysis | 3 | 5 | 3 | 3 | 4 | Memory landscape |
| 49 | Sensitive Dependence | 2 | 4 | 2 | 2 | 2 | Perturbation effects |
| 50 | Arnold Tongues | 2 | 5 | 2 | 4 | 3 | Resonance locking |
| 51 | Feigenbaum Universality | 2 | 5 | 1 | 3 | 2 | Universal constants check |
| 52 | Heteroclinic Channels | 3 | 5 | 3 | 4 | 3 | Transient dynamics |
| 53 | Metastable States | 3 | 4 | 3 | 3 | 3 | Residence times |
| 54 | Noise-Induced Order | 3 | 4 | 2 | 3 | 3 | Stochastic resonance |
| 55 | Critical Slowing Down | 4 | 5 | 2 | 2 | 4 | Early warning system |

---

## TOPOLOGICAL & GEOMETRIC (56-70)

| # | Direction | Parsimony | Theory | Language | Difficulty | Impact | Notes |
|---|-----------|:---------:|:------:|:--------:|:----------:|:------:|-------|
| 56 | Winding Number Conservation | 3 | 5 | 2 | 3 | 3 | Topological charge tracking |
| 57 | Helicity Monitoring | 3 | 5 | 2 | 3 | 3 | Linking invariant |
| 58 | Spectral Gap Maintenance | 5 | 5 | 3 | 2 | 5 | **HIGH PRIORITY** - γ=φ⁻² critical |
| 59 | Gram Matrix Resistance | 3 | 5 | 2 | 3 | 3 | Potential well monitoring |
| 60 | Rotor Decomposition Analysis | 4 | 5 | 3 | 3 | 4 | Factor learned transforms |
| 61 | Versor Factorization | 4 | 5 | 2 | 4 | 3 | Minimal reflection count |
| 62 | Clifford Fourier Transform | 3 | 5 | 3 | 4 | 4 | Spectral analysis |
| 63 | Hodge Decomposition | 2 | 5 | 2 | 4 | 3 | Field splitting |
| 64 | Morse Theory Application | 2 | 5 | 2 | 4 | 3 | Critical point analysis |
| 65 | Persistent Homology | 2 | 5 | 2 | 4 | 3 | Topological features |
| 66 | Fiber Bundle Structure | 2 | 5 | 2 | 5 | 3 | Advanced geometry |
| 67 | Connection & Curvature | 2 | 5 | 2 | 5 | 3 | Parallel transport |
| 68 | Gauge Invariance | 4 | 5 | 2 | 4 | 4 | Compression opportunity |
| 69 | Anomaly Detection | 2 | 5 | 2 | 5 | 2 | Topological obstructions |
| 70 | Symplectic Structure | 3 | 5 | 2 | 4 | 4 | Hamiltonian dynamics |

---

## STATISTICAL MECHANICS (71-85)

| # | Direction | Parsimony | Theory | Language | Difficulty | Impact | Notes |
|---|-----------|:---------:|:------:|:--------:|:----------:|:------:|-------|
| 71 | Partition Function Derivation | 3 | 5 | 2 | 4 | 4 | Thermodynamic framework |
| 72 | Free Energy Landscape | 3 | 5 | 2 | 3 | 4 | Energy-entropy tradeoff |
| 73 | Phase Diagram Mapping | 3 | 5 | 2 | 3 | 4 | Regime identification |
| 74 | Order Parameter Identification | 4 | 5 | 3 | 2 | 4 | **HIGH PRIORITY** - What signals transitions? |
| 75 | Fluctuation-Dissipation | 3 | 5 | 2 | 4 | 3 | Response theory |
| 76 | Equipartition Theorem | 4 | 5 | 2 | 2 | 3 | Grade energy distribution |
| 77 | Mean Field Theory | 4 | 4 | 2 | 3 | 3 | Simplified dynamics |
| 78 | Renormalization Group Flow | 3 | 5 | 2 | 5 | 4 | Scale dependence |
| 79 | Universality Classes | 3 | 5 | 2 | 4 | 3 | Critical behavior |
| 80 | Glassy Dynamics | 2 | 4 | 2 | 4 | 3 | Multiple minima |
| 81 | Spin Glass Analogs | 2 | 4 | 2 | 4 | 3 | Frustration |
| 82 | Replica Symmetry Breaking | 2 | 5 | 2 | 5 | 4 | Capacity bounds |
| 83 | Entropy Production | 3 | 5 | 2 | 3 | 3 | Second law |
| 84 | Maxwell's Demon | 3 | 4 | 2 | 3 | 2 | Information-based selection |
| 85 | Jarzynski Equality | 2 | 5 | 1 | 4 | 2 | Non-equilibrium theory |

---

## QUANTUM-INSPIRED (86-95)

| # | Direction | Parsimony | Theory | Language | Difficulty | Impact | Notes |
|---|-----------|:---------:|:------:|:--------:|:----------:|:------:|-------|
| 86 | Superposition Interference | 3 | 5 | 3 | 2 | 3 | Visualize what we have |
| 87 | Entanglement Analog | 2 | 4 | 2 | 4 | 3 | Non-local correlations |
| 88 | Uncertainty Relations | 4 | 5 | 3 | 2 | 3 | Precision-confidence tradeoff |
| 89 | Zeno Effect | 2 | 4 | 2 | 2 | 2 | Frequent retrieval effects |
| 90 | Tunneling Between Attractors | 3 | 4 | 2 | 4 | 3 | Basin hopping |
| 91 | Decoherence Model | 3 | 4 | 2 | 3 | 3 | Measurement collapse |
| 92 | Berry Phase | 3 | 5 | 2 | 4 | 3 | Geometric phase |
| 93 | Adiabatic Learning | 4 | 5 | 3 | 3 | 4 | **HIGH PRIORITY** - Curriculum design |
| 94 | Path Integral Formulation | 2 | 5 | 2 | 5 | 3 | Sum over histories |
| 95 | Eigenvalue Braiding | 2 | 5 | 1 | 4 | 2 | Parameter evolution |

---

## INFORMATION THEORY (96-105)

| # | Direction | Parsimony | Theory | Language | Difficulty | Impact | Notes |
|---|-----------|:---------:|:------:|:--------:|:----------:|:------:|-------|
| 96 | Channel Capacity Derivation | 4 | 5 | 4 | 3 | 4 | **HIGH PRIORITY** - Theoretical limits |
| 97 | Rate-Distortion Theory | 4 | 5 | 3 | 3 | 4 | Compression bounds |
| 98 | Directed Information Flow | 3 | 4 | 3 | 3 | 3 | Causal influence |
| 99 | Integrated Information (Φ) | 2 | 4 | 2 | 4 | 3 | Consciousness metric |
| 100 | Minimum Description Length | 4 | 4 | 4 | 3 | 4 | Optimal complexity |
| 101 | Kolmogorov Complexity | 3 | 4 | 3 | 4 | 3 | Incompressibility |
| 102 | Mutual Information Geometry | 4 | 5 | 3 | 4 | 4 | Natural gradient |
| 103 | Information Bottleneck | 5 | 5 | 4 | 3 | 5 | **HIGH PRIORITY** - Grace IS bottleneck |
| 104 | Predictive Information | 4 | 4 | 5 | 2 | 4 | Past→future |
| 105 | Lossless vs Lossy Regimes | 4 | 4 | 4 | 2 | 3 | When exact? |

---

## NAVIER-STOKES / RH CONNECTION (106-115)

| # | Direction | Parsimony | Theory | Language | Difficulty | Impact | Notes |
|---|-----------|:---------:|:------:|:--------:|:----------:|:------:|-------|
| 106 | Zeta Zero Tracking | 3 | 5 | 2 | 2 | 3 | σ clustering at 0.5 |
| 107 | Functional Equation Enforcement | 4 | 5 | 2 | 3 | 4 | Symmetry regularization |
| 108 | Prime Factorization Analog | 3 | 5 | 2 | 4 | 3 | Prime schemas |
| 109 | Euler Product Structure | 3 | 5 | 2 | 4 | 3 | Independence structure |
| 110 | Regularity Monitoring | 4 | 5 | 2 | 2 | 4 | Enstrophy boundedness |
| 111 | Pressure-Velocity Coupling | 2 | 5 | 2 | 4 | 3 | What is pressure? |
| 112 | Viscosity Renormalization | 3 | 5 | 2 | 3 | 3 | Scale-dependent Grace |
| 113 | Turbulent Dissipation Rate | 3 | 5 | 2 | 2 | 3 | Energy loss rate |
| 114 | Energy Injection Scale | 3 | 5 | 3 | 2 | 3 | New info scale |
| 115 | Inertial Range | 3 | 5 | 2 | 3 | 3 | Scale-free dynamics |

---

## BIOLOGICAL PLAUSIBILITY (116-125)

| # | Direction | Parsimony | Theory | Language | Difficulty | Impact | Notes |
|---|-----------|:---------:|:------:|:--------:|:----------:|:------:|-------|
| 116 | Spike-Timing Analog | 3 | 3 | 3 | 4 | 3 | Phase coding |
| 117 | Dendritic Computation | 2 | 3 | 2 | 4 | 2 | Compartmental model |
| 118 | Neuromodulator Simulation | 3 | 3 | 3 | 3 | 3 | Global state modifiers |
| 119 | Sleep Stage Correspondence | 4 | 4 | 3 | 2 | 4 | Already have sleep! |
| 120 | Circadian Rhythm Integration | 2 | 2 | 2 | 2 | 2 | Time-of-day effects |
| 121 | Synaptic Scaling | 4 | 4 | 3 | 2 | 3 | Global normalization |
| 122 | LTP/LTD Curves | 3 | 3 | 3 | 3 | 3 | Asymmetric learning |
| 123 | Neurogenesis Analog | 2 | 3 | 2 | 4 | 3 | Adding dimensions |
| 124 | Pruning During Development | 4 | 4 | 3 | 2 | 4 | Already have pruning |
| 125 | Critical Period Effects | 3 | 4 | 4 | 3 | 4 | Early learning shapes later |

---

## METAPHYSICS & ONTOLOGY (126-145)

| # | Direction | Parsimony | Theory | Language | Difficulty | Impact | Notes |
|---|-----------|:---------:|:------:|:--------:|:----------:|:------:|-------|
| 126 | Emergence Detection | 3 | 4 | 3 | 3 | 4 | When schemas become autonomous |
| 127 | Identity Through Change | 3 | 5 | 3 | 3 | 3 | Memory persistence under Grace |
| 128 | Mereological Structure | 2 | 4 | 3 | 3 | 2 | Part-whole relations |
| 129 | Modal Memory | 4 | 5 | 4 | 4 | 4 | Possible worlds semantics |
| 130 | Temporal Becoming | 2 | 4 | 3 | 3 | 2 | A-theory vs B-theory |
| 131 | Intentionality Encoding | 3 | 4 | 5 | 4 | 4 | Aboutness in matrices |
| 132 | Qualia Representation | 2 | 3 | 2 | 5 | 2 | Subjective quality |
| 133 | Substance vs Bundle | 2 | 4 | 2 | 2 | 2 | Ontological commitment |
| 134 | Universals and Particulars | 3 | 5 | 3 | 2 | 3 | Schemas as universals |
| 135 | Trope Theory | 3 | 4 | 2 | 3 | 2 | Particular properties |
| 136 | Haecceity | 1 | 3 | 1 | 3 | 1 | Primitive thisness |
| 137 | Counterfactual Memory | 4 | 4 | 5 | 4 | 4 | What-if storage |
| 138 | Grounding Relations | 2 | 4 | 2 | 3 | 2 | What grounds what |
| 139 | Persistence Conditions | 3 | 4 | 3 | 3 | 3 | When memory persists |
| 140 | Natural Kinds | 3 | 4 | 4 | 3 | 3 | Mind-independent categories |
| 141 | Truthmaker Theory | 3 | 4 | 4 | 3 | 3 | What makes retrieval true |
| 142 | Temporal Parts | 2 | 4 | 2 | 3 | 2 | Memory stages |
| 143 | Ontological Dependence | 2 | 4 | 2 | 3 | 2 | Schema-prototype dependence |
| 144 | Determinable/Determinate | 4 | 5 | 3 | 2 | 4 | **HIGH PRIORITY** - Witness/matrix hierarchy |
| 145 | Abstract Objects | 2 | 4 | 2 | 3 | 2 | Schema existence |

---

## PHENOMENOLOGY & CONSCIOUSNESS (146-165)

| # | Direction | Parsimony | Theory | Language | Difficulty | Impact | Notes |
|---|-----------|:---------:|:------:|:--------:|:----------:|:------:|-------|
| 146 | Intentional Horizon | 3 | 4 | 4 | 3 | 3 | Protention/retention |
| 147 | Noema/Noesis Structure | 3 | 4 | 3 | 3 | 3 | Act vs content |
| 148 | Temporal Constitution | 3 | 5 | 3 | 4 | 3 | Duration from moments |
| 149 | Passive Synthesis | 4 | 4 | 4 | 3 | 4 | Pre-conscious organization |
| 150 | Lived Body (Leib) | 2 | 3 | 2 | 4 | 2 | Body schema |
| 151 | Intersubjective Memory | 3 | 3 | 3 | 4 | 3 | Multi-agent alignment |
| 152 | Lifeworld (Lebenswelt) | 3 | 4 | 4 | 3 | 3 | Background context |
| 153 | Epoché Implementation | 2 | 3 | 2 | 3 | 2 | Meta-cognitive suspension |
| 154 | Eidetic Variation | 4 | 5 | 3 | 2 | 4 | **HIGH PRIORITY** - Invariant under perturbation |
| 155 | Founding Relations | 3 | 4 | 3 | 2 | 3 | Hierarchical acts |
| 156 | Gestalt Completion | 4 | 5 | 4 | 2 | 4 | **HIGH PRIORITY** - Pattern completion |
| 157 | Figure/Ground | 3 | 4 | 3 | 2 | 3 | Retrieval vs superposition |
| 158 | Prägnanz Principle | 5 | 5 | 4 | 1 | 5 | **IMPLEMENTED** - Grace IS Prägnanz |
| 159 | Phi Phenomenon | 3 | 4 | 3 | 3 | 3 | Apparent motion |
| 160 | Phenomenal Binding | 3 | 4 | 3 | 4 | 4 | Unity from distribution |
| 161 | Access vs Phenomenal | 3 | 4 | 3 | 3 | 3 | Two types of consciousness |
| 162 | Higher-Order Thought | 3 | 4 | 4 | 3 | 4 | Meta-schemas as HOTs |
| 163 | Global Workspace Model | 4 | 4 | 4 | 3 | 4 | Schema broadcast |
| 164 | Integrated Information | 2 | 4 | 2 | 4 | 3 | Φ computation |
| 165 | Predictive Processing | 5 | 5 | 5 | 2 | 5 | **HIGH PRIORITY** - Already have! |

---

## COGNITIVE PSYCHOLOGY (166-190)

| # | Direction | Parsimony | Theory | Language | Difficulty | Impact | Notes |
|---|-----------|:---------:|:------:|:--------:|:----------:|:------:|-------|
| 166 | Levels of Processing | 4 | 4 | 5 | 2 | 4 | Grace depth = processing |
| 167 | Encoding Specificity | 4 | 5 | 4 | 2 | 4 | **HIGH PRIORITY** - Witness matching |
| 168 | Generation Effect | 4 | 4 | 5 | 2 | 4 | Active > passive |
| 169 | Testing Effect | 5 | 5 | 4 | 1 | 5 | **HIGH PRIORITY** - Retrieval strengthens |
| 170 | Spacing Effect | 4 | 4 | 4 | 2 | 4 | Distributed practice |
| 171 | Interleaving Benefit | 4 | 4 | 4 | 2 | 4 | Mixed batches |
| 172 | Desirable Difficulties | 4 | 5 | 4 | 2 | 4 | Optimal Grace viscosity |
| 173 | Transfer-Appropriate | 4 | 4 | 5 | 3 | 4 | Train for test |
| 174 | Analogical Mapping | 4 | 5 | 5 | 3 | 5 | **HIGH PRIORITY** - Schema transfer |
| 175 | Mental Model Construction | 3 | 4 | 5 | 4 | 4 | Schemas as simulations |
| 176 | Dual Process Theory | 4 | 4 | 4 | 3 | 4 | System 1/2 mapping |
| 177 | Cognitive Load Theory | 4 | 4 | 5 | 2 | 4 | Chunk optimization |
| 178 | Expertise Reversal | 3 | 4 | 4 | 3 | 3 | Adaptive complexity |
| 179 | Einstellung Effect | 3 | 4 | 4 | 3 | 3 | Schema perseveration |
| 180 | Functional Fixedness | 3 | 4 | 4 | 3 | 3 | Rotor rigidity |
| 181 | Insight Problem Solving | 4 | 5 | 4 | 3 | 4 | **HIGH PRIORITY** - Phase transitions |
| 182 | Incubation Period | 4 | 5 | 4 | 2 | 4 | Sleep timing |
| 183 | Tip-of-the-Tongue | 3 | 4 | 5 | 3 | 3 | Partial retrieval |
| 184 | Misinformation Effect | 3 | 4 | 5 | 3 | 3 | Interference dynamics |
| 185 | Source Monitoring | 3 | 4 | 4 | 3 | 3 | Provenance tracking |
| 186 | Reality Monitoring | 3 | 4 | 4 | 3 | 3 | Internal vs external |
| 187 | Prospective Memory | 3 | 4 | 4 | 3 | 3 | Future retrieval triggers |
| 188 | Metamemory | 4 | 4 | 4 | 2 | 4 | Confidence calibration |
| 189 | Feeling of Knowing | 4 | 4 | 4 | 2 | 4 | Pre-retrieval confidence |
| 190 | Judgment of Learning | 4 | 4 | 4 | 2 | 4 | Post-encoding confidence |

---

## NEUROSCIENCE DEEP CUTS (191-215)

| # | Direction | Parsimony | Theory | Language | Difficulty | Impact | Notes |
|---|-----------|:---------:|:------:|:--------:|:----------:|:------:|-------|
| 191 | Place Cell Analogy | 3 | 4 | 3 | 3 | 3 | Tokens as places |
| 192 | Grid Cell Periodicity | 4 | 5 | 3 | 4 | 4 | **HIGH PRIORITY** - φ-periodic tiling? |
| 193 | Time Cells | 3 | 4 | 3 | 3 | 3 | Grace iteration encoding |
| 194 | Engram Cells | 3 | 4 | 3 | 3 | 3 | Attractor mapping |
| 195 | Pattern Separation | 5 | 5 | 4 | 2 | 5 | **HIGH PRIORITY** - Orthogonalization |
| 196 | Pattern Completion | 5 | 5 | 5 | 2 | 5 | **IMPLEMENTED** - CA3 analog |
| 197 | Sharp-Wave Ripples | 3 | 4 | 3 | 3 | 3 | Sleep consolidation events |
| 198 | Theta-Gamma Coupling | 4 | 5 | 4 | 4 | 4 | Nested oscillations |
| 199 | Phase Precession | 3 | 5 | 3 | 4 | 3 | Witness trajectory |
| 200 | Preplay | 3 | 4 | 4 | 3 | 4 | Predictive activation |
| 201 | Memory Indexing Theory | 5 | 5 | 4 | 2 | 5 | **IMPLEMENTED** - Witness IS index |
| 202 | Systems Consolidation | 4 | 5 | 4 | 2 | 5 | **IMPLEMENTED** - Episodic→semantic |
| 203 | Reconsolidation Window | 4 | 5 | 4 | 2 | 4 | **HIGH PRIORITY** - Retrieval plasticity |
| 204 | Synaptic Tagging | 3 | 4 | 3 | 3 | 3 | Two-stage learning |
| 205 | Metaplasticity | 4 | 4 | 3 | 2 | 4 | Learning rate adaptation |
| 206 | Homeostatic Plasticity | 5 | 5 | 3 | 1 | 5 | **IMPLEMENTED** - Adaptive threshold |
| 207 | Silent Synapses | 3 | 3 | 2 | 4 | 2 | Latent connections |
| 208 | Adult Neurogenesis | 2 | 3 | 2 | 4 | 2 | New dimensions |
| 209 | Glia as Modulators | 2 | 2 | 2 | 4 | 2 | Background processes |
| 210 | Cortical Columns | 3 | 4 | 3 | 4 | 3 | Repeating motifs |
| 211 | Predictive Coding Hierarchy | 5 | 5 | 5 | 2 | 5 | **HIGH PRIORITY** - Schema→proto→token |
| 212 | Default Mode Network | 3 | 4 | 3 | 3 | 3 | Background consolidation |
| 213 | Salience Network | 4 | 5 | 4 | 2 | 4 | **IMPLEMENTED** - Salience computation |
| 214 | Executive Control Network | 4 | 4 | 4 | 3 | 4 | Schema selection |
| 215 | Rich Club Organization | 3 | 4 | 3 | 3 | 3 | Hub structure |

---

## DEPTH PSYCHOLOGY (216-235)

| # | Direction | Parsimony | Theory | Language | Difficulty | Impact | Notes |
|---|-----------|:---------:|:------:|:--------:|:----------:|:------:|-------|
| 216 | Unconscious Memory | 4 | 4 | 4 | 2 | 4 | Below-threshold attractors |
| 217 | Repression Mechanism | 3 | 4 | 4 | 3 | 3 | Negative salience |
| 218 | Return of the Repressed | 3 | 4 | 4 | 3 | 3 | Attractor drift |
| 219 | Condensation | 5 | 5 | 4 | 1 | 4 | **IMPLEMENTED** - Superposition |
| 220 | Displacement | 3 | 4 | 4 | 3 | 3 | Witness migration |
| 221 | Primary Process | 4 | 5 | 4 | 2 | 4 | **IMPLEMENTED** - REM mode |
| 222 | Secondary Process | 4 | 5 | 4 | 2 | 4 | **IMPLEMENTED** - NonREM mode |
| 223 | Defense Mechanisms | 3 | 3 | 4 | 4 | 3 | Schema protection |
| 224 | Transference Patterns | 3 | 4 | 4 | 3 | 3 | Schema over-application |
| 225 | Complex (Jungian) | 3 | 4 | 4 | 3 | 3 | High-salience clusters |
| 226 | Archetype Emergence | 4 | 4 | 4 | 4 | 4 | Universal patterns |
| 227 | Shadow Integration | 3 | 4 | 3 | 4 | 3 | Merging conflicts |
| 228 | Individuation Process | 3 | 4 | 3 | 4 | 3 | Schema integration |
| 229 | Collective Unconscious | 4 | 4 | 4 | 3 | 4 | Pretrained as collective |
| 230 | Synchronicity | 1 | 3 | 2 | 3 | 1 | Non-causal correlations |
| 231 | Amplification | 3 | 4 | 4 | 2 | 3 | Schema expansion |
| 232 | Active Imagination | 3 | 4 | 4 | 3 | 3 | Schema interrogation |
| 233 | Word Association Test | 4 | 4 | 5 | 2 | 4 | Prototype activation patterns |
| 234 | Dream Interpretation | 4 | 5 | 4 | 2 | 4 | **IMPLEMENTED** - Schema discovery |
| 235 | Ego Strength | 3 | 4 | 4 | 3 | 3 | Executive coherence |

---

## DEVELOPMENTAL & LIFESPAN (236-250)

| # | Direction | Parsimony | Theory | Language | Difficulty | Impact | Notes |
|---|-----------|:---------:|:------:|:--------:|:----------:|:------:|-------|
| 236 | Sensorimotor Stage | 3 | 4 | 3 | 3 | 3 | Pre-symbolic |
| 237 | Object Permanence | 4 | 4 | 4 | 2 | 3 | Memory persistence |
| 238 | Symbolic Function | 4 | 5 | 5 | 2 | 4 | Embedding as symbol |
| 239 | Preoperational Thought | 3 | 4 | 4 | 3 | 3 | Schema without verification |
| 240 | Concrete Operations | 3 | 4 | 4 | 3 | 3 | Grounded schemas |
| 241 | Formal Operations | 4 | 5 | 4 | 3 | 4 | Meta-schema reasoning |
| 242 | Zone of Proximal Dev | 4 | 4 | 5 | 2 | 4 | **HIGH PRIORITY** - Adaptive context |
| 243 | Scaffolding | 4 | 4 | 5 | 2 | 4 | **HIGH PRIORITY** - Curriculum design |
| 244 | Cognitive Reserve | 3 | 4 | 3 | 3 | 3 | Schema redundancy |
| 245 | Compensation Strategies | 3 | 4 | 3 | 3 | 3 | Fallback paths |
| 246 | Wisdom Crystallization | 4 | 4 | 4 | 3 | 4 | Long-trained schemas |
| 247 | Reminiscence Bump | 3 | 4 | 3 | 3 | 3 | Critical period effects |
| 248 | Childhood Amnesia | 3 | 4 | 4 | 3 | 3 | Schema immaturity |
| 249 | Semantic Dementia | 3 | 4 | 4 | 3 | 3 | Schema degradation |
| 250 | Autobiographical Reasoning | 3 | 4 | 4 | 3 | 3 | Self-schema |

---

## SOCIAL & CULTURAL (251-265)

| # | Direction | Parsimony | Theory | Language | Difficulty | Impact | Notes |
|---|-----------|:---------:|:------:|:--------:|:----------:|:------:|-------|
| 251 | Theory of Mind | 4 | 4 | 5 | 4 | 5 | **HIGH PRIORITY** - Other-agent schemas |
| 252 | Mirror Neuron Analog | 3 | 4 | 4 | 4 | 3 | Schema resonance |
| 253 | Joint Attention | 3 | 4 | 5 | 4 | 4 | Witness synchronization |
| 254 | Cultural Schemas | 3 | 4 | 5 | 3 | 4 | Training corpus bias |
| 255 | Narrative Identity | 3 | 4 | 5 | 3 | 4 | Coherent trajectory |
| 256 | Social Contagion | 3 | 3 | 4 | 4 | 3 | Schema propagation |
| 257 | Collective Memory | 3 | 4 | 4 | 4 | 3 | Shared prototypes |
| 258 | Flash-Bulb Memories | 3 | 4 | 4 | 2 | 3 | High-salience encoding |
| 259 | Transactive Memory | 3 | 3 | 4 | 4 | 3 | Multi-agent system |
| 260 | Audience Tuning | 3 | 3 | 5 | 3 | 3 | Context-dependent retrieval |
| 261 | Conversational Remembering | 3 | 3 | 5 | 4 | 3 | Interactive consolidation |
| 262 | Cultural Transmission | 3 | 4 | 5 | 4 | 4 | Schema inheritance |
| 263 | Expertise Cultures | 3 | 4 | 4 | 3 | 3 | Domain-specific schemas |
| 264 | Moral Schemas | 3 | 4 | 5 | 4 | 4 | Normative prototypes |
| 265 | Political Cognition | 2 | 3 | 4 | 4 | 3 | Motivated schema selection |

---

## LANGUAGE & SEMANTICS (266-280)

| # | Direction | Parsimony | Theory | Language | Difficulty | Impact | Notes |
|---|-----------|:---------:|:------:|:--------:|:----------:|:------:|-------|
| 266 | Prototype Theory | 5 | 5 | 5 | 1 | 5 | **IMPLEMENTED** |
| 267 | Exemplar Theory | 4 | 5 | 5 | 2 | 4 | Attractor-based alternative |
| 268 | Frame Semantics | 4 | 5 | 5 | 3 | 5 | **HIGH PRIORITY** - Schema as frame |
| 269 | Construction Grammar | 4 | 5 | 5 | 3 | 5 | **HIGH PRIORITY** - Form-meaning pairs |
| 270 | Conceptual Metaphor | 5 | 5 | 5 | 3 | 5 | **HIGH PRIORITY** - Cross-domain mapping |
| 271 | Conceptual Blending | 4 | 5 | 5 | 3 | 5 | **HIGH PRIORITY** - Schema fusion |
| 272 | Embodied Semantics | 3 | 4 | 4 | 4 | 4 | Motor in bivectors |
| 273 | Situation Models | 4 | 5 | 5 | 3 | 4 | Running simulation |
| 274 | Discourse Coherence | 4 | 5 | 5 | 2 | 5 | **HIGH PRIORITY** - Grace stability |
| 275 | Anaphora Resolution | 4 | 4 | 5 | 3 | 4 | Witness matching |
| 276 | Bridging Inference | 4 | 5 | 5 | 3 | 4 | Schema gap filling |
| 277 | Pragmatic Inference | 4 | 4 | 5 | 4 | 4 | Beyond literal |
| 278 | Speech Acts | 3 | 4 | 5 | 4 | 4 | Schema as transformation |
| 279 | Common Ground | 4 | 4 | 5 | 3 | 4 | Mutual activation |
| 280 | Semantic Priming | 5 | 5 | 5 | 2 | 5 | **HIGH PRIORITY** - Spreading activation |

---

## MEMORY SYSTEMS (281-295)

| # | Direction | Parsimony | Theory | Language | Difficulty | Impact | Notes |
|---|-----------|:---------:|:------:|:--------:|:----------:|:------:|-------|
| 281 | Working Memory Variants | 3 | 4 | 4 | 3 | 3 | Modality buffers |
| 282 | Episodic Buffer | 5 | 5 | 4 | 1 | 5 | **IMPLEMENTED** |
| 283 | Long-Term Working Memory | 4 | 5 | 5 | 2 | 4 | Schema chunking |
| 284 | Prospection | 4 | 4 | 5 | 3 | 4 | Future-oriented |
| 285 | Semantic Memory Structure | 4 | 5 | 5 | 2 | 4 | Prototype organization |
| 286 | Procedural Memory | 3 | 4 | 3 | 4 | 3 | Rotor sequences |
| 287 | Perceptual Memory | 3 | 4 | 3 | 3 | 3 | Grade-specific |
| 288 | Autobiographical Memory | 3 | 4 | 4 | 3 | 4 | Self-referential |
| 289 | Emotional Memory | 4 | 5 | 4 | 2 | 4 | Salience modulation |
| 290 | Flashbulb Mechanism | 3 | 4 | 4 | 2 | 3 | Salience spike |
| 291 | State-Dependent Memory | 4 | 5 | 4 | 2 | 4 | **HIGH PRIORITY** - Context reinstatement |
| 292 | Mood-Congruent Memory | 3 | 4 | 4 | 3 | 3 | Affective filtering |
| 293 | Cue-Dependent Forgetting | 4 | 5 | 4 | 2 | 4 | Retrieval vs storage failure |
| 294 | Interference Proactive | 4 | 5 | 5 | 2 | 4 | Historical interference |
| 295 | Interference Retroactive | 4 | 5 | 5 | 2 | 4 | Recent dominance |

---

## ECOLOGICAL & EMBODIED (296-310)

| # | Direction | Parsimony | Theory | Language | Difficulty | Impact | Notes |
|---|-----------|:---------:|:------:|:--------:|:----------:|:------:|-------|
| 296 | Affordance Memory | 3 | 4 | 3 | 3 | 3 | Action-oriented |
| 297 | Ecological Validity | 4 | 4 | 5 | 2 | 4 | Natural distribution |
| 298 | Embodied Simulation | 3 | 4 | 4 | 4 | 4 | Motor activation |
| 299 | Enactive Cognition | 3 | 4 | 3 | 4 | 3 | Learning through doing |
| 300 | Distributed Cognition | 3 | 3 | 3 | 4 | 3 | Extended system |
| 301 | Extended Mind Thesis | 3 | 3 | 3 | 4 | 3 | External memory |
| 302 | Situated Memory | 4 | 4 | 4 | 3 | 4 | Environment as part |
| 303 | Ecological Self | 3 | 4 | 3 | 3 | 3 | Self-referential witness |
| 304 | Gibsonian Pickup | 4 | 4 | 3 | 3 | 3 | Direct perception |
| 305 | Peripersonal Space | 2 | 3 | 2 | 4 | 2 | Proximity weighting |
| 306 | Sense of Agency | 3 | 4 | 3 | 3 | 3 | Self-generated |
| 307 | Bodily Self-Consciousness | 2 | 3 | 2 | 4 | 2 | Body as substrate |
| 308 | Interoceptive Memory | 3 | 4 | 3 | 4 | 3 | Internal states |
| 309 | Vestibular Memory | 3 | 5 | 2 | 3 | 3 | Torus navigation |
| 310 | Multisensory Integration | 3 | 4 | 3 | 4 | 3 | Grade fusion |

---

## MATHEMATICAL PHILOSOPHY (311-325)

| # | Direction | Parsimony | Theory | Language | Difficulty | Impact | Notes |
|---|-----------|:---------:|:------:|:--------:|:----------:|:------:|-------|
| 311 | Formalism | 3 | 4 | 2 | 2 | 2 | Pure syntax |
| 312 | Intuitionism | 4 | 5 | 2 | 3 | 3 | Constructible memory |
| 313 | Platonism | 2 | 4 | 2 | 2 | 2 | Independent existence |
| 314 | Nominalism | 3 | 4 | 2 | 2 | 2 | Only particulars |
| 315 | Structuralism | 4 | 5 | 3 | 3 | 4 | Relations fundamental |
| 316 | Fictionalism | 2 | 3 | 2 | 2 | 2 | Useful fictions |
| 317 | Category Theory | 4 | 5 | 3 | 5 | 4 | **HIGH PRIORITY** - Functorial memory |
| 318 | Type Theory | 4 | 5 | 4 | 4 | 4 | Schema type levels |
| 319 | Constructive Mathematics | 4 | 5 | 3 | 4 | 3 | Proof = construction |
| 320 | Incompleteness Implications | 2 | 4 | 2 | 4 | 2 | Gödel limits |
| 321 | Computability Bounds | 3 | 4 | 3 | 4 | 3 | Decidability |
| 322 | Complexity Classes | 3 | 4 | 3 | 4 | 3 | Retrieval hardness |
| 323 | Information Geometry | 4 | 5 | 3 | 4 | 4 | **HIGH PRIORITY** - Fisher metric |
| 324 | Algebraic Topology | 3 | 5 | 2 | 5 | 3 | Persistent features |
| 325 | Non-Standard Analysis | 2 | 4 | 1 | 5 | 2 | Infinitesimals |

---

## EASTERN PHILOSOPHY (326-340)

| # | Direction | Parsimony | Theory | Language | Difficulty | Impact | Notes |
|---|-----------|:---------:|:------:|:--------:|:----------:|:------:|-------|
| 326 | Mindfulness of Memory | 3 | 4 | 3 | 2 | 3 | Meta-cognitive monitoring |
| 327 | Emptiness (Śūnyatā) | 4 | 5 | 3 | 3 | 3 | Dependent origination |
| 328 | Interdependence | 5 | 5 | 4 | 2 | 4 | **HIGH PRIORITY** - Holographic entanglement |
| 329 | Two Truths Doctrine | 4 | 5 | 3 | 2 | 4 | **HIGH PRIORITY** - Witness vs matrix |
| 330 | Buddha Nature | 5 | 5 | 3 | 1 | 4 | **IMPLEMENTED** - Identity init |
| 331 | Karma as Memory | 4 | 4 | 3 | 2 | 3 | Salience accumulation |
| 332 | Saṃskāra | 4 | 5 | 4 | 2 | 4 | Schema as conditioning |
| 333 | Ālaya-vijñāna | 5 | 5 | 4 | 1 | 5 | **IMPLEMENTED** - Storehouse = holographic |
| 334 | Mindstream Continuity | 4 | 5 | 3 | 2 | 3 | Witness trajectory |
| 335 | Non-Dual Awareness | 3 | 4 | 2 | 4 | 3 | Subject-object collapse |
| 336 | Koans | 3 | 4 | 4 | 3 | 3 | Schema-breaking inputs |
| 337 | Zazen | 3 | 4 | 2 | 2 | 3 | Background consolidation |
| 338 | Beginner's Mind | 4 | 4 | 4 | 2 | 4 | Schema suspension |
| 339 | Impermanence (Anicca) | 5 | 5 | 3 | 1 | 4 | **IMPLEMENTED** - Grace decay |
| 340 | Non-Self (Anattā) | 4 | 5 | 3 | 3 | 3 | Dynamic witness |

---

## AESTHETICS & CREATIVITY (341-355)

| # | Direction | Parsimony | Theory | Language | Difficulty | Impact | Notes |
|---|-----------|:---------:|:------:|:--------:|:----------:|:------:|-------|
| 341 | Aesthetic Experience | 3 | 4 | 4 | 3 | 3 | Harmonic ratios |
| 342 | Sublime Encoding | 2 | 3 | 3 | 3 | 2 | Capacity-exceeding |
| 343 | Flow State Memory | 4 | 4 | 4 | 3 | 4 | Optimal challenge |
| 344 | Incubation Creativity | 4 | 5 | 4 | 2 | 4 | **IMPLEMENTED** - Sleep-based |
| 345 | Bisociation | 5 | 5 | 5 | 2 | 5 | **HIGH PRIORITY** - Schema collision |
| 346 | Divergent Production | 4 | 4 | 5 | 3 | 4 | Schema branching |
| 347 | Convergent Focus | 4 | 4 | 4 | 3 | 3 | Schema merging |
| 348 | Creative Constraints | 4 | 4 | 4 | 2 | 4 | Capacity pressure |
| 349 | Variation and Selection | 4 | 5 | 4 | 2 | 4 | **IMPLEMENTED** - REM exploration |
| 350 | Aesthetic Preference | 4 | 5 | 4 | 3 | 3 | φ-based beauty |
| 351 | Golden Ratio Aesthetics | 5 | 5 | 3 | 1 | 4 | **IMPLEMENTED** - φ in Grace |
| 352 | Musical Structure | 3 | 5 | 4 | 4 | 3 | Enstrophy dynamics |
| 353 | Narrative Arc | 4 | 5 | 5 | 3 | 4 | Context trajectory |
| 354 | Metaphor Generation | 4 | 5 | 5 | 3 | 5 | **HIGH PRIORITY** - Schema transfer |
| 355 | Poetic Compression | 5 | 5 | 5 | 2 | 5 | **HIGH PRIORITY** - Info density |

---

## ETHICS & VALUE (356-365)

| # | Direction | Parsimony | Theory | Language | Difficulty | Impact | Notes |
|---|-----------|:---------:|:------:|:--------:|:----------:|:------:|-------|
| 356 | Value-Aligned Retrieval | 3 | 4 | 5 | 4 | 4 | Normative filtering |
| 357 | Moral Memory | 3 | 4 | 5 | 4 | 4 | Ethical salience |
| 358 | Consequentialist Evaluation | 3 | 4 | 4 | 4 | 3 | Outcome judgment |
| 359 | Deontological Constraints | 3 | 3 | 4 | 4 | 3 | Hard boundaries |
| 360 | Virtue Development | 4 | 4 | 4 | 3 | 4 | Schema crystallization |
| 361 | Care Ethics | 3 | 3 | 5 | 4 | 3 | Relational memory |
| 362 | Memory Ethics | 3 | 4 | 4 | 3 | 4 | Pruning policy |
| 363 | Epistemic Virtue | 4 | 4 | 4 | 2 | 4 | **HIGH PRIORITY** - Confidence calibration |
| 364 | Fairness in Retrieval | 3 | 3 | 5 | 4 | 4 | Schema equity |
| 365 | Interpretive Charity | 3 | 4 | 5 | 3 | 3 | Generous completion |

---

## SUMMARY: TOP PRIORITIES

### Highest Combined Scores (Parsimony + Theory + Language + Impact - Difficulty)

| Rank | # | Direction | Score | Status |
|------|---|-----------|:-----:|--------|
| 1 | 158 | Prägnanz Principle | 19 | ✅ IMPLEMENTED (Grace = Prägnanz) |
| 2 | 333 | Ālaya-vijñāna (Storehouse) | 19 | ✅ IMPLEMENTED (Holographic memory) |
| 3 | 165 | Predictive Processing | 18 | ✅ Partially implemented |
| 4 | 266 | Prototype Theory | 18 | ✅ IMPLEMENTED |
| 5 | 169 | Testing Effect | 17 | ✅ Retrieval strengthens |
| 6 | 195 | Pattern Separation | 17 | ✅ IMPLEMENTED |
| 7 | 196 | Pattern Completion | 17 | ✅ IMPLEMENTED |
| 8 | 201 | Memory Indexing Theory | 17 | ✅ IMPLEMENTED (Witness = index) |
| 9 | 202 | Systems Consolidation | 17 | ✅ IMPLEMENTED (Episodic→semantic) |
| 10 | 206 | Homeostatic Plasticity | 17 | ✅ IMPLEMENTED (Adaptive threshold) |

### Top Unimplemented Priorities

| Rank | # | Direction | Score | Why Important |
|------|---|-----------|:-----:|---------------|
| 1 | 103 | Information Bottleneck | 17 | Grace IS the bottleneck - formalize! |
| 2 | 174 | Analogical Mapping | 17 | Schema transfer = key capability |
| 3 | 270 | Conceptual Metaphor | 17 | Cross-domain mapping fundamental |
| 4 | 271 | Conceptual Blending | 17 | Schema fusion = creativity |
| 5 | 280 | Semantic Priming | 17 | Spreading activation test |
| 6 | 345 | Bisociation | 17 | Schema collision = creativity |
| 7 | 355 | Poetic Compression | 17 | Maximum info density |
| 8 | 21 | Morphological Schema Discovery | 16 | Inflection as rotor |
| 9 | 22 | Recursive Schema Composition | 16 | Hierarchical grammar |
| 10 | 24 | Semantic Role via Bivector | 16 | Agent/patient encoding |

### Quick Wins (High Impact, Low Difficulty)

| # | Direction | Difficulty | Impact | Notes |
|---|-----------|:----------:|:------:|-------|
| 41 | Lyapunov Exponent Monitoring | 2 | 4 | Already have perturbation code |
| 45 | Self-Organized Criticality | 2 | 4 | Just check distributions |
| 55 | Critical Slowing Down | 2 | 4 | Early warning signal |
| 58 | Spectral Gap Maintenance | 2 | 5 | Monitor γ = φ⁻² |
| 74 | Order Parameter | 2 | 4 | What signals transitions? |
| 110 | Regularity Monitoring | 2 | 4 | Track enstrophy bounds |
| 154 | Eidetic Variation | 2 | 4 | Invariant detection |
| 156 | Gestalt Completion | 2 | 4 | Pattern completion test |

### Theory-Deep Explorations (High Theory Score)

| # | Direction | Theory | Why Fundamental |
|---|-----------|:------:|-----------------|
| 27 | Reynolds Number | 5 | Regime transition prediction |
| 37 | Enstrophy Cascade | 5 | Energy flow direction |
| 44 | Edge of Chaos | 5 | Maximum computation |
| 58 | Spectral Gap | 5 | γ = φ⁻² stability |
| 70 | Symplectic Structure | 5 | Hamiltonian dynamics |
| 78 | Renormalization Group | 5 | Scale invariance |
| 102 | Mutual Info Geometry | 5 | Natural gradient |
| 317 | Category Theory | 5 | Functorial structure |

---

## IMPLEMENTATION ROADMAP

### Phase 1: Validation (Easy, High Theory)
- [ ] #41 Lyapunov exponents
- [ ] #45 Self-organized criticality
- [ ] #58 Spectral gap monitoring
- [ ] #74 Order parameter identification
- [ ] #106 Zeta zero clustering

### Phase 2: Language Capabilities (Medium, High Impact)
- [ ] #21 Morphological schemas
- [ ] #24 Semantic role in bivector
- [ ] #268 Frame semantics
- [ ] #270 Conceptual metaphor
- [ ] #274 Discourse coherence

### Phase 3: Deep Theory (Hard, Foundational)
- [ ] #44 Edge of chaos training
- [ ] #96 Channel capacity
- [ ] #103 Information bottleneck formalization
- [ ] #317 Category theory structure
- [ ] #323 Information geometry

### Phase 4: Creativity & Emergence (Medium, Novel)
- [ ] #181 Insight problem solving
- [ ] #345 Bisociation
- [ ] #354 Metaphor generation
- [ ] #355 Poetic compression
- [ ] #126 Emergence detection
