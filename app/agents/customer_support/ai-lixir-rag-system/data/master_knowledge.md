# SERVICE: ADMET Inference System

# FILE: admet_inference.md

# DESCRIPTION: A high-performance, CPU-optimized asynchronous microservice that provides raw molecular property predictions across the five core ADMET dimensions using Message Passing Neural Networks (MPNN).

---

## TOPIC: Overview

**What this service does:**
This service takes one or more molecular structures represented as SMILES strings and asynchronously predicts their raw clinical pharmacokinetic properties across five major categories: Absorption, Distribution, Metabolism, Excretion, and Toxicity (ADMET). It processes calculations using an optimized, concurrent CPU pipeline to deliver raw model output values directly.

**Why it exists in the platform:**
In the drug discovery pipeline, over 90% of candidate molecules fail during clinical trials due to poor pharmacokinetics or unanticipated toxicity. This service serves as an early-stage *in silico* screening filter, allowing researchers to weed out fundamentally flawed compounds long before synthesis, in vitro assays, or in vivo animal studies are conducted.

**Who uses it:**
Medicinal chemists, computational biologists, and translational clinical researchers use this service to prioritize lead candidate structures and guide structural optimization strategies.

---

## TOPIC: Scientific Background

**The core scientific problem this service solves:**
Before a drug can cure a disease, it must travel through the human body, reach its target site in the correct concentration, remain stable long enough to perform its action, and safely clear out without poisoning the patient. Manually assessing these five ADMET dimensions is impossible because molecular behavior inside biological systems depends on complex non-linear interplays of chemical architecture that cannot be inferred by sight alone. This service evaluates thousands of candidate structures concurrently, automating a process that would otherwise require months of bench wet-lab experiments.

**The computational approach used:**
The service utilizes **Message Passing Neural Networks (MPNNs)** via the ChemProp framework. Unlike classic machine learning approaches that convert molecules into static 2D bit-vector fingerprints, MPNNs operate directly on the molecular graph representation. They dynamically learn specific local and global feature representations by passing virtual chemical "messages" across atoms and bonds. This graph-centric approach is chosen because it preserves critical spatial connectivity, structural context, and electronic trends, outperforming old fingerprint methods on property-prediction accuracy.

**Key scientific concepts a user needs to understand:**

* **SMILES** — Simplified Molecular-Input Line-Entry System. A text-based notation string that fully captures a chemical structure's connectivity and atoms (e.g., Ethanol is `CCO`).
* **ADMET** — A critical acronym mapping a drug's clinical journey:
* *Absorption:* How effectively the compound crosses biological membranes (like the gut wall) to enter the bloodstream.
* *Distribution:* How the compound is dispersed throughout bodily tissues and organs.
* *Metabolism:* How metabolic enzymes (typically in the liver) chemically transform the compound.
* *Excretion:* How the body eliminates the compound or its metabolites (usually through renal or biliary routes).
* *Toxicity:* Whether the molecule damages cells, specific organs, or normal physiological systems.


* **MPNN** — Message Passing Neural Network. A type of deep learning framework engineered specifically for graph data structures like chemical compounds.
* **Lipinski's Rule of Five** — A classical reference rule of thumb stating that small-molecule drug absorption is generally favored if molecular weight $< 500$ Da, $\log P < 5$, hydrogen bond donors $< 5$, and hydrogen bond acceptors $< 10$. This service supplements this rule with direct, non-linear ML predictions.

---

## TOPIC: API Endpoints

### Endpoint: `GET /`

**What it does:** Returns core information about the running API version, configuration mode, and supported capability endpoints.
**When to call it:** When verifying initial API reachability or auditing active functional modes.

**Input Parameters:** None.

**Response Fields:**

| Field | Type | What It Means |
| --- | --- | --- |
| `name` | string | Name of the service application |
| `version` | string | Software version string |
| `description` | string | Brief description of service features |
| `mode` | string | Dedicated functional mode (e.g., `inference-only-async`) |
| `output_type` | string | Type of output data generated (e.g., `raw-predictions-only`) |
| `endpoints` | dictionary | Mapping of available path strings to their system purpose |

**Example Request:**

```json
{}

```

**Example Response:**

```json
{
  "name": "ADMET Inference System",
  "version": "3.0.0",
  "description": "REST API for drug ADMET property predictions (Async, Raw Outputs, CPU Optimized)",
  "mode": "inference-only-async",
  "output_type": "raw-predictions-only",
  "endpoints": {
    "docs": "/docs (Swagger UI)",
    "health": "/health",
    "predict": "/predict (Single async prediction)",
    "predict_batch": "/predict_batch (Batch async predictions)",
    "model_status": "/models/status"
  }
}

```

---

### Endpoint: `GET /health`

**What it does:** Verifies that the internal inference architecture and individual models are loaded, returning general system health status.
**When to call it:** For platform readiness checks or automated monitoring scripts.

**Input Parameters:** None.

**Response Fields:**

| Field | Type | What It Means |
| --- | --- | --- |
| `status` | string | Execution health state (`healthy` or throws error) |
| `models_loaded` | integer | Total count of successfully initialized deep learning models |
| `total_models` | integer | Total count of registered ADMET tasks on disk |
| `version` | string | Service release tracker |
| `mode` | string | Processing context |
| `async` | boolean | Confirmation that asynchronous execution is enabled |
| `output` | string | Confirmation of output type formatting |

**Example Request:**

```json
{}

```

**Example Response:**

```json
{
  "status": "healthy",
  "models_loaded": 5,
  "total_models": 5,
  "version": "3.0.0",
  "mode": "inference-only",
  "async": true,
  "output": "raw-predictions"
}

```

---

### Endpoint: `GET /models/status`

**What it does:** Assesses individual sub-model checkpoint health states and reports exactly which task models are online.
**When to call it:** When troubleshooting why a specific prediction category (e.g., Toxicity) returns an empty or null result.

**Input Parameters:** None.

**Response Fields:**

| Field | Type | What It Means |
| --- | --- | --- |
| `models_loaded` | dictionary | A mapping of task names to booleans showing if their checkpoint file loaded |
| `total_models` | integer | Total count of standard tasks evaluated |
| `models_ready` | integer | Total count of task models verified as operational |
| `admet_tasks` | list | Flat list of string task names expected by the service |

**Example Request:**

```json
{}

```

**Example Response:**

```json
{
  "models_loaded": {
    "Absorption": true,
    "Distribution": true,
    "Metabolism": true,
    "Excretion": true,
    "Toxicity": true
  },
  "total_models": 5,
  "models_ready": 5,
  "admet_tasks": ["Absorption", "Distribution", "Metabolism", "Excretion", "Toxicity"]
}

```

---

### Endpoint: `POST /predict`

**What it does:** Evaluates a single molecular compound across all online ADMET models concurrently.
**When to call it:** When checking a single newly sketched molecule or a unique lead compound design.

**Input Parameters:**

| Parameter | Type | Required | Default | Why It Exists (Scientific Reason) |
| --- | --- | --- | --- | --- |
| `smiles` | string | Yes | — | The text-based 1D representation of the compound structure to be evaluated. |

**Response Fields:**

| Field | Type | What It Means |
| --- | --- | --- |
| `smiles` | string | The validated, evaluated molecular structure query. |
| `predictions` | dictionary | Raw property numbers indicating model outputs across the 5 dimensions. |
| `error` | string | Error details if chemical validation or feature extraction fails. |

**Example Request:**

```json
{
  "smiles": "CC(=O)NC1=CC=C(O)C=C1"
}

```

**Example Response:**

```json
{
  "smiles": "CC(=O)NC1=CC=C(O)C=C1",
  "predictions": {
    "Absorption": 0.8921,
    "Distribution": 0.4120,
    "Metabolism": 0.2215,
    "Excretion": 0.5512,
    "Toxicity": 0.0124
  },
  "error": null
}

```

---

### Endpoint: `POST /predict_batch`

**What it does:** Takes an array of multiple structures and executes them in parallel across multi-threaded asynchronous tasks.
**When to call it:** When screening a library subset or batch processing virtual screening files.

**Input Parameters:**

| Parameter | Type | Required | Default | Why It Exists (Scientific Reason) |
| --- | --- | --- | --- | --- |
| `smiles_list` | list of strings | Yes | — | Array of structural strings to process as a virtual chemical library screening batch. |

**Response Fields:**

| Field | Type | What It Means |
| --- | --- | --- |
| `total` | integer | Total count of strings received in the request array. |
| `successful` | integer | Count of molecules that successfully completed MPNN graph inference. |
| `failed` | integer | Count of molecules that hit formatting or pipeline execution failures. |
| `results` | list | Array of `PredictionResponse` objects containing individual chemical records. |
| `processing_time_ms` | float | Wall-clock execution time tracking engine speed for the user's batch. |

**Example Request:**

```json
{
  "smiles_list": [
    "CCO",
    "INVALID_SMILES_STRING",
    "CC(=O)O"
  ]
}

```

**Example Response:**

```json
{
  "total": 3,
  "successful": 2,
  "failed": 1,
  "results": [
    {
      "smiles": "CCO",
      "predictions": {
        "Absorption": 0.9512,
        "Distribution": 0.6120,
        "Metabolism": 0.1042,
        "Excretion": 0.8241,
        "Toxicity": 0.0031
      },
      "error": null
    },
    {
      "smiles": "INVALID_SMILES_STRING",
      "predictions": null,
      "error": "Invalid SMILES format"
    },
    {
      "smiles": "CC(=O)O",
      "predictions": {
        "Absorption": 0.8841,
        "Distribution": 0.5124,
        "Metabolism": 0.1521,
        "Excretion": 0.7712,
        "Toxicity": 0.0011
      },
      "error": null
    }
  ],
  "processing_time_ms": 42.15
}

```

---

### Endpoint: `POST /predict/batch`

> [!NOTE]
> This is a legacy endpoint mirroring `/predict_batch` for backward compatibility. Please update new clients to use `/predict_batch` directly.

---

## TOPIC: Internal Pipeline (Step by Step)

When a prediction endpoint is hit, the microservice executes the following steps:

1. **SMILES Validation**
* *What happens:* The incoming input string is structurally validated using a quick format check to confirm it is not empty.
* *Why it's necessary:* This shields downstream deep learning architectures from running into fatal parsing bugs caused by empty text inputs.


2. **Asynchronous Thread Hand-off**
* *What happens:* The asynchronous router delegates the computational work out to the running system loop via an internal multi-threaded executor pool (`run_in_executor`).
* *Why it's necessary:* Because model calculation loops block Python threads, this hand-off allows the server to keep listening for new API requests without freezing while it works on heavy number-crunching tasks.


3. **Molecular Graph Construction**
* *What happens:* The raw string text gets processed by ChemProp's `SimpleMoleculeMolGraphFeaturizer`, converting the chemical symbols into an explicit numerical graph containing mathematical atom and bond vectors.
* *Why it's necessary:* MPNN neural networks cannot interpret raw text strings directly; they require structured mathematical arrays representing physical connectivity matrices.


4. **Message Passing Graph Inference**
* *What happens:* The generated molecular graph matrices are run through the five deep learning MPNN model checkpoints loaded into the system using PyTorch inference mode (`torch.inference_mode()`).
* *Why it's necessary:* This step performs the core non-linear feature extraction across the five property tracks to generate the raw ADMET prediction numbers.


5. **Output Array Extraction**
* *What happens:* The final tensor values are pulled back from the CPU tensor architecture, unwrapped into plain Python floating-point values, and structured into a standard response object.
* *Why it's necessary:* This strips away deep learning frameworks' internal data wrappers, rendering the information readable by external client applications.



---

## TOPIC: User Questions & Answers

**Q: What do the results mean? How do I interpret them?**
A: This service outputs raw, uninterpreted decimal numbers calculated directly by the underlying model checkpoints. The exact units, scale, and cutoff thresholds depend on how your system models were trained (e.g., probability ranges between 0.0 and 1.0 for classification tasks, or raw target units for regression models). You should check your internal model training parameters to establish your screening cutoffs.

**Q: Why are my ADMET outputs returning `null` values while the rest of the response is successful?**
A: This happens if a particular model checkpoint (e.g., `best_model.ckpt` under the Excretion folder) was missing from the disk during service initialization. The microservice skips missing model checkpoints so that the other active models can keep working, returning `null` for the missing category. Use the `/models/status` endpoint to see which models are online.

**Q: Clarify the difference between in silico prediction and experimental validation.**
A: *In silico* predictions generated by this service are fast, automated, statistical estimates based on historical training data. They cannot replace real lab assays. Instead, they act as an early filter to select the best 10% of candidates for real-world *in vitro* and *in vivo* testing.

**Q: Why does this service run exclusively on the CPU instead of utilizing GPUs?**
A: While GPUs excel at processing huge training datasets, inference for small chemical graphs on optimized MPNN architectures is lightweight. Using an asynchronous parallel CPU design eliminates the system lag of moving data between CPU and GPU memory, ensuring high throughput without requiring expensive server hardware.

**Q: Why does the batch prediction endpoint run so much faster than calling the single prediction endpoint multiple times?**
A: Calling the single endpoint repeatedly creates extra network overhead and processes molecules one after another. The `/predict_batch` endpoint bypasses this by feeding all compounds into Python's asynchronous execution pool simultaneously, maximizing your server's multi-core CPU capacity.

**Q: Does the validation endpoint sanitize structural configurations automatically?**
A: No, the basic request pipeline uses a minimal format verification step to keep processing speeds as high as possible. The service includes a structural sanitization script (`sanitize_smiles`) that leverages RDKit internally, but it is not active in the main API request pipeline. Make sure your SMILES strings are valid and clean before sending them.

---

## TOPIC: Limitations & Warnings

* **No Automatic Interpretation:** The service outputs raw numbers exactly as calculated by the models. It does not flag a compound as "safe," "unsafe," "soluble," or "insoluble." That interpretation layer must be managed by the client application.
* **Minimal API Validation:** The core API endpoints only check that strings are not empty before passing them along. If you send an impossible chemical structure (like a carbon atom with 10 bonds), the validation step won't catch it, which could cause a down-pipeline model processing error.
* **`[VERIFY]` ChemProp Data Parsing Safeguards:** The current internal model pipeline processes batches using a fixed configuration size (`batch_size=1`). While this setup ensures reliable asynchronous processing for individual items, it does not leverage ChemProp's internal native batch optimization for large-scale library evaluations.

---

## TOPIC: Dependencies & Models

| Library / Model | Why It's Used (Not Just What It Is) |
| --- | --- |
| `FastAPI` | High-performance, lightweight web routing framework that supports asynchronous endpoints out of the box. |
| `torch` (PyTorch) | High-performance calculation matrix engine running under inference-only mode on CPU architectures. |
| `chemprop` | Specialized deep-learning chemistry framework used to build and manage structural Message Passing Neural Networks (MPNN). |
| `rdkit` | Classic chemical informatics toolkit included as a helper tool to handle structural sanitization tasks. |
| `pandas` | Data science library included to handle basic CSV file parsing and structural exports. |

---

## TOPIC: Platform Access

**Available on:** Enterprise Platform Tier

**Usage limits:** 100,000 compound processing evaluations per day.

**Restricted features:** Parallel batch screening with more than 500 molecules per concurrent request requires dedicated node deployments.

---

# SERVICE: Chemical RAG System

# FILE: chemical_rag.md

# DESCRIPTION: A drug-discovery grade chemical similarity search engine combining multi-fingerprint FAISS retrieval with LLM-generated structural explanations.

---

## TOPIC: Overview

**What this service does:**
This service enables high-throughput, intelligent chemical similarity exploration across large molecular libraries. It combines ultra-fast binary vector similarity screening with a domain-aware structural scoring fusion layer, followed by automated linguistic structural explanations powered by an LLM (Llama-3.1-8B-Instruct).

**Why it exists in the platform:**
In a drug discovery pipeline, identifying structural analogs (hits) to a query molecule is essential for lead optimization and scaffold hopping. Traditional search systems rely on a single fingerprint type, missing deep topological or structural nuances. This service bridges structural informatics and generative AI, ensuring researchers not only find diverse candidate molecules quickly but also immediately understand their chemical and electronic relationships without manual structural inspection.

**Who uses it:**
Medicinal chemists, computational biologists, and pharmacology researchers who need to screen compound libraries for target-relevant analogs, expand chemical series, or discover alternative scaffolds.

---

## TOPIC: Scientific Background

**The core scientific problem this service solves:**
Molecules that look different in a basic 2D representation can share vital structural or electronic features responsible for bioactivity, while minor structural variations (such as a charge change or fragmentation) can completely alter a drug's efficacy or toxicity (known as an activity cliff). Human researchers cannot manually cross-examine thousands or millions of chemical compounds across multiple fingerprint schemas and geometric attributes simultaneously. This service automates this comprehensive multi-perspective comparison at scale.

**The computational approach used:**
The service leverages **Multi-Fingerprint Fusion** combined with vector indexing. Instead of relying on a single structural descriptor, it computes and blends four complementary chemical representations (Morgan, MACCS, Atom Pairs, and Topological Torsions) to form a robust base structural score. Initial high-speed candidate filtering is performed using binary vector indexes via the FAISS library. Although the service logs and documentation strings reference `FAISS-IVF`, the underlying execution explicitly provisions a `faiss.IndexBinaryFlat` matrix `[VERIFY]`. This approach preserves absolute exact Tanimoto matches during the initial rapid binary screening phase before applying downstream domain-aware reranking, similarity calibration (Z-score to sigmoid mapping), and diversity enforcement via Maximal Marginal Relevance (MMR).

**Key scientific concepts a user needs to understand:**

* **Morgan Fingerprints** — A type of extended-connectivity fingerprint (ECFP) that maps radial atom neighborhoods up to a specific radius (radius=2, equivalent to ECFP4), tracking local atom trajectories into a bit vector.
* **MACCS Keys** — A fixed-length (166-bit) structural key representation that checks for the presence or absence of predefined structural fragments or functional groups (e.g., rings, halogens, carbonyls).
* **Atom Pairs & Topological Torsions** — Distant-dependent descriptors capturing through-bond paths and four-atom sequences respectively, capturing the overall skeleton shape and conformation possibilities.
* **Tanimoto Similarity** — A statistical coefficient used to compare the overlap of bit-vectors. A score of 1.0 means identical bit representations, 0.7+ indicates strong structural similarity, and <0.4 points to weak commonality.
* **Maximal Marginal Relevance (MMR)** — An optimization algorithm that balances relevance (similarity score) against diversity, preventing the return of redundant, identical analogs by forcing chemical scaffold variety in the final output.

---

## TOPIC: API Endpoints

### Endpoint: `POST /search/retrieval-only`

**What it does:** Runs an ultra-fast chemical similarity screening across the indexed library using the multi-fingerprint fusion and chemical reranking pipeline, returning raw metadata without LLM explanations.
**When to call it:** When performing high-throughput screens or programmatic queries where raw data and speed are paramount (<100ms response targets) and human-readable text explanations are unnecessary.

**Input Parameters:**

| Parameter | Type | Required | Default | Why It Exists (Scientific Reason) |
| --- | --- | --- | --- | --- |
| `smiles` | string | Yes | — | The simplified molecular-input line-entry system (SMILES) string representing the query compound. |
| `top_k` | integer | No | 3 | Controls how many structural matches to return (Clamped between 1 and 100). |

**Response Fields:**

| Field | Type | What It Means |
| --- | --- | --- |
| `results` | array | Contains the list of matching compounds sorted by final ranking metrics. |
| `results[].smiles` | string | The canonical SMILES string of the matched library compound. |
| `results[].similarity_score` | float | The blended, chemical-aware score integrating multi-fingerprint and property metrics. |
| `results[].image` | string | An automatically generated URL pointing to the 2D visual depiction of the structure. |
| `results[].cid` | string | PubChem Compound ID or unique database identifier mapped during ingestion. |
| `results[].name` | string | Common chemical or trade name associated with the matched library molecule. |
| `query_smiles` | string | Echoes back the validated query molecule SMILES string. |
| `total_results` | integer | The total count of matched elements returned within this batch. |

**Example Request:**

```json
{
  "smiles": "C(Br)(Br)(Br)Br",
  "top_k": 2
}

```

**Example Response:**

```json
{
  "results": [
    {
      "smiles": "C(Cl)(Cl)(Cl)Br",
      "similarity_score": 0.8145,
      "image": "https://example.com/render/C(Cl)(Cl)(Cl)Br",
      "explanation": null,
      "cid": "10293",
      "name": "Bromotrichloromethane"
    }
  ],
  "query_smiles": "C(Br)(Br)(Br)Br",
  "total_results": 1
}

```

---

### Endpoint: `POST /search/full-rag`

**What it does:** Performs the complete hybrid vector search pipeline and passes top hits to an LLM context layer to append natural language rationales for the structural similarity.
**When to call it:** When an interactive user or medicinal chemist is evaluating hits and requires rapid interpretation of structural overlaps, functional groups, and isosteric replacements.

**Input Parameters:**

| Parameter | Type | Required | Default | Why It Exists (Scientific Reason) |
| --- | --- | --- | --- | --- |
| `smiles` | string | Yes | — | The query SMILES string to analyze. |
| `top_k` | integer | No | 3 | Number of final diverse results to return. |
| `explain` | boolean | No | true | Toggles whether the LLM generation layer should be invoked or bypassed. |

**Response Fields:**

| Field | Type | What It Means |
| --- | --- | --- |
| `results` | array | Contains compound hits along with detailed textual reasoning. |
| `results[].explanation` | string | Human-readable paragraph explaining why the pair is structurally and bioisosterically related. |

**Example Request:**

```json
{
  "smiles": "C(Br)(Br)(Br)Br",
  "top_k": 3,
  "explain": true
}

```

**Example Response:**

```json
{
  "results": [
    {
      "smiles": "C(Cl)(Cl)(Cl)Br",
      "similarity_score": 0.8145,
      "image": "https://example.com/render/C(Cl)(Cl)(Cl)Br",
      "explanation": "The match compound exhibits high similarity by maintaining a tetrahalogenated methane core framework. The structure represents a classic bioisosteric replacement where three bromine atoms from the query are substituted with three chlorine atoms, preserving molecular geometry and volume while subtly shifting electronic distribution.",
      "cid": "10293",
      "name": "Bromotrichloromethane"
    }
  ],
  "query_smiles": "C(Br)(Br)(Br)Br",
  "total_results": 1
}

```

---

## TOPIC: Internal Pipeline (Step by Step)

Each step below occurs when a request enters the chemical search pipeline:

1. **SMILES Validation & Fingerprint Generation**
* *What happens:* The incoming SMILES string is parsed via RDKit into a molecular object. Four tactical fingerprints are generated: Morgan (radius=2, 2048-bit), 166-bit MACCS keys, Hashed Atom Pairs, and Hashed Topological Torsions.
* *Why it's necessary:* Validates structural sanitization up front. Generating diverse bit vectors ensures the molecule is characterized from both local environments and systemic path topologies.


2. **FAISS Binary Flat Filtering**
* *What happens:* The 2048-bit Morgan fingerprint vector is packed into an array of unsigned 8-bit integers (`np.packbits`) and fed into a fast binary flat index search to extract a broad candidate pool ($k_{search} = \min(\max(k \times 20, 200), \text{total\_compounds})$).
* *Why it's necessary:* Prevents computational bottlenecks. Running complex sub-structure alignments directly over millions of rows is impossible in real-time; this acts as a high-speed vector triage step.


3. **Multi-Fingerprint Similarity Fusion**
* *What happens:* For every compound in the candidate pool, Tanimoto similarity metrics are individually computed across all four fingerprint domains and combined linearly using fixed weights:

$$Score_{base} = 0.50 \cdot Morgan + 0.20 \cdot MACCS + 0.20 \cdot AtomPairs + 0.10 \cdot Torsion$$


* *Why it's necessary:* Mitigates representation bias. Morgan fingerprints focus heavily on immediate connectivity pathways, whereas adding MACCS keys and Torsion shapes ensures macro-structural fragments and spatial paths are properly accounted for.


4. **Chemical-Aware Property Reranking**
* *What happens:* Properties (Aromaticity ratio, Ring count variations, Total formal charges, Molecular fragmentation counts) are extracted using cached features. The base score is modulated via specific bonuses and strict domain penalties:

$$Score_{chem} = 0.70 \cdot Score_{base} + 0.15 \cdot Score_{arom} + 0.10 \cdot Score_{ring} - 0.15 \cdot Penalty_{charge} - 0.10 \cdot Penalty_{frag}$$


* *Why it's necessary:* Pure bit-vector matching can miss critical physical attributes. For instance, an extra positive formal charge or an unwanted salt fragment can completely disrupt target binding, which vector models alone might overlook.


5. **Z-Score Similarity Calibration**
* *What happens:* The final chemical-aware scores across the candidate pool are normalized relative to their mean ($\mu$) and standard deviation ($\sigma$). This Z-score is transformed via a standard logistic sigmoid function into a calibrated distribution:

$$P(calibrated) = \frac{1}{1 + e^{-z}}$$


* *Why it's necessary:* Normalizes score values. Different query structures produce varying raw score ranges; mapping to a relative probability distribution establishes a stable foundation for the downstream diversity controls.


6. **Maximal Marginal Relevance (MMR) Redundancy Filtering**
* *What happens:* An iterative selection loop extracts results by prioritizing candidates that exhibit high calibrated similarity to the query while penalizing structures highly identical to molecules *already chosen* for the output list.
* *Why it's necessary:* Enforces chemical diversity. Without MMR, a search for an active lead compound would return dozens of near-identical analogs differing by a single methyl group, hiding entirely different structural families (scaffolds) that could hit the same target.


7. **RDKit Grounded LLM Explanation Generation (For Full RAG)**
* *What happens:* Factual properties (molecular formula, molecular weight, heavy atom count) are calculated via RDKit for the final hit pairs. These numbers are injected into a highly constrained system prompt alongside the SMILES string and dispatched to Llama-3.1-8B-Instruct.
* *Why it's necessary:* Eliminates LLM hallucinations. Forcing the model to ground its structural text observations within verifiable, concrete molecular parameters prevents it from inventing incorrect atom counts or impossible valency structures.



---

## TOPIC: User Questions & Answers

**Q: What do the similarity scores mean? How do I interpret them?**
A: The system outputs two metrics: a `similarity_score` and a `calibrated_score`. The `similarity_score` is the domain-weighted metric combining structural fingerprints and molecular rules (Aromaticity, rings, charge). The `calibrated_score` maps these scores into a localized probability distribution via a sigmoid function. A higher score signifies closer structural, electronic, and fragment alignment to your query compound.

**Q: Why does the system use multiple fingerprints instead of just standard SMILES or Morgan matching?**
A: Relying on a single fingerprint type introduces screening blind spots. Morgan fingerprints are highly effective at mapping immediate atomic environments but lack context regarding overall skeletal pathways. By fusing Morgan fingerprints with MACCS fragments, Atom Pairs, and Topological Torsions, the engine captures local atom types, macroscopic functional groups, and structural frameworks simultaneously.

**Q: Why does the engine penalize formal charges and molecule fragmentation?**
A: In therapeutic design, introducing an unexpected formal charge or selecting a structure split into multiple independent disconnected fragments (salts or counter-ions) can dramatically alter a drug's pharmacokinetics, solubility, and toxicity profiles. The engine applies strict mathematical penalties to these variations to keep hits translationally viable.

**Q: How does the system ensure the LLM does not hallucinate chemical features or bonds?**
A: The pipeline uses a strict grounding guardrail system. Before the text prompt is sent to the LLM, an internal RDKit parser extracts absolute, unyielding chemical metadata (exact molecular formulas, heavy atom counts, and net formal charges). The LLM system prompt mandates that the generated text must remain strictly consistent with these parameters, preventing common errors like inventing non-existent atoms or describing pentavalent carbons.

**Q: What is the purpose of the lambda_param within the search functions?**
A: The `lambda_param` (defaulting to 0.6) dictates the balance within the Maximal Marginal Relevance (MMR) step. A value closer to 1.0 focuses exclusively on returning compounds with the absolute highest similarity score, regardless of redundancy. Lowering this parameter increases the penalty for structural redundancy, forcing the engine to find structurally diverse analogs and alternative scaffolds.

**Q: Why is there a difference between the index types mentioned in the console logs versus the code?**
A: The service logs and internal schema definitions mention a `FAISS-IVF` architecture. However, the core module instantiates a `faiss.IndexBinaryFlat` layout `[VERIFY]`. A flat binary index performs an exhaustive, exact comparison of fingerprint bit vectors rather than an approximate voronoi cell lookup. This ensures perfect recall across the initial screen, though it requires more memory as the database scales.

---

## TOPIC: Limitations & Warnings

* **Discrepancy in Vector Index Type:** The codebase documentation, startup scripts, and API strings explicitly announce a `FAISS-IVF` (Inverted File Index) deployment. However, the code instantiates `faiss.IndexBinaryFlat` `[VERIFY]`. While this guarantees exact match accuracy, it could lead to increased processing times as database sizes grow toward millions of compounds.
* **API Token Dependency for Explanations:** The generative explanation layer requires a valid `HF_TOKEN` environment variable. If this token is missing or if the Hugging Face Router experiences downtime, the system silently drops back to a static, heuristic text block based entirely on general similarity brackets.
* **Fixed Bit Size Limits:** The chemical search engine is built assuming a hardcoded fingerprint vector size of 2048 bits. Attempting to pass or read an index configured with alternative bit depths without rebuilding the database will throw shape mismatch errors.

---

## TOPIC: Dependencies & Models

| Library / Model | Why It's Used (Not Just What It Is) |
| --- | --- |
| `rdkit` | Used as the foundational cheminformatics platform to sanitize structures, calculate distinct multi-fingerprint representations, extract physical properties, and provide grounding parameters for LLM safety. |
| `faiss` | Utilized for high-speed, parallelized binary vector calculations, executing the initial candidate pool triage layer over large chemical spaces in milliseconds. |
| `FastAPI` | Serves as the high-performance asynchronous web framework providing non-blocking execution routines for search processing and thread-pool management. |
| `meta-llama/Llama-3.1-8B-Instruct:fastest` | Selected as the generative explanation layer, interpreting abstract structural overlaps and bioisosteric adjustments into precise, technical text for medicinal chemistry researchers. |

---

## TOPIC: Platform Access

**Available on:** Enterprise Tier

**Usage limits:** High-throughput retrieval screens are uncapped for local instances; however, LLM-based `full-rag` generation requests are subject to upstream Hugging Face Router rate limits and thread-pool constraints.

**Restricted features:** Automated RDKit property grounding, multi-fingerprint fusion logic, and MMR diversity routing are only available when interacting with databases that have gone through the centralized pre-computation ingestion pipeline.

---

# SERVICE: Drug Repurposing

# FILE: drug_repurposing.md

# DESCRIPTION: End-to-end pipeline mapping diseases to biological targets and predicting binding affinities of existing FDA-approved drugs using deep learning.

---

## TOPIC: Overview

**What this service does:**
This service takes a clinical disease name and automatically identifies potential drug treatments from a library of existing FDA-approved compounds. It does this by mapping the disease to specific biological target proteins, fetching their amino acid sequences, and using deep learning to simulate how strongly different drugs will bind to those targets.

**Why it exists in the platform:**
Developing a new drug from scratch takes over a decade and costs billions. This service facilitates *drug repurposing*—finding new therapeutic uses for drugs that have already passed clinical safety trials. Without this automated pipeline, researchers would have to manually cross-reference disease pathways, retrieve protein sequences, and run individual binding simulations, which is impossible at the scale of thousands of drug-target combinations.

**Who uses it:**
Medicinal chemists, computational biologists, and pharmaceutical researchers looking for rapid, safe therapeutic interventions for novel or existing diseases.

---

## TOPIC: Scientific Background

**The core scientific problem this service solves:**
The fundamental challenge of pharmacology is identifying molecules that physically interact with disease-causing proteins. While we have libraries of safe, FDA-approved drugs, testing them all against every human disease protein in a physical lab (*in vitro* or *in vivo*) is prohibitively slow and expensive. This service solves the bottleneck by performing *in silico* (computational) screening at scale to generate high-confidence hypotheses for lab testing.

**The computational approach used:**
The service utilizes the **DeepPurpose MPNN_CNN_BindingDB** deep learning model.

* **MPNN (Message Passing Neural Network):** Used to encode the drug. It treats the drug's SMILES string as a mathematical graph (atoms as nodes, bonds as edges), capturing its 3D topological and chemical features.
* **CNN (Convolutional Neural Network):** Used to encode the target protein. It reads the 1D amino acid sequence to extract localized structural motifs.
This approach was chosen over traditional molecular docking because it is drastically faster (seconds vs. hours per molecule) while maintaining high predictive accuracy for screening large libraries.

**Key scientific concepts a user needs to understand:**

* **Drug Repurposing:** The strategy of investigating existing, safety-approved drugs for new therapeutic purposes, bypassing Phase I clinical trials.
* **Binding Affinity:** A measure of how strongly a drug physically attaches to its target protein. Higher scores generally indicate a higher likelihood of the drug modulating the target's function.
* **SMILES (Simplified Molecular-Input Line-Entry System):** A text-based representation of a chemical molecule's 2D graph structure.
* **EFO ID (Experimental Factor Ontology):** A standardized scientific identifier for diseases and phenotypes, used to strictly define a condition without naming ambiguities.
* **In Silico Prediction:** An experiment or simulation performed entirely by a computer, contrasting with *in vitro* (test tube) or *in vivo* (living organism) experiments.

---

## TOPIC: API Endpoints

### Endpoint: `POST /api/v1/screen`

**What it does:** Runs the complete end-to-end virtual screening pipeline from disease name to ranked drug candidates.
**When to call it:** When you want to discover potential new treatments for a specific disease using a library of existing drugs.

**Input Parameters:**

| Parameter | Type | Required | Default | Why It Exists (Scientific Reason) |
| --- | --- | --- | --- | --- |
| `disease_name` | string | Yes | — | The clinical condition you want to treat (e.g., "Type 2 Diabetes"). |
| `top_n_targets` | int | No | 10 | Limits how many disease-associated proteins are tested. More targets increase computational time but cover more biological pathways. |
| `min_score` | float | No | 0.0 | Filters out weak binders. A higher threshold (e.g., 0.7) ensures only drugs with strong predicted physical interactions are returned. |
| `known_drugs` | list | No | [] | Used to separate ground-truth standard-of-care treatments from novel discoveries in the results list. |

**Response Fields:**

| Field | Type | What It Means |
| --- | --- | --- |
| `top_candidates` | list | An array containing the ranked drug-target pairs. |
| `score` | float | The normalized binding affinity prediction (0 to 1). Values closer to 1 indicate higher confidence in physical binding. |
| `status` | string | Labels the drug as a "Known Treatment" or a "Potential Discovery" based on your inputs. |

**Example Request:**

```json
{
  "disease_name": "Type 2 Diabetes",
  "min_score": 0.5,
  "top_n_targets": 10,
  "known_drugs": ["Metformin"]
}

```

**Example Response:**

```json
{
  "disease_name": "Type 2 Diabetes",
  "total_targets_found": 10,
  "total_drugs_screened": 200,
  "total_pairs_evaluated": 2000,
  "top_candidates": [
    {
      "drug_name": "Drug_DB00838",
      "target_symbol": "GCK",
      "score": 0.92,
      "status": "Potential Discovery"
    }
  ]
}

```

### Endpoint: `POST /api/v1/disease-targets`

**What it does:** Fetches the top proteins biologically associated with a given disease.
**When to call it:** When you are mapping out the biological pathways of a disease but aren't ready to screen drugs yet.

**Input Parameters:**

| Parameter | Type | Required | Default | Why It Exists (Scientific Reason) |
| --- | --- | --- | --- | --- |
| `disease_name` | string | Yes | — | The target disease. |
| `top_n` | int | No | 10 | Caps the number of returned proteins to focus only on the most statistically significant drivers of the disease. |

**Response Fields:**

| Field | Type | What It Means |
| --- | --- | --- |
| `disease_id` | string | The standardized EFO ID for the disease. |
| `targets` | list | A list of target proteins, enriched with their UniProt IDs, sequences, and PDB 3D structure IDs. |

### Endpoint: `GET /api/v1/drug-library`

**What it does:** Returns the current database of FDA-approved drugs being used for screening.
**When to call it:** When you need to audit which compounds are included in the computational search space.

**Input Parameters:** None.

**Response Fields:**

| Field | Type | What It Means |
| --- | --- | --- |
| `total_drugs` | int | Total number of valid chemical structures available. |
| `drugs` | list | Array of drug objects containing their names and SMILES chemical representations. |

---

## TOPIC: Internal Pipeline (Step by Step)

Each step below is what happens inside the service when a full screening request is processed:

1. **Disease Target Mapping (Open Targets API)**
* *What happens:* Translates the user's plain-text disease name into an EFO ID, then queries the Open Targets genetics database to find the top $N$ proteins genetically or clinically associated with the disease.
* *Why it's necessary:* Drugs do not treat abstract diseases; they bind to specific malfunctioning proteins. We must identify these molecular targets first.


2. **Protein Sequence Retrieval (UniProt API)**
* *What happens:* Fetches the raw string of amino acids that make up each identified target protein.
* *Why it's necessary:* The AI model (CNN) cannot process a protein by its name; it requires the actual chemical sequence of amino acids to predict molecular interactions.


3. **Drug Library Loading (TDC)**
* *What happens:* Loads a dataset of FDA-approved drugs (e.g., `Half_Life_Obach`), extracting their chemical structures in SMILES format.
* *Why it's necessary:* This establishes the "repurposing" search space—a constrained list of molecules that are legally and biologically safe for human use.


4. **AI Virtual Screening (DeepPurpose)**
* *What happens:* The system pairs every drug with every disease target. The MPNN-CNN model evaluates each pair, outputting a predicted binding affinity score from 0 to 1.
* *Why it's necessary:* This replaces physical laboratory binding assays, instantly separating compounds that structurally fit the target from those that do not.


5. **Result Processing & Sorting**
* *What happens:* The system filters out weak scores, ranks the remaining pairs from highest to lowest affinity, and checks them against the user's `known_drugs` list to label them as discoveries or known treatments.
* *Why it's necessary:* Researchers need actionable intelligence—a prioritized list of top candidates to take into the physical lab, not a raw data dump.



---

## TOPIC: User Questions & Answers

**Q: What is the logical premise behind computational drug repurposing?**
A: Finding a completely new chemical entity that is safe for humans takes 10–15 years and requires passing rigorous Phase I safety trials. Conversely, an existing FDA-approved drug is already proven safe. If we can computationally identify that an approved drug also binds to a protein responsible for a different disease, we can "repurpose" it and jump directly to Phase II/III efficacy trials, vastly accelerating the availability of treatments.

**Q: Why does the system fetch data from Open Targets and UniProt before making predictions?**
A: To predict a physical interaction, the AI needs physical data. A disease name like "Alzheimer's" is a clinical label, not a molecule. Open Targets translates the clinical label into the actual proteins causing the disease. UniProt then translates those protein names into their exact amino acid sequences. With the protein sequences and the drugs' chemical structures (SMILES), the AI finally has the molecular data required to simulate binding.

**Q: What do the binding scores mean? How do I interpret them?**
A: The AI outputs a normalized score between 0.0 and 1.0 representing the predicted binding affinity.

* **0.0 - 0.3:** Very weak or no binding.
* **0.3 - 0.5:** Weak binding.
* **0.5 - 0.7:** Moderate binding (bioactive).
* **0.7 - 0.9:** Strong binding (likely to work).
* **0.9 - 1.0:** Very strong binding, indicating high confidence that the drug will physically interact with the target.

**Q: Why does the `known_drugs` input parameter exist?**
A: It acts as an internal validation and sorting mechanism. If you input "Metformin" as a known drug for Diabetes, and the AI ranks Metformin highly, it proves the model is working correctly. The service uses this list to tag high-scoring results as either "✅ Known Treatment" (validating the model) or "🆕 Potential Discovery" (highlighting novel candidates for your research).

**Q: Why does the screening process take 30+ seconds to run?**
A: The service is performing heavy computational mathematics. If you screen 200 drugs against 10 targets, the system must process 2,000 unique molecular pairings. For each pair, the AI translates the drug into a graph network, the protein into a convolutional matrix, and calculates their physical interaction. Processing time heavily depends on whether the server has access to GPU acceleration (CUDA) or is relying on standard CPUs.

**Q: Can I use these results directly to treat patients?**
A: Absolutely not. This service generates *in silico* (computational) predictions, which serve as highly informed hypotheses. A high binding score means the drug and protein *should* interact based on their chemical structures, but these findings must be validated *in vitro* (in laboratory assays) and *in vivo* (in clinical settings) before any medical application.

---

## TOPIC: Limitations & Warnings

* **Limitation 1: False Positives in Virtual Screening.** AI models can overpredict binding affinities. A high score indicates structural compatibility, but steric hindrance or biological context in a real human cell may prevent actual binding. Experimental validation is always required.
* **Limitation 2: Computational Hardware Limits.** To ensure stable performance without running out of memory, the system dynamically caps the number of drugs it screens. On CPU deployments, it defaults to limiting the library to ~200 drugs, while GPU deployments can handle ~600+ drugs.
* **Scientific caveat:** The DeepPurpose CNN model treats the protein as a 1D sequence of amino acids rather than a dynamic 3D structure. This is excellent for high-throughput screening but lacks the nuanced 3D pocket dynamics captured by much slower, physics-based molecular dynamics simulations.

---

## TOPIC: Dependencies & Models

| Library / Model | Why It's Used (Not Just What It Is) |
| --- | --- |
| `DeepPurpose (MPNN_CNN_BindingDB)` | The core AI engine. Chosen because it provides state-of-the-art binding affinity predictions using Message Passing on drugs and Convolutions on proteins, offering the best balance of speed and accuracy for library screening. |
| `OpenTargets GraphQL API` | Chosen because it aggregates peer-reviewed genetic and clinical evidence, providing the most statistically reliable links between abstract disease names and specific biological targets. |
| `UniProt REST API` | The definitive global repository for protein data, used to accurately convert target gene symbols into the raw amino acid sequences required by the AI. |
| `TDC (Therapeutic Data Commons)` | Provides high-quality, curated datasets of FDA-approved drugs (specifically ADME/Half_Life_Obach datasets) with validated SMILES strings, ensuring our search space only includes synthesizable, viable drugs. |
| `PyTorch / CUDA` | The underlying deep learning framework. Used to batch array calculations and push them to GPU memory, cutting prediction times from minutes to seconds. |

---

## TOPIC: Platform Access

**Available on:** Standard API integration (Free / Local fallback mode) and Pro / Enterprise (GPU-accelerated production mode).
**Usage limits:** * **CPU Instances:** Recommended limit of 200 drugs and 10 targets per screening request to prevent timeouts (~30s execution).

* **GPU Instances (CUDA):** Scales efficiently up to 600+ drugs per screening (~5s execution).
**Restricted features:** DeepPurpose real predictions require the full ML stack. Without it, the system falls back to a restricted "Mock Mode" utilizing randomized realistic scores and a limited 40-drug fallback library.

---

```markdown
# SERVICE: Ailixir EGFR Molecular Generation Pipeline
# FILE: egfr_generation_rag.md
# DESCRIPTION: AI-powered molecular generation, physicochemical enrichment, affinity prediction, and optional 3D docking for EGFR targets.

---

## TOPIC: Overview

**What this service does:**
This service operates an end-to-end *in silico* pipeline for the discovery of novel Epidermal Growth Factor Receptor (EGFR) inhibitors. It automatically generates new molecular structures, calculates their baseline druglikeness and synthetic accessibility, predicts their binding affinity to EGFR using a deep learning model, and can optionally run physics-based 3D docking to calculate estimated free energy of binding.

**Why it exists in the platform:**
In the drug discovery pipeline, the hit-discovery phase is traditionally bottlenecked by the physical limitations of high-throughput screening. This service exists to computationally explore the vast, uncharted chemical space and prioritize highly synthesizable, high-affinity candidate molecules before any wet-lab resources are spent. 

**Who uses it:**
Medicinal chemists, computational biologists, and early-stage researchers looking to identify novel chemical starting points or scaffolds targeting EGFR.

---

## TOPIC: Scientific Background

**The core scientific problem this service solves:**
The theoretical chemical space contains an estimated $10^{60}$ drug-like molecules. Finding a molecule that safely and effectively binds to a specific target like EGFR—while simultaneously obeying the laws of pharmacokinetics (absorption, distribution, metabolism, excretion)—is extremely difficult to do manually. 

**The computational approach used:**
This service employs a hybrid AI/Physics approach:
* **De Novo Drug Design:** It uses REINVENT 4 to computationally "imagine" entirely new molecules that do not yet exist, sampling from a trained generator model (`egfr_generator.chkpt`).
* **DeepPurpose Affinity Prediction:** It rapidly filters these generated molecules using a neural network (DeepPurpose). DeepPurpose uses **Morgan Fingerprints** (binary vectors that encode the structural features of a molecule) and Amino Acid Composition (AAC) to predict how well the drug will bind to the EGFR sequence without needing 3D physics.
* **AutoDock-GPU (ADGPU):** For the most promising candidates, it converts the 2D structures into 3D using RDKit's ETKDGv3 algorithm and performs physics-based docking against a pre-prepared EGFR grid (`4WKQ_receptor_v5_SBr.maps.fld`).

**Key scientific concepts a user needs to understand:**
* **De novo drug design** — The computational generation of novel molecular entities from scratch, optimized for specific target profiles.
* **SMILES** — Simplified Molecular-Input Line-Entry System. A text-based representation of a 2D chemical structure (e.g., `CCO` for ethanol). Generated outputs must be synthesized or further modeled.
* **QED (Quantitative Estimate of Druglikeness)** — A score from 0 to 1 measuring how "drug-like" a molecule is based on historical data. Higher is better.
* **SA_Score (Synthetic Accessibility)** — An estimate of how difficult it will be for a chemist to physically synthesize the molecule in a lab. Lower scores (e.g., 1-3) mean easier synthesis; higher scores mean it is incredibly difficult or impossible.
* **Binding Affinity (Kd/Ki / pAff)** — A measure of how tightly a ligand binds to a target. Here, higher `pred_pAff_mean` implies stronger predicted binding.

---

## TOPIC: API Endpoints

### Endpoint: `POST /generate`

**What it does:** Submits an asynchronous job to generate molecules, calculate properties, predict affinity, and optionally run docking.
**When to call it:** When you want to trigger a new *de novo* design run to find novel EGFR inhibitors.

**Input Parameters:**

| Parameter | Type | Required | Default | Why It Exists (Scientific Reason) |
|-----------|------|----------|---------|-----------------------------------|
| `preset` | string | No | `egfr_generator` | Dictates the underlying REINVENT generative model to use. |
| `num_molecules` | int | No | 100 | Defines the breadth of chemical space to sample. Bounded between 1 and 5000 to manage compute load. |
| `return_top_k` | int | No | 20 | Filters the vast number of generated structures down to the most promising (sorted by affinity and QED). |
| `docking_mode` | string | No | `off` | Controls whether to run computationally expensive 3D physics simulations (`off`, `top_k`, `all`). |
| `dock_top_k` | int | No | 10 | If `docking_mode` is `top_k`, limits the number of molecules subjected to docking to save GPU time. |

**Response Fields:**

| Field | Type | What It Means |
|-------|------|---------------|
| `job_id` | string | Unique identifier used to poll the job status. |
| `status_url` | string | Endpoint to hit to check if the simulation is running or complete. |

**Example Request:**
```json
{
  "preset": "egfr_generator",
  "num_molecules": 100,
  "return_top_k": 20,
  "docking_mode": "top_k"
}

```

**Example Response:**

```json
{
  "job_id": "gen_20260622_123456_abcdef",
  "status": "queued",
  "message": "Generation job accepted. Poll status_url until completed.",
  "status_url": "http://localhost:8000/jobs/gen_20260622_123456_abcdef",
  "result_url": "http://localhost:8000/jobs/gen_20260622_123456_abcdef/result"
}

```

### Endpoint: `GET /jobs/{job_id}/result`

**What it does:** Retrieves the final processed results of a generation job.
**When to call it:** After polling `GET /jobs/{job_id}` confirms the job `status` is `completed`.

**Input Parameters:** None (passed via URL path).

**Response Fields:**

| Field | Type | What It Means |
| --- | --- | --- |
| `results` | array | Contains the list of generated molecules and their scores. |
| `results[].canonical_smiles` | string | The standardized 2D chemical representation. |
| `results[].pred_pAff_mean` | float | DeepPurpose prediction of binding strength. |
| `results[].docking_score` | float | AutoDock-GPU estimated free energy of binding (kcal/mol). More negative is better. |

*(Note: Additional endpoints include `POST /ligands/export` for converting SMILES to PDB/PDBQT/MOL2, and `POST /reinvent_predict` for internal ML scoring.)*

---

## TOPIC: Internal Pipeline (Step by Step)

Each step below is what happens inside the service when a `POST /generate` request is processed:

1. **Molecule Sampling (REINVENT 4)**
* *What happens:* The `run_reinvent_sampling` function executes a trained neural network checkpoint (`egfr_generator.chkpt`) to generate raw SMILES strings representing novel molecules.
* *Why it's necessary:* This is the core *de novo* engine that constructs new matter rather than searching existing libraries.


2. **Property Calculation & Enrichment**
* *What happens:* `enrich_generated.py` runs RDKit to calculate MW, LogP, TPSA, H-bond donors/acceptors, Rotatable Bonds, QED, and SA_Score. It also calls the DeepPurpose API to predict EGFR binding (`pred_pAff_mean`).
* *Why it's necessary:* Generated molecules are often physically impossible to synthesize or violate Lipinski's Rule of Five. This step filters out pharmacological "junk".


3. **3D Embedding & Conformational Search**
* *What happens:* If docking is enabled, 2D SMILES are converted to 3D structures using RDKit's ETKDGv3 (Experimental-Torsion Knowledge Distance Geometry) and optimized via the UFF force field.
* *Why it's necessary:* Docking requires specific 3D coordinates. A 2D graph (SMILES) cannot be fit into a 3D protein binding pocket.


4. **Ligand Preparation**
* *What happens:* The 3D `.sdf` is converted to a `.pdbqt` file using Meeko (`mk_prepare_ligand.py`), calculating partial charges and identifying rotatable bonds.
* *Why it's necessary:* AutoDock requires a specific file format (PDBQT) that defines the atomic charges and torsions required to simulate molecular flexibility.


5. **Molecular Docking**
* *What happens:* AutoDock-GPU attempts to fit the ligand into the pre-computed EGFR grid (`4WKQ_receptor_v5_SBr.maps.fld`) using genetic algorithms, outputting an Estimated Free Energy of Binding.
* *Why it's necessary:* DeepPurpose (Step 2) is sequence-based and fast. Docking is structure-based, ensuring the molecule physically fits the actual 3D pocket without steric clashes.



---

## TOPIC: User Questions & Answers

**Q: What is the difference between `pred_pAff_mean` and `docking_score`?**
A: `pred_pAff_mean` is a machine-learning prediction generated by DeepPurpose, which looks at the 2D structure of the molecule and the 1D amino acid sequence of EGFR. It's extremely fast but lacks spatial awareness. `docking_score` is a physics-based calculation using AutoDock-GPU that actively tries to fit the 3D shape of your molecule into the 3D shape of the EGFR binding pocket. For `docking_score`, a more negative number (e.g., -11.0 kcal/mol) indicates stronger predicted binding.

**Q: Why does the `docking_mode` parameter exist?**
A: AutoDock-GPU is computationally expensive. Generating 100 molecules and predicting their properties takes seconds, but docking all 100 can take significantly longer. `docking_mode` allows you to save compute time by skipping docking entirely (`off`), only docking the most promising ML-predicted candidates (`top_k`), or brute-forcing the entire set (`all`).

**Q: What do the results mean? How do I interpret them?**
A: Look for molecules with a high `QED` (above 0.3 means it's relatively drug-like) and a low `sa_score` (ideally below 3.5, meaning a chemist can actually synthesize it). From there, prioritize molecules with high `pred_pAff_mean` and highly negative `docking_score` values. Remember that these are *in silico predictions*; a highly ranked molecule still requires experimental validation.

**Q: Why does this take several minutes or hours to run?**
A: While evaluating the ML model (DeepPurpose) is fast, the REINVENT generation phase requires GPU compute to recursively sample chemical space. Furthermore, if you enable docking, the system must generate 3D conformers, assign partial charges, and run extensive physics-based genetic algorithms to find the optimal binding pose for each molecule.

**Q: What are the limitations of this service?**
A: All outputs are computational predictions, not experimental facts. The generated molecules are theoretical. Furthermore, the docking uses a rigid pre-computed grid (`4WKQ`), meaning the simulation assumes the EGFR protein pocket does not move or adapt its shape when the drug binds, which is a simplification of real biology.

**Q: What should I do with the SMILES string?**
A: The SMILES string represents your newly discovered chemical scaffold. You can export its 3D structure (via the `/ligands/export` endpoint) to visualize it in PyMOL/Chimera, use it in a broader chemical similarity search, or hand it off to a synthetic chemist to begin lab preparation.

---

## TOPIC: Limitations & Warnings

* **In silico vs In vitro:** The "Warnings" array in the API explicitly notes that outputs are computational predictions. Experimental validation is required.
* **Rigid Receptor Docking:** The AutoDock-GPU step relies on a pre-generated map file (`4WKQ_receptor_v5_SBr.maps.fld`). It does not account for induced fit or protein flexibility.
* **Mode Collapse Risk:** Generative AI models for chemistry (like REINVENT) can sometimes suffer from mode collapse, producing highly similar or repeating scaffolds if pushed to generate thousands of molecules at once [VERIFY].
* **Synthetic Accessibility Limits:** The `sa_score` is a heuristic estimate. Molecules scored as "easy" may still feature complex stereocenters that challenge real-world synthesis.

---

## TOPIC: Dependencies & Models

| Library / Model | Why It's Used (Not Just What It Is) |
| --- | --- |
| `REINVENT 4` | The core neural network engine for *de novo* molecular generation, capable of navigating chemical space using pre-trained priors. |
| `DeepPurpose` | Used for ultra-fast ligand-target affinity prediction using Morgan Fingerprints and Target Sequence (AAC). It acts as an aggressive filter before expensive docking. |
| `RDKit` | The industry-standard cheminformatics toolkit. Used to canonicalize SMILES, calculate Lipinski/QED properties, and embed 2D graphs into 3D space using ETKDGv3. |
| `Meeko` | Prepares 3D molecules for docking by converting RDKit SDFs into PDBQT formats, ensuring proper charge assignment and rotatable bond definitions. |
| `AutoDock-GPU` | Executes the physics-based conformational search. Used because its CUDA-accelerated genetic algorithm is vastly faster than traditional CPU-bound docking engines. |

---

## TOPIC: Platform Access

**Available on:** [Free / Pro / Enterprise] [VERIFY]
**Usage limits:** Generating >100 molecules or using `docking_mode=all` is computationally intense and is typically rate-limited or gated behind background job queues. Max generation cap is strictly enforced at 5000 molecules per request.
**Restricted features:** Full 3D AutoDock-GPU execution requires an active worker with the `ADGPU_BIN` environment variable correctly set to a CUDA-enabled machine.

---

# SERVICE: Molecular Dynamics (MD) Simulation Service

# FILE: md_simulation_service.md

# DESCRIPTION: Automates the multi-stage pipeline of protein-ligand structure preparation, force field parameterization, explicit solvation, energy minimization, NPT equilibration, and production molecular dynamics simulations using AmberTools and OpenMM.

---

## TOPIC: Overview

**What this service does:**
This service automates the preparation and execution of multi-stage molecular dynamics (MD) simulation workflows for protein-ligand complexes. It accepts raw or partially prepared macromolecular target files and small molecule structures, applies appropriate thermodynamic force fields, solubilizes the system with explicit water molecules and counter-ions, eliminates high-energy steric clashes, equilibrates the system to a specified temperature and pressure, and runs production trajectory loops. It also provides an automated analysis engine to compute geometric and energetic properties over the course of the simulation.

**Why it exists in the platform:**
In the drug discovery pipeline, static molecular docking only provides a snapshot of potential binding configurations and often treats the protein as a rigid object. The MD Simulation Service introduces physical time, temperature, and structural flexibility into the system. This allows researchers to evaluate whether a small molecule ligand will remain stably bound within an active pocket over time, capture induced-fit conformational changes, and observe explicit solvent-mediated interactions, bridging the gap between static *in silico* screening and intensive *in vitro* experimental assays.

**Who uses it:**
Computational biologists, medicinal chemists, and structural drug discovery researchers who need to validate docking hits, analyze target flexibility, or characterize protein-ligand binding mechanisms at an atomistic scale.

---

## TOPIC: Scientific Background

**The core scientific problem this service solves:**
Biological macromolecules and drug-like small molecules are highly dynamic entities whose behaviors are governed by complex thermodynamic landscapes. Determining these behaviors experimentally through techniques such as NMR spectroscopy or time-resolved X-ray crystallography is costly, time-consuming, and difficult to scale. Classical mechanics-based molecular dynamics simulations solve this problem by iteratively calculating the forces acting on every individual atom over small, consecutive increments of time (femtoseconds), producing a detailed trajectory of molecular motion that can be visually and statistically inspected.

**The computational approach used:**
The service uses a hybrid, production-grade toolchain leveraging **AmberTools** binaries for macromolecular topology building, parameterization, and explicit solvation, combined with **OpenMM** for high-performance, GPU-accelerated numerical integration. OpenMM features automatic platform detection (prioritizing CUDA and OpenCL over CPU architectures), allowing simulations involving tens of thousands of explicit water molecules to run efficiently. Trajectory parsing is executed via `pytraj` and `ProLIF` to deliver deep geometric, correlation, and fingerprint diagnostics.

**Key scientific concepts a user needs to understand:**

* **Force Field** — A mathematical model and associated parameters (such as bond lengths, angles, dihedrals, and partial charges) used to calculate the potential energy of a molecular system as a function of its atomic coordinates. This service supports AMBER force fields (`ff14SB` and `ff19SB`) for proteins and GAFF2 for small organic ligands.
* **Explicit Solvation** — Surrounding the protein-ligand complex inside a periodic box with physical, individual water molecules (such as the TIP3P or OPC model). This approach accurately captures critical hydrogen-bonding networks and solvent displacement effects, unlike implicit models that treat water as a uniform mathematical continuum.
* **Energy Minimization** — An optimization step that adjusts atomic coordinates to locate a local potential energy minimum. This eliminates steric clashes or unnatural overlapping distances introduced during file preparation before physical dynamics begin.
* **NPT Ensemble** — A simulation framework where the Number of particles (N), system Pressure (P), and system Temperature (T) are kept constant using specialized barostats and thermostats. This mimics realistic laboratory conditions, enabling the simulation box volume to adjust naturally as the water density changes.
* **RMSD (Root-Mean-Square Deviation)** — A measurement of the average spatial distance between selected atoms (typically protein backbone alpha carbons) across the simulation compared to an initial reference structure. It serves as a metric for tracking structural drift and system equilibration.

---

## TOPIC: API Endpoints

### Endpoint: `POST /process`

**What it does:** Validates input structure files, parses user-specified force fields and simulation parameters, creates a background job workspace, and sends the simulation to an asynchronous execution queue.
**When to call it:** When you want to initiate a new molecular dynamics simulation for a protein-ligand complex.

**Input Parameters:**

Submitted via multipart form data (`request.form` and `request.files`):

| Parameter | Type | Required | Default | Why It Exists (Scientific Reason) |
| --- | --- | --- | --- | --- |
| `protein` | file | Yes | — | Binary or text PDB structure file representing the target protein. |
| `ligand` | file | Yes | — | Binary or text PDB structure file containing the small molecule ligand coordinates. |
| `force_field` | string | No | `"ff19SB"` | Selects either `"ff19SB"` or `"ff14SB"` AMBER parameters to model protein physics. |
| `net_charge` | integer | No | `0` | Specifies the net formal charge of the ligand, which is required for correct AM1-BCC partial charge assignment. |
| `box_size` | float | No | `12.0` | Minimum clearance distance (in Angstroms) from the solute to the edge of the periodic water box boundary. |
| `ion_type` | string | No | `"NaCl"` | Determines the type of salt ions (`"NaCl"` or `"KCl"`) added to neutralize net charges and achieve physiological concentration. |
| `salt_conc` | float | No | `0.15` | The target ionic strength in Molar units, matching physiological salinity conditions. |
| `remove_waters` | boolean | No | `True` | Statically strips existing crystallographic water records from the input structure to ensure clean solvation. |
| `add_hydrogens` | boolean | No | `True` | Uses OpenBabel to assign missing hydrogen atoms to the small organic ligand structure. |
| `equil_time_ns` | float | No | `5.0` | Total duration (in nanoseconds) allocated for restrained pressure and temperature equilibration. |
| `restraint_fc` | float | No | `700.0` | Force constant (in kJ/mol/nm²) applied to restrain macromolecular heavy atoms during equilibration phases. |
| `min_steps` | integer | No | `20000` | Maximum number of iteration steps permitted during initial energy minimization routines. |
| `sim_time_ns` | float | No | `0.1` | Production duration (in nanoseconds) simulated within each sequential loop stride. |
| `n_strides` | integer | No | `1` | Total number of incremental strides to execute during the production stage. |
| `temperature_k` | float | No | `298.0` | Target simulation temperature maintained in Kelvin. |
| `pressure_bar` | float | No | `1.0` | Target simulation pressure maintained in Bar. |
| `dt_fs` | integer | No | `2` | Integration time step size measured in femtoseconds. |
| `savcrd_ps` | integer | No | `10` | Frequency (in picoseconds) at which atomic coordinates are written to the output trajectory files. |
| `print_ps` | integer | No | `10` | Frequency (in picoseconds) at which energy and structural data are written to the simulation logs. |

**Response Fields:**

| Field | Type | What It Means |
| --- | --- | --- |
| `job_id` | string | A unique identifier assigned to track and manage this specific simulation run. |
| `status` | string | Current processing status of the job within the background Celery execution framework (e.g., `"Queued"`). |

**Example Request (cURL command line format):**

```bash
curl -X POST http://localhost:5005/process \
  -F "protein=@receptor.pdb" \
  -F "ligand=@drug_candidate.pdb" \
  -F "force_field=ff19SB" \
  -F "net_charge=-1" \
  -F "n_strides=10" \
  -F "sim_time_ns=1.0"

```

**Example Response:**

```json
{
  "job_id": "4b6ec7f8-9a3d-4c7e-8b1a-2d3e4f5a6b7c",
  "status": "Queued"
}

```

---

### Endpoint: `GET /status/<job_id>`

**What it does:** Returns the current execution status and any available output references for a specific simulation job.
**When to call it:** Periodically (polling) to monitor background simulation progress and retrieve download endpoints once the job finishes.

**Input Parameters:**

| Parameter | Type | Required | Default | Why It Exists (Scientific Reason) |
| --- | --- | --- | --- | --- |
| `job_id` | string | Yes | — | The unique string identifier received when the job was submitted via `/process`. |

**Response Fields:**

| Field | Type | What It Means |
| --- | --- | --- |
| `job_id` | string | Re-states the validated tracking identifier for verification. |
| `status` | string | Detailed execution progress text or state name (e.g., `"Step 3/7 — Energy minimization"`, `"Success"`, or `"Unknown"`). |
| `download_url` | string | *Conditional (Only if complete)*. Relative path used to download the primary simulation results ZIP file. |
| `download_analysis_url` | string | *Conditional (Only if complete)*. Relative path used to download post-simulation analysis data. |
| `pdb_content` | string | *Conditional (Only if complete)*. Text content of the final structural coordinate frame generated by the simulation. |

**Example Request:**

```bash
curl http://localhost:5005/status/4b6ec7f8-9a3d-4c7e-8b1a-2d3e4f5a6b7c

```

**Example Response:**

```json
{
  "job_id": "4b6ec7f8-9a3d-4c7e-8b1a-2d3e4f5a6b7c",
  "status": "Step 3/7 — Energy minimization"
}

```

---

### Endpoint: `GET /download/<job_id>`

**What it does:** Downloads a compressed ZIP archive (`Results.zip`) containing the topology, final coordinates, restart states, and production trajectory files for a completed simulation.
**When to call it:** When `/status/<job_id>` confirms that the job has finished and provides a download URL.

**Input Parameters:**

| Parameter | Type | Required | Default | Why It Exists (Scientific Reason) |
| --- | --- | --- | --- | --- |
| `job_id` | string | Yes | — | Target tracking token mapping to the completed job directory. |

**Response Files Output:** Downloads a standard compressed `application/zip` archive containing files like `*_SYS.prmtop`, `*_SYS.crd`, and `*.dcd` trajectory bundles.

---

### Endpoint: `POST /analyze`

**What it does:** Runs post-simulation trajectory diagnostics, such as RMSD, RMSF, radius of gyration, principal component analysis (PCA), Pearson cross-correlation, interaction energy, and ProLIF interaction fingerprints.
**When to call it:** Once a simulation job completes, use this endpoint to analyze the generated trajectory and evaluate complex stability.

**Input Parameters (JSON Body):**

| Parameter | Type | Required | Default | Why It Exists (Scientific Reason) |
| --- | --- | --- | --- | --- |
| `job_id` | string | Yes | — | Identifies the completed simulation job to analyze. |
| `rmsd_mask` | string | No | `"@CA"` | A `pytraj`-compatible atom selection mask specifying which atoms to include in the RMSD calculation (typically alpha carbons). |
| `cc_mask` | string | No | `"@CA"` | An atom selection mask defining which residues to parse for the Pearson cross-correlation matrix. |
| `skip` | integer | No | `1` | Trajectory frame stride frequency filter. Setting this higher (e.g., `5`) skips intermediate frames to accelerate analysis. |
| `dpi` | integer | No | `300` | Resolution (Dots Per Inch) for output PNG chart graphics. |
| `threshold` | float | No | `0.3` | Minimum interaction occupancy frequency threshold used by ProLIF to filter protein-ligand contact networks. |

**Response Fields:**

| Field | Type | What It Means |
| --- | --- | --- |
| `job_id` | string | Echoes back the assigned job token reference. |
| `download_url` | string | Endpoint path used to fetch the compiled `Analysis.zip` output archive. |
| `outputs` | array | Names of the generated analysis metrics (e.g., `"rmsd"`, `"rmsf"`, `"cross_corr"`, `"prolif"`). |

**Example Request Body:**

```json
{
  "job_id": "4b6ec7f8-9a3d-4c7e-8b1a-2d3e4f5a6b7c",
  "rmsd_mask": "@CA,C,N,O",
  "skip": 2,
  "dpi": 150
}

```

**Example Response:**

```json
{
  "job_id": "4b6ec7f8-9a3d-4c7e-8b1a-2d3e4f5a6b7c",
  "download_url": "/download_analysis/4b6ec7f8-9a3d-4c7e-8b1a-2d3e4f5a6b7c",
  "outputs": ["rmsd", "rmsf", "radgyr", "2d_rmsd", "pca", "cross_corr", "interaction_e"]
}

```

---

### Endpoint: `GET /download_analysis/<job_id>`

**What it does:** Downloads a compressed archive (`Analysis.zip`) containing generated diagnostic charts (PNGs) and data tables (CSVs) from the analysis stage.
**When to call it:** After receiving a successful response from the `/analyze` endpoint.

---

## TOPIC: Internal Pipeline (Step by Step)

When a background execution task begins, the microservice processes the protein-ligand system through the following steps:

1. **Macromolecular Repair via PDBFixer**
* *What happens:* The input protein PDB file is parsed by PDBFixer, which automatically scans for and fills missing residues or sidechain heavy atoms, replaces non-standard residues with standard counterparts, deletes unwanted structural records, and protonates the system at pH 7.0.
* *Why it's necessary:* Crystallized structures often have unresolved loops or missing atoms due to local protein flexibility. Leaving these gaps will cause the force field parameterization step to fail.


2. **Protein Formatting via pdb4amber**
* *What happens:* Runs the AmberTools `pdb4amber` utility to clean and standardize residue nomenclature, append appropriate terminus cap entries, and remove non-standard atomic records.
* *Why it's necessary:* This tool reformats the PDB file to match AMBER naming conventions, ensuring that `tleap` can read and recognize standard amino acids without encountering parsing errors.


3. **Ligand Hydrogen Assignment via OpenBabel**
* *What happens:* If `add_hydrogens` is enabled, OpenBabel evaluates the small organic molecule structure to assign correct chemical protonation states and explicitly add all missing hydrogen atoms.
* *Why it's necessary:* Small molecules must be fully protonated and have correct bond orders specified before calculating their partial charges and electrostatic properties.


4. **Small Molecule Parameterization via Antechamber and parmchk2**
* *What happens:* `antechamber` uses the AM1-BCC semi-empirical charge model to assign point charges and maps ligand topology using General Amber Force Field (GAFF2) atom types. Next, `parmchk2` checks for any missing force field parameters and logs required parameters into an external `.frcmod` file.
* *Why it's necessary:* Standard macromolecular force fields do not contain parameters for novel organic drug molecules. This step dynamically calculates the required parameters to describe the ligand's physics.


5. **System Solvation and Neutralization via tleap**
* *What happens:* Combines the prepared protein and ligand files, loads the specified AMBER parameters (`ff14SB` or `ff19SB`) along with the ligand's `.frcmod` file, adds a periodic solvent box (TIP3P or OPC water), and adds counter-ions (`NaCl` or `KCl`) to neutralize net charges and match the target salinity. It outputs parameter topology (`.prmtop`), coordinate (`.crd`), and reference PDB files.
* *Why it's necessary:* Proteins require an aqueous environment with correct ionic strength to preserve their native structures and binding affinities.


6. **Energy Minimization via OpenMM**
* *What happens:* The system's coordinates are passed into OpenMM, which performs iterative geometric minimization loops to resolve atomic overlaps and steric clashes.
* *Why it's necessary:* Unphysical overlaps or severe steric clashes create high repulsive forces. If these are not relaxed, they can cause the integration math to crash or "explode" at the start of the simulation.


7. **NPT Equilibration via OpenMM**
* *What happens:* The system is heated to the target temperature and adjusted to the target pressure under an NPT ensemble. Harmonic restraints are applied to the solute's heavy atoms during this phase.
* *Why it's necessary:* Restraining the protein and ligand allows the surrounding water molecules to adapt and settle around the solute without distorting the starting binding conformation during initial heating.


8. **Production Stride Loops via OpenMM**
* *What happens:* Removes the harmonic restraints, allowing the entire system to move freely. It runs the simulation in structured blocks called strides, saving an XML state file (`.rst`) at each checkpoint alongside the continuous `.dcd` binary trajectory.
* *Why it's necessary:* This stage generates the actual trajectory data used to observe and analyze protein-ligand dynamics over time.


9. **Results Packaging**
* *What happens:* Collects the topology files, log files, restart configurations, and binary coordinate files, and compiles them into a single `Results.zip` archive inside the job's directory.
* *Why it's necessary:* Consolidates all output files into a single, downloadable archive for downstream analysis or visualization.



---

## TOPIC: User Questions & Answers

**Q: Why does the system default to adding explicit solvent water boxes and ions during preparation?**
A: Biomolecules do not function in a vacuum; their structural configurations, stability, and binding mechanics depend on interactions with the surrounding solvent. Explicit solvation adds individual water molecules to capture these effects, such as hydrogen bonds and water-mediated contacts. Counter-ions (like NaCl or KCl) are added to neutralize net charges and establish physiological salinity (0.15 M by default), which stabilizes electrostatics across periodic boundaries.

**Q: Why does the simulation parameter request include an `n_strides` and `sim_time_ns` per stride value?**
A: Production runs are split into distinct sequential segments called strides to provide checkpointing. For example, setting `n_strides=5` and `sim_time_ns=2.0` results in a total simulation time of 10.0 ns. Each stride saves an independent XML state file (`.rst`). This layout ensures that if a hardware interruption occurs, the simulation can be resumed from the last completed stride rather than restarting from the beginning.

**Q: What do the final results of the post-simulation analysis mean, and how should I interpret the RMSD plots?**
A: Trajectory analysis evaluates the structural stability of the protein-ligand complex over time. Root-Mean-Square Deviation (RMSD) tracks structural drift relative to the initial frame. An RMSD curve that rises and then levels off into a steady plateau indicates that the system has equilibrated into a stable conformation. If the RMSD continues to climb without leveling off, it suggests that the protein is undergoing major structural changes or that the ligand is dissociating from the pocket.

**Q: Why do molecular dynamics simulations take a long time to compute compared to standard molecular docking workflows?**
A: Molecular docking treats the protein as a static target and uses simplified scoring functions to quickly evaluate potential binding poses. In contrast, molecular dynamics simulations calculate explicit Newtonian physics for tens of thousands of individual atoms over millions of steps, using a femtosecond time scale. This rigorous approach requires high-performance GPU acceleration (via CUDA or OpenCL) to run efficiently.

**Q: What are the primary technical limitations of the current implementation of this simulation microservice?**
A: This microservice is designed exclusively for explicit solvent simulations (using TIP3P or OPC models) and does not currently support implicit solvent continuum methods. It is also limited to the AMBER force fields (`ff14SB` and `ff19SB`) for proteins and GAFF2 for ligands. Additionally, it does not support advanced setups such as lipid bilayer membranes, non-standard post-translational modifications, or covalent ligand binding out of the box.

**Q: What does the cross-correlation matrix reveal about the target protein during the post-simulation analysis step?**
A: The cross-correlation matrix measures how much pairs of residues move in tandem across the trajectory. Positive values indicate correlated motion (residues moving in the same direction), which often highlights rigid domain behaviors. Negative values indicate anti-correlated motion (residues moving in opposite directions). This helps researchers map allosteric pathways and long-range structural changes triggered by ligand binding.

---

## TOPIC: Limitations & Warnings

* **Limitation 1 (Fixed Parameter Sets):** The service restricts macromolecular parameters to AMBER `ff14SB` or `ff19SB`, and small molecules to GAFF2. Alternate force field families, such as CHARMM or OPLS, cannot be selected, which may limit compatibility with certain external workflows.
* **Limitation 2 (Explicit Water Overhead):** The pipeline requires explicit solvation boxes. For large macromolecular complexes, the water box can add tens of thousands of atoms, increasing the computational workload and limiting throughput compared to simpler implicit solvent models.
* **Scientific Caveat:** Classical MD force fields rely on fixed atomic point charges (such as the AM1-BCC model used here) and cannot simulate the forming or breaking of covalent bonds. Dynamic chemical processes, such as proton transfer reactions or covalent inhibition mechanisms, cannot be captured without using quantum mechanics (QM/MM) approaches.

---

## TOPIC: Dependencies & Models

| Library / Model | Why It's Used (Not Just What It Is) |
| --- | --- |
| `OpenMM` | Serves as the primary simulation engine, chosen for its high-performance, GPU-accelerated code that automatically leverages CUDA or OpenCL platforms to handle explicit solvent integration efficiently. |
| `AmberTools` | Provides core utilities like `antechamber`, `parmchk2`, and `tleap` to handle automated ligand atom-typing, AM1-BCC charge calculation, parameter validation, and explicit solvation box generation. |
| `PDBFixer` | Automatically inspects input protein PDB files to identify and fix missing residues, heavy atoms, or structural gaps, preventing topology build failures. |
| `pytraj` | An optimized Python interface for the `CPPTRAJ` data processing package, used to rapidly calculate geometric properties across large trajectories, including RMSD, RMSF, and cross-correlation matrices. |
| `ProLIF` | Generates Protein-Ligand Interaction Fingerprints to convert trajectory coordinates into categorical chemical contacts (such as hydrogen bonds, pi-stacking, and ionic interactions) for visualization. |
| `Celery` | An asynchronous task queue used to handle long-running simulation jobs in the background, keeping the main Flask REST API responsive. |

---

## TOPIC: Platform Access

**Available on:** Pro / Enterprise tiers, as explicit solvent MD simulations require dedicated background GPU resources.
**Usage limits:** Managed via background task worker limits (defaults to 4 concurrent pipeline jobs unless configured otherwise). Total trajectory length limits are enforced through maximum values for `n_strides` and `sim_time_ns` parameters.
**Restricted features:** GPU-accelerated computing platforms (CUDA and OpenCL) are restricted to infrastructure tiers equipped with appropriate hardware acceleration.

---

# SERVICE: SciOS Core Orchestrator & Expert Agents Service

# FILE: scios_core_orchestrator.md

# DESCRIPTION: This service orchestrates and routes advanced drug discovery, molecular property predictions, and clinical-grade biomedical queries to specialized expert agent pipelines.

---

## TOPIC: Overview

**What this service does:**
This service functions as the central nervous system of the SciOS Scientific Operating System. It exposes an asynchronous, multi-agent orchestration API that intakes natural language questions from researchers, extracts critical entities (such as compounds, chemical structures, and pathologies), dynamically determines scientific intent, delegates work to isolated computational chemistry and medical reasoning pipelines in parallel, and streams back a unified, language-synthesized response.

**Why it exists in the platform:**
Modern drug discovery requires a unified window into both quantitative chemical computations (such as molecular structure searching and predictive toxicology) and qualitative biological mechanics (such as genomic feasibility and signaling pathways). This service eliminates the silos between computational chemistry tools and biomedical literature. Without it, researchers would have to manually extract data from separate molecular fingerprint indices, machine learning models, and document databases, and then synthesize the connections themselves.

**Who uses it:**
Medicinal chemists, translational medicine researchers, and computational biologists looking for a rapid, intelligent, natural-language interface to evaluate potential therapeutic leads and design drug repurposing pipelines.

---

## TOPIC: Scientific Background

**The core scientific problem this service solves:**
Evaluating an analytical compound for therapeutic potential requires simultaneously analyzing its chemical structural feasibility and its biological interaction profiles. Evaluating these characteristics manually at scale is impossible due to the multi-million compound sizing of molecular libraries and the dense, unstructured nature of medical literature. This service automates the initial phases of virtual screening and automated reasoning.

**The computational approach used:**
The service leverages a Multi-Agent Large Language Model (LLM) router (utilizing models like Qwen3 and Llama 3.3 via high-throughput inference layers) to handle semantic intent categorization.

* To resolve chemical metrics, it hooks into an **In silico ADMET pipeline** powered by Message Passing Neural Networks (MPNNs) that evaluate molecular structures as graph networks.
* For spatial tracking, it delegates molecular search to a **FAISS (Facebook AI Similarity Search)** framework executing Inverse File Frequency (IVF) indexing for ultra-fast, sub-linear vector search across chemical spaces.
* For biological validation, it utilizes a **Medical RAG (Retrieval-Augmented Generation)** strategy built alongside OpenBioLLM prompts to isolate structural biology properties and target pathways without clinical hallucinations.

**Key scientific concepts a user needs to understand:**

* **SMILES (Simplified Molecular Input Line Entry System)** — A linear text notation format that encodes a molecule's exact 2D atom-and-bond topology into a standard ASCII string (e.g., Aspirin is expressed as `CC(=O)Oc1ccccc1C(=O)O`).
* **ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity)** — The five critical pharmacokinetic phases that determine whether a chemical compound can safely enter the human body, travel to its biological destination, break down safely, leave the body, and avoid causing structural organ damage.
* **Drug Repurposing** — The process of identifying new medical indications for existing approved or investigational compounds, saving years of clinical development by leveraging established safety profiles.
* **Mechanism of Action (MoA)** — The specific biochemical pathway and molecular interaction through which a drug compound produces its localized or systemic therapeutic effect on a target pathology.
* **Virtual Screening** — An automated computational simulation technique that evaluates large chemical compound libraries *in silico* to flag candidate structures showing high potential binding capabilities to a disease-linked protein target.

---

## TOPIC: API Endpoints

### Endpoint: `POST /orchestrate`

**What it does:** Receives raw text containing a scientific question, extracts relevant compound names or SMILES signatures, runs appropriate specialized expert agents in parallel, and pipes the compiled data into a final reasoning stream.
**When to call it:** Call this endpoint when you want to execute a comprehensive multi-disciplinary drug discovery query—such as evaluating the toxicity of a custom SMILES structure, finding structural analogs, or assessing whether an existing compound can be repurposed for a new disease target.

**Input Parameters:**

| Parameter | Type | Required | Default | Why It Exists (Scientific Reason) |
| --- | --- | --- | --- | --- |
| `session_id` | string | Yes | — | Tracks the ongoing session token to preserve up to 6 rounds of conversational memory context (e.g., maintaining compound references across follow-up prompts). |
| `user_id` | string | Yes | — | Maps user access credentials for operational security and internal tracking metrics. |
| `text_input` | string | Yes | — | The unstructured natural language scientific question, compound reference, or SMILES data submitted by the investigator. |

**Response Fields:**

Yields an active HTTP token stream (`StreamingResponse` with `text/plain` media type). The output represents consecutive pieces of text synthesizing the multi-agent findings.

**Example Request:**

```json
{
  "session_id": "lab_session_delta_4",
  "user_id": "scientist_71",
  "text_input": "Run virtual screening for Parkinson's disease and evaluate the toxicity metrics of the best candidate."
}

```

**Example Response:**

```text
[Streaming Text Output]
The virtual screening pipeline for Parkinson's disease has identified key target leads. 
Evaluating the primary compound structural analog properties reveals...

```

---

### Endpoint: `GET /`

**What it does:** Serves the core web user interface assets for direct interaction.
**When to call it:** When launching the visual workspace client inside a web browser.

**Input Parameters:**
None.

**Response Fields:**
Returns an `HTMLResponse` containing the raw user interface content.

---

## TOPIC: Internal Pipeline (Step by Step)

When a request arrives at the `/orchestrate` gateway, the following internal processes occur:

1. **Session Memory Recovery**
* *What happens:* The system looks up the provided `session_id` inside an internal memory cache (`SESSION_MEMORY`). It extracts up to the last 6 contextual chat messages.
* *Why it's necessary:* This ensures the orchestrator doesn't lose sight of prior conversational scope, allowing scientists to ask contextual questions like "What is its distribution profile?" without resubmitting structural names.


2. **Intent Classification & Entity Parsing**
* *What happens:* The query text and history are sent to the Orchestrator Brain LLM. The model returns a strictly formatted JSON structure separating the core target intent (`CHEMICAL_SIMILARITY`, `ADMET_ANALYSIS`, `DRUG_REPURPOSING`, `BIOMEDICAL_MECHANISM`, or `APP_HELP`) and a structured entity dictionary containing `compound`, `smiles`, and `disease`.
* *Why it's necessary:* This step transitions unstructured human reasoning into structured parameters required to trigger deep programmatic tools.


3. **Asynchronous Parallel Expert Dispatch**
* *What happens:* The system checks the parsed intent structure and evaluates requirements. If chemical validation is flagged, it instantiates a thread for the `ChemicalAgent`. If disease pathways or repurposing frameworks are flagged, it instantiates a thread for the `MedicalAgent`. Both processes run concurrently using `asyncio.gather`.
* *Why it's necessary:* Running evaluations in parallel reduces first-token latency, ensuring chemical lookups and medical background text parsing occur simultaneously.


4. **Chemical Sub-Pipeline Execution**
* *What happens:* Depending on the routed intent, the `ChemicalAgent` initiates targeted async HTTP calls to independent cluster locations:
* *ADMET:* Calls `ADMET_AI_URL/predict_batch` passing structural SMILES arrays.
* *Virtual Screening:* Calls `DRUG_REPURPOSING_URL/api/v1/screen` passing a targeted pathogenetic entity.
* *Chemical RAG:* Calls `CHEMICAL_AI_URL/search` to perform a FAISS neighborhood analysis on structural vectors.


* *Why it's necessary:* Connects the orchestrator directly to computational microservices that host the 10M+ molecular registry coordinates and MPNN model checkpoints.


5. **Medical Sub-Pipeline Execution**
* *What happens:* The `MedicalAgent` sets its generation parameters to a rigid temperature of `0.0` and queries a specialized domain engine (modeled on OpenBioLLM principles) using precise prompts detailing Drug-Target Interactions, receptor binding, and cellular signaling networks.
* *Why it's necessary:* A temperature configuration of `0.0` maximizes reproducibility and accuracy, preventing the engine from inventing unverified therapeutic associations or mechanism properties.


6. **Unified Synthesis & Secure Streaming Response**
* *What happens:* The individual text logs returned from the Chemical and Medical sub-agents are stitched together into a structured engineering payload (`[Chem Data]` / `[Bio Data]`). This structured block is passed to a final language synthesis pipeline (`temperature=0.3`), which streams the unified answer chunk-by-chunk to the user in their language of choice.
* *Why it's necessary:* Converts disparate computational integers, binding percentages, and literature strings into a cohesive, highly professional scientific summary.



---

## TOPIC: User Questions & Answers

**Q: How does the system determine whether a query belongs to the Chemical Agent or the Medical Agent?**
A: The platform utilizes a central intent orchestrator model that processes user text alongside recent chat context. It maps queries using strict validation guidelines: topics involving exact molecular properties, SMILES parameters, and geometric similarities route to the Chemical Agent, while biological pathways, receptor activities, and disease rationales trigger the Medical Agent. For hybrid queries like drug repurposing, it triggers both agents concurrently.

**Q: Why does the ADMET pipeline require a valid SMILES string instead of a traditional chemical name?**
A: Common or brand names (like "Aspirin") do not map out structural features. The underlying machine learning framework employs Graph Neural Networks (MPNNs) that analyze molecules as exact topological graphs where atoms act as nodes and bonds represent links. A SMILES string maps this exact 2D structure out in text, giving the graph network the exact spatial coordinates required to calculate absorption or toxicological coefficients.

**Q: What do the numerical metrics generated during Virtual Screening represent?**
A: The values generated during screening correspond to calculated binding scores or estimated docking affinities against disease target proteins. Higher numbers signify a stronger structural alignment or calculated binding energy probability within a target protein's functional site. However, these are strictly statistical models and do not guarantee biological efficacy without laboratory verification.

**Q: Why does it sometimes take up to a minute for chemical screening or similarity queries to finish processing?**
A: The service implements a generous 60-second operational timeout window for its internal chemical client calls because virtual screenings and RAG neighbor analysis are highly compute-intensive. Searching across a space of over 10 million registered chemical records via FAISS matrix indices and computing batch properties through multiple neural network graphs requires significant server processing time.

**Q: What are the main limitations of the scientific responses generated by this platform?**
A: All analytical findings provided by this service are entirely *in silico* predictive models. ADMET scores and RAG-driven binding statements are calculated probabilities derived from machine learning datasets and text retrieval blocks. They are not direct substitutes for real-world *in vitro* wet-lab assays, validation cultures, or *in vivo* animal trials. Furthermore, the molecular engines cannot process structural evaluation commands if a molecule cannot be parsed into a clean SMILES string.

**Q: What happens if a server error interrupts an ongoing session midway through execution?**
A: If a severe processing or connection error disrupts the internal execution loops, the core kernel catches the failure, outputs a detailed system debug log to prevent a system crash, and completely clears the local history dictionary for that specific `session_id`. This prevents the system from getting trapped in an unrecoverable loop caused by a corrupted historical state, allowing you to re-initiate the session immediately.

---

## TOPIC: Limitations & Warnings

* **Limitation 1: High Sensitivity to SMILES Parsing errors** — If the orchestrator extracts an incorrect or malformed SMILES string from user input, or if the user includes a typo, the chemical agent cannot execute property predictions, causing the step to fail.
* **Limitation 2: Volatile In-Memory Storage Architecture** — Active session histories are retained inside a standard, local python memory dictionary (`SESSION_MEMORY`). If the underlying web application container experiences a hardware reset, system crash, or restart event, all historical logs are lost.
* **Scientific Caveat: Predictive Screening Disconnect** — High target screening alignment scores or optimal predicted metabolic stats do not equal real-world drug safety or clinical utility. They indicate mathematical likelihoods based on training data and must only be interpreted as computational leads.
* **Structural Intent Mismatch `[VERIFY]**` — The standalone system brain code (`brain.py`) relies on a basic fallback routing architecture that outputs lowercase text intent arrays (`"chemical"`, `"medical"`), whereas the core application loop (`main.py`) enforces strict uppercase intent definitions (`"CHEMICAL_SIMILARITY"`, `"ADMET_ANALYSIS"`). A direct mismatch between these modules could disrupt fallback routing behavior if they are integrated without a mapping layer.

---

## TOPIC: Dependencies & Models

| Library / Model | Why It's Used (Not Just What It Is) |
| --- | --- |
| `FastAPI` | Chosen to establish high-throughput, asynchronous REST endpoints capable of maintaining ongoing HTTP text generation streaming sequences without choking user threads. |
| `AsyncOpenAI` | Serves as the asynchronous client connector linking the core orchestration script to cloud-hosted Groq or Qwen LLM endpoints. |
| `httpx` | An asynchronous HTTP networking client configured with custom 60-second timeouts to connect with remote microservices (ADMET, Repurposing, and Chemical RAG) without causing thread blocking. |
| `Qwen3-32b` `[VERIFY]` | Listed in architecture logs as the primary cognitive orchestration model optimized for sub-300ms first-token responses during zero-shot routing operations. |
| `Llama-3.3-70b-versatile` | Employed as the default orchestrator and language synthesis model via the Groq engine interface configuration. |
| `OpenBioLLM` | Serves as the domain prompt template target for the Medical Agent to guarantee the integration of specialized, clinical-grade biochemical terminology. |
| `FAISS (IVF Indexing)` | Mentioned in system blueprints as the core spatial structural indexing method utilized to run neighborhood lookups across a database of over 10 million chemical combinations. |
| `Graph Neural Networks (MPNNs)` | Mentioned in architectural documentation as the deep learning models used to calculate target ADMET property metrics from molecular inputs. |

---

## TOPIC: Platform Access

**Available on:** Enterprise Lab Infrastructure / Specialized Institutional Pro Tiers `[VERIFY]` (based on the hardware-intensive nature of the integrated 10M+ molecule FAISS indexes, local neural networks, and concurrent multi-agent architecture setup).
**Usage limits:** Restricted by target API rate parameters enforced on the underlying Groq or Qwen token endpoints, along with the max queue limits of the backend ADMET prediction nodes.
**Restricted features:** Automated audio processing functions (such as Whisper Large V3 Turbo and Kokoro / XTTS-v2 speech-to-speech tools) are isolated as utility files (`app/utils/audio.py`) and are not accessible through the core `/orchestrate` REST gateway.



