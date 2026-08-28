import httpx
from app.config import settings

def _fmt(val, decimals=4) -> str:
    """Format numeric values safely without raising TypeError on None."""
    if isinstance(val, (int, float)):
        return f"{val:.{decimals}f}"
    return str(val) if val is not None else "N/A"


class ChemicalAgent:
    def __init__(self):
        self.headers = {"accept": "application/json", "Content-Type": "application/json"}
        # if settings.HF_TOKEN:
        #     self.headers["Authorization"] = f"Bearer {settings.HF_TOKEN}"

    async def run(self, intent: str, entities: dict) -> str:
        # Extract extracted entity variables from the Orchestrator routing output
        compound = entities.get("compound", "")
        smiles = entities.get("smiles", compound) 
        disease = entities.get("disease", "")

        # Use an increased timeout (60.0s) because virtual screening tasks can be compute-intensive
        async with httpx.AsyncClient(timeout=60.0) as client: 
            try:
                # 1. ADMET Pipeline (Batch Prediction)
                if any(k in intent for k in ["admet", "toxicity", "property", "poison", "absorption"]):
                    if not smiles:
                        return "[Chemical Agent] Error: Please provide a valid SMILES string to calculate ADMET properties."
                    
                    url = f"{settings.ADMET_AI_URL.rstrip('/')}/predict_batch"
                    payload = {"smiles_list": [smiles]}
                    
                    response = await client.post(url, json=payload, headers=self.headers)
                    if response.status_code == 200:
                        data = response.json()
                        res = data.get("results", [{}])[0]
                        pred = res.get("predictions", {})
                        
                        return (
                            f"[ADMET Analysis for {res.get('smiles', smiles)}]:\n"
                            f"• Absorption: {_fmt(pred.get('Absorption'))}\n"
                            f"• Distribution: {_fmt(pred.get('Distribution'))}\n"
                            f"• Metabolism: {_fmt(pred.get('Metabolism'))}\n"
                            f"• Excretion: {_fmt(pred.get('Excretion'))}\n"
                            f"• Toxicity: {_fmt(pred.get('Toxicity'))}\n"
                            f"Processing Time: {_fmt(data.get('processing_time_ms'), 2)}ms"
                        )
                    return f"[Chemical Agent Error] Failed to connect to ADMET Space. Status code: {response.status_code}"

                # 2. Virtual Screening Pipeline (Payload trimmed to optimize token efficiency)
                elif any(k in intent for k in ["repurposing", "repurpose", "screen", "target"]):
                    # Fallback if the LLM orchestrator extracted the disease name inside the compound key
                    target_disease = disease if disease else compound 
                    if not target_disease:
                        return "[Chemical Agent] Error: Please specify a target Disease Name to initiate the Virtual Screening pipeline."
                    
                    url = f"{settings.DRUG_REPURPOSING_URL.rstrip('/')}/api/v1/screen"
                    payload = {
                        "disease_name": target_disease,
                        "min_score": 0,
                        "top_n_targets": 5
                    }
                    
                    response = await client.post(url, json=payload, headers=self.headers)
                    if response.status_code == 200:
                        data = response.json()
                        candidates = data.get("top_candidates", [])
                        
                        # Compress screening results into a concise, token-saving single-line text block
                        report = f"Screening '{data.get('disease_name', target_disease)}': "
                        for idx, cand in enumerate(candidates[:3], 1): # Process top 3 candidates only
                            report += f"[{idx}] {cand.get('drug_name', 'Unknown')}->{cand.get('target_symbol', 'Unknown')} (Score: {_fmt(cand.get('binding_score'), 2)}). "
                        return report
                        
                    return f"[Chemical Agent Error] Failed to connect to Drug Repurposing Space. Status code: {response.status_code}"

                # 3. Chemical RAG Pipeline (Full RAG or Retrieval-Only based on context resolution)
                else:
                    if not smiles:
                        return "[Chemical Agent] Error: Please provide a valid SMILES string to execute similarity search queries."
                    
                    endpoint = "/search/full-rag" if "explain" in intent or "detailed" in intent else "/search/retrieval-only"
                    url = f"{settings.CHEMICAL_AI_URL.rstrip('/')}{endpoint}"
                    payload = {"smiles": smiles, "top_k": 3, "explain": True}
                    
                    response = await client.post(url, json=payload, headers=self.headers)
                    if response.status_code == 200:
                        data = response.json()
                        results = data.get("results", [])
                        
                        # Strip raw image URLs, redundant structural metadata, and CIDs to optimize tokens
                        report = f"Query '{data.get('query_smiles', smiles)}': "
                        for idx, res in enumerate(results[:3], 1): # Process top 3 structurally similar compounds only
                            report += f"[{idx}] {res.get('name', 'Compound')} (Sim: {_fmt(res.get('similarity_score'), 2)}). "
                            if res.get("explanation"):
                                report += f"Note: {res.get('explanation')} "
                        return report
                        
                    return f"[Chemical Agent Error] Failed to connect to Chemical RAG Space. Status code: {response.status_code}"

            except httpx.RequestError as exc:
                return f"[Chemical Agent Exception] A network communication error occurred with remote cloud spaces: {exc}"