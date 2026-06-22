import httpx
from app.config import settings

class ChemicalAgent:
    def __init__(self):
        self.headers = {"accept": "application/json", "Content-Type": "application/json"}
        # if settings.HF_TOKEN:
        #     self.headers["Authorization"] = f"Bearer {settings.HF_TOKEN}"

    async def run(self, intent: str, entities: dict) -> str:
        # استخراج المتغيرات من الـ Orchestrator
        compound = entities.get("compound", "")
        smiles = entities.get("smiles", compound) 
        disease = entities.get("disease", "")

        async with httpx.AsyncClient(timeout=60.0) as client: # زيادة الـ timeout لأن الـ screening بياخد وقت
            try:
                #  1. ADMET (Predict Batch)
                if any(k in intent for k in ["admet", "toxicity", "property", "poison", "absorption"]):
                    if not smiles:
                        return "[Chemical Agent] خطأ: يرجى تزويدي بصيغة SMILES لحساب خصائص الـ ADMET."
                    
                    url = f"{settings.ADMET_AI_URL.rstrip('/')}/predict_batch"
                    payload = {"smiles_list": [smiles]}
                    
                    response = await client.post(url, json=payload, headers=self.headers)
                    if response.status_code == 200:
                        data = response.json()
                        res = data.get("results", [{}])[0]
                        pred = res.get("predictions", {})
                        
                        return (
                            f"[ADMET Analysis for {res.get('smiles')}]:\n"
                            f"• Absorption: {pred.get('Absorption'):.4f}\n"
                            f"• Distribution: {pred.get('Distribution'):.4f}\n"
                            f"• Metabolism: {pred.get('Metabolism'):.4f}\n"
                            f"• Excretion: {pred.get('Excretion'):.4f}\n"
                            f"• Toxicity: {pred.get('Toxicity'):.4f}\n"
                            f"Processing Time: {data.get('processing_time_ms'):.2f}ms"
                        )
                    return f"[Chemical Agent Error] فشل الاتصال بـ ADMET Space. كود: {response.status_code}"

                #  2.(Virtual Screening Pipeline)
                elif any(k in intent for k in ["repurposing", "repurpose", "screen", "target"]):
                    target_disease = disease if disease else compound # Fallback لو الـ LLM استخرج اسم المرض في خانة الـ compound
                    if not target_disease:
                        return "[Chemical Agent] خطأ: يرجى تحديد اسم المرض (Disease Name) لبدء عملية الـ Virtual Screening."
                    
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
                        
                        report = (
                            f"[Virtual Screening Results for {data.get('disease_name')}]:\n"
                            f"• Total Targets Found: {data.get('total_targets_found')}\n"
                            f"• Total Drugs Screened: {data.get('total_drugs_screened')}\n"
                            f"• Top Repurposing Candidates:\n"
                        )
                        for idx, cand in enumerate(candidates[:3], 1): # عرض أعلى 3 نتائج
                            report += f"  {idx}. Drug: {cand.get('drug_name')} -> Target: {cand.get('target_symbol')} (Binding Score: {cand.get('binding_score'):.4f}, Status: {cand.get('status')})\n"
                        return report
                    return f"[Chemical Agent Error] فشل الاتصال بـ Drug Repurposing Space. كود: {response.status_code}"

                #  3.Chemical RAG (Full Rag / Retrieval Only)
                else:
                    if not smiles:
                        return "[Chemical Agent] خطأ: يرجى تزويدي بصيغة SMILES لإجراء البحث عن مركبات شبيهة."
                        
                    endpoint = "/search/full-rag" if "explain" in intent or "detailed" in intent else "/search/retrieval-only"
                    url = f"{settings.CHEMICAL_AI_URL.rstrip('/')}{endpoint}"
                    payload = {"smiles": smiles, "top_k": 3, "explain": True}
                    
                    response = await client.post(url, json=payload, headers=self.headers)
                    if response.status_code == 200:
                        data = response.json()
                        results = data.get("results", [])
                        
                        report = f"Chemical RAG found {data.get('total_results', 0)} similar compounds for '{data.get('query_smiles')}':\n"
                        for idx, res in enumerate(results, 1):
                            report += f"  {idx}. Name: {res.get('name')} (CID: {res.get('cid')}), Similarity: {res.get('similarity_score'):.4f}\n"
                            if res.get("explanation"):
                                report += f"     Explanation: {res.get('explanation')}\n"
                        return report
                    return f"[Chemical Agent Error] فشل الاتصال بـ Chemical RAG Space. كود: {response.status_code}"

            except httpx.RequestError as exc:
                return f"[Chemical Agent Exception] حدث خطأ أثناء الاتصال بالخوادم السحابية: {exc}"