import os
import re
import json
import time
import urllib.request
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


class DINSBaselineEvaluator:
    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        benchmark_json_path: str = "/Developer/wildfire_ai/SBFire_SJSU-20260730T030001Z-1-001/benchmark_results/dins_vlm_benchmark_50.json",
        output_dir: str = "/Developer/wildfire_ai/SBFire_SJSU-20260730T030001Z-1-001/benchmark_results",
    ):
        self.model_name = model_name or os.environ.get("LLM_MODEL", "MiniMax-M3")
        self.is_claude = self.model_name.lower().startswith("claude")
        self.is_qwen = "qwen" in self.model_name.lower()

        if self.is_claude:
            self.endpoint_url = endpoint_url or "https://api.anthropic.com/v1/messages"
            self.api_key = api_key or self._get_claude_api_key()
        elif self.is_qwen:
            default_qwen_url = os.environ.get("TB_REMOTE_QWEN_URL", "http://REMOTE_HOST:28000/v1")
            if not default_qwen_url.endswith("/chat/completions"):
                default_qwen_url = default_qwen_url.rstrip("/") + "/chat/completions"
            self.endpoint_url = endpoint_url or default_qwen_url
            self.api_key = api_key or os.environ.get("QWEN_REMOTE_KEY", "REDACTED_KEY")
        else:
            self.endpoint_url = endpoint_url or os.environ.get("LLM_ENDPOINT", "https://api.minimaxi.com/v1/chat/completions")
            self.api_key = api_key or os.environ.get("MINIMAX_API_KEY", "")

        self.benchmark_json_path = Path(benchmark_json_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_claude_api_key(self) -> str:
        k = os.environ.get("CLAUDE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
        if not k:
            try:
                with open(os.path.expanduser("~/.bashrc")) as f:
                    for line in f:
                        if "CLAUDE_API_KEY" in line and "=" in line:
                            k = line.split("=")[1].strip().strip("'\"")
                            break
            except Exception:
                pass
        return k

    def _format_prompt(self, item: Dict[str, Any]) -> str:
        prompt = f"""You are an expert wildfire structural protection and fire forensic analyst.
Analyze the structural attributes and physical wildfire environment of the following residential building in a California Wildland-Urban Interface (WUI) fire:

[Building Physical Attributes]
- Structure Type: {item.get('structure_type', 'Single Family Residence')}
- Roof Material: {item.get('roof_construction', 'Unknown')}
- Exterior Siding: {item.get('exterior_siding', 'Unknown')}
- Eaves Design: {item.get('eaves', 'Unknown')}
- Vent Screening: {item.get('vent_screen', 'Unknown')}
- Ground Deck / Porch: {item.get('deck_on_grade', 'Unknown')}
- Elevated Deck: {item.get('deck_elevated', 'Unknown')}
- Attached Fence: {item.get('fence_attached', 'Unknown')}
- Year Built: {item.get('year_built', 'Unknown')}

[Wildfire Incident Environment]
- Fire Incident: {item.get('incident', 'Wildfire')} Fire ({item.get('county', 'California')} County, CA)
- Topography / Location: Lat {item.get('latitude')}, Lon {item.get('longitude')}
- Weather & Fuel: Red Flag Warning conditions, wind gusts 25-35 mph, dead fuel moisture (DFM) < 6%, surrounding chaparral/timber brush.

[Requirements]
Provide your final assessment immediately on the first three lines in this exact format:
PREDICTION: Destroyed (or Survived)
PRIMARY_VULNERABILITY: Siding (or Roof, Deck, Vent, Eaves, Fence, None)
FAILURE_MECHANISM: Embers (or Direct Flame, Radiant Heat, None)

Followed by a concise 1-2 paragraph physical vulnerability and fire spread analysis.
"""
        return prompt

    def query_model(self, prompt: str) -> Dict[str, Any]:
        """Query LLM/VLM API endpoint with optional Bearer or Anthropic auth."""
        t0 = time.perf_counter()
        try:
            if self.is_claude:
                payload = {
                    "model": self.model_name,
                    "max_tokens": 1500,
                    "messages": [{"role": "user", "content": prompt}],
                }
                data = json.dumps(payload).encode("utf-8")
                headers = {
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                }
                req = urllib.request.Request(self.endpoint_url, data=data, headers=headers)
                with urllib.request.urlopen(req, timeout=60) as resp:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    latency = round(time.perf_counter() - t0, 2)
                    text_parts = []
                    thinking_parts = []
                    for block in res_json.get("content", []):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "thinking":
                            thinking_parts.append(block.get("thinking", ""))
                    if thinking_parts:
                        full_text = f"<think>\n{''.join(thinking_parts)}\n</think>\n\n" + "\n".join(text_parts)
                    else:
                        full_text = "\n".join(text_parts)
                    return {"success": True, "text": full_text, "latency": latency}
            else:
                payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1500,
                    "temperature": 0.1,
                }
                data = json.dumps(payload).encode("utf-8")
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"

                req = urllib.request.Request(self.endpoint_url, data=data, headers=headers)
                with urllib.request.urlopen(req, timeout=60) as resp:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    latency = round(time.perf_counter() - t0, 2)
                    choice = res_json.get("choices", [{}])[0]
                    msg = choice.get("message", {})
                    content = msg.get("content") or ""
                    reasoning = msg.get("reasoning") or ""
                    full_text = (reasoning + "\n" + content) if reasoning else content
                    return {"success": True, "text": full_text, "latency": latency}
        except Exception as e:
            return {"success": False, "error": str(e), "latency": round(time.perf_counter() - t0, 2)}

    def parse_prediction(self, response_text: str) -> Dict[str, str]:
        """Parse structured prediction labels and thinking trace from response."""
        pred = "Unknown"
        vuln = "Unknown"
        mech = "Unknown"
        thinking_trace = ""
        analysis_text = ""

        # Extract thinking trace if present
        m_think = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL | re.IGNORECASE)
        if m_think:
            thinking_trace = m_think.group(1).strip()
            # The body after </think>
            analysis_text = response_text[m_think.end():].strip()
        else:
            analysis_text = response_text.strip()

        # Match PREDICTION
        m_pred = re.search(r"PREDICTION:\s*\[?(Destroyed|Survived)\]?", response_text, re.IGNORECASE)
        if m_pred:
            pred = "Destroyed" if "destroyed" in m_pred.group(1).lower() else "Survived"
        else:
            if "destroyed" in response_text.lower() and "survived" not in response_text.lower():
                pred = "Destroyed"
            elif "survived" in response_text.lower():
                pred = "Survived"

        # Match PRIMARY_VULNERABILITY
        m_vuln = re.search(r"PRIMARY_VULNERABILITY:\s*\[?([A-Za-z\s]+)\]?", response_text, re.IGNORECASE)
        if m_vuln:
            vuln_val = m_vuln.group(1).strip()
            # Clean up extra words if any
            for candidate in ["Roof", "Deck", "Vent", "Eaves", "Siding", "Fence", "None"]:
                if candidate.lower() in vuln_val.lower():
                    vuln = candidate
                    break
            if vuln == "Unknown":
                vuln = vuln_val.split()[0] if vuln_val else "Unknown"

        # Match FAILURE_MECHANISM
        m_mech = re.search(r"FAILURE_MECHANISM:\s*\[?([A-Za-z\s]+)\]?", response_text, re.IGNORECASE)
        if m_mech:
            mech_val = m_mech.group(1).strip()
            for candidate in ["Embers", "Direct Flame", "Radiant Heat", "None"]:
                if candidate.lower() in mech_val.lower():
                    mech = candidate
                    break
            if mech == "Unknown":
                mech = mech_val.split()[0] if mech_val else "Unknown"

        return {
            "predicted_survival": pred,
            "predicted_vulnerability": vuln,
            "predicted_mechanism": mech,
            "thinking_trace": thinking_trace,
            "analysis_text": analysis_text[:800],
        }

    def _eval_single_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single item."""
        case_id = item["id"]
        gt_survival = item["survival_label"]
        gt_locus = item.get("ignition_point", "None")
        prompt = self._format_prompt(item)

        res = self.query_model(prompt)
        if not res["success"]:
            return {
                "case_id": case_id,
                "success": False,
                "error": res.get("error"),
                "gt_survival": gt_survival,
            }

        parsed = self.parse_prediction(res["text"])
        pred_survival = parsed["predicted_survival"]
        pred_vuln = parsed["predicted_vulnerability"]

        is_correct = (pred_survival == gt_survival)

        # Locus match
        locus_match = False
        if gt_locus and gt_locus.lower() not in ["unknown", "none"]:
            if gt_locus.lower() in pred_vuln.lower() or pred_vuln.lower() in gt_locus.lower():
                locus_match = True

        # Physical hallucination detection:
        # e.g., Model claims roof failed when roof was Class A tile/metal/asphalt and ground truth locus was deck/vents
        hallucinated = False
        roof_material = str(item.get("roof_construction", "")).lower()
        if "roof" in pred_vuln.lower() and ("tile" in roof_material or "asphalt" in roof_material or "metal" in roof_material):
            if gt_locus.lower() not in ["roof", "unknown"]:
                hallucinated = True

        return {
            "case_id": case_id,
            "success": True,
            "incident": item["incident"],
            "structure_type": item["structure_type"],
            "ground_truth": {
                "survival": gt_survival,
                "raw_damage": item["raw_damage"],
                "ignition_point": gt_locus,
                "fire_source": item.get("ignition_source"),
                "defensive_actions": item.get("defensive_actions"),
                "roof": item.get("roof_construction"),
                "vents": item.get("vent_screen"),
                "deck": item.get("deck_on_grade"),
            },
            "model_output": {
                "predicted_survival": pred_survival,
                "predicted_vulnerability": pred_vuln,
                "predicted_mechanism": parsed["predicted_mechanism"],
                "is_survival_correct": is_correct,
                "locus_match": locus_match,
                "hallucinated_feature": hallucinated,
                "latency_sec": res["latency"],
                "thinking_trace_snippet": parsed["thinking_trace"][:500],
                "analysis_text": parsed["analysis_text"][:500],
            },
        }

    def evaluate_benchmark(self, max_cases: int = 50, workers: int = 4) -> Dict[str, Any]:
        """Run full evaluation across benchmark cases using concurrent workers."""
        with open(self.benchmark_json_path, "r", encoding="utf-8") as f:
            catalog = json.load(f)

        items = catalog.get("items", [])[:max_cases]
        print(f"[*] Starting Baseline Evaluation: Model = {self.model_name}, Total Cases = {len(items)}, Workers = {workers}")

        results = []
        tp, fp, tn, fn = 0, 0, 0, 0
        locus_correct = 0
        hallucination_cases = 0

        completed_count = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_item = {executor.submit(self._eval_single_item, item): item for item in items}
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                completed_count += 1
                try:
                    res = future.result()
                    if not res.get("success", False):
                        print(f"[{completed_count}/{len(items)}] Case {item['id']} FAILED: {res.get('error')}")
                        continue

                    mo = res["model_output"]
                    gt = res["ground_truth"]
                    is_correct = mo["is_survival_correct"]

                    if gt["survival"] == "Destroyed":
                        if mo["predicted_survival"] == "Destroyed":
                            tp += 1
                        else:
                            fn += 1
                    else:
                        if mo["predicted_survival"] == "Survived":
                            tn += 1
                        else:
                            fp += 1

                    if mo["locus_match"]:
                        locus_correct += 1
                    if mo["hallucinated_feature"]:
                        hallucination_cases += 1

                    results.append(res)
                    print(
                        f"[{completed_count:>2}/{len(items)}] Case {res['case_id']}: "
                        f"GT={gt['survival']:<9} | Pred={mo['predicted_survival']:<9} | "
                        f"Vuln={mo['predicted_vulnerability']:<7} | "
                        f"{'✓ Correct' if is_correct else '✗ Miss'} ({mo['latency_sec']}s)"
                    )
                except Exception as exc:
                    print(f"[{completed_count}/{len(items)}] Case {item['id']} generated exception: {exc}")

        total = tp + fp + tn + fn
        accuracy = round((tp + tn) / total, 4) if total > 0 else 0.0
        precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0
        fp_rate = round(fp / (tn + fp), 4) if (tn + fp) > 0 else 0.0
        fn_rate = round(fn / (tp + fn), 4) if (tp + fn) > 0 else 0.0

        eval_summary = {
            "model_tested": self.model_name,
            "benchmark_dataset": "WildfireReason-DINS-50",
            "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_evaluated": total,
            "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
            "metrics": {
                "accuracy": accuracy,
                "precision_destroyed": precision,
                "recall_destroyed": recall,
                "f1_score": f1,
                "false_alarm_rate_FP": fp_rate,
                "miss_rate_FN": fn_rate,
                "locus_attribution_accuracy": round(locus_correct / max(tp, 1), 4),
                "hallucination_rate": round(hallucination_cases / max(total, 1), 4),
            },
            "cases": results,
        }

        # Save results to internal folder
        model_slug = self.model_name.lower().replace("-", "_").replace(".", "_")
        results_path = self.output_dir / f"baseline_evaluation_{model_slug}.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(eval_summary, f, indent=2)

        print("\n" + "=" * 60)
        print(f"[*] BASELINE BENCHMARK EVALUATION SUMMARY: {self.model_name}")
        print("=" * 60)
        print(f"  • Total Evaluated Cases:  {total}")
        print(f"  • Accuracy:               {accuracy * 100:.2f}%")
        print(f"  • Precision (Destroyed):  {precision * 100:.2f}%")
        print(f"  • Recall (Destroyed):     {recall * 100:.2f}%")
        print(f"  • F1-Score:               {f1:.4f}")
        print(f"  • Confusion Matrix:       TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        print(f"  • 'Everything Burns' FP:  {fp}/{tn+fp} ({fp_rate * 100:.1f}%) [False Alarm Rate]")
        print(f"  • Locus Attribution Acc:  {locus_correct}/{max(tp, 1)} ({eval_summary['metrics']['locus_attribution_accuracy'] * 100:.1f}%)")
        print(f"  • Feature Hallucination:  {hallucination_cases}/{total} ({eval_summary['metrics']['hallucination_rate'] * 100:.1f}%)")
        print(f"  • Saved JSON Artifact:    {results_path}")
        print("=" * 60 + "\n")

        return eval_summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Wildfire VLM/LLM Baseline Evaluation")
    parser.add_argument("--model", type=str, default=None, help="Model name (e.g. claude-haiku-5-5, MiniMax-M3)")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent worker threads")
    parser.add_argument("--max-cases", type=int, default=50, help="Maximum benchmark cases to evaluate")
    args = parser.parse_args()

    evaluator = DINSBaselineEvaluator(model_name=args.model)
    evaluator.evaluate_benchmark(max_cases=args.max_cases, workers=args.workers)


