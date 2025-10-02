#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import math
import time
import pickle
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------- Verbose switch -----------------------------
VERBOSE = True  # use --quiet to disable

# ----------------------------- Keys -----------------------------
PHQ8_KEYS = [
    "PHQ8_NoInterest","PHQ8_Depressed","PHQ8_Sleep","PHQ8_Tired",
    "PHQ8_Appetite","PHQ8_Failure","PHQ8_Concentrating","PHQ8_Moving"
]

# ----------------------------- Utils -----------------------------
def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _log(msg: str):
    if VERBOSE:
        print(f"[{_now()}] {msg}")

def _l2norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def _strip_json_block(s: str) -> str:
    t = s.strip()
    if "<answer>" in t and "</answer>" in t:
        t = t.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    if t.startswith("```json"):
        t = t[len("```json"):].strip()
        if t.endswith("```"):
            t = t[:-3].strip()
    elif t.startswith("```"):
        t = t[len("```"):].strip()
        if t.endswith("```"):
            t = t[:-3].strip()
    if "{" in t and "}" in t:
        t = t[t.find("{"): t.rfind("}") + 1]
    t = t.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    t = re.sub(r",\s*([}\]])", r"\1", t)
    return t

def _tolerant_fixups(s: str) -> str:
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = re.sub(r",\s*([}\]])", r"\1", s)
    # Fix common Appetite spillover: move reason/score back inside object
    s = re.sub(
        r'("PHQ8_Appetite"\s*:\s*\{\s*"evidence"\s*:\s*(?:\[[^\]]*\]|"[^"]*"|\'[^\']*\')\s*\})\s*,\s*"reason"\s*:\s*(".*?"|\'.*?\')\s*,\s*"score"\s*:\s*("N/A"|[0-3])\s*([},])',
        r'\1, "reason": \2, "score": \3\4',
        s,
        flags=re.DOTALL
    )
    return s

def _normalize_item(v: Any) -> Dict[str, Any]:
    if not isinstance(v, dict):
        v = {}
    ev = v.get("evidence", "No relevant evidence found")
    if isinstance(ev, list):
        ev = [str(x).strip() for x in ev if str(x).strip()]
        ev_str = ev[0] if ev else "No relevant evidence found"
    else:
        ev_str = str(ev).strip() or "No relevant evidence found"
    reason = v.get("reason", "")
    if not isinstance(reason, str):
        reason = str(reason)
    score = v.get("score", "N/A")
    if isinstance(score, str):
        s = score.strip().upper()
        if s == "N/A":
            score = "N/A"
        else:
            try:
                score = int(s)
            except Exception:
                score = "N/A"
    if isinstance(score, int):
        score = max(0, min(3, score))
    elif score != "N/A":
        score = "N/A"
    return {"evidence": ev_str, "reason": reason, "score": score}

def _validate_and_normalize(obj: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in PHQ8_KEYS:
        out[k] = _normalize_item(obj.get(k, {}))
    return out

def _compute_total_and_severity(res: Dict[str, Any]) -> Tuple[int, str]:
    total = 0
    for k in PHQ8_KEYS:
        sc = res[k].get("score", "N/A")
        if isinstance(sc, int):
            total += sc
    if total <= 4:
        sev = "minimal"
    elif total <= 9:
        sev = "mild"
    elif total <= 14:
        sev = "moderate"
    elif total <= 19:
        sev = "mod_severe"
    else:
        sev = "severe"
    return total, sev

# ----------------------------- Ollama clients -----------------------------
def ollama_chat(host: str, model: str, system_prompt: str, user_prompt: str, timeout=180) -> str:
    url = f"http://{host}:11434/api/chat"
    payload = {
        "model": model,
        "messages": (
            [{"role": "system", "content": system_prompt}] if system_prompt else []
        ) + [{"role": "user", "content": user_prompt}],
        "stream": False,
        "options": {"temperature": 0.2, "top_k": 20, "top_p": 0.8}
    }
    _log(f"[CHAT] host={host} model={model}")
    _log("[CHAT] System prompt (exact):\n" + (system_prompt or "(empty)"))
    _log("[CHAT] User prompt (exact):\n" + user_prompt)
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    content = r.json()["message"]["content"]
    _log("[CHAT] Raw model output (first 1200 chars):\n" + content[:1200])
    return content

def ollama_embed(host: str, model: str, text: str, dim: Optional[int] = None, timeout=120) -> List[float]:
    url = f"http://{host}:11434/api/embeddings"
    r = requests.post(url, json={"model": model, "prompt": text}, timeout=timeout)
    r.raise_for_status()
    emb = r.json()["embedding"]
    if dim is not None:
        emb = emb[:dim]
    n = math.sqrt(sum(x * x for x in emb))
    if n > 0:
        emb = [x / n for x in emb]
    _log(f"[EMB] host={host} model={model} dim={len(emb)} text_len={len(text)}")
    _log(f"[EMB] head: {emb[:8]}")
    return emb

# ----------------------------- Prompts (unchanged) -----------------------------
EVIDENCE_EXTRACT_PROMPT = """Analyze the following therapy transcript and extract specific text chunks that provide evidence for each PHQ-8 domain. Based on the evidence found, determine the appropriate PHQ-8 score for each domain.

PHQ-8 scoring reference:
- 0 = not at all (0-1 days)
- 1 = several days (2-6 days)
- 2 = more than half the days (7-11 days)
- 3 = nearly every day (12-14 days)

For each domain below, extract direct quotes from the transcript that indicate the presence, absence, frequency, or severity of symptoms. Then determine the most appropriate score based on the evidence.

PHQ-8 domains:
- nointerest: little interest or pleasure in activities
- depressed: feeling down, depressed, or hopeless
- sleep: sleep problems (trouble falling/staying asleep or sleeping too much)
- tired: feeling tired or having little energy
- appetite: appetite changes (poor appetite or overeating)
- failure: negative self-perception or feeling like a failure
- concentrating: trouble concentrating on tasks
- moving: psychomotor changes (moving/speaking slowly or restlessness)

Return a JSON object with arrays of relevant transcript quotes for each domain. Each quote should be a direct excerpt from the transcript that supports your scoring decision. Include as many relevant quotes as you find - this could range from 0 quotes (if no evidence) to multiple quotes per domain.

Therapy transcript:
{transcript}

Respond with valid JSON matching this structure:
{{
    "PHQ8_NoInterest": ["evidence_1", "evidence_2"],
    "PHQ8_Depressed": ["evidence_1"],
    "PHQ8_Sleep": ["evidence_1", "evidence_2"],
    "PHQ8_Tired": ["evidence_1"],
    "PHQ8_Appetite": [],
    "PHQ8_Failure": ["evidence_1"],
    "PHQ8_Concentrating": ["evidence_1"],
    "PHQ8_Moving": ["evidence_1"]
}}

Important: Extract UNIQUE quotes only and do not reformat them.
"""

SYSTEM_PROMPT = """You are a AI psychiatrist assessment tool specialized in analyzing interview transcripts to evaluate PHQ-8 (Patient Health Questionnaire-8) scores. The PHQ-8 is a validated depression screening instrument that assesses eight of the nine DSM-IV criteria for depressive disorders over the past 2 weeks.

PHQ-8 Scoring System:
- 0 = Not at all (0-1 days in the past 2 weeks)
- 1 = Several days (2-6 days)
- 2 = More than half the days (7-11 days)
- 3 = Nearly every day (12-14 days)

Key Principles:
1. Base all assessments strictly on evidence from the transcript
2. When no relevant evidence exists for a symptom, output "N/A" rather than assuming absence (score of 0)
3. Consider both explicit statements and behavioral indicators in the interview
4. Be conservative - only assign scores when there is clear evidence of frequency
5. Remember that absence of mention does not necessarily mean absence of symptom
6. When reference examples are provided, use them for cross-analysis to inform scoring decisions
7. Weight reference examples based on their similarity to the current evidence - highly similar references should be considered more heavily
8. Follow the format specified exactly
"""

def make_scoring_user_prompt(transcript: str, reference_bundle: str) -> str:
    return f"""Analyze the following interview transcript and predict PHQ-8 scores for each symptom domain.

Interview transcript:
<transcript>
{transcript}
</transcript>

{reference_bundle}

Analyze each symptom using the following approach in <thinking> tags:
1. Search for direct quotes or behavioral evidence related to each PHQ-8 symptom
2. When reference examples are provided, compare the current evidence with similar reference cases
3. Evaluate the frequency/severity based on available evidence and reference comparisons
4. Consider how similar the reference examples are to the current evidence - if highly similar, give more weight to the reference scores; if less similar, rely more on direct analysis
5. If no relevant evidence exists, mark as "N/A" rather than assuming absence
6. Only assign numeric scores (0-3) when evidence clearly indicates frequency

After your analysis, provide your final assessment in <answer> tags as a JSON object.

For each symptom, provide:
1. "evidence": exact quotes from transcript (use "No relevant evidence found" if not discussed)
2. "reason": explanation of scoring decision, including cross-reference analysis when applicable and why N/A if applicable
3. "score": integer 0-3 based on evidence, or "N/A" if no relevant evidence

Return ONLY a JSON object in <answer> tags with these exact keys:
- "PHQ8_NoInterest": {{evidence, reason, score}}
- "PHQ8_Depressed": {{evidence, reason, score}}
- "PHQ8_Sleep": {{evidence, reason, score}}
- "PHQ8_Tired": {{evidence, reason, score}}
- "PHQ8_Appetite": {{evidence, reason, score}}
- "PHQ8_Failure": {{evidence, reason, score}}
- "PHQ8_Concentrating": {{evidence, reason, score}}
- "PHQ8_Moving": {{evidence, reason, score}}"""

# ----------------------------- Retrieval -----------------------------
def _find_similar_chunks(
    q_emb: List[float],
    pet: Dict[int, List[Tuple[str, List[float]]]],
    top_k: int
) -> List[Dict[str, Any]]:
    sims = []
    q = [q_emb]
    for pid, pairs in pet.items():
        for raw, emb in pairs:
            sim = cosine_similarity(q, [emb])[0][0]
            sims.append({"participant_id": int(pid), "raw_text": raw, "similarity": float(sim)})
    sims.sort(key=lambda x: x["similarity"], reverse=True)
    return sims[:top_k]

def _build_reference_for_item(
    evidence_quotes: List[str],
    item_key: str,
    ollama_host: str,
    emb_model: str,
    top_k: int,
    pet: Dict[int, List[Tuple[str, List[float]]]],
    phq_df: pd.DataFrame,
    dim: Optional[int] = None,
    min_chars: int = 15
) -> Tuple[str, List[str]]:
    text = "\n".join(evidence_quotes or [])
    if len(text) < min_chars:
        _log(f"[RETRIEVE] item={item_key} no valid evidence; skipping retrieval")
        return "<Reference Examples>\nNo valid evidence found\n<Reference Examples>", []
    emb = ollama_embed(ollama_host, emb_model, text, dim=dim)
    hits = _find_similar_chunks(emb, pet, top_k)
    _log(f"[RETRIEVE] item={item_key} top_k={top_k} hits={len(hits)}")
    lines, sims = [], []
    for h in hits:
        pid = h["participant_id"]
        sims.append(f"{h['similarity']:.4f}")
        row = phq_df.loc[phq_df["Participant_ID"] == pid]
        if row.empty:
            continue
        val = row[item_key].values[0]
        try:
            val = int(val)
        except Exception:
            val = "N/A"
        lines.append(f"({item_key} Score: {val})\n{h['raw_text']}")
    if not lines:
        return "<Reference Examples>\nNo valid evidence found\n<Reference Examples>", sims
    block = "<Reference Examples>\n\n" + "\n\n".join(lines) + "\n\n<Reference Examples>"
    _log(f"[RETRIEVE] item={item_key} cosine: {', '.join(sims) if sims else 'N/A'}")
    return block, sims

# ----------------------------- Main class -----------------------------
class QuantitativeAssessor:
    def __init__(
        self,
        ollama_host: str = "127.0.0.1",
        chat_model: str = "llama3",
        emb_model: str  = "dengcao/Qwen3-Embedding-8B:Q4_K_M",
        pickle_path: str = "chunk_8_step_2_participant_embedded_transcripts.pkl",
        gt_train_csv: str = "train_split_Depression_AVEC2017.csv",
        gt_dev_csv: str   = "dev_split_Depression_AVEC2017.csv",
        top_k: int = 3,
        dim: Optional[int] = None
    ):
        self.ollama_host = ollama_host
        self.chat_model  = chat_model
        self.emb_model   = emb_model
        self.top_k       = top_k
        self.dim         = dim

        df_train = pd.read_csv(gt_train_csv)
        df_dev   = pd.read_csv(gt_dev_csv)
        df = pd.concat([df_train, df_dev], ignore_index=True)
        if "Participant_ID" not in df.columns:
            raise ValueError("CSV must contain 'Participant_ID' column.")
        df["Participant_ID"] = df["Participant_ID"].astype(int)
        if not all(c in df.columns for c in PHQ8_KEYS):
            raise ValueError("CSV must contain PHQ8_* item columns (0–3).")
        self.phq_df = df.sort_values("Participant_ID").reset_index(drop=True)

        p = Path(pickle_path)
        if not p.exists():
            raise FileNotFoundError(f"Pickle not found: {p.resolve()}")
        with open(p, "rb") as f:
            pet_raw = pickle.load(f)

        pet: Dict[int, List[Tuple[str, List[float]]]] = {}
        for pid, pairs in pet_raw.items():
            pid_int = int(pid)
            norm_pairs = []
            for raw, emb in pairs:
                v = np.asarray(emb, dtype=np.float32)
                if self.dim is not None:
                    v = v[: self.dim]
                v = _l2norm(v)
                norm_pairs.append((raw, v.tolist()))
            pet[pid_int] = norm_pairs
        self.participant_embedded_transcripts = pet

    # Evidence extraction (LLM → JSON arrays of quotes per item)
    def extract_evidence(self, transcript: str) -> Dict[str, List[str]]:
        user_prompt = EVIDENCE_EXTRACT_PROMPT.format(transcript=transcript)
        _log("[STEP] Evidence extraction prompt (exact):")
        _log(user_prompt)
        raw = ollama_chat(self.ollama_host, self.chat_model, system_prompt="", user_prompt=user_prompt)
        try:
            txt = _strip_json_block(raw)
            obj = json.loads(txt)
        except Exception as e:
            _log(f"[WARN] Evidence JSON parse failed: {e}")
            obj = {}
        out = {}
        for k in PHQ8_KEYS:
            arr = obj.get(k, []) if isinstance(obj, dict) else []
            if not isinstance(arr, list):
                arr = []
            seen, uniq = set(), []
            for q in arr:
                qs = str(q).strip()
                if qs and qs not in seen:
                    seen.add(qs)
                    uniq.append(qs)
            out[k] = uniq
        _log("[STEP] Evidence dict:")
        _log(json.dumps(out, ensure_ascii=False, indent=2))
        return out

    # Build reference bundle per item
    def build_reference_bundle(self, evidence_dict: Dict[str, List[str]]) -> Tuple[str, Dict[str, List[str]]]:
        blocks = []
        sim_scores: Dict[str, List[str]] = {}
        for item in PHQ8_KEYS:
            block, sims = _build_reference_for_item(
                evidence_quotes=evidence_dict.get(item, []),
                item_key=item,
                ollama_host=self.ollama_host,
                emb_model=self.emb_model,
                top_k=self.top_k,
                pet=self.participant_embedded_transcripts,
                phq_df=self.phq_df,
                dim=self.dim
            )
            blocks.append(f"[{item}]\n{block}")
            sim_scores[item] = sims
        bundle = "\n\n".join(blocks)
        _log("[STEP] Reference bundle (truncated 1500 chars):\n" + bundle[:1500])
        return bundle, sim_scores

    # Score using system+user prompts
    def score_with_references(self, transcript: str, reference_bundle: str) -> Dict[str, Any]:
        user_prompt = make_scoring_user_prompt(transcript, reference_bundle)
        _log("[STEP] Scoring system prompt (exact):")
        _log(SYSTEM_PROMPT)
        _log("[STEP] Scoring user prompt (exact):")
        _log(user_prompt)
        raw = ollama_chat(self.ollama_host, self.chat_model, SYSTEM_PROMPT, user_prompt)
        try:
            txt = _strip_json_block(raw)
            fixed = _tolerant_fixups(txt)
            try:
                obj = json.loads(fixed)
            except Exception as e2:
                _log(f"[WARN] Primary parse failed: {e2}. Trying raw txt...")
                obj = json.loads(txt)
            obj = _validate_and_normalize(obj)
            _log("[STEP] Parsed+validated JSON (without cosine_similarity):")
            _log(json.dumps(obj, ensure_ascii=False, indent=2))
            return obj
        except Exception as e:
            raise RuntimeError(f"Scoring JSON parse/validate failed: {e}\nRAW:\n{raw[:1500]}")

    # Public API
    def assess(self, interview_text: str) -> Dict[str, Any]:
        _log("[STEP] Interview transcript:")
        _log(interview_text.strip())
        evidence = self.extract_evidence(interview_text)
        ref_bundle, _ = self.build_reference_bundle(evidence)
        result = self.score_with_references(interview_text, ref_bundle)
        total, sev = _compute_total_and_severity(result)
        result["_total"] = total
        result["_severity"] = sev
        _log("[STEP] Final result:")
        _log(json.dumps(result, ensure_ascii=False, indent=2))
        return result

# ----------------------------- CLI -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama_host", default="127.0.0.1")
    parser.add_argument("--chat_model", default="llama3")
    parser.add_argument("--emb_model",  default="dengcao/Qwen3-Embedding-8B:Q4_K_M")
    parser.add_argument("--pickle_path", default="chunk_8_step_2_participant_embedded_transcripts.pkl")
    parser.add_argument("--gt_train_csv", default="train_split_Depression_AVEC2017.csv")
    parser.add_argument("--gt_dev_csv",   default="dev_split_Depression_AVEC2017.csv")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--quiet", action="store_true", help="Disable verbose prints")
    args = parser.parse_args()

    if args.quiet:
        VERBOSE = False

    qa = QuantitativeAssessor(
        ollama_host=args.ollama_host,
        chat_model=args.chat_model,
        emb_model=args.emb_model,
        pickle_path=args.pickle_path,
        gt_train_csv=args.gt_train_csv,
        gt_dev_csv=args.gt_dev_csv,
        top_k=args.top_k,
        dim=args.dim
    )

    demo = """
- Hello, Alison Wells, do you want to come and have a seat? Hi I’m Dr Taylor one
of the GP’s at this surgery, what would you like me to call you?
- Alison will be fine
- Ok, so what’s brought you here today Alison?
- My sister's noticed, I’m just a bit fed up really. My sister said I should come.
- Right ok. Has this been going on for some time?
- Yeah, few months really.
- Ok, do you want to tell me a bit more about what’s been going on?
- Just things, things seem to be piling up. - Right - I just don’t seem to be coping with things,
- Right - the kids and things. - Right, ok.
Would it be OK Alison for me to ask you a few more detailed questions
about how you’ve been feeling?
- Uhuh
- Ok
"""
    try:
        out = qa.assess(demo)
        print(json.dumps(out, ensure_ascii=False, indent=2))
    except Exception as e:
        _log(f"[ERROR] {e}")
        raise
