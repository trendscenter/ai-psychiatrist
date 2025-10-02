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

# ----------------------------- Keyword fallback -----------------------------
DOMAIN_KEYWORDS = {
    "PHQ8_NoInterest": ["can't be bothered", "no interest", "nothing really", "not enjoy", "no pleasure", "what's the point", "can’t be bothered", "cant be bothered"],
    "PHQ8_Depressed": ["fed up", "miserable", "depressed", "very black", "hopeless", "low"],
    "PHQ8_Sleep": ["sleep", "fall asleep", "wake up", "insomnia", "clock", "tired in the morning"],
    "PHQ8_Tired": ["exhausted", "tired", "little energy", "fatigue", "no energy"],
    "PHQ8_Appetite": ["appetite", "weight", "lost weight", "eat", "eating", "don’t bother", "dont bother", "looser"],
    "PHQ8_Failure": ["useless", "failure", "bad about myself", "burden"],
    "PHQ8_Concentrating": ["concentrat", "memory", "forgot", "thinking of something else", "focus"],
    "PHQ8_Moving": ["moving slowly", "restless", "fidget", "speaking slowly", "psychomotor"]
}

def _sentences(txt: str) -> List[str]:
    parts = re.split(r'(?<=[\.\?\!])\s+|\n+', txt.strip())
    return [p.strip(" \t-") for p in parts if p and len(p.strip()) > 0]

def _keyword_backfill(transcript: str, current: Dict[str, List[str]], per_item_cap: int = 3) -> Dict[str, List[str]]:
    sents = _sentences(transcript.lower())
    orig_sents = _sentences(transcript)
    out = {k: list(v) for k, v in current.items()}
    for key, kws in DOMAIN_KEYWORDS.items():
        need = max(0, per_item_cap - len(out.get(key, [])))
        if need == 0:
            continue
        hits = []
        for idx, s in enumerate(sents):
            if any(kw in s for kw in kws):
                hits.append(orig_sents[idx].strip())
            if len(hits) >= need:
                break
        if hits:
            seen = set(out.get(key, []))
            merged = out.get(key, []) + [h for h in hits if h not in seen]
            out[key] = merged[:per_item_cap]
    return out

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
    _log("[CHAT] User prompt (exact):\n" + user_prompt)
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    content = r.json()["message"]["content"]
    _log("[CHAT] Raw model output (first 800 chars):\n" + content[:800])
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
    return emb

# ----------------------------- Prompts (filled) -----------------------------
quantitative_prompt = f"""You are a AI psychiatrist assessment tool specialized in analyzing interview transcripts to evaluate PHQ-8 (Patient Health Questionnaire-8) scores. The PHQ-8 is a validated depression screening instrument that assesses eight of the nine DSM-IV criteria for depressive disorders over the past 2 weeks.

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
Analyze the following interview transcript and predict PHQ-8 scores for each symptom domain.

Interview transcript:
<transcript>
{{patient_transcript}}
</transcript>

{{reference_evidence}}

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
- "PHQ8_NoInterest": {{evidence, reason, score}} for little interest or pleasure in doing things (anhedonia)
- "PHQ8_Depressed": {{evidence, reason, score}} for feeling down, depressed, or hopeless (depressed mood)
- "PHQ8_Sleep": {{evidence, reason, score}} for trouble falling or staying asleep, or sleeping too much (sleep problems)
- "PHQ8_Tired": {{evidence, reason, score}} for feeling tired or having little energy (fatigue)
- "PHQ8_Appetite": {{evidence, reason, score}} for poor appetite or overeating (appetite/weight changes)
- "PHQ8_Failure": {{evidence, reason, score}} for feeling bad about yourself or that you are a failure (negative self-perception)
- "PHQ8_Concentrating": {{evidence, reason, score}} for trouble concentrating on things like reading or watching TV (concentration problems)
- "PHQ8_Moving": {{evidence, reason, score}} for moving or speaking slowly, or being fidgety/restless (psychomotor changes)"""

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
{
    "PHQ8_NoInterest": ["evidence_1", "evidence_2"],
    "PHQ8_Depressed": ["evidence_1"],
    "PHQ8_Sleep": ["evidence_1", "evidence_2"],
    "PHQ8_Tired": ["evidence_1"],
    "PHQ8_Appetite": [],
    "PHQ8_Failure": ["evidence_1"],
    "PHQ8_Concentrating": ["evidence_1"],
    "PHQ8_Moving": ["evidence_1"]
}

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
def _find_similar_chunks(q_emb: List[float], pet: Dict[int, List[Tuple[str, List[float]]]], top_k: int) -> List[Dict[str, Any]]:
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
    min_chars: int = 8
) -> Tuple[str, List[str]]:
    text = "\n".join(evidence_quotes or [])
    if len(text) < min_chars:
        return "<Reference Examples>\nNo valid evidence found\n<Reference Examples>", []
    emb = ollama_embed(ollama_host, emb_model, text, dim=dim)
    hits = _find_similar_chunks(emb, pet, top_k)
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
        df["Participant_ID"] = df["Participant_ID"].astype(int)
        self.phq_df = df.sort_values("Participant_ID").reset_index(drop=True)

        with open(pickle_path, "rb") as f:
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

    def extract_evidence(self, transcript: str) -> Dict[str, List[str]]:
        user_prompt = EVIDENCE_EXTRACT_PROMPT.replace("{transcript}", transcript)
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
        enriched = _keyword_backfill(transcript, out, per_item_cap=3)
        _log("[STEP] Evidence dict (with keyword backfill):")
        _log(json.dumps(enriched, ensure_ascii=False, indent=2))
        return enriched

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
        return "\n\n".join(blocks), sim_scores

    def score_with_references(self, transcript: str, reference_bundle: str) -> Dict[str, Any]:
        user_prompt = make_scoring_user_prompt(transcript, reference_bundle)
        raw = ollama_chat(self.ollama_host, self.chat_model, SYSTEM_PROMPT, user_prompt)
        try:
            txt = _strip_json_block(raw)
            fixed = _tolerant_fixups(txt)
            obj = json.loads(fixed)
            return _validate_and_normalize(obj)
        except Exception as e:
            raise RuntimeError(f"Scoring JSON parse failed: {e}\nRAW:\n{raw[:1000]}")

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

# # ----------------------------- CLI -----------------------------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--ollama_host", default="127.0.0.1")
#     parser.add_argument("--chat_model", default="llama3")
#     parser.add_argument("--emb_model",  default="dengcao/Qwen3-Embedding-8B:Q4_K_M")
#     parser.add_argument("--pickle_path", default="chunk_8_step_2_participant_embedded_transcripts.pkl")
#     parser.add_argument("--gt_train_csv", default="train_split_Depression_AVEC2017.csv")
#     parser.add_argument("--gt_dev_csv",   default="dev_split_Depression_AVEC2017.csv")
#     parser.add_argument("--top_k", type=int, default=3)
#     parser.add_argument("--dim", type=int, default=None)
#     parser.add_argument("--quiet", action="store_true")
#     args = parser.parse_args()
#
#     if args.quiet:
#         VERBOSE = False
#
#     qa = QuantitativeAssessor(
#         ollama_host=args.ollama_host,
#         chat_model=args.chat_model,
#         emb_model=args.emb_model,
#         pickle_path=args.pickle_path,
#         gt_train_csv=args.gt_train_csv,
#         gt_dev_csv=args.gt_dev_csv,
#         top_k=args.top_k,
#         dim=args.dim
#     )
#
#     demo = """
# - Hello, Alison Wells, do you want to come and have a seat? Hi I’m Dr Taylor one
# of the GP’s at this surgery, what would you like me to call you?
# - Alison will be fine
# - Ok, so what’s brought you here today Alison?
# - My sister's noticed, I’m just a bit fed up really. My sister said I should come.
# - Right ok. Has this been going on for some time?
# - Yeah, few months really.
# - Ok, do you want to tell me a bit more about what’s been going on?
# - Just things, things seem to be piling up. - Right - I just don’t seem to be coping with things,
# - Right - the kids and things. - Right, ok.
# Would it be OK Alison for me to ask you a few more detailed questions
# about how you’ve been feeling?
# - Uhuh
# - Ok
# Well if we start with asking you a bit about your mood.
# How have you been feeling in yourself?
# - I say a bit fed up. I get up in the morning, everything seems very black. - Right
# - It’s like, it's just like swimming in, in treacle really and I just don't, I think by tea time when the kids get home,
# I’ve been having fairly decent conversation with them but...
# - Right, and can I just check Alison when you say things feel very black, do you feel very miserable?
# - Fed up, miserable.
# - Right, OK. And what about sort of feeling tearful? Has that been happening?
# - I dropped some sugar the other day and I just burst into tears.
# - Right, OK.
# - Thanks
# - So is it the slightest thing that will make you tearful, things that perhaps
# wouldn’t ordinarily bother you?
# - Yeah my sister's noticed it as well.
# - Right, OK.
# So you’ve been feeling very low with episodes of tearfulness, what about
# other things, your energy levels are you managing to keep up with things?
# - I used to do a lot with the kids I used to go swimming, playing but now I just spend
# the day on the sofa unless I have to go to work.
# - Right, just remind me, what is it you do for your job?
# - I work in a supermarket.
# - Right, so how have you been managing at work?
# - I’ve not been going in as much cause I just feel so exhausted...
# ...but I'm just not...the supermarket's been
# taken over and they’ve cut the wages - Right - and I’ve had problems with the bills
# - Right - and it's like catalogues just writing me letters, - Uhuh
# - you know the kids they want all these new games and stuff
# - Yeah sure
# - and it’s just y'know.
# - Things are difficult all round then. With all this going on how are you sleeping Alison?
# - It just takes me ages to go to sleep, I used to read a book - Right
# - and just drop off, - Right - but now I just spend my time
# looking at the clock as it goes round and round.
# - So from actually getting off to bed and getting off to sleep how long is that taking?
# - Couple of hours probably.
# - Right, OK.
# - Once you’re asleep are you waking up much during the night?
# - I wake up about...last night I think it was about 4 o clock I woke up. - Right
# - And can you get back to sleep from that time?
# - No, no.
# - And then you’re actually getting out of bed in the morning, are you still feeling tired at that point?
# - I’m just exhausted, I feel like my brain's not been switched off. I'm just exhausted the next day.
# - OK
# - What about eating what’s your appetite been like while you’ve been feeling like this?
# - I used to have quite a weight problem, but the last couple of months this is a bit looser.
# - Right
# - Er, I just...
# - Do you know how much weight you’ve lost?
# - No, no.
# - The kids come in from school and they make their own stuff and I just don’t bother really.
# - Ok, Ok so you’re appetite's gone down as well. What about things like concentrating and your memory
# both when you’re watching TV at home or when you’re out doing your job.
# - How have those things been?
# - Well I mean pretty useless with the kids, I forgot the swimming money last week, PE
# kit and parents evening even. - Right, ok. - I just start one job, and, I’m not explaining myself very
# well. It's like the television, I used to like watching the soaps, EastEnders
# or something and now 10 minutes later I’m thinking of something else.
# - Ok, ok.
# - And what about things Alison you used to enjoy, are there things in life that
# you still enjoy at the moment?
# - Nothing really, as I say, a bit useless with the kids. I used to enjoy going out, I used
# to go out with my friends, the pictures and things, but of course now I can’t be bothered.
# - Is it that you can’t be bothered and you don’t feel like it as well?
# - What’s the point really? - Right, ok. - You know.
# - And I was going to ask, how old are your children now?
# - I had them a bit later in life, it's took a a long time to have them. I got a girl and
# a boy, one's 11 and one's 9.
# - And looking after children takes a lot of time and energy, how are you managing to keep
# up with that feeling as low as you do?
# - Well they’re a bit self-sufficient really the kids, they come in from school get their
# own tea. I should be doing more for them really but I’m not I’m just a
# bit useless at that at the moment.
# - Ok, and what about looking after yourself?
# - Well you can see I’m just a mess. Dave used to say, that’s my boyfriend,
# - Right
# - he used to, you know, not have much
# money but I’d take a bit of pride in what I was doing. My hair and stuff but I can't
# can't be bothered with that now, there's no point really.
# - Ok, ok.
# - And you mentioned Dave, that’s your current partner, how long have you and Dave been together?
# About a year I met him at work.
# - And how are things, because often when people feel
# really down it has an impact on everything including their
# relationships so how are things with you and Dave at the moment?
# He’s not ringing as much, he used to text, he’s getting fed up with me not wanting
# to go out and things.
# - It’s a slightly embarrassing thing to ask about but I guess it's important, often when people are
# really feeling very low it affects everything in the relationship including things like their
# sex life. Have you noticed any changes there for you?
# He’s always trying to pressure me a little bit - Right - and stuff,
# but I’m really not into that at the moment.
# - Right, ok. You just don’t feel like that at the moment.
# - No. - Ok.
# So can I just a recap Alison to check I’ve got this right, for the last few months you’ve been
# feeling really down, no energy, problems with your sleeping and eating,
# problems with concentrating,
# not really enjoying things and actually struggling a bit with the kids and perhaps some difficulties
# in your relationship with Dave. Have I got that right?
# - Hmmmm.
# - OK.
# Can I ask Alison, in the past
# have there ever been episodes where you’ve felt like this?
# - When my husband left, I was always crying then for no particular reason. - Right
# - I haven’t told anybody this before but I took some tablets.
# - Right
# So how long ago are we talking was this a few years ago?
# - About four years ago.
# - About four years ago.
# Ok, so you took some tablets, - Yeah - Can you tell me,
# is it alright to tell me a little bit more about that?
# - You know what it’s like, the kids are in bed and you’re on your own and I had a few
# glasses of wine and I just took these tablets.
# - Right, OK.
# Can you remember what you actually took at the time what sort of tablets they were?
# - They were just in the bathroom cabinet, it was paracetamol.
# - Right, OK.
# So you took some paracetamol, can you remember roughly how many you took?
# - About 2 strips, about 12.
# - Right, OK.
# - And you’d had a couple of glasses of wine; did you take anything else, any other tablets with it?
# - No. - Ok.
# Ok. And was this something Alison that you’d thought about for a while or was it a spur of the
# moment that evening?
# - As I say I was crying a lot but I think it was just the wine. - Right, ok.
# - But you know, I’m just a bit of a burden to everybody really.
# - And was there any other things that you did around the time, sometimes when people take tablets
# they leave a note, or do other kinda final acts, get their affairs in order?
# Did you do any of those things?
# - No, I just thought, that you know, I'd take the tablets and I'd just go to sleep.
# - Right, ok.
# So did you have any thoughts about what taking the tablets would do? Did you..
# - I just thought I’d go to sleep and not wake up, but I woke up a couple of hours later
# and was sick everywhere. - Right, ok. God I was sick.
# - Ok, so I actually just want to check I get this right because it's important.
# You actually thought that they would kill you at the time?
# - I just didn’t want to wake up, - Right, ok
# - As I say, I’m just useless, I’m a useless mum now and I was then.
# - Right ok, so you took the tablets and you were very sick in the night
# did you seek any medical help at the time?
# - No, no. - Ok.
# And then were you OK the following day?
# - Well, yeah, I just felt a bit of a twit really.
# - Right, did you feel pleased you were still alive?
# - Yeah, I think you know I realised it was me just being silly.
# - Right, Ok, Ok.
# So that was a few years ago, if we just come back to how you are feeling at the moment,
# you talked about feeling very low… Have there been times currently when you’ve thought
# about either taking an overdose or doing something else to harm yourself in any way?
# - You know at night when, when you’re watching the clock and you know, you’re on your own -Yeah
# - and the kids are in bed. It is, just everything’s so hard, and yeah I suppose, you know,
# it just feels easy you know? - Right
# And has that just been something you’ve thought about or have
# you actually made any plans, got any tablets in or done anything else?
# - No, no, - Ok - nothing like that.
# Ok, and I guess it’s a difficult question to ask but one we would ask everybody in your situation.
# Have things ever been so bad you felt so low that you’ve not only thought about harming
# yourself or perhaps killing yourself but you’ve also wondered whether
# the best thing might be to take the children with you?
# -No, I’d never do anything with my kids I love my kids. No I wouldn’t hurt them.
# - Ok.
# - And what about the other side of that, positive things, things to live for...
# ...things that you feel good about?
# - Not much at the moment I suppose, kids sometimes you know. They do things that make you think,
# you know, what’s good about life - Right - and things but...
# - And are there other things that help at the moment,
# I’m thinking about people that could be supportive?
# - My sister as I say she said to get down here and she’s always there, she comes down - Yeah
# - and rings. I’ve got a couple of friends they’re quite good, - Right
# - but and Dave when he’s in the mood of course. But I don’t think he’s not going to be around for much longer.
# - Right.
# And do you think, or do you feel able to keep yourself safe at the moment from hurting yourself?
# - I think so yeah, I know I can come here now; you’ve been very good today
# - Ok.
# Do you think if that was to change so that you didn’t feel able to keep yourself safe,
# you’d be able to let anybody know?
# - All I know is what happened last time that was nothing
# and I was silly then so I know to come here.
# - Right, Ok.
# OK Alison, well thanks for going through all that, I can appreciate it must be very painful. It does
# sound to me that you are suffering from symptoms that strongly suggest that you are actually depressed
# at the moment. Now I’m not sure how much you know about depression?
# - Not much really, not much but I know I just don’t feel right at the moment.
# - Ok, ok
# Well I guess just briefly depression can cause a number of problems for people and traditionally
# we think about people feeling very low and very miserable and often you know thinking
# about hurting themselves. But it can also affect all other areas of life in terms of
# problems with eating and sleeping and the other problems you’ve noticed is that concentration
# and perhaps not really managing as well as normal. I guess the positive side is that you’ve done something
# about it and you’ve come to talk to me about it today and I think there are almost certainly a
# range of things we can put in place to help you and treatments that are available. So I guess
# what I’m thinking is it might be worth us spending a few minutes just thinking about
# those options for you so that we can start to improve things for you.
# Would that be alright with you?
# - I need to do this, yes; I think that’s sensible yeah. - Ok
#     """
#     out = qa.assess(demo)
#     print(json.dumps(out, ensure_ascii=False, indent=2))
