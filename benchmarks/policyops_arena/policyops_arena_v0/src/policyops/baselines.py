from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from .env import PolicyOpsEnv
from .world import evaluate_context


class LLMClient:
    def generate(self, prompt: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class DummyClient(LLMClient):
    def generate(self, prompt: str) -> str:
        return json.dumps(
            {
                "decision": "needs_more_info",
                "conditions": [],
                "evidence": [],
            }
        )


def _build_prompt(ticket: str, clauses: List[Dict[str, Any]]) -> str:
    header = (
        "You are a policy assistant. Respond with JSON containing: "
        "decision (allow|deny|require_condition|needs_more_info), "
        "conditions (list of keys), evidence (list of clause_id).\n"
    )
    body = ["Ticket:", ticket, "", "Clauses:"]
    for clause in clauses:
        body.append(f"[{clause['clause_id']}] {clause['text']}")
    body.append("")
    body.append("Return JSON only.")
    return header + "\n".join(body)


def _parse_prediction(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            data = {}
        else:
            try:
                data = json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                data = {}
    decision = data.get("decision", "needs_more_info")
    conditions = data.get("conditions", []) or []
    evidence = data.get("evidence", []) or []
    if not isinstance(conditions, list):
        conditions = []
    if not isinstance(evidence, list):
        evidence = []
    return {"decision": decision, "conditions": conditions, "evidence": evidence}


def run_topk_rag(task: Any, env: PolicyOpsEnv, client: LLMClient) -> Tuple[Dict[str, Any], str, List[str]]:
    results = env.search(task.user_ticket, top_k=10)
    opened: List[Dict[str, Any]] = []
    opened_ids: List[str] = []
    for item in results:
        if len(opened_ids) >= env.open_budget:
            break
        clause = env.open(item["clause_id"])
        opened.append(clause)
        opened_ids.append(clause["clause_id"])
    prompt = _build_prompt(task.user_ticket, opened)
    raw = client.generate(prompt)
    prediction = _parse_prediction(raw)
    return prediction, prompt, opened_ids


def run_full_history(task: Any, env: PolicyOpsEnv, client: LLMClient) -> Tuple[Dict[str, Any], str, List[str]]:
    results = env.search(task.user_ticket, top_k=10)
    expanded_query = f"{task.user_ticket} update supersede exception definition"
    results += env.search(expanded_query, top_k=10)

    seen = set()
    opened: List[Dict[str, Any]] = []
    opened_ids: List[str] = []
    for item in results:
        clause_id = item["clause_id"]
        if clause_id in seen:
            continue
        seen.add(clause_id)
        if len(opened_ids) >= env.open_budget:
            break
        clause = env.open(clause_id)
        opened.append(clause)
        opened_ids.append(clause_id)

    prompt = _build_prompt(task.user_ticket, opened)
    raw = client.generate(prompt)
    prediction = _parse_prediction(raw)
    return prediction, prompt, opened_ids


def run_goc_heuristic(task: Any, env: PolicyOpsEnv) -> Tuple[Dict[str, Any], str, List[str]]:
    seed_results = env.search(task.user_ticket, top_k=20)
    expanded_query = f"{task.user_ticket} exception supersede definition effective"
    expanded_results = env.search(expanded_query, top_k=20)

    merged = {item["clause_id"]: item for item in seed_results}
    for item in expanded_results:
        merged.setdefault(item["clause_id"], item)

    def score_item(item: Dict[str, Any]) -> Tuple[int, float]:
        snippet = item.get("snippet", "").lower()
        priority = 0
        if any(word in snippet for word in ["supersede", "effective", "update"]):
            priority += 3
        if "definition" in snippet:
            priority += 2
        if "exception" in snippet:
            priority += 1
        return (priority, item.get("score", 0.0))

    ranked = sorted(merged.values(), key=score_item, reverse=True)
    opened: List[str] = []
    for item in ranked:
        if len(opened) >= env.open_budget:
            break
        clause = env.open(item["clause_id"])
        opened.append(clause["clause_id"])

    decision, conditions, _, _ = evaluate_context(env.world, task.context)
    prediction = {
        "decision": decision,
        "conditions": conditions,
        "evidence": opened,
    }
    prompt = ""  # heuristic does not build an LLM prompt
    return prediction, prompt, opened
