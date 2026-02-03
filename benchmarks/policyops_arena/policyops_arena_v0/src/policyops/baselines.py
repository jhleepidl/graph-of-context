from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from .env import PolicyOpsEnv


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


def _build_goc_prompt(ticket: str, clauses: List[Dict[str, Any]], clause_ids: List[str]) -> str:
    header = (
        "너는 정책 담당 에이전트. 아래 티켓을 처리해라.\n"
        "아래 제공된 clauses 중에서만 근거를 선택할 것.\n"
        "다음 JSON 스키마를 엄격히 준수해라:\n"
        "decision: allow|deny|needs_more_info\n"
        "required_conditions: list of {key, details}\n"
        "evidence: list of clause_id (최소 2개, 제공된 clause_id 중에서만 선택)\n"
        "customer_message: 1~3문장\n"
    )
    body = ["", "Ticket:", ticket, "", "Available clause_ids:", ", ".join(clause_ids), "", "Clauses:"]
    for clause in clauses:
        body.append(f"[{clause['clause_id']}] {clause['text']}")
    body.append("")
    body.append("Return JSON only.")
    return header + "\n".join(body)


def _parse_goc_prediction(raw: str, opened_ids: List[str]) -> Dict[str, Any]:
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
    if decision not in {"allow", "deny", "needs_more_info"}:
        decision = "needs_more_info"

    conditions: List[str] = []
    if isinstance(data.get("required_conditions"), list):
        for item in data.get("required_conditions", []):
            if isinstance(item, dict) and item.get("key"):
                conditions.append(str(item["key"]))
            elif isinstance(item, str):
                conditions.append(item)
    elif isinstance(data.get("conditions"), list):
        for item in data.get("conditions", []):
            if isinstance(item, str):
                conditions.append(item)

    evidence = data.get("evidence", []) or []
    if not isinstance(evidence, list):
        evidence = []
    if opened_ids:
        evidence = [cid for cid in evidence if cid in opened_ids]

    return {"decision": decision, "conditions": conditions, "evidence": evidence}


def run_topk_rag(task: Any, env: PolicyOpsEnv, client: LLMClient) -> Tuple[Dict[str, Any], List[str], str]:
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
    return prediction, opened_ids, prompt


def run_full_history(task: Any, env: PolicyOpsEnv, client: LLMClient) -> Tuple[Dict[str, Any], List[str], str]:
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
    return prediction, opened_ids, prompt


def run_goc_heuristic(
    task: Any, env: PolicyOpsEnv, client: LLMClient
) -> Tuple[Dict[str, Any], List[str], str]:
    seed_results = env.search(task.user_ticket, top_k=20)
    expanded_query = f\"{task.user_ticket} exception supersede revoke definition effective immediately\"
    expanded_results = env.search(expanded_query, top_k=20)

    merged = {item[\"clause_id\"]: item for item in seed_results}
    for item in expanded_results:
        merged.setdefault(item[\"clause_id\"], item)

    def score_item(item: Dict[str, Any]) -> Tuple[int, float]:
        snippet = item.get(\"snippet\", \"\").lower()
        if any(word in snippet for word in [\"effective\", \"supersede\", \"revoke\", \"update\", \"amend\"]):
            priority = 3
        elif any(word in snippet for word in [\"definition\", \"means\", \"for the purposes of\"]):
            priority = 2
        elif any(word in snippet for word in [\"except\", \"unless\", \"however\", \"provided that\"]):
            priority = 1
        else:
            priority = 0
        return (priority, item.get(\"score\", 0.0))

    ranked = sorted(merged.values(), key=score_item, reverse=True)
    opened: List[Dict[str, Any]] = []
    opened_ids: List[str] = []
    for item in ranked:
        if len(opened_ids) >= env.open_budget:
            break
        clause = env.open(item[\"clause_id\"])
        opened.append(clause)
        opened_ids.append(clause[\"clause_id\"])

    prompt = _build_goc_prompt(task.user_ticket, opened, opened_ids)
    raw = client.generate(prompt)
    prediction = _parse_goc_prediction(raw, opened_ids)
    return prediction, opened_ids, prompt
