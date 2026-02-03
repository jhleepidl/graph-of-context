from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from .env import PolicyOpsEnv
from .controller import Controller

ALLOWED_DECISIONS = {"allow", "deny", "require_condition", "needs_more_info"}


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
                "customer_message": "",
            }
        )


def _load_dotenv(path: str = ".env", override: bool = False) -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if not override and key in os.environ:
                continue
            os.environ[key] = value


class OpenAIClient(LLMClient):
    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: int = 60,
        dotenv_path: str = ".env",
    ) -> None:
        if api_key is None:
            _load_dotenv(dotenv_path)
            api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required (set it in .env or env vars).")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        payload = {"model": self.model, "input": prompt}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/responses",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read().decode("utf-8")
        response = json.loads(raw)
        text = response.get("output_text")
        if isinstance(text, str) and text.strip():
            return text
        output = response.get("output", [])
        chunks: List[str] = []
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content", [])
                if not isinstance(content, list):
                    continue
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if isinstance(part.get("text"), str):
                        chunks.append(part["text"])
        if chunks:
            return "\n".join(chunks)
        choices = response.get("choices", [])
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message") if isinstance(choices[0], dict) else None
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"]
        return ""

def _build_prompt(ticket: str, clauses: List[Dict[str, Any]]) -> str:
    header = (
        "You are a policy assistant. Respond with JSON only using this schema:\n"
        '{ "decision": "allow|deny|require_condition|needs_more_info", '
        '"conditions": ["KEY1", "KEY2"], '
        '"evidence": ["C-..."], '
        '"customer_message": "..." }\n'
        "Use only the provided clauses as evidence.\n"
    )
    body = ["Ticket:", ticket, "", "Clauses:"]
    for clause in clauses:
        body.append(f"[{clause['clause_id']}] {clause['text']}")
    body.append("")
    body.append("Return JSON only.")
    return header + "\n".join(body)


def _extract_conditions(*candidates: Any) -> List[str]:
    conditions: List[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, list):
            continue
        for item in candidate:
            key: Optional[str] = None
            if isinstance(item, str):
                key = item
            elif isinstance(item, dict):
                if "key" in item:
                    key = str(item.get("key"))
                elif "name" in item:
                    key = str(item.get("name"))
            if key and key not in seen:
                conditions.append(key)
                seen.add(key)
    return conditions


def _normalize_prediction(data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        data = {}
    decision = data.get("decision", "needs_more_info")
    if decision not in ALLOWED_DECISIONS:
        decision = "needs_more_info"
    conditions = _extract_conditions(data.get("conditions"), data.get("required_conditions"))
    evidence = data.get("evidence", []) or []
    if not isinstance(evidence, list):
        evidence = []
    evidence = [str(item) for item in evidence if isinstance(item, (str, int))]
    customer_message = data.get("customer_message", "")
    if not isinstance(customer_message, str):
        customer_message = ""
    return {
        "decision": decision,
        "conditions": conditions,
        "evidence": evidence,
        "customer_message": customer_message,
    }


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
    return _normalize_prediction(data)


def _build_goc_prompt(ticket: str, clauses: List[Dict[str, Any]], clause_ids: List[str]) -> str:
    header = (
        "You are a policy agent. Resolve the ticket below.\n"
        "Use only the provided clauses as evidence.\n"
        "Return JSON only using this schema:\n"
        '{ "decision": "allow|deny|require_condition|needs_more_info", '
        '"conditions": ["KEY1", "KEY2"], '
        '"evidence": ["C-..."], '
        '"customer_message": "..." }\n'
    )
    body = ["", "Ticket:", ticket, "", "Available clause_ids:", ", ".join(clause_ids), "", "Clauses:"]
    for clause in clauses:
        body.append(f"[{clause['clause_id']}] {clause['text']}")
    body.append("")
    body.append("Return JSON only.")
    return header + "\n".join(body)


def _ensure_min_evidence(evidence: List[str], opened_ids: List[str], min_count: int = 2) -> List[str]:
    if len(evidence) < min_count:
        for clause_id in opened_ids:
            if clause_id not in evidence:
                evidence.append(clause_id)
            if len(evidence) >= min_count:
                break
    return evidence


def _parse_goc_prediction(raw: str, opened_ids: List[str]) -> Tuple[Dict[str, Any], str | None]:
    raw = raw.strip()
    parse_error: str | None = None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            return {}, "parse_error: no_json_object"
        try:
            data = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return {}, "parse_error: invalid_json"

    prediction = _normalize_prediction(data)
    return prediction, parse_error


def run_topk_rag(
    task: Any, env: PolicyOpsEnv, client: LLMClient
) -> Tuple[Dict[str, Any], List[str], str, str | None]:
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
    return prediction, opened_ids, prompt, raw


def run_full_history(
    task: Any, env: PolicyOpsEnv, client: LLMClient
) -> Tuple[Dict[str, Any], List[str], str, str | None]:
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
    return prediction, opened_ids, prompt, raw


def run_goc_heuristic(
    task: Any,
    env: PolicyOpsEnv,
    client: LLMClient,
    controller: Optional[Controller] = None,
    controller_mode: str = "off",
) -> Tuple[Dict[str, Any], List[str], str, str | None, str | None, str | None]:
    errors: List[str] = []
    fallback_prediction = {
        "decision": "needs_more_info",
        "conditions": [],
        "evidence": [],
        "customer_message": "",
    }
    try:
        seed_results = env.search(task.user_ticket, top_k=20)
    except Exception as exc:  # noqa: BLE001 - defensive
        seed_results = []
        errors.append(f"search_error: {exc}")
    expanded_query = f"{task.user_ticket} exception supersede revoke definition effective immediately"
    try:
        expanded_results = env.search(expanded_query, top_k=20)
    except Exception as exc:  # noqa: BLE001 - defensive
        expanded_results = []
        errors.append(f"search_error: {exc}")

    merged = {item["clause_id"]: item for item in seed_results}
    for item in expanded_results:
        merged.setdefault(item["clause_id"], item)

    controller_action: str | None = None
    ordered_clause_ids: List[str] = []
    if controller and controller_mode != "off":
        context_features = controller.build_context_features(task.context, env.open_budget)
        controller_action = controller.select_action(
            context_features, explore=controller_mode == "train"
        )
        ordered_clause_ids = controller.rank_clause_ids(
            controller_action, seed_results, expanded_results
        )

    def score_item(item: Dict[str, Any]) -> Tuple[int, float]:
        snippet = item.get("snippet", "").lower()
        if any(word in snippet for word in ["effective", "supersede", "revoke", "update", "amend"]):
            priority = 3
        elif any(word in snippet for word in ["definition", "means", "for the purposes of"]):
            priority = 2
        elif any(word in snippet for word in ["except", "unless", "however", "provided that"]):
            priority = 1
        else:
            priority = 0
        return (priority, item.get("score", 0.0))

    ranked = sorted(merged.values(), key=score_item, reverse=True)
    opened: List[Dict[str, Any]] = []
    opened_ids: List[str] = []
    opened_set: set[str] = set()
    if ordered_clause_ids:
        ranked_ids = ordered_clause_ids
    else:
        ranked_ids = [item["clause_id"] for item in ranked]
    for clause_id in ranked_ids:
        if len(opened_set) >= env.open_budget:
            break
        if clause_id in opened_set:
            continue
        try:
            clause = env.open(clause_id)
        except RuntimeError as exc:
            errors.append(f"open_error: {exc}")
            if "budget" in str(exc):
                break
            continue
        except Exception as exc:  # noqa: BLE001 - defensive
            errors.append(f"open_error: {exc}")
            continue
        opened.append(clause)
        opened_set.add(clause["clause_id"])
        opened_ids.append(clause["clause_id"])
    prompt = _build_goc_prompt(task.user_ticket, opened, opened_ids)
    raw_output: str | None = None
    try:
        raw_output = client.generate(prompt)
        prediction, parse_error = _parse_goc_prediction(raw_output, opened_ids)
        if parse_error:
            errors.append(parse_error)
        if not prediction:
            prediction = dict(fallback_prediction)
        prediction.setdefault("decision", "needs_more_info")
        prediction.setdefault("conditions", [])
        prediction.setdefault("evidence", [])
        prediction.setdefault("customer_message", "")
    except Exception as exc:  # noqa: BLE001 - defensive
        errors.append(f"generate_error: {exc}")
        prediction = dict(fallback_prediction)

    if controller and controller_mode == "train" and controller_action:
        gold_ids = set(task.gold.gold_evidence or [])
        opened_id_set = set(opened_ids)
        coverage = len(opened_id_set & gold_ids) / max(1, len(gold_ids))
        bonus = 0.0
        critical_kinds = {"update", "definition", "exception"}
        gold_kinds = {
            env.world.clauses[cid].kind
            for cid in gold_ids
            if cid in env.world.clauses and env.world.clauses[cid].kind in critical_kinds
        }
        for kind in gold_kinds:
            if any(
                env.world.clauses.get(cid) and env.world.clauses[cid].kind == kind
                for cid in opened_id_set
            ):
                bonus += 0.1
        penalty = -0.01 * len(opened_ids)
        reward = coverage + bonus + penalty
        context_features = controller.build_context_features(task.context, env.open_budget)
        controller.update(controller_action, context_features, reward)

    error = "; ".join(errors) if errors else None
    return prediction, opened_ids, prompt, raw_output, error, controller_action
