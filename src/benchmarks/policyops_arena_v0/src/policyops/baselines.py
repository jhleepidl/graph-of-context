from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from .env import PolicyOpsEnv
from .controller import Controller, RerankController
from .tools import query_rewrite
from .query_facets import extract_facets
from .world import evaluate_context

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


def run_engine_oracle(task: Any, env: PolicyOpsEnv) -> Dict[str, Any]:
    decision, conditions, evidence, _ = evaluate_context(env.world, task.context)
    return {
        "decision": decision,
        "conditions": conditions,
        "evidence": evidence,
        "customer_message": "",
    }

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


def _enrich_search_results(env: PolicyOpsEnv, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for item in results:
        clause_id = item.get("clause_id")
        clause = env.world.clauses.get(clause_id) if clause_id else None
        enriched.append(
            {
                **item,
                "kind": clause.kind if clause else None,
                "slot": clause.slot if clause else None,
                "published_at": clause.published_at if clause else None,
                "authority": clause.authority if clause else None,
                "is_bridge_doc": item.get("is_bridge_doc"),
                "bridge_for_slot": item.get("bridge_for_slot"),
                "query_contains_canonical": item.get("query_contains_canonical"),
                "query_contains_update_kw": item.get("query_contains_update_kw"),
            }
        )
    return enriched


def _merge_search_results(result_sets: List[List[Dict[str, Any]]], top_k: int) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for results in result_sets:
        for item in results:
            clause_id = item.get("clause_id")
            if not clause_id:
                continue
            score = float(item.get("score", 0.0))
            snippet = item.get("snippet", "")
            current = merged.get(clause_id)
            if current is None or score > float(current.get("score", 0.0)):
                merged[clause_id] = {"clause_id": clause_id, "score": score, "snippet": snippet}
    ranked = sorted(merged.values(), key=lambda it: it.get("score", 0.0), reverse=True)
    return ranked[:top_k]


def _merge_hybrid_results(
    base_results: List[Dict[str, Any]],
    struct_results: List[Dict[str, Any]],
    top_k: int,
    merge_rank_fusion: str = "max",
) -> List[Dict[str, Any]]:
    base_rank = {item.get("clause_id"): idx + 1 for idx, item in enumerate(base_results) if item.get("clause_id")}
    struct_rank = {
        item.get("clause_id"): idx + 1 for idx, item in enumerate(struct_results) if item.get("clause_id")
    }
    merged: Dict[str, Dict[str, Any]] = {}
    for item in base_results:
        clause_id = item.get("clause_id")
        if not clause_id:
            continue
        merged.setdefault(
            clause_id,
            {
                "clause_id": clause_id,
                "base_score": float(item.get("score", 0.0)),
                "struct_score": None,
                "base_rank": base_rank.get(clause_id),
                "struct_rank": struct_rank.get(clause_id),
            },
        )
        merged[clause_id]["base_score"] = float(item.get("score", 0.0))
    for item in struct_results:
        clause_id = item.get("clause_id")
        if not clause_id:
            continue
        merged.setdefault(
            clause_id,
            {
                "clause_id": clause_id,
                "base_score": None,
                "struct_score": float(item.get("score", 0.0)),
                "base_rank": base_rank.get(clause_id),
                "struct_rank": struct_rank.get(clause_id),
            },
        )
        merged[clause_id]["struct_score"] = float(item.get("score", 0.0))

    for clause_id, entry in merged.items():
        bscore = entry.get("base_score")
        sscore = entry.get("struct_score")
        if merge_rank_fusion == "rrf":
            rrf = 0.0
            if entry.get("base_rank"):
                rrf += 1.0 / (60.0 + float(entry["base_rank"]))
            if entry.get("struct_rank"):
                rrf += 1.0 / (60.0 + float(entry["struct_rank"]))
            merge_score = rrf
        else:
            merge_score = max([s for s in [bscore, sscore] if s is not None] or [0.0])
        entry["merge_score"] = merge_score
        entry["score"] = merge_score
        entry["in_base_topk"] = entry.get("base_rank") is not None
        entry["in_struct_topk"] = entry.get("struct_rank") is not None

    ranked = sorted(merged.values(), key=lambda it: it.get("score", 0.0), reverse=True)
    return ranked[:top_k]


def _search_with_rewrite(
    env: PolicyOpsEnv,
    ticket_text: str,
    task_meta: Dict[str, Any],
    top_k: int,
    use_query_rewrite: bool,
    rewrite_queries_count: int = 3,
    rewrite_mode: str = "expanded",
    merge_rank_fusion: str = "max",
) -> Tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
    if not use_query_rewrite:
        return env.search(ticket_text, top_k=top_k), [ticket_text], []

    queries = query_rewrite(ticket_text, task_meta, mode=rewrite_mode)[: max(1, rewrite_queries_count)]

    if rewrite_mode == "hybrid":
        base_query = queries[0] if queries else ticket_text
        struct_query = queries[1] if len(queries) > 1 else base_query
        try:
            base_results = env.search(base_query, top_k=top_k)
        except Exception:
            base_results = []
        try:
            struct_results = env.search(struct_query, top_k=top_k)
        except Exception:
            struct_results = []
        merged = _merge_hybrid_results(
            base_results,
            struct_results,
            top_k=top_k,
            merge_rank_fusion=merge_rank_fusion,
        )
        variants = [
            {"stage": "base", "query": base_query, "results": base_results},
            {"stage": "structured", "query": struct_query, "results": struct_results},
            {"stage": "hybrid_merged", "query": "hybrid_merged", "results": merged},
        ]
        return merged, [base_query, struct_query], variants

    results: List[List[Dict[str, Any]]] = []
    for q in queries:
        try:
            results.append(env.search(q, top_k=top_k))
        except Exception:
            results.append([])
    merged = _merge_search_results(results, top_k=top_k)
    return merged, queries, []


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
        "Decision meanings:\n"
        "- deny: explicitly prohibited by a clause.\n"
        "- require_condition: allowed only if conditions are met; include required conditions.\n"
        "Return JSON only using this schema:\n"
        '{ "decision": "allow|deny|require_condition|needs_more_info", '
        '"conditions": ["KEY1", "KEY2"], '
        '"evidence": ["C-..."], '
        '"customer_message": "..." }\n'
        "Example: {\"decision\":\"require_condition\",\"conditions\":[\"DPA_APPROVAL\"],\"evidence\":[\"C-001\"],\"customer_message\":\"Allowed once conditions are met.\"}\n"
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


def _extract_canonical_from_text(text: str) -> List[str]:
    lowered = text.lower()
    if "includes" in lowered:
        parts = text.split("'")
        if len(parts) >= 3:
            return [parts[1]]
    return []


def run_topk_rag(
    task: Any,
    env: PolicyOpsEnv,
    client: LLMClient,
    primary_top_k: int = 20,
    use_query_rewrite: bool = True,
    rewrite_queries_count: int = 3,
    rewrite_mode: str = "expanded",
    merge_rank_fusion: str = "max",
) -> Tuple[Dict[str, Any], List[str], str, str | None, Dict[str, Any]]:
    results, rewrite_queries, search_variants = _search_with_rewrite(
        env,
        task.user_ticket,
        task.context,
        top_k=primary_top_k,
        use_query_rewrite=use_query_rewrite,
        rewrite_queries_count=rewrite_queries_count,
        rewrite_mode=rewrite_mode,
        merge_rank_fusion=merge_rank_fusion,
    )
    opened_for_prompt_clauses: List[Dict[str, Any]] = []
    opened_ids: List[str] = []
    for item in results:
        if len(opened_ids) >= env.open_budget:
            break
        clause = env.open(item["clause_id"])
        opened_for_prompt_clauses.append(clause)
        opened_ids.append(clause["clause_id"])
    prompt = _build_prompt(task.user_ticket, opened_for_prompt_clauses)
    raw = client.generate(prompt)
    prediction = _parse_prediction(raw)
    diag = {
        "primary_search_results": _enrich_search_results(env, results),
        "primary_search_top_k": primary_top_k,
        "primary_search_query": task.user_ticket,
        "rewrite_used": use_query_rewrite,
        "rewrite_queries": rewrite_queries,
    }
    if search_variants:
        diag["primary_search_variants"] = search_variants
    return prediction, opened_ids, prompt, raw, diag


def run_full_history(
    task: Any,
    env: PolicyOpsEnv,
    client: LLMClient,
    primary_top_k: int = 20,
    use_query_rewrite: bool = True,
    rewrite_queries_count: int = 3,
    rewrite_mode: str = "expanded",
    merge_rank_fusion: str = "max",
) -> Tuple[Dict[str, Any], List[str], str, str | None, Dict[str, Any]]:
    primary_results, rewrite_queries, search_variants = _search_with_rewrite(
        env,
        task.user_ticket,
        task.context,
        top_k=primary_top_k,
        use_query_rewrite=use_query_rewrite,
        rewrite_queries_count=rewrite_queries_count,
        rewrite_mode=rewrite_mode,
        merge_rank_fusion=merge_rank_fusion,
    )
    expanded_query = f"{task.user_ticket} update supersede exception definition"
    expanded_results = env.search(expanded_query, top_k=primary_top_k)
    results = list(primary_results) + list(expanded_results)

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
    diag = {
        "primary_search_results": _enrich_search_results(env, primary_results),
        "primary_search_top_k": primary_top_k,
        "primary_search_query": task.user_ticket,
        "rewrite_used": use_query_rewrite,
        "rewrite_queries": rewrite_queries,
        "secondary_search_results": _enrich_search_results(env, expanded_results),
        "secondary_search_query": expanded_query,
    }
    if search_variants:
        diag["primary_search_variants"] = search_variants
    return prediction, opened_ids, prompt, raw, diag


def run_oracle(
    task: Any,
    env: PolicyOpsEnv,
    client: LLMClient,
    primary_top_k: int = 20,
    use_query_rewrite: bool = True,
    rewrite_queries_count: int = 3,
    rewrite_mode: str = "expanded",
    merge_rank_fusion: str = "max",
) -> Tuple[Dict[str, Any], List[str], str, str | None, Dict[str, Any]]:
    gold_ids = list(task.gold.gold_evidence or [])
    opened: List[Dict[str, Any]] = []
    opened_ids: List[str] = []
    opened_set: set[str] = set()
    used_search_fallback = False
    rewrite_queries = [task.user_ticket]
    search_variants: List[Dict[str, Any]] = []

    for clause_id in gold_ids:
        if len(opened_ids) >= env.open_budget:
            break
        if clause_id in opened_set:
            continue
        try:
            clause = env.open(clause_id)
        except RuntimeError as exc:
            if "budget" in str(exc):
                break
            continue
        except Exception:
            continue
        opened.append(clause)
        opened_ids.append(clause["clause_id"])
        opened_set.add(clause["clause_id"])

    primary_results: List[Dict[str, Any]] = []
    if len(opened_ids) < env.open_budget:
        try:
            results, rewrite_queries, search_variants = _search_with_rewrite(
                env,
                task.user_ticket,
                task.context,
                top_k=primary_top_k,
                use_query_rewrite=use_query_rewrite,
                rewrite_queries_count=rewrite_queries_count,
                rewrite_mode=rewrite_mode,
                merge_rank_fusion=merge_rank_fusion,
            )
            used_search_fallback = True
        except Exception:
            results = []
            rewrite_queries = [task.user_ticket]
            search_variants = []
        primary_results = results
        for item in results:
            if len(opened_ids) >= env.open_budget:
                break
            clause_id = item.get("clause_id")
            if not clause_id or clause_id in opened_set:
                continue
            try:
                clause = env.open(clause_id)
            except RuntimeError as exc:
                if "budget" in str(exc):
                    break
                continue
            except Exception:
                continue
            opened.append(clause)
            opened_ids.append(clause["clause_id"])
            opened_set.add(clause["clause_id"])

    prompt = _build_prompt(task.user_ticket, opened)
    raw_output = client.generate(prompt)
    prediction = _parse_prediction(raw_output)
    gold_set = set(gold_ids)
    opened_gold_count = len(set(opened_ids) & gold_set)
    coverage = opened_gold_count / max(1, len(gold_set))
    oracle_meta = {
        "oracle_opened_gold_count": opened_gold_count,
        "oracle_gold_coverage": coverage,
        "oracle_used_search_fallback": used_search_fallback,
        "primary_search_results": _enrich_search_results(env, primary_results),
        "primary_search_top_k": primary_top_k,
        "primary_search_query": task.user_ticket,
        "rewrite_used": use_query_rewrite,
        "rewrite_queries": rewrite_queries,
    }
    if search_variants:
        oracle_meta["primary_search_variants"] = search_variants
    return prediction, opened_ids, prompt, raw_output, oracle_meta


def summarize_clause_history(clauses: List[Clause], max_items: int = 8) -> str:
    if not clauses:
        return "No prior clauses available."
    lines: List[str] = []
    for clause in clauses[:max_items]:
        decision = clause.effect.get("decision") if clause.effect else None
        conditions = clause.conditions or []
        cond_text = ", ".join(conditions[:3]) if conditions else "none"
        lines.append(
            f"{clause.clause_id}: {clause.kind} decision={decision} conditions={cond_text}"
        )
    return " | ".join(lines)


def run_similarity_only(
    task: Any,
    env: PolicyOpsEnv,
    client: LLMClient,
    primary_top_k: int = 20,
    use_query_rewrite: bool = True,
    rewrite_queries_count: int = 3,
    rewrite_mode: str = "expanded",
    merge_rank_fusion: str = "max",
) -> Tuple[Dict[str, Any], List[str], str, str | None, Dict[str, Any]]:
    # Fallback to topk retrieval for non-threaded usage.
    prediction, opened_ids, prompt, raw, diag = run_topk_rag(
        task,
        env,
        client,
        primary_top_k=primary_top_k,
        use_query_rewrite=use_query_rewrite,
        rewrite_queries_count=rewrite_queries_count,
        rewrite_mode=rewrite_mode,
        merge_rank_fusion=merge_rank_fusion,
    )
    diag["baseline_mode"] = "similarity_only"
    return prediction, opened_ids, prompt, raw, diag


def run_agent_fold(
    task: Any,
    env: PolicyOpsEnv,
    client: LLMClient,
    primary_top_k: int = 20,
    use_query_rewrite: bool = True,
    rewrite_queries_count: int = 3,
    rewrite_mode: str = "expanded",
    merge_rank_fusion: str = "max",
) -> Tuple[Dict[str, Any], List[str], str, str | None, Dict[str, Any]]:
    prediction, opened_ids, prompt, raw, diag = run_topk_rag(
        task,
        env,
        client,
        primary_top_k=primary_top_k,
        use_query_rewrite=use_query_rewrite,
        rewrite_queries_count=rewrite_queries_count,
        rewrite_mode=rewrite_mode,
        merge_rank_fusion=merge_rank_fusion,
    )
    diag["baseline_mode"] = "agent_fold"
    return prediction, opened_ids, prompt, raw, diag

def run_goc_heuristic(
    task: Any,
    env: PolicyOpsEnv,
    client: LLMClient,
    controller: Optional[Controller] = None,
    controller_mode: str = "off",
    primary_top_k: int = 20,
    force_open_top_n: int = 1,
    force_open_source: str = "primary",
    use_query_rewrite: bool = True,
    rewrite_queries_count: int = 3,
    controller_policy: str = "bandit",
    rewrite_mode: str = "expanded",
    merge_rank_fusion: str = "max",
    agent_query_policy: str = "single_hop",
    hop1_query_mode: str = "stripped",
    bridge_reward_bonus: float = 0.0,
    open_split_mode: str = "all_union_rank",
    open_split_hop1: int = 0,
    open_policy: str = "current",
    save_internal_graph: bool = False,
    internal_budget_active: int = 1200,
    internal_budget_unfold: int = 650,
    internal_unfold_k: int = 8,
) -> Tuple[Dict[str, Any], List[str], str, str | None, str | None, str | None, Dict[str, Any]]:
    errors: List[str] = []
    fallback_prediction = {
        "decision": "needs_more_info",
        "conditions": [],
        "evidence": [],
        "customer_message": "",
    }
    reward_breakdown: Dict[str, float] | None = None
    hop1_query = task.user_ticket
    hop2_query = ""
    hop2_candidate_query = ""
    used_canonical_terms: List[str] = []
    used_update_kw = False
    hop2_executed = False
    hop2_query_contains_canonical = False
    hop2_skip_reason: str | None = None
    extracted_facets: Dict[str, List[str]] = {}
    bridge_probe_clause_id: str | None = None
    bridge_gold_clause_id: str | None = getattr(task, "bridge_clause_id", None)
    opened_probe_ids: List[str] = []
    opened_for_prompt_ids: List[str] = []
    opened_total_ids: List[str] = []
    opened_set: set[str] = set()
    opened_probe_clauses: List[Dict[str, Any]] = []
    opened_for_prompt_clauses: List[Dict[str, Any]] = []
    bridge_opened_in_probe = False
    bridge_opened_in_prompt = False
    bridge_probe_is_slot_specific = False
    bridge_needed = bool(getattr(task, "bridge_clause_id", None))
    scenario_mode = getattr(task, "scenario_mode", "v0")
    bridge_clause = None  # selected bridge clause (if any)
    hop1_results: List[Dict[str, Any]] = []
    hop2_results: List[Dict[str, Any]] = []
    bridge_open_cap_hit = False
    meta_avoided_count = 0
    fallback_reason: str | None = None
    hop2_pool_used_count = 0
    goc_internal_snapshots: List[Dict[str, Any]] = []
    goc_internal_mem = None

    if save_internal_graph:
        try:
            from src.memory import GoCMemory  # type: ignore
        except Exception:
            GoCMemory = None  # type: ignore
        if GoCMemory is not None:
            try:
                goc_internal_mem = GoCMemory(
                    budget_active=int(internal_budget_active),
                    budget_unfold=int(internal_budget_unfold),
                    unfold_k=int(internal_unfold_k),
                )
            except Exception:
                goc_internal_mem = None

    def _append_internal_snapshot(kind: str) -> None:
        if goc_internal_mem is None:
            return
        try:
            snap = goc_internal_mem.snapshot()  # type: ignore[attr-defined]
            if not isinstance(snap, dict):
                return
            snap["snapshot_kind"] = str(kind)
            snap["snapshot_idx"] = int(len(goc_internal_snapshots) + 1)
            goc_internal_snapshots.append(snap)
        except Exception:
            pass

    def _record_internal_tool(
        tool_name: str,
        args_obj: Dict[str, Any],
        observation: str,
        *,
        docids: Optional[List[str]] = None,
        storage_text: Optional[str] = None,
    ) -> None:
        if goc_internal_mem is None:
            return
        try:
            goc_internal_mem.record_tool(  # type: ignore[attr-defined]
                tool_name=str(tool_name),
                args=dict(args_obj or {}),
                observation=str(observation or ""),
                docids=[str(d) for d in (docids or []) if d],
                storage_text=storage_text,
            )
            _append_internal_snapshot("step")
        except Exception:
            pass

    if goc_internal_mem is not None:
        _orig_search = env.search
        _orig_open = env.open

        def _search_wrapped(
            query: str,
            filters: Optional[Dict[str, Any]] = None,
            top_k: int = 10,
        ) -> List[Dict[str, Any]]:
            results = _orig_search(query=query, filters=filters, top_k=top_k)
            try:
                lines: List[str] = []
                q_docids: List[str] = []
                for item in results[:20]:
                    cid = item.get("clause_id")
                    if cid:
                        q_docids.append(str(cid))
                    score = item.get("score")
                    snippet = str(item.get("snippet") or "").replace("\n", " ").strip()
                    if len(snippet) > 220:
                        snippet = snippet[:220] + "..."
                    lines.append(f"clause_id={cid} | score={score} | {snippet}")
                _record_internal_tool(
                    "search",
                    {"query": query, "filters": filters or {}, "top_k": int(top_k)},
                    "\n".join(lines) if lines else "[]",
                    docids=q_docids,
                )
            except Exception:
                pass
            return results

        def _open_wrapped(clause_id: str) -> Dict[str, Any]:
            opened = _orig_open(clause_id=clause_id)
            try:
                clause_obj = env.world.clauses.get(clause_id)
                doc_ref = str(getattr(clause_obj, "doc_id", "") or "")
                obs_lines = [f"CLAUSE_ID: {clause_id}"]
                if doc_ref:
                    obs_lines.append(f"DOC_ID: {doc_ref}")
                obs_lines.append(f"TEXT: {str(opened.get('text') or '')[:800]}")
                docids = [str(clause_id)]
                if doc_ref:
                    docids.append(doc_ref)
                _record_internal_tool(
                    "open_page",
                    {"clause_id": clause_id},
                    "\n".join(obs_lines),
                    docids=docids,
                    storage_text=str(opened.get("text") or ""),
                )
            except Exception:
                pass
            return opened

        env.search = _search_wrapped  # type: ignore[assignment]
        env.open = _open_wrapped  # type: ignore[assignment]

    def _strip_ticket(text: str) -> str:
        lowered = text.lower()
        for prefix in [
            "customer asks about",
            "please advise",
            "need to know if",
            "we want to",
            "can we",
            "is it allowed to",
        ]:
            if lowered.startswith(prefix):
                text = text[len(prefix) :].strip()
                break
        tokens = [t for t in re.findall(r"[a-zA-Z0-9_']+", text.lower()) if len(t) > 2]
        return " ".join(tokens[:16]) if tokens else text

    def _llm_keywords(ticket: str) -> str:
        prompt = (
            "Extract key search keywords from this ticket. "
            'Return JSON: {"keywords":["..."],"slot_guess":"...","needs_update":true|false}.\n'
            f"Ticket: {ticket}\nJSON:"
        )
        try:
            raw = client.generate(prompt)
            data = json.loads(raw)
            keywords = data.get("keywords", []) if isinstance(data, dict) else []
            if isinstance(keywords, list) and keywords:
                return " ".join([str(k) for k in keywords][:16])
        except Exception:
            pass
        return _strip_ticket(ticket)

    def _build_hop1_query(ticket: str) -> str:
        if hop1_query_mode == "raw":
            return ticket
        if hop1_query_mode == "llm_keywords":
            return _llm_keywords(ticket)
        return _strip_ticket(ticket)
    if agent_query_policy == "two_hop_bridge":
        try:
            hop1_query = _build_hop1_query(task.user_ticket)
            hop1_results = env.search(hop1_query, top_k=primary_top_k)
        except Exception as exc:  # noqa: BLE001 - defensive
            hop1_results = []
            errors.append(f"search_error: {exc}")
        # Open first bridge doc to extract canonical terms.
        if scenario_mode == "bridged_v1_1" and bridge_needed:
            preferred = [
                item
                for item in hop1_results
                if item.get("is_bridge_doc") and item.get("bridge_for_slot") == task.context.get("slot")
            ]
            fallback = [item for item in hop1_results if item.get("is_bridge_doc")]
            for item in preferred or fallback:
                cid = item.get("clause_id")
                clause = env.world.clauses.get(cid) if cid else None
                if not clause:
                    continue
                if len(opened_set) >= env.open_budget:
                    break
                if cid in opened_set:
                    continue
                try:
                    opened = env.open(cid)
                    opened_probe_clauses.append(opened)
                except Exception as exc:
                    errors.append(f"open_error: {exc}")
                    continue
                opened_probe_ids.append(cid)
                opened_total_ids.append(cid)
                opened_set.add(cid)
                bridge_probe_clause_id = cid
                bridge_clause = clause
                bridge_opened_in_probe = True
                bridge_probe_is_slot_specific = (
                    clause.bridge_for_slot == task.context.get("slot")
                    if clause.bridge_for_slot
                    else False
                )
                if clause.canonical_terms:
                    used_canonical_terms = list(clause.canonical_terms)
                else:
                    used_canonical_terms = _extract_canonical_from_text(clause.text)
                break
        extracted_facets = extract_facets(task.user_ticket)
        if bridge_needed:
            hop2_terms: List[str] = []
            for term in used_canonical_terms[:2]:
                if term and term not in hop2_terms:
                    hop2_terms.append(term)
            slot_term = task.context.get("slot")
            if slot_term and slot_term not in hop2_terms:
                hop2_terms.append(slot_term)
            facet_terms: List[str] = []
            for key in ["region", "purpose", "data_type", "product", "tier", "action"]:
                facet_terms.extend(extracted_facets.get(key, []))
            for term in facet_terms:
                if term and term not in hop2_terms:
                    hop2_terms.append(term)
                if len(hop2_terms) >= 6:
                    break
            if getattr(task, "needs_update_resolution", False):
                hop2_terms.extend(["effective immediately", "supersedes"])
                used_update_kw = True
            hop2_candidate_query = " ".join(hop2_terms).strip()
            hop2_query_contains_canonical = any(
                term.lower() in hop2_candidate_query.lower() for term in used_canonical_terms
            )
            hop2_token_count = len(re.findall(r"[A-Za-z0-9_']+", hop2_candidate_query))
            hop2_has_signal = bool(hop2_query_contains_canonical or used_update_kw)
            if env.open_budget <= 0:
                hop2_skip_reason = "budget"
            elif not used_canonical_terms:
                hop2_skip_reason = "no_canonical_extracted"
            elif hop2_token_count < 3:
                hop2_skip_reason = "too_short"
            elif not hop2_has_signal:
                hop2_skip_reason = "other"
            if (
                hop2_candidate_query
                and hop2_token_count >= 3
                and hop2_has_signal
                and env.open_budget > 0
            ):
                hop2_executed = True
                hop2_query = hop2_candidate_query
                try:
                    hop2_results = env.search(hop2_query, top_k=primary_top_k)
                except Exception as exc:  # noqa: BLE001 - defensive
                    hop2_results = []
                    errors.append(f"search_error: {exc}")
            else:
                hop2_query = ""
                hop2_results = []
                if hop2_skip_reason is None:
                    hop2_skip_reason = "other"
        else:
            hop2_skip_reason = "disabled"
            hop2_query = ""
            hop2_results = []
        seed_results = _merge_hybrid_results(
            hop1_results,
            hop2_results,
            top_k=primary_top_k,
            merge_rank_fusion=merge_rank_fusion,
        )
        rewrite_queries = [hop1_query, hop2_query] if hop2_query else [hop1_query]
        search_variants = [
            {"stage": "hop1_base", "query": hop1_query, "results": hop1_results},
        ]
        if hop2_executed:
            search_variants.append(
                {
                    "stage": "hop2_canonical",
                    "query": hop2_query,
                    "results": hop2_results,
                    "query_meta": {
                        "used_canonical_terms": used_canonical_terms,
                        "used_update_kw": used_update_kw,
                        "extracted_facets": extracted_facets,
                        "hop2_executed": hop2_executed,
                    },
                    "depends_on": f"query:{task.task_id}:hop1_base",
                }
            )
        search_variants.append({"stage": "hybrid_merged", "query": "hybrid_merged", "results": seed_results})
    else:
        try:
            seed_results, rewrite_queries, search_variants = _search_with_rewrite(
                env,
                task.user_ticket,
                task.context,
                top_k=primary_top_k,
                use_query_rewrite=use_query_rewrite,
                rewrite_queries_count=rewrite_queries_count,
                rewrite_mode=rewrite_mode,
                merge_rank_fusion=merge_rank_fusion,
            )
            # Treat the primary search results as hop1 for single-hop variants.
            hop1_results = list(seed_results)
        except Exception as exc:  # noqa: BLE001 - defensive
            seed_results = []
            errors.append(f"search_error: {exc}")
            rewrite_queries = [task.user_ticket]
            search_variants = []
        hop2_skip_reason = "disabled"
    expanded_query = None
    expanded_results: List[Dict[str, Any]] = []
    if agent_query_policy != "two_hop_bridge":
        expanded_query = f"{task.user_ticket} exception supersede revoke definition effective immediately"
        try:
            expanded_results = env.search(expanded_query, top_k=primary_top_k)
        except Exception as exc:  # noqa: BLE001 - defensive
            expanded_results = []
            errors.append(f"search_error: {exc}")

    merged = {item["clause_id"]: item for item in seed_results}
    for item in expanded_results:
        merged.setdefault(item["clause_id"], item)

    controller_action: str | None = None
    ordered_clause_ids: List[str] = []
    rerank_scores: Dict[str, float] | None = None
    if controller and controller_mode != "off":
        if controller_policy == "rerank" and isinstance(controller, RerankController):
            rerank_scores = controller.score_candidates(
                seed_results, expanded_results, env.world, task.context
            )
            ordered_clause_ids = [
                cid for cid, _ in sorted(rerank_scores.items(), key=lambda kv: kv[1], reverse=True)
            ]
        else:
            search_stats = controller.compute_search_stats(seed_results)
            context_features = controller.build_context_features(task.context, env.open_budget, search_stats)
            controller_action = controller.select_action(
                context_features, explore=controller_mode == "train"
            )
            ordered_clause_ids = controller.rank_clause_ids(
                controller_action, seed_results, expanded_results, clause_lookup=env.world.clauses
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
    forced_open_ids: List[str] = []
    hop1_id_set = {item.get("clause_id") for item in hop1_results if item.get("clause_id")}
    hop2_id_set = {item.get("clause_id") for item in hop2_results if item.get("clause_id")}
    open_from_hop1_ids: List[str] = []
    open_from_hop2_ids: List[str] = []

    def _is_bridge_clause(cid: str) -> bool:
        clause_obj = env.world.clauses.get(cid)
        if not clause_obj:
            return False
        return bool(
            getattr(clause_obj, "is_bridge_doc", False)
            or getattr(clause_obj, "bridge_for_slot", None)
            or getattr(clause_obj, "kind", None) in {"definition", "glossary"}
        )

    def _is_meta_clause(cid: str) -> bool:
        clause_obj = env.world.clauses.get(cid)
        if not clause_obj:
            return False
        return getattr(clause_obj, "kind", None) == "priority" or getattr(clause_obj, "slot", None) == "meta"

    if force_open_top_n > 0:
        if force_open_source == "merged":
            force_candidates = ranked
        else:
            force_candidates = sorted(seed_results, key=lambda item: item.get("score", 0.0), reverse=True)
        force_has_non_meta = any(
            item.get("clause_id") and not _is_meta_clause(item.get("clause_id"))
            for item in force_candidates
        )
        for item in force_candidates[:force_open_top_n]:
            if len(opened_set) >= env.open_budget:
                break
            clause_id = item.get("clause_id")
            if not clause_id or clause_id in opened_set:
                continue
            if open_policy == "bridge_one_only":
                if _is_bridge_clause(clause_id) and any(_is_bridge_clause(cid) for cid in opened_set):
                    bridge_open_cap_hit = True
                    continue
                if _is_meta_clause(clause_id) and force_has_non_meta:
                    meta_avoided_count += 1
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
            opened_for_prompt_clauses.append(clause)
            opened_set.add(clause["clause_id"])
            opened_for_prompt_ids.append(clause["clause_id"])
            opened_total_ids.append(clause["clause_id"])
            forced_open_ids.append(clause["clause_id"])
            if clause_id in hop2_id_set:
                open_from_hop2_ids.append(clause_id)
            elif clause_id in hop1_id_set:
                open_from_hop1_ids.append(clause_id)
    hop1_ranked_ids = [
        item.get("clause_id")
        for item in sorted(hop1_results, key=lambda it: it.get("score", 0.0), reverse=True)
        if item.get("clause_id")
    ]
    hop2_ranked_ids = [
        item.get("clause_id")
        for item in sorted(hop2_results, key=lambda it: it.get("score", 0.0), reverse=True)
        if item.get("clause_id")
    ]

    if open_policy == "core_first_heuristic":
        open_policy = "soft_core_rerank"

    def _open_from_candidates(
        candidates: List[str],
        max_count: int,
        origin: str | None = None,
        allow_bridge: bool = True,
    ) -> None:
        nonlocal opened_for_prompt_clauses
        pending = list(candidates)
        safety = 0
        while pending:
            if len(opened_set) >= env.open_budget:
                break
            if max_count <= 0:
                break
            safety += 1
            if safety > len(candidates) * 3:
                break
            clause_id = pending.pop(0)
            if clause_id in opened_set:
                continue
            clause_obj = env.world.clauses.get(clause_id)
            if not allow_bridge and clause_obj and clause_obj.is_bridge_doc:
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
            opened_for_prompt_clauses.append(clause)
            opened_set.add(clause["clause_id"])
            opened_for_prompt_ids.append(clause["clause_id"])
            opened_total_ids.append(clause["clause_id"])
            if origin == "hop1":
                open_from_hop1_ids.append(clause_id)
            elif origin == "hop2":
                open_from_hop2_ids.append(clause_id)
            else:
                if clause_id in hop2_id_set:
                    open_from_hop2_ids.append(clause_id)
                elif clause_id in hop1_id_set:
                    open_from_hop1_ids.append(clause_id)
            max_count -= 1

    if open_policy == "oracle_open_if_in_union":
        from .diagnostics import merge_search_results_union

        union_results = merge_search_results_union(hop1_results, hop2_results)
        union_ids = [item.get("clause_id") for item in union_results if item.get("clause_id")]
        winning_clause = task.gold.gold_evidence[0] if task.gold.gold_evidence else None
        core_ids = list(getattr(task.gold, "gold_evidence_core", None) or task.gold.gold_evidence or [])
        preferred: List[str] = []
        if winning_clause and winning_clause in union_ids:
            preferred.append(winning_clause)
        for cid in union_ids:
            if cid in core_ids and cid not in preferred:
                preferred.append(cid)
        _open_from_candidates(preferred, env.open_budget, origin=None)
    elif open_policy == "bridge_one_only":
        if ordered_clause_ids:
            ranked_ids = ordered_clause_ids
        else:
            ranked_ids = [item["clause_id"] for item in ranked]
        if agent_query_policy == "two_hop_bridge" and hop2_ranked_ids:
            candidate_ids = hop2_ranked_ids + [cid for cid in ranked_ids if cid not in hop2_ranked_ids]
        else:
            candidate_ids = list(ranked_ids)
            if agent_query_policy == "two_hop_bridge" and not hop2_ranked_ids:
                fallback_reason = "hop2_empty"
        non_meta_candidates = [cid for cid in candidate_ids if not _is_meta_clause(cid)]
        only_meta = len(non_meta_candidates) == 0
        if only_meta:
            fallback_reason = "only_meta" if fallback_reason is None else f"{fallback_reason}|only_meta"
        filtered_ids: List[str] = []
        bridge_used = sum(1 for cid in opened_set if _is_bridge_clause(cid))
        for cid in candidate_ids:
            if len(opened_set) + len(filtered_ids) >= env.open_budget:
                break
            if cid in opened_set:
                continue
            if _is_bridge_clause(cid):
                if bridge_used >= 1:
                    bridge_open_cap_hit = True
                    continue
            if _is_meta_clause(cid) and not only_meta:
                meta_avoided_count += 1
                continue
            filtered_ids.append(cid)
            if _is_bridge_clause(cid):
                bridge_used += 1
        _open_from_candidates(filtered_ids, env.open_budget - len(opened_set), origin=None, allow_bridge=True)
    elif open_policy == "hop2_priority" and agent_query_policy == "two_hop_bridge":
        remaining = max(0, env.open_budget - len(opened_set))
        if hop2_ranked_ids:
            _open_from_candidates(hop2_ranked_ids, remaining, origin="hop2", allow_bridge=False)
        else:
            _open_from_candidates(hop1_ranked_ids, remaining, origin="hop1", allow_bridge=False)
    elif open_split_mode == "split_hop1_hop2" and agent_query_policy == "two_hop_bridge":
        hop1_quota = max(0, min(open_split_hop1, env.open_budget))
        hop2_quota = max(0, env.open_budget - hop1_quota)
        _open_from_candidates(hop1_ranked_ids, hop1_quota, origin="hop1")
        _open_from_candidates(hop2_ranked_ids, hop2_quota, origin="hop2")
    else:
        if ordered_clause_ids:
            ranked_ids = ordered_clause_ids
        else:
            ranked_ids = [item["clause_id"] for item in ranked]
        if agent_query_policy == "two_hop_bridge" and hop2_id_set:
            pref = [cid for cid in hop2_results[: min(3, primary_top_k)] if cid.get("clause_id")]
            hop2_pref_ids = [item.get("clause_id") for item in pref if item.get("clause_id")]
            ranked_ids = hop2_pref_ids + [cid for cid in ranked_ids if cid not in hop2_pref_ids]
        if open_policy == "soft_core_rerank":
            base_ranks = {cid: idx for idx, cid in enumerate(ranked_ids)}
            fixed = ranked_ids[:2]

            def _soft_bonus(cid: str) -> tuple[int, int]:
                clause_obj = env.world.clauses.get(cid)
                bonus = 0
                if clause_obj:
                    if clause_obj.kind in {"update", "exception"}:
                        bonus += 3
                    elif clause_obj.kind == "rule":
                        bonus += 1
                    decision = clause_obj.effect.get("decision") if clause_obj.effect else None
                    if decision in {"deny", "allow"}:
                        bonus += 1
                    text = clause_obj.text.lower()
                    slot_hint = (task.slot_hint_alias or "").lower()
                    canonical = (task.canonical_slot_term or "").lower()
                    if slot_hint and slot_hint in text:
                        bonus += 1
                    if canonical and canonical in text:
                        bonus += 1
                item = merged.get(cid)
                snippet = (item.get("snippet", "") if item else "").lower()
                if any(word in snippet for word in ["update", "revoke", "supersede", "amend", "effective"]):
                    bonus += 1
                if any(word in snippet for word in ["except", "unless", "however", "provided that"]):
                    bonus += 1
                return (bonus, -base_ranks.get(cid, 0))

            rest = [cid for cid in ranked_ids if cid not in fixed]
            rest_sorted = sorted(rest, key=_soft_bonus, reverse=True)
            ranked_ids = fixed + rest_sorted
        _open_from_candidates(ranked_ids, env.open_budget, origin=None, allow_bridge=True)

    if bridge_probe_clause_id:
        bridge_opened_in_prompt = bridge_probe_clause_id in opened_for_prompt_ids

    hop2_pool_used_count = len(open_from_hop2_ids)
    opened_ids = list(opened_total_ids)
    prompt = _build_goc_prompt(task.user_ticket, opened_for_prompt_clauses, opened_for_prompt_ids)
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

    if controller and controller_mode == "train":
        if controller_policy == "rerank" and isinstance(controller, RerankController):
            controller.update_weights(seed_results, expanded_results, env.world, task.context, task.gold)
        elif controller_action:
            gold_ids = set(task.gold.gold_evidence or [])
            opened_id_set = set(opened_total_ids or opened_ids)
            winning_clause = task.gold.gold_evidence[0] if task.gold.gold_evidence else None
            opened_has_winning_clause = 1.0 if winning_clause and winning_clause in opened_id_set else 0.0
            coverage = len(opened_id_set & gold_ids) / max(1, len(gold_ids))
            r_win = 1.0 * opened_has_winning_clause
            r_cov = 0.5 * coverage
            r_open_penalty = -0.01 * float(env.open_count)
            r_bridge = 0.0
            if bridge_reward_bonus and bridge_probe_clause_id and bridge_probe_clause_id in opened_id_set:
                r_bridge = float(bridge_reward_bonus)
            reward = r_win + r_cov + r_open_penalty + r_bridge
            search_stats = controller.compute_search_stats(seed_results)
            context_features = controller.build_context_features(task.context, env.open_budget, search_stats)
            controller.update(controller_action, context_features, reward)
            reward_breakdown = {
                "r_win": r_win,
                "r_cov": r_cov,
                "r_open_penalty": r_open_penalty,
                "r_bridge": r_bridge,
                "r_total": reward,
            }

    error = "; ".join(errors) if errors else None
    opened_id_set = set(opened_total_ids or opened_ids)
    bridge_opened_any = any(
        env.world.clauses.get(cid)
        and (
            env.world.clauses[cid].kind in {"definition", "glossary"}
            or env.world.clauses[cid].bridge_for_slot
            or getattr(env.world.clauses[cid], "is_bridge_doc", False)
        )
        for cid in opened_id_set
    )
    diag = {
        "primary_search_results": _enrich_search_results(env, seed_results),
        "primary_search_top_k": primary_top_k,
        "primary_search_query": task.user_ticket,
        "hop1_search_results": _enrich_search_results(env, hop1_results),
        "hop2_search_results": _enrich_search_results(env, hop2_results),
        "secondary_search_results": _enrich_search_results(env, expanded_results),
        "secondary_search_query": expanded_query,
        "forced_open_ids": forced_open_ids,
        "rewrite_used": use_query_rewrite,
        "rewrite_queries": rewrite_queries,
        "agent_query_policy": agent_query_policy,
        "bridge_gold_clause_id": bridge_gold_clause_id,
        "bridge_probe_clause_id": bridge_probe_clause_id,
        "bridge_probe_is_slot_specific": bridge_probe_is_slot_specific,
        "bridge_needed": bridge_needed,
        "bridge_opened_any": bridge_opened_any,
        "bridge_opened_gold": bool(bridge_gold_clause_id and bridge_gold_clause_id in opened_id_set),
        "canonical_used_in_query2": hop2_query_contains_canonical,
        "used_canonical_terms": used_canonical_terms,
        "used_update_kw": used_update_kw,
        "hop2_candidate_query": hop2_candidate_query,
        "hop2_query": hop2_query,
        "hop2_executed": hop2_executed,
        "hop2_query_contains_canonical": hop2_query_contains_canonical,
        "hop2_skip_reason": hop2_skip_reason,
        "extracted_facets": extracted_facets,
        "hop1_query_mode": hop1_query_mode,
        "hop1_query_text": hop1_query if agent_query_policy == "two_hop_bridge" else "",
        "hop1_query_contains_alias": bool(getattr(task, "slot_hint_alias", None))
        and getattr(task, "slot_hint_alias", "") in (hop1_query or ""),
        "hop1_query_contains_canonical": bool(getattr(task, "canonical_slot_term", None))
        and getattr(task, "canonical_slot_term", "") in (hop1_query or ""),
        "open_split_mode": open_split_mode,
        "open_split_hop1": open_split_hop1,
        "opened_probe_clause_ids": opened_probe_ids,
        "opened_for_prompt_clause_ids": opened_for_prompt_ids,
        "opened_total_clause_ids": opened_total_ids or opened_ids,
        "open_from_hop1_count": len(open_from_hop1_ids),
        "open_from_hop2_count": len(open_from_hop2_ids),
        "prompt_includes_from_hop2": bool(open_from_hop2_ids),
        "opened_doc_ids_hop1": open_from_hop1_ids[:5],
        "opened_doc_ids_hop2": open_from_hop2_ids[:5],
        "bridge_opened": bool(bridge_probe_clause_id and bridge_probe_clause_id in opened_id_set),
        "bridge_opened_in_probe": bridge_opened_in_probe,
        "bridge_opened_in_prompt": bridge_opened_in_prompt,
        "bridge_open_cap_hit": bridge_open_cap_hit,
        "meta_avoided_count": meta_avoided_count,
        "hop2_pool_used_count": hop2_pool_used_count,
        "fallback_reason": fallback_reason,
    }
    if reward_breakdown:
        diag["controller_reward_breakdown"] = reward_breakdown
    if search_variants:
        diag["primary_search_variants"] = search_variants
    if rerank_scores is not None:
        diag["rerank_used"] = True
        diag["rerank_top_score"] = max(rerank_scores.values()) if rerank_scores else None
    if save_internal_graph:
        _append_internal_snapshot("final")
        diag["goc_internal_snapshots"] = list(goc_internal_snapshots)
    return prediction, opened_ids, prompt, raw_output, error, controller_action, diag
