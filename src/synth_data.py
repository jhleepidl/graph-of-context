from __future__ import annotations
from typing import List, Dict, Any
import random
import json
import string

ATTRS = ["start_year", "headquarters", "lead", "code_name", "key_number"]
ALIAS_PREFIXES = [
    "Beacon", "Nimbus", "Vector", "Harbor", "Kite", "Juniper", "Quartz", "Atlas",
    "Signal", "Delta", "Falcon", "Mosaic", "Cinder", "Echo", "Nova", "Pioneer",
]


def gen_long_description(rng: random.Random, name: str, n_words: int = 260) -> str:
    """Generate a long, mostly-irrelevant description to inflate tool observations.
    This makes FullHistory expensive while keeping facts in compact key-value lines.
    """
    words: List[str] = []
    for _ in range(max(0, n_words)):
        if rng.random() < 0.06:
            words.append(name)
        else:
            words.append(rand_word(rng, rng.randint(3, 8)))
    chunks = []
    step = 60
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i:i + step]))
    return "\n\n".join(chunks)


def rand_word(rng: random.Random, n: int = 6) -> str:
    letters = string.ascii_uppercase
    return "".join(rng.choice(letters) for _ in range(n))


def gen_alias_handle(rng: random.Random, i: int) -> str:
    prefix = rng.choice(ALIAS_PREFIXES)
    return f"{prefix}-{i:03d}-{rng.randint(10, 99)}"


def gen_policy_tag(rng: random.Random, task_id: str) -> str:
    short = rand_word(rng, 4)
    return f"POLICY-{task_id.replace('_', '')}-{short}"


def gen_ticket_id(rng: random.Random, prefix: str, i: int) -> str:
    return f"{prefix}-{i:04d}-{rand_word(rng, 3)}"


def gen_entity(rng: random.Random, i: int) -> Dict[str, Any]:
    name = f"Project_{i:04d}"
    headquarters = f"City_{rng.randint(1, 60)}"
    relocation_city = f"City_{rng.randint(1, 60)}"
    while relocation_city == headquarters:
        relocation_city = f"City_{rng.randint(1, 60)}"
    return {
        "name": name,
        "alias_handle": gen_alias_handle(rng, i),
        "start_year": rng.randint(1995, 2024),
        "headquarters": headquarters,
        "relocation_year": rng.randint(2005, 2025),
        "relocation_city": relocation_city,
        "lead": f"Lead_{rand_word(rng, 4)}",
        "code_name": f"Codename_{rand_word(rng, 5)}",
        "key_number": rng.randint(10, 9999),
    }


def doc_truth(entity: Dict[str, Any], docid: str) -> Dict[str, Any]:
    content = (
        f"{entity['name']} OFFICIAL PROFILE\n"
        f"start_year: {entity['start_year']}\n"
        f"headquarters: {entity['headquarters']}\n"
        f"lead: {entity['lead']}\n"
        f"code_name: {entity['code_name']}\n"
        f"key_number: {entity['key_number']}\n"
        f"related_projects: {', '.join(entity.get('related_projects', []))}\n"
        f"description: {entity.get('description', '')}\n"
        f"relocation_note: relocated_to {entity.get('relocation_city')} in {entity.get('relocation_year')}\n"
        f"NOTE: This document is authoritative."
    )
    return {"docid": docid, "url": f"https://local/{docid}", "title": f"{entity['name']} Official", "content": content}


def doc_distractor(entity: Dict[str, Any], docid: str, rng: random.Random) -> Dict[str, Any]:
    wrong_hq = f"City_{rng.randint(61, 90)}"
    wrong_year = entity["start_year"] + rng.choice([-3, -2, -1, 1, 2, 3])
    repeats = " ".join([entity["name"]] * rng.randint(8, 14))
    wrong_reloc_city = f"City_{rng.randint(61, 90)}"
    wrong_reloc_year = int(entity.get("relocation_year", 2015)) + rng.choice([-3, -2, -1, 1, 2, 3])

    content = (
        f"{repeats}\n"
        f"start_year: {wrong_year}\n"
        f"headquarters: {wrong_hq}\n"
        f"lead: {entity['lead']}\n"
        f"code_name: {entity['code_name']}\n"
        f"key_number: {entity['key_number']}\n"
        f"related_projects: {', '.join(entity.get('related_projects', []))}\n"
        f"description: {entity.get('description', '')}\n"
        f"relocation_note: relocated_to {wrong_reloc_city} in {wrong_reloc_year}\n"
        f"Rumor blog; may be outdated."
    )
    return {"docid": docid, "url": f"https://local/{docid}", "title": f"{entity['name']} Rumor", "content": content}


def doc_alias_note(entity: Dict[str, Any], docid: str) -> Dict[str, Any]:
    content = (
        "FIELD NOTE\n"
        f"handle: {entity['alias_handle']}\n"
        f"canonical_project: {entity['name']}\n"
        "This note only resolves the handle to the canonical project record.\n"
        "Do not use this note for factual attributes like start_year or headquarters."
    )
    return {
        "docid": docid,
        "url": f"https://local/{docid}",
        "title": f"{entity['alias_handle']} Field Note",
        "content": content,
    }


def doc_policy_memo(
    task_id: str,
    docid: str,
    cutoff_year: int,
    archived: bool = False,
    policy_tag: str | None = None,
    scope_handles: List[str] | None = None,
    scope_projects: List[str] | None = None,
) -> Dict[str, Any]:
    status = "archived" if archived else "active"
    title_suffix = "Archived" if archived else "Current"
    tag = str(policy_tag or task_id)
    handles_line = f"scope_handles: {', '.join(scope_handles or [])}" if scope_handles else "scope_handles: none"
    projects_line = f"scope_projects: {', '.join(scope_projects or [])}" if scope_projects else "scope_projects: none"
    content = (
        f"CITY RESOLUTION POLICY MEMO ({status.upper()})\n"
        f"policy_id: {task_id}\n"
        f"policy_tag: {tag}\n"
        f"policy_state: {status}\n"
        f"{handles_line}\n"
        f"{projects_line}\n"
        f"When reporting the current operating city for the selected project, use relocation_city if relocation_year >= {cutoff_year}; otherwise use headquarters.\n"
        "Only ACTIVE/CURRENT memos should be used for final answers.\n"
        "This memo does not contain project-specific facts."
    )
    return {
        "docid": docid,
        "url": f"https://local/{docid}",
        "title": f"Current Operating City Policy [{tag}] {title_suffix}",
        "content": content,
    }


def doc_operating_city_approval(entity: Dict[str, Any], docid: str) -> Dict[str, Any]:
    content = (
        "OPERATING CITY APPROVAL\n"
        f"project: {entity['name']}\n"
        f"handle: {entity['alias_handle']}\n"
        f"approval_ticket: {entity['approval_ticket']}\n"
        f"effective_year: {entity['relocation_year']}\n"
        f"approved_operating_city: {entity['relocation_city']}\n"
        "This approval supersedes the default headquarters for operating-city reporting unless a later active exception notice overrides it.\n"
        "Use later exception or revocation notices for the same project if present."
    )
    return {
        "docid": docid,
        "url": f"https://local/{docid}",
        "title": f"Operating City Approval - {entity['name']}",
        "content": content,
    }


def doc_legacy_exception_notice(entity: Dict[str, Any], docid: str) -> Dict[str, Any]:
    content = (
        "LEGACY OPERATING EXCEPTION NOTICE\n"
        f"project: {entity['name']}\n"
        f"handle: {entity['alias_handle']}\n"
        f"exception_ticket: {entity['exception_ticket']}\n"
        f"issue_year: {entity['exception_year']}\n"
        f"override_city: {entity['headquarters']}\n"
        "This notice overrides the operating city approval while it remains active.\n"
        "A later Exception Revocation notice for the same project and exception_ticket cancels this exception."
    )
    return {
        "docid": docid,
        "url": f"https://local/{docid}",
        "title": f"Legacy Operating Exception - {entity['name']}",
        "content": content,
    }


def doc_exception_revocation_notice(entity: Dict[str, Any], docid: str, exception_docid: str) -> Dict[str, Any]:
    content = (
        "EXCEPTION REVOCATION NOTICE\n"
        f"project: {entity['name']}\n"
        f"handle: {entity['alias_handle']}\n"
        f"revocation_ticket: {entity['revocation_ticket']}\n"
        f"revokes_exception_ticket: {entity['exception_ticket']}\n"
        f"revokes_docid: {exception_docid}\n"
        f"revocation_year: {entity['revocation_year']}\n"
        "After this revocation, the operating city approval becomes current again."
    )
    return {
        "docid": docid,
        "url": f"https://local/{docid}",
        "title": f"Exception Revocation - {entity['name']}",
        "content": content,
    }


def doc_archived_status_board(entity: Dict[str, Any], docid: str, reported_city: str) -> Dict[str, Any]:
    content = (
        "ARCHIVED STATUS BOARD\n"
        f"project: {entity['name']}\n"
        f"handle: {entity['alias_handle']}\n"
        f"snapshot_year: {entity['archived_snapshot_year']}\n"
        f"reported_operating_city: {reported_city}\n"
        "status: archived\n"
        "This board is a stale snapshot and should not be used for current reporting decisions."
    )
    return {
        "docid": docid,
        "url": f"https://local/{docid}",
        "title": f"Archived Status Board - {entity['name']}",
        "content": content,
    }


def _final_operating_city(entity: Dict[str, Any]) -> str:
    if str(entity.get('exception_state', 'none')) == 'active':
        return str(entity.get('headquarters'))
    return str(entity.get('relocation_city'))


def doc_noise(docid: str, rng: random.Random) -> Dict[str, Any]:
    words = [rand_word(rng, rng.randint(3, 8)) for _ in range(rng.randint(200, 380))]
    content = " ".join(words)
    return {"docid": docid, "url": f"https://local/{docid}", "title": "Noise", "content": content}


def _operating_city(entity: Dict[str, Any], cutoff_year: int) -> str:
    return str(entity.get("relocation_city")) if int(entity.get("relocation_year", 0)) >= int(cutoff_year) else str(entity.get("headquarters"))


def _policy_cutoffs_for_target(target: Dict[str, Any]) -> tuple[int, int]:
    reloc_year = int(target.get("relocation_year", 2018))
    active_cutoff = reloc_year
    archived_cutoff = reloc_year + 2
    return active_cutoff, archived_cutoff


def _ensure_current_policy_docs(
    corpus: List[Dict[str, Any]],
    task_id: str,
    target: Dict[str, Any],
    rng: random.Random,
    scope_handles: List[str] | None = None,
    scope_projects: List[str] | None = None,
) -> tuple[str, str, int, str]:
    active_cutoff, archived_cutoff = _policy_cutoffs_for_target(target)
    active_docid = f"D_POLICY_{task_id}_CUR"
    archived_docid = f"D_POLICY_{task_id}_ARC"
    policy_tag = gen_policy_tag(rng, task_id)
    corpus.append(doc_policy_memo(task_id=task_id, docid=active_docid, cutoff_year=active_cutoff, archived=False, policy_tag=policy_tag, scope_handles=scope_handles, scope_projects=scope_projects))
    corpus.append(doc_policy_memo(task_id=task_id, docid=archived_docid, cutoff_year=archived_cutoff, archived=True, policy_tag=policy_tag, scope_handles=scope_handles, scope_projects=scope_projects))
    return active_docid, archived_docid, active_cutoff, policy_tag


def _hard_task_meta(task_type: str, active_policy_docid: str, archived_policy_docid: str, cutoff_year: int, policy_tag: str, benchmark_profile_name: str = "hard", extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "benchmark_profile": str(benchmark_profile_name or "hard"),
        "task_type": task_type,
        "task_slice": "memory_necessary",
        "needs_alias_resolution": True,
        "needs_rule_doc": True,
        "needs_multi_support": True,
        "has_stale_rule_distractor": True,
        "active_policy_docid": active_policy_docid,
        "archived_policy_docid": archived_policy_docid,
        "active_policy_cutoff_year": int(cutoff_year),
        "policy_tag": policy_tag,
    }
    if extra:
        base.update(extra)
    return base


def make_corpus_and_tasks(
    out_corpus_path: str,
    out_tasks_path: str,
    n_entities: int = 80,
    n_tasks: int = 50,
    distractors_per_entity: int = 2,
    noise_docs: int = 120,
    seed: int = 7,
    long_horizon: bool = False,
    long_desc_words: int = 320,
    related_degree: int = 3,
    n_projects_per_task: int = 10,
    hop_steps: int = 4,
    long_task_ratio: float = 0.7,
    late_binding: bool = False,
    late_binding_ratio: float = 0.5,
    late_binding_topn: int = 2,
    branch_merge: bool = False,
    branch_merge_ratio: float = 0.35,
    branch_merge_group_min: int = 2,
    benchmark_profile: str = "standard",
    hard_mode: bool = False,
    hard_compare_ratio: float = 0.35,
    hard_late_binding_ratio: float = 0.35,
    hard_branch_merge_ratio: float = 0.30,
    hard_compare_candidates: int | None = None,
    hard_late_candidates: int | None = None,
    hard_branch_candidates: int | None = None,
    structured_dependency_ratio: float = 0.35,
    structured_branch_ratio: float = 0.35,
    structured_compare_candidates: int | None = None,
    structured_dependency_candidates: int | None = None,
):
    rng = random.Random(seed)
    profile_name = str(benchmark_profile or "standard").strip().lower()
    hard_mode = bool(hard_mode or profile_name in {"hard", "hard_lite", "hard_extreme"})
    structured_mode = bool(profile_name in {"structured_lite", "structured", "structured_extreme"})
    if hard_mode:
        if profile_name == "hard_lite":
            default_compare_candidates = 4
            default_late_candidates = 5
            default_branch_candidates = 4
        elif profile_name == "hard_extreme":
            default_compare_candidates = max(8, int(n_projects_per_task))
            default_late_candidates = max(8, int(n_projects_per_task))
            default_branch_candidates = max(8, int(n_projects_per_task))
        else:
            default_compare_candidates = 5
            default_late_candidates = 6
            default_branch_candidates = 6
        hard_compare_candidates = int(hard_compare_candidates or default_compare_candidates)
        hard_late_candidates = int(hard_late_candidates or default_late_candidates)
        hard_branch_candidates = int(hard_branch_candidates or default_branch_candidates)
    if structured_mode:
        if profile_name == "structured_lite":
            default_compare_candidates = 4
            default_dependency_candidates = 2
        elif profile_name == "structured_extreme":
            default_compare_candidates = 6
            default_dependency_candidates = 4
        else:
            default_compare_candidates = 5
            default_dependency_candidates = 3
        structured_compare_candidates = int(structured_compare_candidates or default_compare_candidates)
        structured_dependency_candidates = int(structured_dependency_candidates or default_dependency_candidates)
    entities = [gen_entity(rng, i) for i in range(n_entities)]

    for e in entities:
        e["description"] = gen_long_description(rng, e["name"], n_words=long_desc_words) if long_horizon else ""
    if long_horizon:
        names = [e["name"] for e in entities]
        for e in entities:
            candidates = [n for n in names if n != e["name"]]
            rng.shuffle(candidates)
            e["related_projects"] = candidates[:max(0, int(related_degree))]

    name_to_entity: Dict[str, Dict[str, Any]] = {e["name"]: e for e in entities}
    corpus: List[Dict[str, Any]] = []
    gold_map: Dict[str, str] = {}
    alias_map: Dict[str, str] = {}
    approval_map: Dict[str, str] = {}
    exception_map: Dict[str, str] = {}
    revocation_map: Dict[str, str] = {}
    archived_map: Dict[str, str] = {}

    for i, e in enumerate(entities):
        truth_id = f"D_TRUTH_{i:04d}"
        corpus.append(doc_truth(e, truth_id))
        gold_map[e["name"]] = truth_id
        for j in range(distractors_per_entity):
            did = f"D_DIST_{i:04d}_{j}"
            corpus.append(doc_distractor(e, did, rng))
        if hard_mode or structured_mode:
            alias_id = f"D_ALIAS_{i:04d}"
            corpus.append(doc_alias_note(e, alias_id))
            alias_map[e["name"]] = alias_id
        if structured_mode:
            e["approval_ticket"] = gen_ticket_id(rng, "APR", i)
            e["exception_ticket"] = gen_ticket_id(rng, "EXC", i)
            e["revocation_ticket"] = gen_ticket_id(rng, "REV", i)
            e["exception_year"] = max(int(e.get("relocation_year", 2018)), int(e.get("start_year", 2000)) + 1)
            if e["exception_year"] == int(e.get("relocation_year", 2018)):
                e["exception_year"] += 1
            state_roll = rng.random()
            if state_roll < 0.34:
                e["exception_state"] = "none"
            elif state_roll < 0.67:
                e["exception_state"] = "active"
            else:
                e["exception_state"] = "revoked"
            e["revocation_year"] = int(e["exception_year"]) + 1
            e["archived_snapshot_year"] = max(int(e.get("start_year", 2000)), int(e.get("relocation_year", 2018)) - 1)
            archived_city = str(e.get("headquarters")) if rng.random() < 0.5 else str(e.get("relocation_city"))

            approval_id = f"D_APPROVAL_{i:04d}"
            corpus.append(doc_operating_city_approval(e, approval_id))
            approval_map[e["name"]] = approval_id

            archived_id = f"D_ARCHIVED_{i:04d}"
            corpus.append(doc_archived_status_board(e, archived_id, reported_city=archived_city))
            archived_map[e["name"]] = archived_id

            if e["exception_state"] in {"active", "revoked"}:
                exc_id = f"D_EXCEPTION_{i:04d}"
                corpus.append(doc_legacy_exception_notice(e, exc_id))
                exception_map[e["name"]] = exc_id
                if e["exception_state"] == "revoked":
                    rev_id = f"D_REVOCATION_{i:04d}"
                    corpus.append(doc_exception_revocation_notice(e, rev_id, exc_id))
                    revocation_map[e["name"]] = rev_id

    for k in range(noise_docs):
        corpus.append(doc_noise(f"D_NOISE_{k:04d}", rng))

    rng.shuffle(corpus)

    def pick_entities(k: int) -> List[Dict[str, Any]]:
        return rng.sample(entities, k)

    tasks: List[Dict[str, Any]] = []

    def _docid_for(name: str) -> str:
        return gold_map[name]

    def _alias_docid_for(name: str) -> str:
        return alias_map[name]

    def _pick_entities(k: int) -> List[Dict[str, Any]]:
        return pick_entities(k)

    def _code_initial(e: Dict[str, Any]) -> str:
        cn = str(e.get("code_name", ""))
        return cn[len("Codename_"):len("Codename_") + 1] if cn.startswith("Codename_") and len(cn) > len("Codename_") else (cn[:1] or "X")

    def _approval_docid_for(name: str) -> str:
        return approval_map[name]

    def _archived_docid_for(name: str) -> str:
        return archived_map[name]

    def _exception_docid_for(name: str) -> str | None:
        return exception_map.get(name)

    def _revocation_docid_for(name: str) -> str | None:
        return revocation_map.get(name)

    def _alias_family(entity: Dict[str, Any]) -> str:
        return str(entity.get("alias_handle", "")).split("-", 1)[0]

    def _pick_structured_dependency_entities(k: int, allow_active: bool) -> List[Dict[str, Any]]:
        pool = [e for e in entities if allow_active or str(e.get("exception_state", "none")) in {"none", "revoked"}]
        if len(pool) < k:
            pool = list(entities)
        best: List[Dict[str, Any]] | None = None
        best_score: tuple[int, int, int] | None = None
        for _ in range(64):
            es = rng.sample(pool, k)
            families = {_alias_family(e) for e in es}
            years = sorted(int(e.get("start_year", 0)) for e in es)
            min_gap = min((b - a) for a, b in zip(years, years[1:])) if len(years) > 1 else 99
            score = (len(families), min_gap, len({int(e.get("start_year", 0)) for e in es}))
            if len(families) == len(es) and min_gap >= 2:
                return es
            if best_score is None or score > best_score:
                best = es
                best_score = score
        return best or rng.sample(pool, k)

    def _structured_support_docids(name: str) -> List[str]:
        docs = [_docid_for(name), _approval_docid_for(name), _archived_docid_for(name)]
        exc = _exception_docid_for(name)
        rev = _revocation_docid_for(name)
        if exc:
            docs.append(exc)
        if rev:
            docs.append(rev)
        return docs

    def _structured_dependency_docids(
        name: str,
        *,
        include_exception: bool,
        include_revocation: bool,
    ) -> List[str]:
        docs = [_docid_for(name), _approval_docid_for(name)]
        exc = _exception_docid_for(name)
        rev = _revocation_docid_for(name)
        if include_exception and exc:
            docs.append(exc)
        if include_revocation and rev:
            docs.append(rev)
        return docs

    def _structured_meta(task_type: str, task_slice: str, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "benchmark_profile": str(benchmark_profile or "structured"),
            "task_type": task_type,
            "task_slice": task_slice,
            "needs_alias_resolution": True,
            "supports_current_city_chain": True,
        }
        if extra:
            meta.update(extra)
        return meta

    def _structured_retrieval_task(task_id: str) -> Dict[str, Any]:
        k = max(4, int(structured_compare_candidates or 4))
        es = _pick_entities(k)
        earliest = min(es, key=lambda x: x["start_year"])
        names = [e["name"] for e in es]
        q = (
            "You must use evidence from opened pages.\n"
            f"Given the following canonical projects: {', '.join(names)}. "
            "Open OFFICIAL PROFILE pages and identify the project with the earliest start_year. "
            "Answer exactly as '<ProjectName> | <Headquarters>'."
        )
        return {
            "id": task_id,
            "question": q,
            "entities": names,
            "required": ["start_year", "headquarters"],
            "answer": f"{earliest['name']} | {earliest['headquarters']}",
            "gold_docids": [_docid_for(n) for n in names],
            **_structured_meta(
                task_type="structured_retrieval",
                task_slice="retrieval_sufficient",
                extra={"n_candidates": len(names), "required_support_count": 1},
            ),
        }

    def _structured_dependency_task(task_id: str) -> Dict[str, Any]:
        profile_name_local = str(benchmark_profile or "structured").strip().lower()
        lite_mode = profile_name_local == "structured_lite"
        min_candidates = 2 if lite_mode else 3
        k = max(min_candidates, int(structured_dependency_candidates or min_candidates))
        es = _pick_structured_dependency_entities(k=k, allow_active=not lite_mode)
        target = min(es, key=lambda x: x["start_year"])
        handles = [e["alias_handle"] for e in es]
        projects = [e["name"] for e in es]
        final_city = _final_operating_city(target)
        exception_state = str(target.get("exception_state", "none"))
        include_exception = (not lite_mode) and bool(_exception_docid_for(target["name"]))
        include_revocation = bool(_revocation_docid_for(target["name"]))
        dependency_docids = _structured_dependency_docids(
            target["name"],
            include_exception=include_exception,
            include_revocation=include_revocation,
        )
        gold_docids = [_alias_docid_for(e["name"]) for e in es] + [_docid_for(e["name"]) for e in es] + dependency_docids
        if lite_mode:
            q = (
                "You must use evidence from opened pages.\n"
                f"Candidate handles ({len(handles)} total): {', '.join(handles)}.\n"
                "Resolve each handle via its FIELD NOTE page, then open the matching OFFICIAL PROFILE pages and identify the project with the earliest start_year.\n"
                "For that selected project, determine the CURRENT operating city using the Operating City Approval notice. If there is a later Exception Revocation notice for the same project, treat the approval as current again.\n"
                "Answer exactly as '<ProjectName> | <City>'."
            )
            required = ["alias_handle", "start_year", "approval_ticket"]
            support_variant = "approval_or_revoked"
        else:
            q = (
                "You must use evidence from opened pages.\n"
                f"Candidate handles ({len(handles)} total): {', '.join(handles)}.\n"
                "First resolve each handle via its FIELD NOTE page to the canonical project. Then open the OFFICIAL PROFILE pages and identify the project with the earliest start_year.\n"
                "For that selected project, determine the CURRENT operating city using the OFFICIAL PROFILE plus any Operating City Approval, Legacy Operating Exception, and Exception Revocation notices for the same project.\n"
                "Answer exactly as '<ProjectName> | <City>'."
            )
            required = ["alias_handle", "start_year", "headquarters", "approval_ticket", "exception_ticket"]
            support_variant = "full_chain"
        return {
            "id": task_id,
            "question": q,
            "entities": projects,
            "required": required,
            "answer": f"{target['name']} | {final_city}",
            "gold_docids": list(dict.fromkeys(gold_docids)),
            **_structured_meta(
                task_type="structured_dependency_current_city",
                task_slice="dependency_necessary",
                extra={
                    "candidate_handles": handles,
                    "target_project": target["name"],
                    "target_exception_state": exception_state,
                    "n_candidates": len(projects),
                    "required_support_count": len(dependency_docids),
                    "support_variant": support_variant,
                },
            ),
        }

    def _structured_branch_task(task_id: str) -> Dict[str, Any]:
        branch_candidates = [e for e in entities if str(e.get("exception_state")) in {"active", "revoked"}]
        target = rng.choice(branch_candidates or entities)
        handle = target["alias_handle"]
        final_city = _final_operating_city(target)
        gold_docids = [_alias_docid_for(target["name"])] + _structured_support_docids(target["name"])
        q = (
            "You must use evidence from opened pages.\n"
            f"Resolve handle {handle} via its FIELD NOTE page, then determine the CURRENT operating city for the canonical project.\n"
            "Use the OFFICIAL PROFILE, the Operating City Approval notice, and any Legacy Operating Exception / Exception Revocation notices for that same project.\n"
            "Archived status boards are stale snapshots and should not be used for the final answer.\n"
            "Answer exactly as '<ProjectName> | <City>'."
        )
        return {
            "id": task_id,
            "question": q,
            "entities": [target["name"]],
            "required": ["alias_handle", "approval_ticket", "exception_ticket", "relocation_note"],
            "answer": f"{target['name']} | {final_city}",
            "gold_docids": list(dict.fromkeys(gold_docids)),
            **_structured_meta(
                task_type="structured_branch_resolution",
                task_slice="branch_resolution",
                extra={
                    "target_handle": handle,
                    "target_project": target["name"],
                    "exception_state": target.get("exception_state", "none"),
                    "required_support_count": len(_structured_support_docids(target["name"])),
                },
            ),
        }

    def _hard_compare_task(task_id: str) -> Dict[str, Any]:
        k = max(4, int(hard_compare_candidates))
        es = _pick_entities(k)
        target = min(es, key=lambda x: x["start_year"])
        handles = [e["alias_handle"] for e in es]
        projects = [e["name"] for e in es]
        active_policy_docid, archived_policy_docid, active_cutoff, policy_tag = _ensure_current_policy_docs(
            corpus, task_id=task_id, target=target, rng=rng, scope_handles=handles, scope_projects=projects
        )
        answer = f"{target['name']} | {_operating_city(target, active_cutoff)}"
        gold_docids = [_alias_docid_for(e["name"]) for e in es] + [_docid_for(e["name"]) for e in es] + [active_policy_docid]
        q = (
            "HARD BENCHMARK TASK. You must use evidence from opened pages.\n"
            f"Candidate handles ({len(handles)} total): {', '.join(handles)}.\n"
            "Resolve EACH handle via its FIELD NOTE page (title pattern: '<Handle> Field Note') to the canonical project.\n"
            "Then open the OFFICIAL PROFILE page for each canonical project (title pattern: '<ProjectName> Official') and identify the project with the earliest start_year.\n"
            f"The relevant policy memo family is tagged {policy_tag}. Search for that exact tag, then use the CURRENT/ACTIVE memo (ignore archived/stale policy memos) to decide whether to report headquarters or relocation_city for the selected project.\n"
            "Answer exactly as '<ProjectName> | <City>'."
        )
        return {
            "id": task_id,
            "question": q,
            "entities": projects,
            "required": ["alias_handle", "start_year", "headquarters", "relocation_note", "policy_memo"],
            "answer": answer,
            "gold_docids": gold_docids,
            **_hard_task_meta(
                task_type="hard_compare_policy_alias",
                active_policy_docid=active_policy_docid,
                archived_policy_docid=archived_policy_docid,
                cutoff_year=active_cutoff,
                policy_tag=policy_tag,
                benchmark_profile_name=str(benchmark_profile or "hard"),
                extra={"candidate_handles": handles, "n_candidates": len(es), "active_city_source": "relocation_city"},
            ),
        }

    def _hard_late_binding_task(task_id: str) -> Dict[str, Any]:
        k = max(5, int(hard_late_candidates))
        for _ in range(30):
            es = _pick_entities(k)
            groups: Dict[str, List[Dict[str, Any]]] = {}
            for e in es:
                groups.setdefault(_code_initial(e), []).append(e)
            candidates = [(c, lst) for c, lst in groups.items() if len(lst) >= max(2, int(late_binding_topn))]
            if candidates:
                init, filtered = rng.choice(candidates)
                break
        else:
            es = _pick_entities(k)
            init = _code_initial(es[0])
            filtered = [e for e in es if _code_initial(e) == init]
        topn = max(2, int(late_binding_topn))
        shortlist = sorted(filtered, key=lambda x: x["key_number"], reverse=True)[:topn]
        target = max(shortlist, key=lambda x: int(x.get("relocation_year", 0)))
        handles = [e["alias_handle"] for e in es]
        projects = [e["name"] for e in es]
        shortlist_handles = [e["alias_handle"] for e in shortlist]
        active_policy_docid, archived_policy_docid, active_cutoff, policy_tag = _ensure_current_policy_docs(
            corpus, task_id=task_id, target=target, rng=rng, scope_handles=handles, scope_projects=projects
        )
        q1 = (
            "HARD BENCHMARK TASK (part 1/2). Do NOT finish yet.\n"
            "You must use evidence from opened pages.\n"
            f"Candidate handles ({len(handles)} total): {', '.join(handles)}.\n"
            "Resolve handles via FIELD NOTE pages (title pattern: '<Handle> Field Note').\n"
            f"Then keep only the canonical projects whose code_name starts with 'Codename_{init}'.\n"
            f"Among those filtered canonical projects, SHORTLIST the TOP {topn} by key_number (highest first).\n"
            "Then call the return tool with the shortlisted CANONICAL project names in order."
        )
        q2 = (
            "FOLLOW-UP (hard late binding):\n"
            "Using ONLY the shortlisted canonical projects you identified earlier, open their OFFICIAL PROFILE pages (title pattern: '<ProjectName> Official') and locate relocation_note.\n"
            f"Then search for policy tag {policy_tag} and use the CURRENT/ACTIVE memo (ignore archived/stale policy memos) to choose the shortlisted project with the MOST RECENT relocation_year.\n"
            "Answer exactly as '<ProjectName> | <City>', where the city must follow the CURRENT policy memo."
        )
        answer = f"{target['name']} | {_operating_city(target, active_cutoff)}"
        gold_docids = [_alias_docid_for(e["name"]) for e in es] + [_docid_for(e["name"]) for e in shortlist] + [active_policy_docid]
        return {
            "id": task_id,
            "question": q1,
            "turns": [q1, q2],
            "entities": projects,
            "required": ["alias_handle", "code_name", "key_number", "relocation_note", "policy_memo"],
            "answer": answer,
            "gold_docids": gold_docids,
            **_hard_task_meta(
                task_type="hard_late_binding_policy_alias",
                active_policy_docid=active_policy_docid,
                archived_policy_docid=archived_policy_docid,
                cutoff_year=active_cutoff,
                policy_tag=policy_tag,
                benchmark_profile_name=str(benchmark_profile or "hard"),
                extra={
                    "candidate_handles": handles,
                    "shortlist_handles": shortlist_handles,
                    "late_binding_topn": topn,
                    "code_initial": init,
                    "n_turns": 2,
                    "late_binding_style": "alias_shortlist_then_policy",
                },
            ),
        }

    def _hard_branch_merge_task(task_id: str) -> Dict[str, Any]:
        k = max(4, int(hard_branch_candidates))
        gmin = max(2, int(branch_merge_group_min))
        for _ in range(40):
            es = _pick_entities(k)
            groups: Dict[str, List[Dict[str, Any]]] = {}
            for e in es:
                groups.setdefault(_code_initial(e), []).append(e)
            candidates = [(c, lst) for c, lst in groups.items() if len(lst) >= 2 * gmin]
            if not candidates:
                continue
            init, filtered = rng.choice(candidates)
            filtered_shuf = list(filtered)
            rng.shuffle(filtered_shuf)
            cut = max(gmin, len(filtered_shuf) // 2)
            group_a = filtered_shuf[:cut]
            group_b = filtered_shuf[cut:]
            if len(group_b) < gmin:
                need = gmin - len(group_b)
                if need > 0 and len(group_a) > gmin:
                    group_b = group_a[-need:] + group_b
                    group_a = group_a[:-need]
            if len(group_a) >= gmin and len(group_b) >= gmin:
                break
        else:
            return _hard_late_binding_task(task_id)

        a_winner = max(group_a, key=lambda x: x["key_number"])
        b_winner = max(group_b, key=lambda x: x["key_number"])
        target = a_winner if int(a_winner.get("relocation_year", 0)) >= int(b_winner.get("relocation_year", 0)) else b_winner
        handles = [e["alias_handle"] for e in es]
        projects = [e["name"] for e in es]
        active_policy_docid, archived_policy_docid, active_cutoff, policy_tag = _ensure_current_policy_docs(
            corpus, task_id=task_id, target=target, rng=rng, scope_handles=handles, scope_projects=projects
        )
        group_a_handles = [e["alias_handle"] for e in group_a]
        group_b_handles = [e["alias_handle"] for e in group_b]
        q1 = (
            "HARD BENCHMARK TASK (part 1/3). Do NOT finish yet.\n"
            "Resolve GROUP A handles via FIELD NOTE pages (title pattern: '<Handle> Field Note'), then inspect OFFICIAL PROFILE pages.\n"
            f"GROUP A handles: {', '.join(group_a_handles)}.\n"
            f"Among canonical projects in GROUP A whose code_name starts with 'Codename_{init}', return the SINGLE canonical project with the largest key_number as 'A_WINNER: <ProjectName>'."
        )
        q2 = (
            "HARD BENCHMARK TASK (part 2/3). Do NOT finish yet.\n"
            "Resolve GROUP B handles via FIELD NOTE pages (title pattern: '<Handle> Field Note'), then inspect OFFICIAL PROFILE pages.\n"
            f"GROUP B handles: {', '.join(group_b_handles)}.\n"
            f"Among canonical projects in GROUP B whose code_name starts with 'Codename_{init}', return the SINGLE canonical project with the largest key_number as 'B_WINNER: <ProjectName>'."
        )
        q3 = (
            "HARD BENCHMARK TASK (part 3/3) — MERGE.\n"
            "Using ONLY A_WINNER and B_WINNER from earlier, open BOTH OFFICIAL PROFILE pages and compare relocation_note.\n"
            f"Then search for policy tag {policy_tag} and use the CURRENT/ACTIVE policy memo (ignore archived/stale memos) to produce the city.\n"
            "Answer exactly as '<ProjectName> | <City>'."
        )
        answer = f"{target['name']} | {_operating_city(target, active_cutoff)}"
        gold_docids = [_alias_docid_for(e["name"]) for e in es] + [_docid_for(a_winner["name"]), _docid_for(b_winner["name"]), active_policy_docid]
        return {
            "id": task_id,
            "question": q1,
            "turns": [q1, q2, q3],
            "entities": projects,
            "required": ["alias_handle", "code_name", "key_number", "relocation_note", "policy_memo"],
            "answer": answer,
            "gold_docids": gold_docids,
            **_hard_task_meta(
                task_type="hard_branch_merge_policy_alias",
                active_policy_docid=active_policy_docid,
                archived_policy_docid=archived_policy_docid,
                cutoff_year=active_cutoff,
                policy_tag=policy_tag,
                benchmark_profile_name=str(benchmark_profile or "hard"),
                extra={
                    "candidate_handles": handles,
                    "group_a_handles": group_a_handles,
                    "group_b_handles": group_b_handles,
                    "code_initial": init,
                    "n_turns": 3,
                    "late_binding_style": "branch_merge_then_policy",
                },
            ),
        }

    for t in range(n_tasks):
        task_id = f"TASK_{t:04d}"
        if structured_mode and long_horizon:
            r = rng.random()
            retrieval_ratio = max(0.0, 1.0 - float(structured_dependency_ratio) - float(structured_branch_ratio))
            if r < retrieval_ratio:
                tasks.append(_structured_retrieval_task(task_id))
            elif r < retrieval_ratio + float(structured_dependency_ratio):
                tasks.append(_structured_dependency_task(task_id))
            else:
                tasks.append(_structured_branch_task(task_id))
            continue
        if hard_mode and long_horizon:
            r = rng.random()
            if r < float(hard_compare_ratio):
                tasks.append(_hard_compare_task(task_id))
            elif r < float(hard_compare_ratio + hard_late_binding_ratio):
                tasks.append(_hard_late_binding_task(task_id))
            else:
                tasks.append(_hard_branch_merge_task(task_id))
            continue

        if not long_horizon:
            es = pick_entities(3)
            earliest = min(es, key=lambda x: x["start_year"])
            q = (
                f"Given {es[0]['name']}, {es[1]['name']}, and {es[2]['name']}, "
                f"which project has the earliest start_year, and what is its headquarters? "
                f"Answer exactly as '<ProjectName> | <Headquarters>'."
            )
            answer = f"{earliest['name']} | {earliest['headquarters']}"
            gold_docids = [_docid_for(e["name"]) for e in es]
            tasks.append({
                "id": task_id,
                "question": q,
                "entities": [e["name"] for e in es],
                "required": ["start_year", "headquarters"],
                "answer": answer,
                "gold_docids": gold_docids
            })
            continue

        task_type = "compare_many" if rng.random() < (1.0 - long_task_ratio) else ("two_phase" if rng.random() < 0.55 else "hop")

        if task_type == "compare_many":
            k = max(6, int(n_projects_per_task))
            es = _pick_entities(k)
            earliest = min(es, key=lambda x: x["start_year"])
            names = [e["name"] for e in es]
            q = (
                f"You must use evidence from opened pages.\n"
                f"Given the following projects: {', '.join(names)}. "
                f"Which project has the earliest start_year, and what is its headquarters? "
                f"Answer exactly as '<ProjectName> | <Headquarters>'."
            )
            answer = f"{earliest['name']} | {earliest['headquarters']}"
            gold_docids = [_docid_for(n) for n in names]
            tasks.append({
                "id": task_id,
                "question": q,
                "entities": names,
                "required": ["start_year", "headquarters"],
                "answer": answer,
                "gold_docids": gold_docids,
                "task_type": task_type,
                "benchmark_profile": str(benchmark_profile or "standard"),
                "task_slice": "retrieval_sufficient",
            })

        elif task_type == "two_phase":
            k = max(8, int(n_projects_per_task))
            for _ in range(30):
                es = _pick_entities(k)
                groups: Dict[str, List[Dict[str, Any]]] = {}
                for e in es:
                    groups.setdefault(_code_initial(e), []).append(e)
                min_needed = 4 if branch_merge else 2
                candidates = [(c, lst) for c, lst in groups.items() if len(lst) >= min_needed]
                if candidates:
                    init, lst = rng.choice(candidates)
                    break
            else:
                init = _code_initial(es[0])

            filtered = [e for e in es if _code_initial(e) == init]

            if (
                branch_merge
                and late_binding
                and (rng.random() < float(branch_merge_ratio))
                and (len(filtered) >= 2 * int(branch_merge_group_min))
            ):
                filtered_shuf = list(filtered)
                rng.shuffle(filtered_shuf)
                gmin = max(2, int(branch_merge_group_min))
                cut = max(gmin, len(filtered_shuf) // 2)
                group_a = filtered_shuf[:cut]
                group_b = filtered_shuf[cut:]
                if len(group_b) < gmin:
                    need = gmin - len(group_b)
                    if need > 0 and len(group_a) > gmin:
                        group_b = group_a[-need:] + group_b
                        group_a = group_a[:-need]

                if len(group_a) >= gmin and len(group_b) >= gmin:
                    a_winner = max(group_a, key=lambda x: x["key_number"])
                    b_winner = max(group_b, key=lambda x: x["key_number"])
                    def _rel_y(e: Dict[str, Any]) -> int:
                        return int(e.get("relocation_year", 0))
                    if _rel_y(a_winner) != _rel_y(b_winner):
                        target = a_winner if _rel_y(a_winner) > _rel_y(b_winner) else b_winner
                    else:
                        target = a_winner if int(a_winner["key_number"]) >= int(b_winner["key_number"]) else b_winner

                    names = [e["name"] for e in es]
                    group_a_names = [e["name"] for e in group_a]
                    group_b_names = [e["name"] for e in group_b]

                    q1 = (
                        "MULTI-TURN TASK (part 1/3). Do NOT finish yet.\n"
                        "You must use evidence from opened pages.\n"
                        f"We will focus on GROUP A first. GROUP A projects: {', '.join(group_a_names)}.\n"
                        f"Among GROUP A projects whose code_name starts with 'Codename_{init}', find the SINGLE project with the largest key_number.\n"
                        "Open OFFICIAL PROFILE pages (prefer docid D_TRUTH_####) to verify.\n"
                        "IMPORTANT: Once you identify the winner, open the winner's OFFICIAL PROFILE again using open_page with args find='relocation_note' (or section='tail') to capture the `relocation_note:` line now (you will need it later).\n"
                        "Then call the `return` tool with message exactly: 'A_WINNER: <ProjectName>'."
                    )

                    q2 = (
                        "MULTI-TURN TASK (part 2/3). Do NOT finish yet.\n"
                        "You must use evidence from opened pages.\n"
                        f"Now focus on GROUP B. GROUP B projects: {', '.join(group_b_names)}.\n"
                        f"Among GROUP B projects whose code_name starts with 'Codename_{init}', find the SINGLE project with the largest key_number.\n"
                        "Open OFFICIAL PROFILE pages (prefer docid D_TRUTH_####) to verify.\n"
                        "IMPORTANT: Once you identify the winner, open the winner's OFFICIAL PROFILE again using open_page with args find='relocation_note' (or section='tail') to capture the `relocation_note:` line now (you will need it later).\n"
                        "Then call the `return` tool with message exactly: 'B_WINNER: <ProjectName>'."
                    )

                    q3 = (
                        "MULTI-TURN TASK (part 3/3) — MERGE (late binding):\n"
                        "Using ONLY the A_WINNER and B_WINNER you returned earlier (do NOT redo global search over all projects),\n"
                        "open BOTH winners' OFFICIAL PROFILE pages (use the page that starts with '<Project> OFFICIAL PROFILE') and locate the line `relocation_note:` which is placed deep in the page.\n"
                        "Tip: use open_page with args find='relocation_note' (or section='tail') to access deep lines.\n"
                        "Choose the winner with the MOST RECENT relocation_year.\n"
                        "Finish with the headquarters of that project.\n"
                        "Answer exactly as '<ProjectName> | <Headquarters>'."
                    )

                    answer = f"{target['name']} | {target['headquarters']}"
                    gold_docids = [_docid_for(n) for n in names]
                    tasks.append({
                        "id": task_id,
                        "question": q1,
                        "turns": [q1, q2, q3],
                        "entities": names,
                        "required": ["code_name", "key_number", "relocation_note", "headquarters"],
                        "answer": answer,
                        "gold_docids": gold_docids,
                        "task_type": "late_binding_branch_merge",
                        "benchmark_profile": str(benchmark_profile or "standard"),
                        "task_slice": "memory_necessary",
                        "late_binding": {
                            "init": init,
                            "group_a": group_a_names,
                            "group_b": group_b_names,
                            "a_winner": a_winner["name"],
                            "b_winner": b_winner["name"],
                            "tie_breaker": "max(relocation_year) over {A_WINNER,B_WINNER}",
                        },
                    })
                    continue

            if late_binding and (len(filtered) >= max(2, int(late_binding_topn))) and (rng.random() < float(late_binding_ratio)):
                topn = max(2, int(late_binding_topn))
                shortlist = sorted(filtered, key=lambda x: x["key_number"], reverse=True)[:topn]
                target = max(shortlist, key=lambda x: int(x.get("relocation_year", 0)))

                names = [e["name"] for e in es]
                shortlist_names = [e["name"] for e in shortlist]
                q1 = (
                    "You will receive a follow-up question later. Do NOT finish yet.\n"
                    "You must use evidence from opened pages.\n"
                    f"Given these projects: {', '.join(names)}.\n"
                    f"Step 1) Select projects whose code_name starts with 'Codename_{init}'.\n"
                    f"Step 2) Among the selected projects, SHORTLIST the TOP {topn} projects by key_number (highest first).\n"
                    "Then call the `return` tool with a message listing the shortlisted project names (in order)."
                )
                q2 = (
                    "FOLLOW-UP (late binding):\n"
                    f"Using ONLY the shortlisted projects you identified earlier (do NOT redo global search over all projects),\n"
                    f"open their OFFICIAL PROFILE pages (use the page that starts with '<Project> OFFICIAL PROFILE') and read the line `relocation_note:` which is placed deep in the page.\n"
                    "Tip: use open_page with args find='relocation_note' (or section='tail') to access deep lines.\n"
                    f"Choose the shortlisted project with the MOST RECENT relocation_year.\n"
                    "Finally, finish with the headquarters of that project.\n"
                    "Answer exactly as '<ProjectName> | <Headquarters>'."
                )
                answer = f"{target['name']} | {target['headquarters']}"
                gold_docids = [_docid_for(n) for n in names]
                tasks.append({
                    "id": task_id,
                    "question": q1,
                    "turns": [q1, q2],
                    "entities": names,
                    "required": ["code_name", "key_number", "relocation_note", "headquarters"],
                    "answer": answer,
                    "gold_docids": gold_docids,
                    "task_type": "late_binding",
                    "benchmark_profile": str(benchmark_profile or "standard"),
                    "task_slice": "memory_necessary",
                    "late_binding": {
                        "init": init,
                        "topn": topn,
                        "shortlist": shortlist_names,
                        "tie_breaker": "max(relocation_year)",
                    },
                })
                continue

            winner = max(filtered, key=lambda x: x["key_number"])
            names = [e["name"] for e in es]
            q = (
                f"You must use evidence from opened pages.\n"
                f"Given these projects: {', '.join(names)}.\n"
                f"Step 1) Select projects whose code_name starts with 'Codename_{init}'.\n"
                f"Step 2) Among the selected projects, choose the one with the largest key_number, "
                f"and report its headquarters.\n"
                f"Answer exactly as '<ProjectName> | <Headquarters>'."
            )
            answer = f"{winner['name']} | {winner['headquarters']}"
            gold_docids = [_docid_for(n) for n in names]
            tasks.append({
                "id": task_id,
                "question": q,
                "entities": names,
                "required": ["code_name", "key_number", "headquarters"],
                "answer": answer,
                "gold_docids": gold_docids,
                "task_type": task_type,
                "benchmark_profile": str(benchmark_profile or "standard"),
                "task_slice": "retrieval_sufficient",
            })

        else:
            start_e = rng.choice(entities)
            visited = [start_e["name"]]
            cur = start_e
            for _ in range(max(1, int(hop_steps))):
                rel = list(cur.get("related_projects", []))
                if not rel:
                    break
                nxt_name = rel[0]
                visited.append(nxt_name)
                cur = name_to_entity[nxt_name]
            visited_entities = [name_to_entity[n] for n in visited]
            earliest = min(visited_entities, key=lambda x: x["start_year"])
            q = (
                f"You must use evidence from opened pages.\n"
                f"Start from {start_e['name']}. Open its OFFICIAL PROFILE and read the 'related_projects' list. "
                f"Follow the chain for {max(1, int(hop_steps))} hops by taking the FIRST listed related project each time.\n"
                f"Among all visited projects (including the start), which has the earliest start_year, and what is its headquarters? "
                f"Answer exactly as '<ProjectName> | <Headquarters>'."
            )
            answer = f"{earliest['name']} | {earliest['headquarters']}"
            gold_docids = [_docid_for(n) for n in visited]
            tasks.append({
                "id": task_id,
                "question": q,
                "entities": visited,
                "required": ["related_projects", "start_year", "headquarters"],
                "answer": answer,
                "gold_docids": gold_docids,
                "task_type": task_type,
                "benchmark_profile": str(benchmark_profile or "standard"),
                "task_slice": "memory_necessary",
            })
    json.dump(corpus, open(out_corpus_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump(tasks, open(out_tasks_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    return {"n_docs": len(corpus), "n_tasks": len(tasks), "seed": seed, "benchmark_profile": str(benchmark_profile or ('hard' if hard_mode else 'standard'))}
