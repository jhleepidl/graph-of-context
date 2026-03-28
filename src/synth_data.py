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


def doc_policy_memo(task_id: str, docid: str, cutoff_year: int, archived: bool = False) -> Dict[str, Any]:
    status = "archived" if archived else "active"
    title_suffix = "Archived" if archived else "Current"
    content = (
        f"CITY RESOLUTION POLICY MEMO ({status.upper()})\n"
        f"policy_id: {task_id}\n"
        f"policy_state: {status}\n"
        f"When reporting the current operating city for the selected project, use relocation_city if relocation_year >= {cutoff_year}; otherwise use headquarters.\n"
        "Only ACTIVE/CURRENT memos should be used for final answers.\n"
        "This memo does not contain project-specific facts."
    )
    return {
        "docid": docid,
        "url": f"https://local/{docid}",
        "title": f"Current Operating City Policy {title_suffix}",
        "content": content,
    }


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


def _ensure_current_policy_docs(corpus: List[Dict[str, Any]], task_id: str, target: Dict[str, Any]) -> tuple[str, str, int]:
    active_cutoff, archived_cutoff = _policy_cutoffs_for_target(target)
    active_docid = f"D_POLICY_{task_id}_CUR"
    archived_docid = f"D_POLICY_{task_id}_ARC"
    corpus.append(doc_policy_memo(task_id=task_id, docid=active_docid, cutoff_year=active_cutoff, archived=False))
    corpus.append(doc_policy_memo(task_id=task_id, docid=archived_docid, cutoff_year=archived_cutoff, archived=True))
    return active_docid, archived_docid, active_cutoff


def _hard_task_meta(task_type: str, active_policy_docid: str, archived_policy_docid: str, cutoff_year: int, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "benchmark_profile": "hard",
        "task_type": task_type,
        "task_slice": "memory_necessary",
        "needs_alias_resolution": True,
        "needs_rule_doc": True,
        "needs_multi_support": True,
        "has_stale_rule_distractor": True,
        "active_policy_docid": active_policy_docid,
        "archived_policy_docid": archived_policy_docid,
        "active_policy_cutoff_year": int(cutoff_year),
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
):
    rng = random.Random(seed)
    hard_mode = bool(hard_mode or str(benchmark_profile).strip().lower() == "hard")
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

    for i, e in enumerate(entities):
        truth_id = f"D_TRUTH_{i:04d}"
        corpus.append(doc_truth(e, truth_id))
        gold_map[e["name"]] = truth_id
        for j in range(distractors_per_entity):
            did = f"D_DIST_{i:04d}_{j}"
            corpus.append(doc_distractor(e, did, rng))
        if hard_mode:
            alias_id = f"D_ALIAS_{i:04d}"
            corpus.append(doc_alias_note(e, alias_id))
            alias_map[e["name"]] = alias_id

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

    def _hard_compare_task(task_id: str) -> Dict[str, Any]:
        k = max(6, int(n_projects_per_task))
        es = _pick_entities(k)
        target = min(es, key=lambda x: x["start_year"])
        active_policy_docid, archived_policy_docid, active_cutoff = _ensure_current_policy_docs(corpus, task_id=task_id, target=target)
        handles = [e["alias_handle"] for e in es]
        answer = f"{target['name']} | {_operating_city(target, active_cutoff)}"
        gold_docids = [_alias_docid_for(e["name"]) for e in es] + [_docid_for(e["name"]) for e in es] + [active_policy_docid]
        q = (
            "HARD BENCHMARK TASK. You must use evidence from opened pages.\n"
            f"Candidate handles: {', '.join(handles)}.\n"
            "First resolve EACH handle via its FIELD NOTE to the canonical project.\n"
            "Then open the OFFICIAL PROFILE for each canonical project and identify the project with the earliest start_year.\n"
            "Finally, use the CURRENT OPERATING CITY POLICY memo (ignore archived/stale policy memos) to decide whether to report headquarters or relocation_city for that selected project.\n"
            "Answer exactly as '<ProjectName> | <City>'."
        )
        return {
            "id": task_id,
            "question": q,
            "entities": [e["name"] for e in es],
            "required": ["alias_handle", "start_year", "headquarters", "relocation_note", "policy_memo"],
            "answer": answer,
            "gold_docids": gold_docids,
            **_hard_task_meta(
                task_type="hard_compare_policy_alias",
                active_policy_docid=active_policy_docid,
                archived_policy_docid=archived_policy_docid,
                cutoff_year=active_cutoff,
                extra={
                    "candidate_handles": handles,
                    "n_candidates": len(es),
                    "active_city_source": "relocation_city",
                },
            ),
        }

    def _hard_late_binding_task(task_id: str) -> Dict[str, Any]:
        k = max(8, int(n_projects_per_task))
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
        active_policy_docid, archived_policy_docid, active_cutoff = _ensure_current_policy_docs(corpus, task_id=task_id, target=target)
        handles = [e["alias_handle"] for e in es]
        shortlist_handles = [e["alias_handle"] for e in shortlist]
        q1 = (
            "HARD BENCHMARK TASK (part 1/2). Do NOT finish yet.\n"
            "You must use evidence from opened pages.\n"
            f"Candidate handles: {', '.join(handles)}.\n"
            f"Resolve the handles via FIELD NOTE pages, then keep only the canonical projects whose code_name starts with 'Codename_{init}'.\n"
            f"Among those filtered canonical projects, SHORTLIST the TOP {topn} by key_number (highest first).\n"
            "Then call the return tool with the shortlisted CANONICAL project names in order."
        )
        q2 = (
            "FOLLOW-UP (hard late binding):\n"
            "Using ONLY the shortlisted canonical projects you identified earlier, open their OFFICIAL PROFILE pages and locate relocation_note.\n"
            "Then consult the CURRENT OPERATING CITY POLICY memo (ignore archived/stale policy memos) and choose the shortlisted project with the MOST RECENT relocation_year.\n"
            "Answer exactly as '<ProjectName> | <City>', where the city must follow the CURRENT policy memo."
        )
        answer = f"{target['name']} | {_operating_city(target, active_cutoff)}"
        gold_docids = [_alias_docid_for(e["name"]) for e in es] + [_docid_for(e["name"]) for e in shortlist] + [active_policy_docid]
        return {
            "id": task_id,
            "question": q1,
            "turns": [q1, q2],
            "entities": [e["name"] for e in es],
            "required": ["alias_handle", "code_name", "key_number", "relocation_note", "policy_memo"],
            "answer": answer,
            "gold_docids": gold_docids,
            **_hard_task_meta(
                task_type="hard_late_binding_policy_alias",
                active_policy_docid=active_policy_docid,
                archived_policy_docid=archived_policy_docid,
                cutoff_year=active_cutoff,
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
        k = max(8, int(n_projects_per_task))
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
        active_policy_docid, archived_policy_docid, active_cutoff = _ensure_current_policy_docs(corpus, task_id=task_id, target=target)
        handles = [e["alias_handle"] for e in es]
        group_a_handles = [e["alias_handle"] for e in group_a]
        group_b_handles = [e["alias_handle"] for e in group_b]
        q1 = (
            "HARD BENCHMARK TASK (part 1/3). Do NOT finish yet.\n"
            "Resolve GROUP A handles via FIELD NOTE pages, then inspect OFFICIAL PROFILE pages.\n"
            f"GROUP A handles: {', '.join(group_a_handles)}.\n"
            f"Among canonical projects in GROUP A whose code_name starts with 'Codename_{init}', return the SINGLE canonical project with the largest key_number as 'A_WINNER: <ProjectName>'."
        )
        q2 = (
            "HARD BENCHMARK TASK (part 2/3). Do NOT finish yet.\n"
            "Resolve GROUP B handles via FIELD NOTE pages, then inspect OFFICIAL PROFILE pages.\n"
            f"GROUP B handles: {', '.join(group_b_handles)}.\n"
            f"Among canonical projects in GROUP B whose code_name starts with 'Codename_{init}', return the SINGLE canonical project with the largest key_number as 'B_WINNER: <ProjectName>'."
        )
        q3 = (
            "HARD BENCHMARK TASK (part 3/3) — MERGE.\n"
            "Using ONLY A_WINNER and B_WINNER from earlier, open BOTH OFFICIAL PROFILE pages and compare relocation_note.\n"
            "Select the winner with the MOST RECENT relocation_year, then use the CURRENT OPERATING CITY POLICY memo (ignore archived/stale memos) to produce the city.\n"
            "Answer exactly as '<ProjectName> | <City>'."
        )
        answer = f"{target['name']} | {_operating_city(target, active_cutoff)}"
        gold_docids = [_alias_docid_for(e["name"]) for e in es] + [_docid_for(a_winner["name"]), _docid_for(b_winner["name"]), active_policy_docid]
        return {
            "id": task_id,
            "question": q1,
            "turns": [q1, q2, q3],
            "entities": [e["name"] for e in es],
            "required": ["alias_handle", "code_name", "key_number", "relocation_note", "policy_memo"],
            "answer": answer,
            "gold_docids": gold_docids,
            **_hard_task_meta(
                task_type="hard_branch_merge_policy_alias",
                active_policy_docid=active_policy_docid,
                archived_policy_docid=archived_policy_docid,
                cutoff_year=active_cutoff,
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

    return {"n_docs": len(corpus), "n_tasks": len(tasks), "seed": seed, "benchmark_profile": "hard" if hard_mode else str(benchmark_profile or 'standard')}
