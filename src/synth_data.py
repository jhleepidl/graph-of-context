from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import random
import json
import string

ATTRS = ["start_year", "headquarters", "lead", "code_name", "key_number"]

def gen_long_description(rng: random.Random, name: str, n_words: int = 260) -> str:
    """Generate a long, mostly-irrelevant description to inflate tool observations.
    This makes FullHistory expensive while keeping facts in compact key-value lines.
    """
    # Mix the entity name with pseudo-words to create long but easy-to-ignore text.
    words: List[str] = []
    for _ in range(max(0, n_words)):
        if rng.random() < 0.06:
            words.append(name)
        else:
            words.append(rand_word(rng, rng.randint(3, 8)))
    # Insert a few separators so it looks like paragraphs.
    chunks = []
    step = 60
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i:i+step]))
    return "\n\n".join(chunks)


def rand_word(rng: random.Random, n=6) -> str:
    letters = string.ascii_uppercase
    return "".join(rng.choice(letters) for _ in range(n))

def gen_entity(rng: random.Random, i: int) -> Dict[str, Any]:
    name = f"Project_{i:04d}"
    return {
        "name": name,
        "start_year": rng.randint(1995, 2024),
        "headquarters": f"City_{rng.randint(1, 60)}",
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
        f"NOTE: This document is authoritative."
    )
    return {"docid": docid, "url": f"https://local/{docid}", "title": f"{entity['name']} Official", "content": content}

def doc_distractor(entity: Dict[str, Any], docid: str, rng: random.Random) -> Dict[str, Any]:
    # Wrong headquarters or start_year, but very query-relevant (repeats entity name)
    wrong_hq = f"City_{rng.randint(61, 90)}"
    wrong_year = entity["start_year"] + rng.choice([-3, -2, -1, 1, 2, 3])
    # Make it rank high by repeating entity name and keywords
    repeats = " ".join([entity["name"]] * rng.randint(8, 14))
    content = (
        f"{repeats}\n"
        f"start_year: {wrong_year}\n"
        f"headquarters: {wrong_hq}\n"
        f"lead: {entity['lead']}\n"
        f"code_name: {entity['code_name']}\n"
        f"key_number: {entity['key_number']}\n"
        f"related_projects: {', '.join(entity.get('related_projects', []))}\n"
        f"description: {entity.get('description', '')}\n"
        f"Rumor blog; may be outdated."
    )
    return {"docid": docid, "url": f"https://local/{docid}", "title": f"{entity['name']} Rumor", "content": content}

def doc_noise(docid: str, rng: random.Random) -> Dict[str, Any]:
    words = [rand_word(rng, rng.randint(3, 8)) for _ in range(rng.randint(200, 380))]
    content = " ".join(words)
    return {"docid": docid, "url": f"https://local/{docid}", "title": "Noise", "content": content}

def make_corpus_and_tasks(
    out_corpus_path: str,
    out_tasks_path: str,
    n_entities: int = 80,
    n_tasks: int = 50,
    distractors_per_entity: int = 2,
    noise_docs: int = 120,
    seed: int = 7,
    # Long-horizon extensions (optional)
    long_horizon: bool = False,
    long_desc_words: int = 320,
    related_degree: int = 3,
    n_projects_per_task: int = 10,
    hop_steps: int = 4,
    long_task_ratio: float = 0.7,
):
    rng = random.Random(seed)
    entities = [gen_entity(rng, i) for i in range(n_entities)]

    # Fill descriptions / related_projects for long-horizon settings
    for e in entities:
        e["description"] = gen_long_description(rng, e["name"], n_words=long_desc_words) if long_horizon else ""
    if long_horizon:
        names = [e["name"] for e in entities]
        for e in entities:
            # Sample a small set of related projects (excluding self)
            candidates = [n for n in names if n != e["name"]]
            rng.shuffle(candidates)
            e["related_projects"] = candidates[:max(0, int(related_degree))]



    # Fast lookup for hop-traversal tasks
    name_to_entity: Dict[str, Dict[str, Any]] = {e["name"]: e for e in entities}

    corpus: List[Dict[str, Any]] = []
    gold_map: Dict[str, str] = {}  # entity -> truth docid

    # entity docs
    for i, e in enumerate(entities):
        truth_id = f"D_TRUTH_{i:04d}"
        corpus.append(doc_truth(e, truth_id))
        gold_map[e["name"]] = truth_id

        for j in range(distractors_per_entity):
            did = f"D_DIST_{i:04d}_{j}"
            corpus.append(doc_distractor(e, did, rng))

    # noise docs
    for k in range(noise_docs):
        corpus.append(doc_noise(f"D_NOISE_{k:04d}", rng))

    rng.shuffle(corpus)

    def pick_entities(k: int) -> List[Dict[str, Any]]:
        return rng.sample(entities, k)

    tasks: List[Dict[str, Any]] = []

    def _docid_for(name: str) -> str:
        return gold_map[name]

    def _pick_entities(k: int) -> List[Dict[str, Any]]:
        # Deterministic-ish selection with rng
        return pick_entities(k)

    def _code_initial(e: Dict[str, Any]) -> str:
        cn = str(e.get("code_name", ""))
        return cn[len("Codename_"):len("Codename_")+1] if cn.startswith("Codename_") and len(cn) > len("Codename_") else (cn[:1] or "X")

    for t in range(n_tasks):
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
                "id": f"TASK_{t:04d}",
                "question": q,
                "entities": [e["name"] for e in es],
                "required": ["start_year", "headquarters"],
                "answer": answer,
                "gold_docids": gold_docids
            })
            continue

        # Long-horizon: mix multi-open + late dependency tasks
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
                "id": f"TASK_{t:04d}",
                "question": q,
                "entities": names,
                "required": ["start_year", "headquarters"],
                "answer": answer,
                "gold_docids": gold_docids,
                "task_type": task_type
            })

        elif task_type == "two_phase":
            # Phase 1: filter by code_name initial letter (ensured >=2 matches)
            k = max(8, int(n_projects_per_task))
            for _ in range(30):
                es = _pick_entities(k)
                groups: Dict[str, List[Dict[str, Any]]] = {}
                for e in es:
                    groups.setdefault(_code_initial(e), []).append(e)
                candidates = [(c, lst) for c, lst in groups.items() if len(lst) >= 2]
                if candidates:
                    init, lst = rng.choice(candidates)
                    break
            else:
                # Fallback: just use first letter from first entity
                init = _code_initial(es[0])

            filtered = [e for e in es if _code_initial(e) == init]
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
                "id": f"TASK_{t:04d}",
                "question": q,
                "entities": names,
                "required": ["code_name", "key_number", "headquarters"],
                "answer": answer,
                "gold_docids": gold_docids,
                "task_type": task_type
            })

        else:
            # hop traversal: follow related_projects for hop_steps hops using "first listed" rule
            start_e = rng.choice(entities)
            visited = [start_e["name"]]
            cur = start_e
            for _ in range(max(1, int(hop_steps))):
                rel = list(cur.get("related_projects", []))
                if not rel:
                    break
                nxt_name = rel[0]  # first listed
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
                "id": f"TASK_{t:04d}",
                "question": q,
                "entities": visited,
                "required": ["related_projects", "start_year", "headquarters"],
                "answer": answer,
                "gold_docids": gold_docids,
                "task_type": task_type
            })
    json.dump(corpus, open(out_corpus_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump(tasks, open(out_tasks_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    return {"n_docs": len(corpus), "n_tasks": len(tasks), "seed": seed}
