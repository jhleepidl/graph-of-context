from __future__ import annotations

from typing import Any, Dict, List, Optional


def compute_retrieval_diagnostics(
    opened_ids: List[str],
    gold_ids: List[str],
    search_results: List[Dict[str, Any]],
    top_k_used: int,
    save_snapshot: bool = False,
    snapshot_k: int = 20,
) -> Dict[str, Any]:
    results = list(search_results or [])
    results.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)

    opened_set = set(opened_ids or [])
    gold_set = set(gold_ids or [])
    opened_gold = opened_set & gold_set
    opened_gold_count = len(opened_gold)
    opened_gold_coverage = opened_gold_count / max(1, len(gold_set))
    winning_clause = gold_ids[0] if gold_ids else None
    opened_has_winning_clause = bool(winning_clause and winning_clause in opened_set)

    gold_ranks: List[int] = []
    winning_clause_rank: Optional[int] = None
    best_gold_score: Optional[float] = None
    best_non_gold_score: Optional[float] = None

    for idx, item in enumerate(results, start=1):
        clause_id = item.get("clause_id")
        score = float(item.get("score", 0.0))
        if clause_id in gold_set:
            gold_ranks.append(idx)
            if best_gold_score is None or score > best_gold_score:
                best_gold_score = score
        else:
            if best_non_gold_score is None or score > best_non_gold_score:
                best_non_gold_score = score
        if winning_clause and clause_id == winning_clause and winning_clause_rank is None:
            winning_clause_rank = idx

    min_gold_rank = min(gold_ranks) if gold_ranks else None
    gold_in_search_topk = bool(min_gold_rank and min_gold_rank <= top_k_used)

    gold_score_gap = None
    if best_gold_score is not None and best_non_gold_score is not None:
        gold_score_gap = best_non_gold_score - best_gold_score

    diag: Dict[str, Any] = {
        "opened_gold_count": opened_gold_count,
        "opened_gold_coverage": opened_gold_coverage,
        "opened_has_winning_clause": opened_has_winning_clause,
        "gold_in_search_topk": gold_in_search_topk,
        "min_gold_rank": min_gold_rank,
        "winning_clause_rank": winning_clause_rank,
        "best_gold_score": best_gold_score,
        "best_non_gold_score": best_non_gold_score,
        "gold_score_gap": gold_score_gap,
        "num_search_results": len(results),
    }
    if save_snapshot:
        diag["search_topk_clause_ids"] = [
            item.get("clause_id") for item in results[:snapshot_k]
        ]
    return diag
