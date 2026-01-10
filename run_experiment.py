import argparse
from pathlib import Path
from src.synth_data import make_corpus_and_tasks
from src.runner import run_experiment

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n_tasks", type=int, default=50)
    ap.add_argument("--n_entities", type=int, default=80)
    ap.add_argument("--distractors_per_entity", type=int, default=2)
    ap.add_argument("--noise_docs", type=int, default=120)

    ap.add_argument("--budget_active", type=int, default=2000)
    ap.add_argument("--budget_unfold", type=int, default=800)
    ap.add_argument("--unfold_k", type=int, default=6)

    ap.add_argument("--summary_keep_fields", type=int, default=1,
                    help="How many non-start_year fields to keep in branch return summary (lossy fold). 0/1 tends to amplify differences.")
    ap.add_argument("--linear_summary_every", type=int, default=8)
    ap.add_argument("--agentfold_fold_chunk", type=int, default=10)

    args = ap.parse_args()

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    corpus_path = data_dir / "corpus.json"
    tasks_path = data_dir / "tasks.json"

    make_corpus_and_tasks(
        out_corpus_path=str(corpus_path),
        out_tasks_path=str(tasks_path),
        n_entities=args.n_entities,
        n_tasks=args.n_tasks,
        distractors_per_entity=args.distractors_per_entity,
        noise_docs=args.noise_docs,
        seed=args.seed,
    )

    out_results = Path("results.jsonl")
    out_report = Path("report.md")
    res = run_experiment(
        corpus_path=str(corpus_path),
        tasks_path=str(tasks_path),
        out_results_path=str(out_results),
        out_report_path=str(out_report),
        budget_active=args.budget_active,
        budget_unfold=args.budget_unfold,
        unfold_k=args.unfold_k,
        summary_keep_fields=args.summary_keep_fields,
        linear_summary_every=args.linear_summary_every,
        agentfold_fold_chunk=args.agentfold_fold_chunk,
    )
    print("Done:", res)
    print("Report:", out_report.resolve())

if __name__ == "__main__":
    main()
