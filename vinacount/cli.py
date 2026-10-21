"""Command-line interface: vinacount {review, download, predict, evaluate, train}."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vinacount import config

DETECTORS = ("lora", "base", "rules", "vllm", "api")


def _detector(args: argparse.Namespace):
    from vinacount.detector import make_detector

    return make_detector(args.detector, adapter=args.adapter, device=args.device, batch_size=args.batch_size,
                         vllm_url=args.vllm_url, vllm_model=args.vllm_model, api_model=args.api_model,
                         api_base_url=args.api_base_url, api_key_env=args.api_key_env)


def _corpus(args: argparse.Namespace) -> list[dict]:
    from vinacount.data import load_corpus, split_records

    path = args.data_dir / "corpus.jsonl"
    if not path.exists():
        sys.exit(f"{path} not found; run `vinacount download` first")
    records = load_corpus(path, args.corpus)
    return split_records(records, args.split) if getattr(args, "split", None) else records


def cmd_review(args: argparse.Namespace) -> None:
    from vinacount.pipeline import load_filing_pair, review

    result = review(load_filing_pair(args.filing_pair), _detector(args))
    out = args.output or Path(f"{args.filing_pair.stem}.review.json")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    flagged = [f["signal_id"] for f in result["tool_findings"] if f["flag"]]
    print(f"flagged checks: {', '.join(flagged) or 'none'}")
    for c, g in zip(result["candidate_risks"], result["guard"]):
        label = g["assessment"]["support_level"] if g["status"] == "accepted" else f"rejected ({g['error']})"
        print(f"  {c['risk_category']:<42} {label}")
    report = result["report"]
    print(f"report: {len(report['grouped_findings'])} grouped findings, {len(report['limited_review_items'])} "
          f"limited-review items, {len(report['evidence_gaps'])} evidence gaps -> {out}")


def cmd_download(args: argparse.Namespace) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=config.DATASET_REPO, repo_type="dataset", revision=config.DATASET_REVISION,
                      local_dir=args.data_dir)
    from vinacount.data import load_corpus, split_records

    for corpus in ("final", "selection"):
        rows = load_corpus(args.data_dir / "corpus.jsonl", corpus)
        print(corpus, {s: len(split_records(rows, s)) for s in ("train", "validation", "test")})


def cmd_predict(args: argparse.Namespace) -> None:
    from vinacount.data import read_jsonl

    records = _corpus(args)[: args.limit]
    out = args.output or Path("outputs") / f"{args.split}_{args.corpus}_{args.detector}.jsonl"
    done = {r["example_id"] for r in read_jsonl(out)} if out.exists() else set()
    todo = [r for r in records if r["example_id"] not in done]
    print(f"{len(done)} saved, {len(todo)} to run -> {out}")
    if not todo:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    detector = _detector(args)
    step = max(args.batch_size, 1)
    for start in range(0, len(todo), step):
        chunk = todo[start : start + step]
        texts = detector.generate([r["messages"][:2] for r in chunk])
        with open(out, "a", encoding="utf-8") as f:
            for r, text in zip(chunk, texts):
                f.write(json.dumps({"example_id": r["example_id"], "response": text}, ensure_ascii=False) + "\n")
        print(f"{len(done) + start + len(chunk)}/{len(done) + len(todo)}", flush=True)


def cmd_evaluate(args: argparse.Namespace) -> None:
    from vinacount.data import read_jsonl
    from vinacount.scoring import HANDLINGS, LABELS, bootstrap, confusion, evaluate, scores

    responses = {r["example_id"]: r["response"] for r in read_jsonl(args.responses)}
    records = _corpus(args)
    if args.present_only:
        records = [r for r in records if r["example_id"] in responses]
    result = {}
    print(f"{args.responses.name}: {len(records)} packets ({args.split}, {args.corpus} corpus)")
    print(f"{'handling':>10}  valid    acc  macro-F1  weak-F1")
    for handling in HANDLINGS:
        gold, pred = evaluate(records, responses, handling)
        s = scores(gold, pred)
        result[handling] = {**s, "confusion": confusion(gold, pred)}
        if args.bootstrap:
            result[handling]["bootstrap_95"] = bootstrap(gold, pred, reps=args.bootstrap, seed=config.BOOTSTRAP_SEED)
        print(f"{handling:>10}  {s['valid_rate']:.3f}  {s['accuracy']:.3f}  {s['macro_f1']:.3f}     {s['weak_f1']:.3f}")
    if args.bootstrap:
        ci = result["guarded"]["bootstrap_95"]
        print("guarded 95% intervals: " + ", ".join(f"{k} [{v[0]:.3f}, {v[1]:.3f}]" for k, v in ci.items()))
    print("guarded confusion (rows gold, columns predicted: S W N I invalid)")
    for g in LABELS:
        print(f"  {g:<22} " + " ".join(f"{n:>4}" for n in result["guarded"]["confusion"][g].values()))
    if args.output:
        args.output.write_text(json.dumps(result, indent=1) + "\n")


def cmd_train(args: argparse.Namespace) -> None:
    from vinacount.training import train_lora

    overrides = {k: v for k, v in {"epochs": args.epochs, "eval_steps": args.eval_steps,
                                   "learning_rate": args.learning_rate}.items() if v is not None}
    result = train_lora(_corpus(args), args.output_dir, **overrides)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="vinacount", description="Evidence-bounded accounting review.")
    sub = parser.add_subparsers(dest="command", required=True)

    def detector_options(p: argparse.ArgumentParser, default: str) -> None:
        g = p.add_argument_group("detector")
        g.add_argument("--detector", choices=DETECTORS, default=default,
                       help="lora: released adapter (or --adapter); base: no adapter; rules: no language model; "
                            "vllm: a vLLM server; api: an OpenAI-compatible chat API")
        g.add_argument("--adapter", help="local adapter directory or Hugging Face repo")
        g.add_argument("--device", default="auto", help="auto, cuda, or cpu")
        g.add_argument("--batch-size", type=int, default=1)
        g.add_argument("--vllm-url", default="http://127.0.0.1:8000/v1")
        g.add_argument("--vllm-model", default="vinacount-detector", help="served adapter name")
        g.add_argument("--api-model", default="deepseek-v4-pro")
        g.add_argument("--api-base-url", default="https://api.deepseek.com")
        g.add_argument("--api-key-env", default="DEEPSEEK_API_KEY", help="environment variable holding the key")

    def corpus_options(p: argparse.ArgumentParser, split: str | None) -> None:
        p.add_argument("--data-dir", type=Path, default=Path("data"))
        p.add_argument("--corpus", choices=("final", "selection"), default="final")
        if split:
            p.add_argument("--split", choices=("validation", "test"), default=split)

    p = sub.add_parser("review", help="review a filing pair and write the report")
    p.add_argument("filing_pair", type=Path, help="JSON file with case_id, target, and prior_year ReportMemory")
    p.add_argument("-o", "--output", type=Path)
    detector_options(p, "lora")
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("download", help="download the synthetic corpus and saved responses")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("predict", help="run a detector on a corpus split and save raw responses")
    corpus_options(p, "test")
    p.add_argument("-o", "--output", type=Path)
    p.add_argument("--limit", type=int)
    detector_options(p, "lora")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("evaluate", help="score saved responses (strict, guarded, label-only)")
    p.add_argument("responses", type=Path)
    corpus_options(p, "test")
    p.add_argument("--present-only", action="store_true", help="score only packets that have a response")
    p.add_argument("--bootstrap", type=int, default=0, metavar="N", help="packet-level bootstrap replicates")
    p.add_argument("-o", "--output", type=Path, help="write metrics as JSON")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("train", help="fine-tune a LoRA detector")
    corpus_options(p, None)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/train"))
    p.add_argument("--epochs", type=int)
    p.add_argument("--eval-steps", type=int)
    p.add_argument("--learning-rate", type=float)
    p.set_defaults(func=cmd_train)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
