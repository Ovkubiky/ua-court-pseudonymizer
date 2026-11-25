"""Evaluate the hybrid layer (scripts/hybrid.py) on synthetic splits.

    # rules alone: per-type and per-rule precision / recall
    python scripts/hybrid_eval.py rules --truth data/synthetic/dev.jsonl data/synthetic_v1/dev.jsonl

    # ablation on raw model spans written by `ner_baseline.py infer --raw-output ...`
    python scripts/hybrid_eval.py ablation --truth data/synthetic/dev.jsonl --raw baselines/.../dev.raw.jsonl

The reference test is refused unless --final is given: it must be scored once, after every choice is fixed.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hybrid  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as h:
        return [json.loads(line) for line in h if line.strip()]


def evaluator():
    spec = importlib.util.spec_from_file_location("court_evaluator", ROOT / "data" / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["court_evaluator"] = mod
    spec.loader.exec_module(mod)
    return mod


def assign(spans: list[tuple[int, int, str]], text: str) -> list[dict]:
    """Same pseudonym for the same normalised surface (per category); good enough for NER metrics."""
    keys: dict[tuple[str, str], int] = {}
    counters = collections.Counter()
    out = []
    for s, e, c in sorted(spans):
        k = (c, " ".join(text[s:e].lower().split()))
        if k not in keys:
            counters[c] += 1
            keys[k] = counters[c]
        out.append({"start": s, "end": e, "pseudonym": f"{c}_{keys[k]}"})
    return out


def guard(paths: list[Path], final: bool) -> None:
    for p in paths:
        if "test/court_cases" in str(p).replace("\\", "/") and not final:
            raise SystemExit(f"{p} is the reference test: pass --final only for the single final evaluation")


def table(res: dict, title: str) -> None:
    print(f"\n{title}: relaxed macro F1 {res['ner_relaxed']['macro']['f1']:.4f}, strict {res['ner_strict']['macro']['f1']:.4f}, "
          f"deanon risk {res['deanonymization_risk_by_name']:.4f}")
    print(f"  {'type':<12}{'P':>7}{'R':>7}{'F1':>7}{'sP':>7}{'sR':>7}{'sF1':>7}")
    for t in hybrid.CATEGORIES:
        r, s = res["ner_relaxed"]["per_type"][t], res["ner_strict"]["per_type"][t]
        print(f"  {t:<12}{r['precision']:7.3f}{r['recall']:7.3f}{r['f1']:7.3f}{s['precision']:7.3f}{s['recall']:7.3f}{s['f1']:7.3f}")


def cmd_rules(args) -> None:
    ev = evaluator()
    for path in args.truth:
        gold = load_jsonl(path)
        per_rule = collections.defaultdict(lambda: [0, 0])  # rule -> [matched, total]
        preds = []
        for row in gold:
            text = row["original"]
            gspans = [(a["start"], a["end"], a["pseudonym"].split("_")[0].strip()) for a in row["annotations"]]
            persons = [g for g in gspans if g[2] == "ОСОБА"] if args.gold_persons else []
            rs = hybrid.rule_spans(text, persons)
            chosen = hybrid.merge_with_rules([], rs)
            preds.append({"id": row["id"], "annotations": assign(chosen, text)})
            for s, e, c, name in rs:
                hit = any(gc == c and min(e, ge) - max(s, gs) >= .5 * (max(e, ge) - min(s, gs)) for gs, ge, gc in gspans)
                per_rule[name][0] += hit
                per_rule[name][1] += 1
        res = ev.evaluate(gold, preds)
        table(res, f"rules only - {path}")
        print("  per-rule precision (IoU >= 0.5, same category):")
        for name, (hit, tot) in sorted(per_rule.items(), key=lambda x: -x[1][1]):
            print(f"    {name:<14}{hit:6d}/{tot:<6d} {hit / tot:.3f}")


def cmd_ablation(args) -> None:
    guard(args.truth, args.final)
    ev = evaluator()
    gold = load_jsonl(args.truth[0])
    raw = {str(r["id"]): [tuple(s) for s in r["spans"]] for r in load_jsonl(args.raw)}
    variants = {"model (raw)": (False, False), "+ snap": (True, False), "+ rules": (False, True), "+ snap + rules": (True, True)}
    for name, (use_snap, use_rules) in variants.items():
        preds = []
        for row in gold:
            spans = hybrid.combine(row["original"], raw.get(str(row["id"]), []), use_snap, use_rules)
            preds.append({"id": row["id"], "annotations": assign(spans, row["original"])})
        table(ev.evaluate(gold, preds), name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rules")
    r.add_argument("--truth", type=Path, nargs="+", required=True)
    r.add_argument("--gold-persons", action="store_true", help="give the date rule the gold person spans (upper bound)")
    r.set_defaults(func=cmd_rules)
    a = sub.add_parser("ablation")
    a.add_argument("--truth", type=Path, nargs=1, required=True)
    a.add_argument("--raw", type=Path, required=True)
    a.add_argument("--final", action="store_true")
    a.set_defaults(func=cmd_ablation)
    args = ap.parse_args()
    if args.cmd == "rules":
        guard(args.truth, False)
    args.func(args)


if __name__ == "__main__":
    main()
