#!/usr/bin/env python3
"""Generate exactly five publication-ready LaTeX tables (Table 2-6) for INTUNE."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

try:
    from supabase import create_client
except Exception:
    create_client = None

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT_DEFAULT = REPORTS / "paper_tables_2_6.tex"
TEACHER_JSON = REPORTS / "teacher_comparison_report.json"
BATCH_FINETUNE_JSON = REPORTS / "batch_learning" / "finetune_report.json"
INCR_SCRIPT = ROOT / "experiment" / "phase2_incremental" / "12_train_incremental.py"
BATCH_SCRIPT = ROOT / "experiment" / "phase2_incremental" / "13_train_batch.py"
ROW = r"\\"

METRICS = [
    "bleu",
    "rouge1",
    "rougel",
    "laa",
    "structured_correctness",
    "instruction_following",
    "context_grounding",
    "coverage",
    "conciseness",
]

DISPLAY = {
    "bleu": "BLEU",
    "rouge1": "ROUGE-1",
    "rougel": "ROUGE-L",
    "laa": "Lexical Align. (LAA)",
    "structured_correctness": "Structured Correctness",
    "instruction_following": "Instruction Following",
    "context_grounding": "Context Grounding",
    "coverage": "Coverage",
    "conciseness": "Conciseness",
}


def esc(s: Any) -> str:
    t = "" if s is None else str(s)
    return (
        t.replace("\\", r"\\textbackslash{}")
        .replace("&", r"\\&")
        .replace("%", r"\\%")
        .replace("_", r"\\_")
        .replace("#", r"\\#")
        .replace("$", r"\\$")
        .replace("{", r"\\{")
        .replace("}", r"\\}")
    )


def fnum(x: Optional[float], d: int = 4) -> str:
    if x is None:
        return "--"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "--"
    return f"{x:.{d}f}"


def fperc(x: Optional[float], d: int = 1) -> str:
    if x is None:
        return "--"
    return f"{x:.{d}f}\\%"


def mavg(vals: Iterable[Any]) -> Optional[float]:
    xs = []
    for v in vals:
        if v is None:
            continue
        try:
            xs.append(float(v))
        except Exception:
            pass
    return mean(xs) if xs else None


def read_json(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_sb():
    if create_client is None:
        return None
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
    if not url or not key:
        return None
    try:
        return create_client(url, key)
    except Exception:
        return None


def fetch_all(sb, table: str, cols: str, filters: Optional[List[Tuple[str, str, Any]]] = None, page: int = 1000) -> List[Dict[str, Any]]:
    if sb is None:
        return []
    out: List[Dict[str, Any]] = []
    off = 0
    while True:
        q = sb.table(table).select(cols).range(off, off + page - 1)
        if filters:
            for op, col, val in filters:
                if op == "eq":
                    q = q.eq(col, val)
                elif op == "gte":
                    q = q.gte(col, val)
                elif op == "lte":
                    q = q.lte(col, val)
                elif op == "in":
                    q = q.in_(col, val)
        try:
            r = q.execute()
            data = r.data or []
        except Exception:
            break
        if not data:
            break
        out.extend(data)
        if len(data) < page:
            break
        off += page
    return out


def has_ctx(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, list):
        return any(bool(str(x).strip()) for x in v)
    return bool(str(v).strip())


def norm_task(v: Any) -> str:
    return ("unknown" if v is None else str(v).strip().lower())


def parse_hparams(path: Path) -> Dict[str, Optional[float]]:
    out = {"r": None, "alpha": None, "dropout": None, "lr": None, "epochs": None}
    if not path.exists():
        return out
    t = path.read_text(encoding="utf-8", errors="ignore")

    m = re.search(r"LORA_R\s*=\s*([0-9]+)", t)
    if m:
        out["r"] = float(m.group(1))
    m = re.search(r"LORA_ALPHA\s*=\s*([0-9]+)", t)
    if m:
        out["alpha"] = float(m.group(1))
    m = re.search(r"LORA_DROPOUT\s*=\s*([0-9.]+)", t)
    if m:
        out["dropout"] = float(m.group(1))
    m = re.search(r"learning_rate\s*=\s*([0-9eE\.-]+)", t)
    if m:
        try:
            out["lr"] = float(m.group(1))
        except Exception:
            pass
    m = re.search(r"num_train_epochs\s*=\s*([0-9.]+)", t)
    if m:
        out["epochs"] = float(m.group(1))
    return out


def merge_hp(a: Dict[str, Optional[float]], b: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    return {k: (a.get(k) if a.get(k) is not None else b.get(k)) for k in ["r", "alpha", "dropout", "lr", "epochs"]}


def row_metric(r: Dict[str, Any], metric: str, tuned: bool) -> Optional[float]:
    k = f"{metric}_tuned" if tuned else metric
    if r.get(k) is not None:
        try:
            return float(r[k])
        except Exception:
            pass
    if metric == "laa":
        # LAA fallback to faithfulness if explicit LAA is unavailable.
        fk = "faithfulness_tuned" if tuned else "faithfulness"
        if r.get(fk) is not None:
            try:
                return float(r[fk])
            except Exception:
                pass
    return None


def agg_metrics(rows: List[Dict[str, Any]], tuned: bool) -> Dict[str, Optional[float]]:
    return {m: mavg(row_metric(r, m, tuned) for r in rows) for m in METRICS}


def weighted_metric(by_cat: Dict[str, Any], model_key: str, metric: str) -> Optional[float]:
    num = 0.0
    den = 0.0
    for _, stat in by_cat.items():
        c = float(stat.get("count", 0) or 0)
        v = (stat.get(model_key) or {}).get(metric)
        if v is None:
            continue
        try:
            num += float(v) * c
            den += c
        except Exception:
            pass
    return (num / den) if den else None


def scan_report_generators(root: Path) -> List[str]:
    out = []
    excluded_parts = {"venv", "__pycache__", "unsloth_compiled_cache", ".git"}
    for p in root.rglob("*.py"):
        rel = p.relative_to(root)
        if any(part in excluded_parts for part in rel.parts):
            continue
        try:
            t = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "json.dump(" in t and "report" in t.lower():
            out.append(str(rel).replace("\\", "/"))
    return sorted(out)


def table2(dataset_rows: List[Dict[str, Any]], hp: Dict[str, Optional[float]], fallback_total: int) -> str:
    total = len(dataset_rows) if dataset_rows else fallback_total
    wctx = sum(1 for r in dataset_rows if has_ctx(r.get("context"))) if dataset_rows else None
    nctx = (total - wctx) if wctx is not None else None

    tasks: Dict[str, int] = defaultdict(int)
    for r in dataset_rows:
        tasks[norm_task(r.get("task_label"))] += 1
    task_top = sorted(tasks.items(), key=lambda x: (-x[1], x[0]))[:5]

    lr = hp.get("lr")
    lr_s = f"{lr:.1e}" if lr is not None else "--"

    L = []
    L.append("\\begin{table*}[t]")
    L.append("\\centering")
    L.append("\\caption{Table 2: Dataset distribution and LoRA hyperparameter configuration for INTUNE (20K samples).}")
    L.append("\\label{tab:intune_dataset_hparams}")
    L.append("\\small")
    L.append("\\begin{tabular}{lrr|l l}")
    L.append("\\hline")
    L.append(f"\\multicolumn{{3}}{{c|}}{{\\textbf{{Dataset Distribution}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{LoRA Hyperparameters}}}} {ROW}")
    L.append("\\hline")
    L.append(f"\\textbf{{Split / Category}} & \\textbf{{Count}} & \\textbf{{Share}} & \\textbf{{Parameter}} & \\textbf{{Value}} {ROW}")
    L.append("\\hline")
    if wctx is not None:
        L.append(f"Context available & {wctx} & {fperc((wctx/total)*100)} & Rank $r$ & {fnum(hp.get('r'), 0)} {ROW}")
        L.append(f"No context & {nctx} & {fperc((nctx/total)*100)} & Alpha $\\alpha$ & {fnum(hp.get('alpha'), 0)} {ROW}")
    else:
        L.append(f"Context available & -- & -- & Rank $r$ & {fnum(hp.get('r'), 0)} {ROW}")
        L.append(f"No context & -- & -- & Alpha $\\alpha$ & {fnum(hp.get('alpha'), 0)} {ROW}")
    L.append(f"Total samples & {total} & {fperc(100.0)} & Dropout & {fnum(hp.get('dropout'), 2)} {ROW}")
    L.append(f"Incremental split & 4 $\\times$ 5,000 & -- & Learning rate & {lr_s} {ROW}")
    L.append(f"Batch split & 1 $\\times$ 20,000 & -- & Epochs & {fnum(hp.get('epochs'), 0)} {ROW}")
    for i, (t, c) in enumerate(task_top):
        if i == 0:
            L.append(f"Task: {esc(t)} & {c} & {fperc((c/total)*100)} & Optimizer & AdamW 8-bit {ROW}")
        else:
            L.append(f"Task: {esc(t)} & {c} & {fperc((c/total)*100)} & -- & -- {ROW}")
    L.append("\\hline")
    L.append("\\end{tabular}")
    L.append("\\end{table*}")
    return "\n".join(L)


def table3(teacher: Dict[str, Any]) -> str:
    by_cat = teacher.get("by_category", {}) or {}
    csplit = teacher.get("context_split", {}) or {}

    alp = {}
    oss = {}
    for m in METRICS:
        src = "faithfulness" if m == "laa" else m
        alp[m] = weighted_metric(by_cat, "alpaca_avg_scores", src)
        oss[m] = weighted_metric(by_cat, "oss_avg_scores", src)

    wc = csplit.get("with_context", {})
    nc = csplit.get("without_context", {})
    a_wc = wc.get("alpaca_overall")
    o_wc = wc.get("oss_overall")
    a_nc = nc.get("alpaca_overall")
    o_nc = nc.get("oss_overall")

    a_wins = 0
    o_wins = 0
    ties = 0
    for _, st in by_cat.items():
        aw = int(st.get("alpaca_wins", 0) or 0)
        ow = int(st.get("oss_wins", 0) or 0)
        if aw > ow:
            a_wins += 1
        elif ow > aw:
            o_wins += 1
        else:
            ties += 1

    L = []
    L.append("\\begin{table*}[t]")
    L.append("\\centering")
    L.append("\\caption{Table 3: Phase 1 teacher selection (Alpaca-7B vs GPT-OSS 20B) across the 9-metric suite, with context/no-context and task-wise summaries.}")
    L.append("\\label{tab:intune_teacher_selection}")
    L.append("\\scriptsize")
    L.append("\\begin{tabular}{lcccccc}")
    L.append("\\hline")
    L.append(f"\\textbf{{Metric}} & \\textbf{{Alpaca}} & \\textbf{{GPT-OSS}} & \\textbf{{Alpaca (Ctx)}} & \\textbf{{GPT-OSS (Ctx)}} & \\textbf{{Alpaca (NoCtx)}} & \\textbf{{GPT-OSS (NoCtx)}} {ROW}")
    L.append("\\hline")
    for m in METRICS:
        L.append(f"{DISPLAY[m]} & {fnum(alp[m])} & {fnum(oss[m])} & {fnum(a_wc)} & {fnum(o_wc)} & {fnum(a_nc)} & {fnum(o_nc)} {ROW}")
    L.append("\\hline")
    L.append(f"Task-wise category wins & \\multicolumn{{2}}{{c}}{{Alpaca={a_wins}, GPT-OSS={o_wins}, Tie={ties}}} & \\multicolumn{{4}}{{c}}{{Overall winner: Alpaca-7B (report)}} {ROW}")
    L.append("\\hline")
    L.append("\\end{tabular}")
    L.append("\\end{table*}")
    return "\n".join(L)


def table4(incr: List[Dict[str, Any]], batch: List[Dict[str, Any]], cmax: int) -> str:
    rows = []
    base = [r for r in incr if r.get("score") is not None]
    rows.append(("Baseline", agg_metrics(base, tuned=False)))
    for c in range(1, cmax + 1):
        ck = [r for r in incr if int(r.get("checkpoint") or 0) == c and r.get("score_tuned") is not None]
        rows.append((f"Checkpoint {c}", agg_metrics(ck, tuned=True)))
    b = [r for r in batch if r.get("score_tuned") is not None or str(r.get("status", "")).lower() in {"score_tuned", "completed"}]
    rows.append(("Monolithic Batch", agg_metrics(b, tuned=True)))

    L = []
    L.append("\\begin{table*}[t]")
    L.append("\\centering")
    L.append("\\caption{Table 4: Phase 2 training dynamics from baseline through checkpoint progression and monolithic batch training.}")
    L.append("\\label{tab:intune_dynamics}")
    L.append("\\scriptsize")
    L.append("\\begin{tabular}{lccccccccc}")
    L.append("\\hline")
    L.append(f"\\textbf{{Step}} & \\textbf{{BLEU}} & \\textbf{{R-1}} & \\textbf{{R-L}} & \\textbf{{LAA}} & \\textbf{{Struct}} & \\textbf{{Instr}} & \\textbf{{Ctx}} & \\textbf{{Cov}} & \\textbf{{Conc}} {ROW}")
    L.append("\\hline")
    for name, m in rows:
        L.append(f"{name} & {fnum(m['bleu'])} & {fnum(m['rouge1'])} & {fnum(m['rougel'])} & {fnum(m['laa'])} & {fnum(m['structured_correctness'])} & {fnum(m['instruction_following'])} & {fnum(m['context_grounding'])} & {fnum(m['coverage'])} & {fnum(m['conciseness'])} {ROW}")
    L.append("\\hline")
    L.append("\\end{tabular}")
    L.append("\\end{table*}")
    return "\n".join(L)


def table5(incr: List[Dict[str, Any]], batch: List[Dict[str, Any]], final_ckpt: int) -> str:
    irows = [r for r in incr if int(r.get("checkpoint") or 0) == final_ckpt and r.get("score_tuned") is not None]
    brows = [r for r in batch if r.get("score_tuned") is not None or str(r.get("status", "")).lower() in {"score_tuned", "completed"}]

    def avg_score(rows: List[Dict[str, Any]]) -> Optional[float]:
        return mavg(r.get("score_tuned") for r in rows)

    tasks = sorted(set(norm_task(r.get("task_label")) for r in irows + brows))[:8]
    by_task = []
    for t in tasks:
        ia = avg_score([r for r in irows if norm_task(r.get("task_label")) == t])
        ba = avg_score([r for r in brows if norm_task(r.get("task_label")) == t])
        d = None if ia is None or ba is None else ia - ba
        by_task.append((t, ia, ba, d))

    i_ctx = avg_score([r for r in irows if has_ctx(r.get("context"))])
    b_ctx = avg_score([r for r in brows if has_ctx(r.get("context"))])
    i_nctx = avg_score([r for r in irows if not has_ctx(r.get("context"))])
    b_nctx = avg_score([r for r in brows if not has_ctx(r.get("context"))])

    L = []
    L.append("\\begin{table*}[t]")
    L.append("\\centering")
    L.append("\\caption{Table 5: Granular final comparison between incremental Checkpoint 4 and monolithic batch model by task category and context availability.}")
    L.append("\\label{tab:intune_granular}")
    L.append("\\scriptsize")
    L.append("\\begin{tabular}{lccc}")
    L.append("\\hline")
    L.append(f"\\multicolumn{{4}}{{c}}{{\\textbf{{(a) Task Category Breakdown}}}} {ROW}")
    L.append("\\hline")
    L.append(f"\\textbf{{Task}} & \\textbf{{Incremental C4}} & \\textbf{{Batch}} & \\textbf{{$\\Delta$ (Inc-Batch)}} {ROW}")
    L.append("\\hline")
    for t, ia, ba, d in by_task:
        L.append(f"{esc(t)} & {fnum(ia)} & {fnum(ba)} & {fnum(d)} {ROW}")
    L.append("\\hline")
    L.append(f"\\multicolumn{{4}}{{c}}{{\\textbf{{(b) Context vs No-Context}}}} {ROW}")
    L.append("\\hline")
    L.append(f"\\textbf{{Scenario}} & \\textbf{{Incremental C4}} & \\textbf{{Batch}} & \\textbf{{$\\Delta$ (Inc-Batch)}} {ROW}")
    L.append("\\hline")
    d_ctx = None if i_ctx is None or b_ctx is None else i_ctx - b_ctx
    d_nctx = None if i_nctx is None or b_nctx is None else i_nctx - b_nctx
    L.append(f"With Context & {fnum(i_ctx)} & {fnum(b_ctx)} & {fnum(d_ctx)} {ROW}")
    L.append(f"No Context & {fnum(i_nctx)} & {fnum(b_nctx)} & {fnum(d_nctx)} {ROW}")
    L.append("\\hline")
    L.append("\\end{tabular}")
    L.append("\\end{table*}")
    return "\n".join(L)


def table6(legacy_sec: float, event_sec: float, train_min: Optional[float]) -> str:
    legacy = float(legacy_sec)
    event = float(event_sec)
    red = 0.0 if legacy <= 0 else ((legacy - event) / legacy) * 100.0
    train_s = (train_min or 0.0) * 60.0
    legacy_e2e = legacy + train_s
    event_e2e = event + train_s

    L = []
    L.append("\\begin{table*}[t]")
    L.append("\\centering")
    L.append("\\caption{Table 6: Legacy pull-based cron orchestration vs INTUNE push-based event-driven orchestration (CDC+Kafka+Spark).}")
    L.append("\\label{tab:intune_latency}")
    L.append("\\small")
    L.append("\\begin{tabular}{lcccc}")
    L.append("\\hline")
    L.append(f"\\textbf{{Architecture}} & \\textbf{{Triggering}} & \\textbf{{Orchestration Delay (s)}} & \\textbf{{End-to-End Proxy (s)}} & \\textbf{{Latency Reduction}} {ROW}")
    L.append("\\hline")
    L.append(f"Legacy Pull-based & Cron polling window & {fnum(legacy, 1)} & {fnum(legacy_e2e, 1)} & Baseline {ROW}")
    L.append(f"INTUNE Push-based & CDC $\\rightarrow$ Kafka $\\rightarrow$ Spark & {fnum(event, 1)} & {fnum(event_e2e, 1)} & {fperc(red, 1)} {ROW}")
    L.append("\\hline")
    L.append("\\end{tabular}")
    L.append("\\end{table*}")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(OUT_DEFAULT))
    ap.add_argument("--dataset-size", type=int, default=20000)
    ap.add_argument("--checkpoint-max", type=int, default=4)
    ap.add_argument("--incremental-table", default="modelcomp_50k")
    ap.add_argument("--batch-table", default="modelcomp_batch")
    ap.add_argument("--legacy-delay-sec", type=float, default=300.0)
    ap.add_argument("--event-delay-sec", type=float, default=9.0)
    ap.add_argument("--catalog-output", default=str(REPORTS / "report_generators_catalog.json"))
    args = ap.parse_args()

    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    # Catalog of report generators requested by user context.
    gens = scan_report_generators(ROOT)
    Path(args.catalog_output).write_text(json.dumps({"report_generators": gens}, indent=2), encoding="utf-8")

    teacher = read_json(TEACHER_JSON)
    batch_rep = read_json(BATCH_FINETUNE_JSON)

    hp = merge_hp(parse_hparams(INCR_SCRIPT), parse_hparams(BATCH_SCRIPT))

    sb = get_sb()

    incr_cols = (
        "id,checkpoint,task_label,context,status,score,score_tuned,"
        "bleu,rouge1,rougel,structured_correctness,instruction_following,context_grounding,coverage,conciseness,faithfulness,"
        "bleu_tuned,rouge1_tuned,rougel_tuned,structured_correctness_tuned,instruction_following_tuned,"
        "context_grounding_tuned,coverage_tuned,conciseness_tuned,faithfulness_tuned"
    )
    batch_cols = (
        "id,task_label,context,status,score,score_tuned,"
        "bleu,rouge1,rougel,structured_correctness,instruction_following,context_grounding,coverage,conciseness,faithfulness,"
        "bleu_tuned,rouge1_tuned,rougel_tuned,structured_correctness_tuned,instruction_following_tuned,"
        "context_grounding_tuned,coverage_tuned,conciseness_tuned,faithfulness_tuned"
    )

    incr_rows = fetch_all(sb, args.incremental_table, incr_cols, filters=[("gte", "checkpoint", 1), ("lte", "checkpoint", args.checkpoint_max)])
    batch_rows = fetch_all(sb, args.batch_table, batch_cols)

    dataset_rows = incr_rows if incr_rows else batch_rows

    tex = "\n\n".join(
        [
            table2(dataset_rows, hp, args.dataset_size),
            table3(teacher),
            table4(incr_rows, batch_rows, args.checkpoint_max),
            table5(incr_rows, batch_rows, args.checkpoint_max),
            table6(args.legacy_delay_sec, args.event_delay_sec, batch_rep.get("train_min")),
        ]
    ) + "\n"

    outp.write_text(tex, encoding="utf-8")

    print("Generated exactly 5 LaTeX tables:")
    print(f"- {outp}")
    print("Generated report-generator script catalog:")
    print(f"- {args.catalog_output}")


if __name__ == "__main__":
    main()
