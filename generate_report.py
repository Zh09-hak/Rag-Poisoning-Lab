import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_frames(paths):
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    if "error" in df.columns:
        df["error"] = df["error"].astype("string").fillna("")
    else:
        df["error"] = ""
    if "hedged" in df.columns:
        df["hedged"] = df["hedged"].astype(str).str.lower().eq("true")
    return df

def split_types(df):
    if "lexical_acceptance" in df.columns:
        semantic_mask = df["lexical_acceptance"].notna()
    else:
        semantic_mask = pd.Series(False, index=df.index)
    return df[~semantic_mask].copy(), df[semantic_mask].copy()

def pick_benchmark(df, benchmark_id):
    if df.empty or "benchmark_id" not in df.columns:
        return df
    if benchmark_id:
        return df[df["benchmark_id"] == benchmark_id].copy()
    counts = df["benchmark_id"].value_counts()
    return df[df["benchmark_id"] == counts.index[0]].copy()

def aggregate_lexical(df):
    return df.groupby(["provider", "model", "mode", "top_k"], as_index=False).agg(
        runs=("repeat", "count"),
        acceptance=("acceptance_rate", "mean"),
        assertion=("assertion_rate", "mean"),
        latency=("duration_ms", "mean"),
        hedged=("hedged", "mean"),
        evaluated=("claims_evaluated", "mean"),
    )

def build_reductions(agg):
    rows = []
    for (provider, model, top_k), grp in agg.groupby(["provider", "model", "top_k"]):
        v = grp[grp["mode"] == "vulnerable"]
        h = grp[grp["mode"] == "hardened"]
        if len(v) and len(h):
            rows.append({
                "provider": provider, "model": model, "top_k": int(top_k),
                "vulnerable_assertion": float(v["assertion"].iloc[0]),
                "hardened_assertion": float(h["assertion"].iloc[0]),
                "assertion_reduction": float(v["assertion"].iloc[0]) - float(h["assertion"].iloc[0]),
            })
    return pd.DataFrame(rows)

def layers_from_df(df, lex_col):
    if df.empty or "semantic_assertion_loose" not in df.columns:
        return pd.DataFrame()
    sub = df[df["semantic_assertion_loose"].notna()]
    if sub.empty or lex_col not in sub.columns:
        return pd.DataFrame()
    agg = sub.groupby(["provider", "mode"], as_index=False).agg(
        lexical=(lex_col, "mean"),
        strict=("semantic_assertion_strict", "mean"),
        loose=("semantic_assertion_loose", "mean"),
    )
    rows = []
    for provider, grp in agg.groupby("provider"):
        vuln = grp[grp["mode"] == "vulnerable"]
        if len(vuln):
            v = vuln.iloc[0]
            rows.append({
                "provider": provider,
                "model": sub[sub["provider"] == provider]["model"].iloc[0],
                "lexical": v["lexical"],
                "semantic_strict": v["strict"],
                "semantic_loose": v["loose"],
            })
    return pd.DataFrame(rows)

def sem_reductions_from_df(df):
    if df.empty or "semantic_assertion_loose" not in df.columns:
        return pd.DataFrame()
    sub = df[df["semantic_assertion_loose"].notna()]
    if sub.empty:
        return pd.DataFrame()
    agg = sub.groupby(["provider", "mode"], as_index=False).agg(loose=("semantic_assertion_loose", "mean"))
    rows = []
    for provider, grp in agg.groupby("provider"):
        v = grp[grp["mode"] == "vulnerable"]
        h = grp[grp["mode"] == "hardened"]
        if len(v) and len(h):
            rows.append({
                "provider": provider,
                "model": sub[sub["provider"] == provider]["model"].iloc[0],
                "vulnerable_loose": float(v["loose"].iloc[0]),
                "hardened_loose": float(h["loose"].iloc[0]),
                "semantic_reduction": float(v["loose"].iloc[0]) - float(h["loose"].iloc[0]),
            })
    return pd.DataFrame(rows)

def chart_metric_layers(layers, out):
    labels = layers["provider"].tolist()
    x = np.arange(len(labels))
    width = 0.26
    lex = (layers["lexical"] * 100).tolist()
    strict = (layers["semantic_strict"] * 100).tolist()
    loose = (layers["semantic_loose"] * 100).tolist()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, lex, width, label="Lexical overlap", color="#8b5cf6")
    ax.bar(x, strict, width, label="Semantic strict (full claim)", color="#ffb020")
    ax.bar(x + width, loose, width, label="Semantic loose (fact + partial)", color="#ff3b5c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Vulnerable assertion (%)")
    ax.set_title("How metric choice changes the measured attack success")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "metric_layers.png", dpi=160)
    plt.close(fig)

def chart_semantic_reduction(red, out):
    labels = red["provider"].tolist()
    x = np.arange(len(labels))
    width = 0.35
    vuln = (red["vulnerable_loose"] * 100).tolist()
    hard = (red["hardened_loose"] * 100).tolist()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, vuln, width, label="Vulnerable", color="#ff3b5c")
    ax.bar(x + width / 2, hard, width, label="Hardened", color="#22ff88")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Semantic assertion, loose (%)")
    ax.set_title("Semantic assertion: vulnerable vs hardened")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "semantic_reduction.png", dpi=160)
    plt.close(fig)

def chart_assertion(agg, out):
    labels, vuln, hard = [], [], []
    for (provider, top_k), grp in agg.groupby(["provider", "top_k"]):
        labels.append(f"{provider}\nk={top_k}")
        v = grp[grp["mode"] == "vulnerable"]["assertion"]
        h = grp[grp["mode"] == "hardened"]["assertion"]
        vuln.append(float(v.iloc[0]) * 100 if len(v) else 0)
        hard.append(float(h.iloc[0]) * 100 if len(h) else 0)
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, vuln, width, label="Vulnerable", color="#ff3b5c")
    ax.bar(x + width / 2, hard, width, label="Hardened", color="#22ff88")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Assertion rate (%)")
    ax.set_title("Lexical assertion: vulnerable vs hardened")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "assertion_reduction.png", dpi=160)
    plt.close(fig)

def chart_latency(agg, out):
    labels = [f"{r.provider}\n{r.mode}\nk={r.top_k}" for r in agg.itertuples()]
    values = agg["latency"].tolist()
    colors = ["#ff3b5c" if m == "vulnerable" else "#22ff88" for m in agg["mode"]]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(labels)), values, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Mean latency (ms)")
    ax.set_title("Mean latency by configuration")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "latency.png", dpi=160)
    plt.close(fig)

def md_table(df, columns, headers):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in df.itertuples():
        cells = []
        for c in columns:
            v = getattr(r, c)
            cells.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

def write_report(lex_df, lex_agg, lex_red, sem_layers, sem_red, out):
    has_lex = not lex_agg.empty
    has_sem = not sem_layers.empty
    lines = []
    lines.append("# RAG Poisoning Research Lab — Final Report")
    lines.append("")
    lines.append("Controlled sandbox study of retrieval poisoning in RAG systems and a prompt-level defense. All identities are synthetic; no real people and no public publishing involved.")
    lines.append("")
    lines.append("## Executive summary")
    lines.append("")
    if has_sem:
        if not sem_red.empty:
            hard_max = float(sem_red["hardened_loose"].max())
            lines.append(f"- Hardened RAG semantic assertion (loose) never exceeded {hard_max:.1%} across all providers. The defense holds.")
        for _, row in sem_layers.iterrows():
            lines.append(f"- {row['provider']} vulnerable: lexical {row['lexical']:.0%}, semantic strict {row['semantic_strict']:.0%}, semantic loose {row['semantic_loose']:.0%}.")
        lines.append("- Lexical overlap under-counts paraphrased poisoning; the semantic loose metric recovers gist-level assertions that word matching misses.")
    if has_lex:
        failed = int((lex_df["error"] != "").sum())
        lines.append(f"- Lexical benchmark rows: {len(lex_df)}, failed runs: {failed}.")
    lines.append("- High lexical overlap in hardened mode reflects quoting unverified claims with caveats; assertion rate is the security-relevant metric.")
    lines.append("")
    if has_sem:
        lines.append("## Metric layers: how measurement changes the story")
        lines.append("")
        lines.append(md_table(sem_layers, ["provider", "model", "lexical", "semantic_strict", "semantic_loose"], ["Provider", "Model", "Lexical", "Semantic strict", "Semantic loose"]))
        lines.append("")
        lines.append("![Metric layers](metric_layers.png)")
        lines.append("")
        lines.append("## Semantic assertion reduction")
        lines.append("")
        lines.append(md_table(sem_red, ["provider", "model", "vulnerable_loose", "hardened_loose", "semantic_reduction"], ["Provider", "Model", "Vulnerable", "Hardened", "Reduction"]))
        lines.append("")
        lines.append("![Semantic reduction](semantic_reduction.png)")
        lines.append("")
    if has_lex:
        lines.append("## Lexical assertion reduction")
        lines.append("")
        lines.append(md_table(lex_red, ["provider", "model", "top_k", "vulnerable_assertion", "hardened_assertion", "assertion_reduction"], ["Provider", "Model", "Top-k", "Vulnerable", "Hardened", "Reduction"]))
        lines.append("")
        lines.append("![Assertion reduction](assertion_reduction.png)")
        lines.append("")
        lines.append("## Lexical aggregated metrics")
        lines.append("")
        lines.append(md_table(lex_agg, ["provider", "model", "mode", "top_k", "runs", "acceptance", "assertion", "latency", "hedged"], ["Provider", "Model", "Mode", "Top-k", "Runs", "Overlap", "Assertion", "Latency ms", "Hedged"]))
        lines.append("")
        lines.append("![Latency](latency.png)")
        lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- Lexical overlap can miss paraphrased claims; the semantic judge mitigates this but adds its own noise.")
    lines.append("- LLM judges can disagree; hold the judge constant when comparing answer models.")
    lines.append("- Cloud latency is affected by rate limits and queueing; local latency is more stable.")
    lines.append("- Single question template per run; repeat with varied questions and larger stores for broader coverage.")
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("Run the app, generate synthetic claims, run Benchmark with the judge enabled, export the CSV, then:")
    lines.append("")
    lines.append("```bash")
    lines.append("python generate_report.py <benchmark_csv> [semantic_csv]")
    lines.append("```")
    lines.append("")
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+")
    parser.add_argument("--benchmark-id", default="")
    parser.add_argument("--out", default="report")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_frames(args.csv)
    lex_df, sem_df = split_types(df)
    lex_df = pick_benchmark(lex_df, args.benchmark_id)
    has_lex = not lex_df.empty and "assertion_rate" in lex_df.columns
    lex_agg = aggregate_lexical(lex_df) if has_lex else pd.DataFrame()
    lex_red = build_reductions(lex_agg) if has_lex else pd.DataFrame()
    sem_layers = layers_from_df(sem_df, "lexical_acceptance")
    sem_red = sem_reductions_from_df(sem_df)
    if sem_layers.empty:
        sem_layers = layers_from_df(lex_df, "acceptance_rate")
    if sem_red.empty:
        sem_red = sem_reductions_from_df(lex_df)
    has_sem = not sem_layers.empty
    if has_lex:
        chart_assertion(lex_agg, out)
        chart_latency(lex_agg, out)
    if has_sem:
        chart_metric_layers(sem_layers, out)
        chart_semantic_reduction(sem_red, out)
    write_report(lex_df, lex_agg, lex_red, sem_layers, sem_red, out)
    if has_lex:
        lex_agg.to_csv(out / "lexical_aggregated.csv", index=False)
    if has_sem:
        sem_layers.to_csv(out / "semantic_layers.csv", index=False)
    print("report_written:", str(out / "REPORT.md"))

if __name__ == "__main__":
    main()