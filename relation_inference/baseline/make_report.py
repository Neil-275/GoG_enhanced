"""Render the report.json files written by run_baseline into one Markdown table set.

    python -m relation_inference.baseline.make_report results/baseline/family/family_finetune/report.json ...
"""
import json
import sys
from pathlib import Path

QA = ("Hits@Any", "Hits@Hard", "F1", "HHR")
SEL = ("r@1", "r@3", "rd@1", "rd@3", "oracle_in_C")
GROUPS = ("all", "present", "absent")
STAGES = (("text", "text only (λ=γ=0)"), ("presence", "+ presence"), ("full", "+ prior (full)"))


def fmt(x):
    return "–" if x is None else f"{x:.3f}"


def cfg_str(c):
    return f"T={c['T']} λ={c['lam']} γ={c['gamma']} k_b={c['k_b']} k_p={c['k_p']} m={c['m']} K={c['K']}"


def dataset_section(path: Path) -> list[str]:
    r = json.loads(path.read_text())
    ds = path.parent.parent.name
    counts = r["counts"]["test"]
    lines = [f"## {ds}", "",
             f"Prior: `{r['prior_path']}`. Test: {counts['n']} questions "
             f"({counts.get('present', 0)} present, {counts.get('absent', 0)} absent).", ""]

    lines += ["### QA on test", "",
              "| Row | Target | Config | Group | " + " | ".join(QA) + " |",
              "|---|---|---|---|" + "---|" * len(QA)]
    for stage, label in STAGES:
        if stage not in r:
            continue
        for target in ("F1", "HHR"):
            t = r[stage][target]["test"]
            for g in GROUPS:
                m = t["qa"][g]
                lines.append(f"| {label} | {target} | {cfg_str(t['config']) if g == 'all' else ''} | {g} | "
                             + " | ".join(fmt(m.get(k)) for k in QA) + " |")
    if "oracle" in r:
        picked_K = sorted({r[s][t]["test"]["config"]["K"] for s, _ in STAGES if s in r for t in ("F1", "HHR")})
        for K in picked_K or [int(k) for k in r["oracle"]["test"]]:
            for g in GROUPS:
                m = r["oracle"]["test"][str(K)][g]
                lines.append(f"| oracle (r, d) | – | K={K}, all gold slots | {g} | "
                             + " | ".join(fmt(m.get(k)) for k in QA) + " |")
    lines.append("")

    lines += ["### Relation selection on test (F1 config of each stage)", "",
              "| Row | Group | " + " | ".join(SEL) + " |", "|---|---|" + "---|" * len(SEL)]
    for stage, label in STAGES:
        if stage in r:
            for g in GROUPS:
                s = r[stage]["F1"]["test"]["selection"][g]
                lines.append(f"| {label} | {g} | " + " | ".join(fmt(s.get(k)) for k in SEL) + " |")
    lines.append("")

    lines += ["### Winner sources on test (F1 config of each stage, all questions)", ""]
    for stage, label in STAGES:
        if stage in r:
            ws = r[stage]["F1"]["test"]["winner_sources"]["all"]
            lines.append(f"- **{label}**: " + ", ".join(f"{k}: {v}" for k, v in ws.items()))
    lines.append("")
    return lines


def main():
    out = ["# Propose–Predict baseline (no agent)", ""]
    for p in sys.argv[1:]:
        out += dataset_section(Path(p))
    text = "\n".join(out)
    target = Path("results/baseline/REPORT.md")
    target.write_text(text, encoding="utf-8")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
