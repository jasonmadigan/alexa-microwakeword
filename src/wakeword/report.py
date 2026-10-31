from __future__ import annotations


def _pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def _point_line(name: str, label: str, p: dict) -> str:
    house = "" if p.get("household_ambient_faph") is None else f", household ambient {p['household_ambient_faph']:.2f}/h"
    return (
        f"| {name} | {label} | {p['probability_cutoff']:.2f} | {p['sliding_window_size']} | "
        f"{p['ambient_faph']:.2f}{house} | {_pct(p['worst_group_recall'])} | {_pct(p['mean_group_recall'])} | "
        f"{_pct(p.get('confusable_accept_rate'))} |"
    )


def markdown(report: dict) -> str:
    models = report["models"]
    lines = [
        "# Evaluation",
        "",
        f"Ambient audio: {report['ambient_hours']:.2f} h. Target: at most {report['target_faph']} false accepts per hour"
        + ("" if report.get("max_confusable_accept") is None else f" and {100 * report['max_confusable_accept']:.1f} % of confusable phrases accepted")
        + ".",
        "Detection follows the firmware: sliding-window mean over uint8 probabilities, 1 s cool-off, VAD gate.",
        "Bold recall is under the configured minimum for that group.",
        "",
        "| model | point | cutoff | window | false accepts/h | worst group | mean group | confusables accepted |",
        "|-|-|-|-|-|-|-|-|",
    ]
    for name, m in models.items():
        if "configured" in m:
            lines.append(_point_line(name, "configured", m["configured"]))
        lines.append(_point_line(name, "best", m["best"]))

    for name, m in models.items():
        if name != "candidate" or not m.get("frontier"):
            continue
        lines += [
            "",
            "## Operating points (candidate)",
            "",
            "Every 0.05 of cutoff per window. Confusables: mean accept rate over all confusable-phrase clips.",
            "",
            "| cutoff | window | false accepts/h | worst group | mean group | confusables accepted |",
            "|-|-|-|-|-|-|",
        ]
        for p in sorted(m["frontier"], key=lambda p: (p["sliding_window_size"], p["probability_cutoff"])):
            lines.append(
                f"| {p['probability_cutoff']:.2f} | {p['sliding_window_size']} | {p['ambient_faph']:.2f} | "
                f"{_pct(p['worst_group_recall'])} | {_pct(p['mean_group_recall'])} | {_pct(p['confusable_accept_rate'])} |"
            )

    columns = [(name, "configured" if "configured" in m else "best") for name, m in models.items()]
    first = models[columns[0][0]][columns[0][1]]
    conditions = [c for c in next(iter(first["groups"].values())) if c != "n"]
    header = " | ".join(f"{n} ({pt}) {c}" for n, pt in columns for c in conditions)
    lines += ["", "## Recall by group", "", f"| group | n | {header} |", "|-|-|" + "-|" * (len(columns) * len(conditions))]
    for group in first["groups"]:
        cells = []
        for n, pt in columns:
            point = models[n][pt]
            g = point["groups"].get(group, {})
            # bold marks a group under evaluation.min_group_recall at that point
            low = group in point["groups_below_min_recall"]
            cells += [f"**{_pct(g.get(c))}**" if low else _pct(g.get(c)) for c in conditions]
        lines.append(f"| {group} | {first['groups'][group]['n']} | " + " | ".join(cells) + " |")

    hard = first["hard_negative_accept_rate"]
    if hard:
        lines += ["", "## Confusable phrases accepted", "", f"| phrase | {header} |", "|-|" + "-|" * (len(columns) * len(conditions))]
        for phrase in sorted(hard):
            cells = []
            for n, pt in columns:
                h = models[n][pt]["hard_negative_accept_rate"].get(phrase, {})
                cells += [_pct(h.get(c)) for c in conditions]
            lines.append(f"| {phrase.removeprefix('phrase/')} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
