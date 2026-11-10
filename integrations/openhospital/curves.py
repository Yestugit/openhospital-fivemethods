from __future__ import annotations
import csv
import html
from pathlib import Path
from typing import Any, Mapping, Sequence


def _write_curve_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    x_key: str,
    series: Sequence[tuple[str, str]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([x_key, *(key for key, _ in series)])
        for row in rows:
            writer.writerow([row.get(x_key), *(row.get(key) for key, _ in series)])


def _write_curve_svg(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    x_key: str,
    x_label: str,
    y_label: str,
    title: str,
    series: Sequence[tuple[str, str]],
    y_range: tuple[float, float] | None = None,
) -> None:
    width, height = 900, 520
    left, right, top, bottom = 82, 28, 50, 68
    plot_width = width - left - right
    plot_height = height - top - bottom
    colors = ("#2563eb", "#dc2626", "#059669", "#7c3aed")
    samples = [
        (float(row[x_key]), row)
        for row in rows
        if isinstance(row.get(x_key), (int, float))
    ]
    values = [
        float(row[key])
        for _, row in samples
        for key, _ in series
        if isinstance(row.get(key), (int, float))
    ]
    if samples:
        x_min = min(value for value, _ in samples)
        x_max = max(value for value, _ in samples)
    else:
        x_min, x_max = 0.0, 1.0
    if x_max == x_min:
        x_max = x_min + 1.0
    if y_range is None:
        y_min = min(0.0, min(values, default=0.0))
        y_max = max(values, default=1.0)
        if y_max == y_min:
            y_max = y_min + 1.0
        y_max *= 1.05
    else:
        y_min, y_max = y_range

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="Arial" font-size="18">{html.escape(title)}</text>',
    ]
    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = sy(value)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="11">{value:.2f}</text>')
    parts.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#111827"/>',
        f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="13">{html.escape(x_label)}</text>',
        f'<text x="20" y="{height / 2}" text-anchor="middle" font-family="Arial" font-size="13" transform="rotate(-90 20 {height / 2})">{html.escape(y_label)}</text>',
    ])
    for index, (key, label) in enumerate(series):
        points = [
            (sx(x_value), sy(float(row[key])))
            for x_value, row in samples if isinstance(row.get(key), (int, float))
        ]
        color = colors[index % len(colors)]
        if len(points) >= 2:
            coordinates = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
            parts.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in points:
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{color}"/>')
        legend_x = left + index * 210
        parts.append(f'<line x1="{legend_x}" y1="{height - 42}" x2="{legend_x + 24}" y2="{height - 42}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x + 31}" y="{height - 38}" font-family="Arial" font-size="11">{html.escape(label)}</text>')
    for x_value, _ in samples:
        x = sx(x_value)
        parts.append(f'<text x="{x:.2f}" y="{height - bottom + 18}" text-anchor="middle" font-family="Arial" font-size="10">{x_value:g}</text>')
    if not samples:
        parts.append(f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle" font-family="Arial" font-size="14">No data</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_openhospital_curves(
    points: Sequence[Mapping[str, Any]], output_dir: str | Path
) -> dict[str, str]:
    target = Path(output_dir).expanduser().resolve() / "curves"
    target.mkdir(parents=True, exist_ok=True)
    batches = sorted(
        (row for row in points if isinstance(row.get("batch_index"), int)),
        key=lambda row: int(row["batch_index"]),
    )
    checkpoints = sorted(
        (
            row for row in points
            if row.get("split") == "primary_test" and isinstance(row.get("checkpoint"), int)
        ),
        key=lambda row: int(row["checkpoint"]),
    )
    clinical = (
        ("examination_precision", "Examination precision"),
        ("diagnosis_accuracy", "Diagnosis accuracy"),
        ("treatment_overall_score", "Treatment score"),
    )
    tokens = (("combined_input_tokens", "Total input tokens"),)
    checkpoint_metrics = (
        ("examination_precision", "Examination precision"),
        ("diagnosis_accuracy", "Diagnosis accuracy"),
        ("treatment_overall_score", "Treatment score"),
    )
    specs = (
        ("batch_clinical", batches, "batch_index", "Training batch", "Score", "Training-batch clinical performance", clinical, (0.0, 1.0)),
        ("batch_input_tokens", batches, "batch_index", "Training batch", "Input tokens", "Training-batch input tokens", tokens, None),
        ("checkpoint_learning", checkpoints, "checkpoint", "Completed training encounters", "Score", "Fixed primary-test learning curve", checkpoint_metrics, (0.0, 1.0)),
    )
    artifacts: dict[str, str] = {}
    for name, rows, x_key, x_label, y_label, title, series, y_range in specs:
        csv_path = target / f"{name}.csv"
        svg_path = target / f"{name}.svg"
        _write_curve_csv(csv_path, rows, x_key=x_key, series=series)
        _write_curve_svg(
            svg_path,
            rows,
            x_key=x_key,
            x_label=x_label,
            y_label=y_label,
            title=title,
            series=series,
            y_range=y_range,
        )
        artifacts[f"{name}_csv"] = str(csv_path)
        artifacts[f"{name}_svg"] = str(svg_path)
    return artifacts
