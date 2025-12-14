#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch


def to_native(obj):
    """Convierte tipos numpy/pandas a tipos nativos para JSON (y evita int64/float64)."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(x) for x in obj]
    if obj is pd.NA:
        return None
    if isinstance(obj, np.ndarray):
        return [to_native(x) for x in obj.tolist()]
    return obj


def fmt_ms(x):
    return f"{float(x):,.0f} ms"


def fmt_pct(x):
    return f"{float(x) * 100:,.2f}%"


def fmt_num(x):
    return f"{float(x):,.2f}"


def parse_jtl(jtl_path: Path) -> pd.DataFrame:
    df = pd.read_csv(jtl_path)
    # columnas típicas: timeStamp, elapsed, label, success, responseCode, responseMessage, ...
    df["elapsed"] = pd.to_numeric(df["elapsed"], errors="coerce")
    df = df.dropna(subset=["elapsed"])
    # success puede venir como True/False o "true"/"false"
    if df["success"].dtype != bool:
        df["success"] = df["success"].astype(str).str.lower().isin(["true", "1", "yes"])
    return df


def compute_metrics(df: pd.DataFrame):
    t0 = int(df["timeStamp"].min())
    t1 = int(df["timeStamp"].max())
    duration_s = max(1.0, (t1 - t0) / 1000.0)

    total = int(len(df))
    errors = int((~df["success"]).sum())
    successes = total - errors
    error_rate = (errors / total) if total else 0.0
    throughput_rps = (total / duration_s) if duration_s else 0.0

    kpis = {
        "samples": total,
        "successes": successes,
        "errors": errors,
        "error_rate": float(error_rate),
        "throughput_rps": float(throughput_rps),
        "avg_ms": float(df["elapsed"].mean()),
        "min_ms": float(df["elapsed"].min()),
        "p50_ms": float(df["elapsed"].quantile(0.50)),
        "p90_ms": float(df["elapsed"].quantile(0.90)),
        "p95_ms": float(df["elapsed"].quantile(0.95)),
        "p99_ms": float(df["elapsed"].quantile(0.99)),
        "max_ms": float(df["elapsed"].max()),
        "duration_seconds": float(duration_s),
        "start_timestamp_ms": t0,
        "end_timestamp_ms": t1,
    }

    # por transacción
    rows = []
    for label, g in df.groupby("label", dropna=False):
        g_total = int(len(g))
        g_errors = int((~g["success"]).sum())
        g_err = (g_errors / g_total) if g_total else 0.0
        rows.append({
            "label": str(label),
            "samples": g_total,
            "errors": g_errors,
            "error_rate": float(g_err),
            "avg_ms": float(g["elapsed"].mean()),
            "p95_ms": float(g["elapsed"].quantile(0.95)),
            "p99_ms": float(g["elapsed"].quantile(0.99)),
            "min_ms": float(g["elapsed"].min()),
            "max_ms": float(g["elapsed"].max()),
        })
    txn_df = pd.DataFrame(rows).sort_values(["p95_ms", "error_rate"], ascending=[False, False])

    # top errores
    top_errors = []
    err_df = df[~df["success"]].copy()
    if len(err_df) > 0:
        sig_parts = []
        # arma firma con lo que exista
        for col in ["responseCode", "responseMessage", "failureMessage", "label"]:
            if col in err_df.columns:
                sig_parts.append(col)

        def signature(r):
            parts = []
            for col in sig_parts:
                v = r.get(col, "")
                if pd.notna(v) and str(v).strip():
                    parts.append(str(v))
            return " | ".join(parts) if parts else "Error"

        err_df["signature"] = err_df.apply(signature, axis=1)
        vc = err_df["signature"].value_counts().head(10)
        for sig, cnt in vc.items():
            top_errors.append({
                "signature": str(sig),
                "count": int(cnt),
                "pct": float(cnt / total) if total else 0.0
            })

    return kpis, txn_df, top_errors


def build_custom_html(out_dir: Path, kpis: dict, txn_df: pd.DataFrame, top_errors: list):
    top_slow = txn_df.head(10).to_dict(orient="records")

    html_path = out_dir / "performance_report_custom.html"

    html = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Reporte de Performance</title>
<style>
  :root{{ --ink:#0b1220; --muted: rgba(11,18,32,.72); --line: rgba(11,18,32,.10); --accent:#0b3cff; --bg:#f7f9fc; --card:#fff;}}
  body{{font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin:0; background:var(--bg); color:var(--ink);}}
  .wrap{{max-width:1100px; margin:32px auto; padding:0 18px;}}
  .hero{{background:linear-gradient(135deg, rgba(11,60,255,.10), rgba(11,18,32,.02)); border:1px solid var(--line); border-radius:18px; padding:22px;}}
  .row{{display:grid; grid-template-columns: repeat(12, 1fr); gap:14px; margin-top:14px;}}
  .card{{background:var(--card); border:1px solid var(--line); border-radius:16px; padding:16px; box-shadow: 0 6px 18px rgba(11,18,32,.06);}}
  .kpi .v{{font-size:22px; font-weight:800;}}
  .kpi .k{{font-size:12px; color:var(--muted); letter-spacing:.2px; margin-bottom:6px;}}
  h1{{margin:0; font-size:22px;}} h2{{margin:0 0 8px; font-size:16px;}}
  .sub{{color:var(--muted); font-size:13px; margin-top:6px;}}
  table{{width:100%; border-collapse:collapse; font-size:13px;}}
  th, td{{border-bottom:1px solid var(--line); padding:10px 8px; text-align:left;}}
  th{{color:var(--muted); font-weight:700; font-size:12px; text-transform:uppercase; letter-spacing:.3px;}}
  .mono{{font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;}}
  .grid-3{{grid-column: span 4;}} .grid-4{{grid-column: span 3;}} .grid-6{{grid-column: span 6;}} .grid-12{{grid-column: span 12;}}
  @media (max-width: 900px){{ .grid-3,.grid-4,.grid-6{{grid-column: span 12;}} }}
</style>
</head>
<body>
<div class="wrap">

  <div class="hero">
    <h1>Reporte de Pruebas de Performance</h1>
    <div class="sub">
      Generado: <b>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</b> ·
      Ventana: <span class="mono">{kpis["start_timestamp_ms"]}</span> → <span class="mono">{kpis["end_timestamp_ms"]}</span> ·
      Duración: <b>{fmt_num(kpis["duration_seconds"])} s</b>
    </div>
  </div>

  <div class="row">
    <div class="card kpi grid-4"><div class="k">Muestras</div><div class="v">{kpis["samples"]:,}</div></div>
    <div class="card kpi grid-4"><div class="k">Throughput</div><div class="v">{fmt_num(kpis["throughput_rps"])} rps</div></div>
    <div class="card kpi grid-4"><div class="k">Error Rate</div><div class="v">{fmt_pct(kpis["error_rate"])}</div></div>

    <div class="card kpi grid-3"><div class="k">Promedio</div><div class="v">{fmt_ms(kpis["avg_ms"])}</div></div>
    <div class="card kpi grid-3"><div class="k">p90</div><div class="v">{fmt_ms(kpis["p90_ms"])}</div></div>
    <div class="card kpi grid-3"><div class="k">p95</div><div class="v">{fmt_ms(kpis["p95_ms"])}</div></div>
    <div class="card kpi grid-3"><div class="k">p99</div><div class="v">{fmt_ms(kpis["p99_ms"])}</div></div>
  </div>

  <div class="row">
    <div class="card grid-12">
      <h2>Top 10 transacciones más lentas (por p95)</h2>
      <table>
        <tr><th>Transacción</th><th>Muestras</th><th>Errores</th><th>Error rate</th><th>p95</th><th>p99</th></tr>
"""
    for r in top_slow:
        html += f"""<tr>
<td>{r["label"]}</td>
<td>{int(r["samples"])}</td>
<td>{int(r["errors"])}</td>
<td>{fmt_pct(r["error_rate"])}</td>
<td>{fmt_ms(r["p95_ms"])}</td>
<td>{fmt_ms(r["p99_ms"])}</td>
</tr>"""
    html += """
      </table>
    </div>
  </div>

  <div class="row">
    <div class="card grid-12">
      <h2>Top errores</h2>
"""
    if top_errors:
        html += "<table><tr><th>Firma</th><th>Conteo</th><th>% del total</th></tr>"
        for e in top_errors:
            html += f"""<tr>
<td class="mono">{e["signature"]}</td>
<td>{e["count"]}</td>
<td>{fmt_pct(e["pct"])}</td>
</tr>"""
        html += "</table>"
    else:
        html += '<div class="sub">No se registraron errores (success=true en todas las muestras).</div>'

    html += """
    </div>
  </div>

</div>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    return html_path


def build_pdf(out_dir: Path, kpis: dict, txn_df: pd.DataFrame, top_errors: list):
    pdf_path = out_dir / "performance_report_custom.pdf"

    top_slow = txn_df.head(5).to_dict(orient="records")

    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    width, height = letter

    def title(y, text):
        c.setFont("Helvetica-Bold", 16)
        c.drawString(0.8 * inch, y, text)
        return y - 0.35 * inch

    def kv(y, k, v):
        c.setFont("Helvetica-Bold", 10)
        c.drawString(0.8 * inch, y, k)
        c.setFont("Helvetica", 10)
        c.drawString(2.8 * inch, y, v)
        return y - 0.22 * inch

    y = height - 0.8 * inch
    y = title(y, "Reporte de Pruebas de Performance")
    c.setFont("Helvetica", 10)
    c.drawString(0.8 * inch, y, f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 0.25 * inch

    y = title(y, "Resumen (KPIs)")
    y = kv(y, "Muestras", f"{kpis['samples']:,}")
    y = kv(y, "Throughput", f"{kpis['throughput_rps']:.2f} rps")
    y = kv(y, "Error rate", f"{kpis['error_rate']*100:.2f}%")
    y = kv(y, "Avg", f"{kpis['avg_ms']:.0f} ms")
    y = kv(y, "p95", f"{kpis['p95_ms']:.0f} ms")
    y = kv(y, "p99", f"{kpis['p99_ms']:.0f} ms")
    y -= 0.15 * inch

    y = title(y, "Top 5 transacciones más lentas (p95)")
    c.setFont("Helvetica-Bold", 9)
    c.drawString(0.8 * inch, y, "Transacción")
    c.drawString(3.6 * inch, y, "p95")
    c.drawString(4.4 * inch, y, "p99")
    c.drawString(5.2 * inch, y, "Err%")
    y -= 0.18 * inch

    c.setFont("Helvetica", 9)
    for r in top_slow:
        c.drawString(0.8 * inch, y, str(r["label"])[:45])
        c.drawRightString(4.2 * inch, y, f"{r['p95_ms']:.0f} ms")
        c.drawRightString(5.0 * inch, y, f"{r['p99_ms']:.0f} ms")
        c.drawRightString(6.6 * inch, y, f"{r['error_rate']*100:.2f}%")
        y -= 0.18 * inch
        if y < 1.2 * inch:
            c.showPage()
            y = height - 0.8 * inch
            c.setFont("Helvetica", 9)

    y -= 0.15 * inch
    y = title(y, "Top errores")
    c.setFont("Helvetica", 9)
    if top_errors:
        for e in top_errors[:5]:
            line = f"{e['signature']}  (x{e['count']}, {e['pct']*100:.2f}%)"
            max_chars = 95
            for i in range(0, len(line), max_chars):
                c.drawString(0.8 * inch, y, line[i:i + max_chars])
                y -= 0.16 * inch
                if y < 1.0 * inch:
                    c.showPage()
                    y = height - 0.8 * inch
                    c.setFont("Helvetica", 9)
    else:
        c.drawString(0.8 * inch, y, "No se registraron errores.")
        y -= 0.2 * inch

    c.save()
    return pdf_path


def main():
    ap = argparse.ArgumentParser(description="Genera reporte HTML+PDF personalizado desde un results.jtl de JMeter.")
    ap.add_argument("--jtl", required=True, help="Ruta al results.jtl (CSV) generado por JMeter")
    ap.add_argument("--out", required=True, help="Directorio de salida (ej: target)")
    ap.add_argument("--export-csv", action="store_true", help="Exporta transactions_metrics.csv y errors_top.csv")
    args = ap.parse_args()

    jtl_path = Path(args.jtl).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not jtl_path.exists():
        raise SystemExit(f"No existe el JTL: {jtl_path}")

    df = parse_jtl(jtl_path)
    kpis, txn_df, top_errors = compute_metrics(df)

    # JSON auxiliar de KPIs (útil para CI)
    (out_dir / "kpis.json").write_text(json.dumps(to_native(kpis), indent=2), encoding="utf-8")

    html_path = build_custom_html(out_dir, kpis, txn_df, top_errors)
    pdf_path = build_pdf(out_dir, kpis, txn_df, top_errors)

    if args.export_csv:
        txn_df.to_csv(out_dir / "transactions_metrics.csv", index=False)
        pd.DataFrame(top_errors).to_csv(out_dir / "errors_top.csv", index=False)

    print("OK")
    print(f"HTML: {html_path}")
    print(f"PDF : {pdf_path}")
    print(f"KPIs: {out_dir / 'kpis.json'}")


if __name__ == "__main__":
    main()
