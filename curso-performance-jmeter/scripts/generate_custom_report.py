# scripts/generate_custom_report.py
import argparse
import os
import json
from datetime import datetime

import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)

import matplotlib.pyplot as plt


def read_jtl(jtl_path: str) -> pd.DataFrame:
    """
    Lee JTL CSV típico de JMeter.
    Columnas esperadas (pueden variar): timeStamp, elapsed, label, responseCode, success, bytes, sentBytes, Latency, Connect
    """
    df = pd.read_csv(jtl_path)

    # Normaliza columnas comunes
    colmap = {c.lower(): c for c in df.columns}
    def pick(name, default=None):
        return colmap.get(name.lower(), default)

    # timestamps
    ts_col = pick("timeStamp") or pick("timestamp") or pick("time")
    if ts_col:
        df["ts"] = pd.to_datetime(df[ts_col], unit="ms", errors="coerce")
    else:
        df["ts"] = pd.NaT

    # elapsed
    el_col = pick("elapsed")
    if el_col:
        df["elapsed"] = pd.to_numeric(df[el_col], errors="coerce")
    else:
        raise ValueError("No encuentro columna 'elapsed' en el JTL.")

    # label
    lab_col = pick("label")
    df["label"] = df[lab_col] if lab_col else "ALL"

    # success / responseCode
    suc_col = pick("success")
    df["success"] = df[suc_col].astype(str).str.lower().isin(["true", "1", "yes"]) if suc_col else True

    rc_col = pick("responseCode")
    df["responseCode"] = df[rc_col].astype(str) if rc_col else ""

    # responseMessage (opcional)
    rm_col = pick("responseMessage")
    df["responseMessage"] = df[rm_col].astype(str) if rm_col else ""

    # limpia
    df = df.dropna(subset=["elapsed"])
    return df


def pct(series: pd.Series, q: float) -> float:
    if series.empty:
        return float("nan")
    return float(np.percentile(series.values, q))


def build_kpis(df: pd.DataFrame) -> dict:
    start = df["ts"].min()
    end = df["ts"].max()
    duration_s = float((end - start).total_seconds()) if pd.notna(start) and pd.notna(end) else float("nan")

    samples = int(len(df))
    errors = int((~df["success"]).sum())
    error_rate = (errors / samples * 100.0) if samples else 0.0

    avg = float(df["elapsed"].mean()) if samples else float("nan")
    p90 = pct(df["elapsed"], 90)
    p95 = pct(df["elapsed"], 95)
    p99 = pct(df["elapsed"], 99)
    mx = float(df["elapsed"].max()) if samples else float("nan")
    mn = float(df["elapsed"].min()) if samples else float("nan")

    throughput = (samples / duration_s) if duration_s and not np.isnan(duration_s) and duration_s > 0 else float("nan")

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "start": str(start) if pd.notna(start) else "",
        "end": str(end) if pd.notna(end) else "",
        "duration_s": duration_s,
        "samples": samples,
        "errors": errors,
        "error_rate": error_rate,
        "throughput_rps": throughput,
        "avg_ms": avg,
        "p90_ms": p90,
        "p95_ms": p95,
        "p99_ms": p99,
        "min_ms": mn,
        "max_ms": mx,
    }


def top_slowest(df: pd.DataFrame, n=10) -> pd.DataFrame:
    g = df.groupby("label").agg(
        samples=("elapsed", "size"),
        errors=("success", lambda s: int((~s).sum())),
        p95=("elapsed", lambda s: np.percentile(s.values, 95) if len(s) else np.nan),
        p99=("elapsed", lambda s: np.percentile(s.values, 99) if len(s) else np.nan),
        avg=("elapsed", "mean")
    ).reset_index()

    g["error_rate"] = np.where(g["samples"] > 0, g["errors"] / g["samples"] * 100.0, 0.0)
    g = g.sort_values(["p95", "p99"], ascending=False).head(n)
    return g


def top_errors(df: pd.DataFrame, n=10) -> pd.DataFrame:
    e = df.loc[~df["success"]].copy()
    if e.empty:
        return pd.DataFrame(columns=["label", "responseCode", "responseMessage", "count"])
    g = e.groupby(["label", "responseCode", "responseMessage"]).size().reset_index(name="count")
    return g.sort_values("count", ascending=False).head(n)


def save_plot_timeline(df: pd.DataFrame, out_png: str):
    if df["ts"].isna().all():
        return False

    # bucket por segundo
    d = df.dropna(subset=["ts"]).copy()
    d["sec"] = d["ts"].dt.floor("S")
    g = d.groupby("sec")["elapsed"].median().reset_index()

    plt.figure()
    plt.plot(g["sec"], g["elapsed"])
    plt.title("Mediana de tiempo de respuesta (ms) por segundo")
    plt.xlabel("Tiempo")
    plt.ylabel("ms")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    return True


def save_plot_hist(df: pd.DataFrame, out_png: str):
    plt.figure()
    plt.hist(df["elapsed"].values, bins=30)
    plt.title("Distribución de tiempos de respuesta (ms)")
    plt.xlabel("ms")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    return True


def save_plot_top_slowest(slowest_df: pd.DataFrame, out_png: str):
    if slowest_df.empty:
        return False
    plt.figure()
    plt.bar(slowest_df["label"].astype(str).values, slowest_df["p95"].values)
    plt.title("Top transacciones más lentas (p95 ms)")
    plt.xlabel("Transacción")
    plt.ylabel("p95 (ms)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    return True


def fmt_ms(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:.0f} ms"


def fmt_num(x, nd=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:.{nd}f}"


def build_pdf(out_pdf: str, kpis: dict, slowest: pd.DataFrame, errors: pd.DataFrame, charts: list[str]):
    doc = SimpleDocTemplate(out_pdf, pagesize=A4,
                            leftMargin=1.6*cm, rightMargin=1.6*cm,
                            topMargin=1.6*cm, bottomMargin=1.6*cm)

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], spaceAfter=8)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], spaceBefore=12, spaceAfter=6)
    normal = styles["BodyText"]

    story = []

    # PORTADA
    story.append(Paragraph("Reporte de Pruebas de Performance", h1))
    story.append(Spacer(1, 6))

    meta = [
        ["Generado", kpis["generated_at"]],
        ["Ventana", f'{kpis["start"]} → {kpis["end"]}'],
        ["Duración", f'{fmt_num(kpis["duration_s"], 2)} s'],
    ]
    t = Table(meta, colWidths=[4*cm, 12*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ("GRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    # KPI “cards”
    story.append(Paragraph("KPIs", h2))
    kpi_table = [
        ["Muestras", str(kpis["samples"]), "Throughput", f'{fmt_num(kpis["throughput_rps"], 2)} rps'],
        ["Errores", str(kpis["errors"]), "Error rate", f'{fmt_num(kpis["error_rate"], 2)} %'],
        ["Promedio", fmt_ms(kpis["avg_ms"]), "p90", fmt_ms(kpis["p90_ms"])],
        ["p95", fmt_ms(kpis["p95_ms"]), "p99", fmt_ms(kpis["p99_ms"])],
        ["Min", fmt_ms(kpis["min_ms"]), "Max", fmt_ms(kpis["max_ms"])],
    ]
    kt = Table(kpi_table, colWidths=[3.4*cm, 4.6*cm, 3.4*cm, 4.6*cm])
    kt.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 10),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(kt)

    # GRÁFICOS
    if charts:
        story.append(Spacer(1, 12))
        story.append(Paragraph("Gráficos", h2))
        for c in charts:
            if os.path.exists(c):
                story.append(Image(c, width=16.5*cm, height=9.0*cm))
                story.append(Spacer(1, 8))

    # TABLAS
    story.append(PageBreak())
    story.append(Paragraph("Top 10 transacciones más lentas (por p95)", h2))

    if not slowest.empty:
        data = [["Transacción", "Muestras", "Errores", "Error rate", "Avg", "p95", "p99"]]
        for _, r in slowest.iterrows():
            data.append([
                str(r["label"]),
                str(int(r["samples"])),
                str(int(r["errors"])),
                f'{r["error_rate"]:.2f}%',
                fmt_ms(r["avg"]),
                fmt_ms(r["p95"]),
                fmt_ms(r["p99"]),
            ])
        tt = Table(data, colWidths=[5.5*cm, 2.2*cm, 2.0*cm, 2.3*cm, 2.0*cm, 2.0*cm, 2.0*cm])
        tt.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
            ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
            ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
            ("FONTSIZE", (0,0), (-1,-1), 9),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ]))
        story.append(tt)
    else:
        story.append(Paragraph("No hay datos para mostrar.", normal))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Top errores", h2))
    if errors.empty:
        story.append(Paragraph("No se registraron errores.", normal))
    else:
        data = [["Transacción", "Código", "Mensaje", "Cantidad"]]
        for _, r in errors.iterrows():
            msg = (r["responseMessage"] or "")[:80]
            data.append([str(r["label"]), str(r["responseCode"]), msg, str(int(r["count"]))])
        et = Table(data, colWidths=[5.0*cm, 2.0*cm, 7.0*cm, 2.0*cm])
        et.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
            ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
            ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
            ("FONTSIZE", (0,0), (-1,-1), 8),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
        ]))
        story.append(et)

    # CONCLUSIONES Y RECOMENDACIONES (AUTO)
    story.append(PageBreak())
    story.append(Paragraph("Conclusiones y Recomendaciones", h2))

    conclusions = []
    if kpis["error_rate"] >= 1.0:
        conclusions.append(f"- Se detecta **{kpis['error_rate']:.2f}%** de errores. Revisar estabilidad y códigos de respuesta.")
    else:
        conclusions.append(f"- Error rate **{kpis['error_rate']:.2f}%** (ok).")

    if not np.isnan(kpis["p95_ms"]) and kpis["p95_ms"] > 800:
        conclusions.append(f"- p95 = **{kpis['p95_ms']:.0f} ms** (alto). Posible cuello de botella en backend o DB.")
    else:
        conclusions.append(f"- p95 = **{kpis['p95_ms']:.0f} ms** (dentro de lo esperado).")

    if not np.isnan(kpis["throughput_rps"]) and kpis["throughput_rps"] < 1:
        conclusions.append(f"- Throughput bajo (**{kpis['throughput_rps']:.2f} rps**). Validar ramp-up, timers y capacidad.")
    else:
        conclusions.append(f"- Throughput = **{kpis['throughput_rps']:.2f} rps**.")

    story.append(Paragraph("<br/>".join(conclusions), normal))
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        "Recomendaciones sugeridas:<br/>"
        "- Definir SLA/SLI (p95, error rate) por endpoint y validar contra baseline.<br/>"
        "- Ejecutar 3 corridas y comparar variación (consistencia).<br/>"
        "- Correlacionar con métricas del servidor (CPU, memoria, DB, GC).<br/>"
        "- Separar escenarios: smoke (rápido), carga (estable), stress (límite).",
        normal
    ))

    doc.build(story)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jtl", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = read_jtl(args.jtl)
    kpis = build_kpis(df)
    slow = top_slowest(df, 10)
    err = top_errors(df, 10)

    # Guardar KPIs en json (útil para Jenkins/IA después)
    kpis_path = os.path.join(args.out, "kpis.json")
    with open(kpis_path, "w", encoding="utf-8") as f:
        json.dump(kpis, f, ensure_ascii=False, indent=2)

    # Charts
    charts = []
    p1 = os.path.join(args.out, "chart_timeline.png")
    if save_plot_timeline(df, p1):
        charts.append(p1)
    p2 = os.path.join(args.out, "chart_hist.png")
    save_plot_hist(df, p2); charts.append(p2)
    p3 = os.path.join(args.out, "chart_top_slowest.png")
    if save_plot_top_slowest(slow, p3):
        charts.append(p3)

    out_html = os.path.join(args.out, "performance_report_custom.html")
    out_pdf  = os.path.join(args.out, "performance_report_custom.pdf")

    # HTML simple (si ya tienes uno mejor, déjalo; aquí solo mantenemos compatibilidad)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(f"""<!doctype html><html><head><meta charset="utf-8"><title>Reporte</title></head>
<body>
<h1>Reporte de Pruebas de Performance</h1>
<p><b>Generado:</b> {kpis["generated_at"]}</p>
<ul>
<li>Muestras: {kpis["samples"]}</li>
<li>Throughput: {kpis["throughput_rps"]:.2f} rps</li>
<li>Error rate: {kpis["error_rate"]:.2f}%</li>
<li>Avg: {kpis["avg_ms"]:.0f} ms</li>
<li>p95: {kpis["p95_ms"]:.0f} ms</li>
<li>p99: {kpis["p99_ms"]:.0f} ms</li>
</ul>
</body></html>""")

    build_pdf(out_pdf, kpis, slow, err, charts)

    print("OK")
    print("HTML:", os.path.abspath(out_html))
    print("PDF :", os.path.abspath(out_pdf))
    print("KPIs:", os.path.abspath(kpis_path))


if __name__ == "__main__":
    main()
