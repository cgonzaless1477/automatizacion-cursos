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


# =========================
# Data
# =========================
def read_jtl(jtl_path: str) -> pd.DataFrame:
    df = pd.read_csv(jtl_path)

    colmap = {c.lower(): c for c in df.columns}

    def pick(name: str):
        return colmap.get(name.lower())

    # timestamp
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
    df["label"] = df[lab_col].astype(str) if lab_col else "ALL"

    # success
    suc_col = pick("success")
    if suc_col:
        df["success"] = df[suc_col].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        df["success"] = True

    # response code/message (opcionales)
    rc_col = pick("responseCode")
    df["responseCode"] = df[rc_col].astype(str) if rc_col else ""

    rm_col = pick("responseMessage")
    df["responseMessage"] = df[rm_col].astype(str) if rm_col else ""

    df = df.dropna(subset=["elapsed"]).copy()
    return df


def pct(series: pd.Series, q: float) -> float:
    if series.empty:
        return float("nan")
    return float(np.percentile(series.values, q))


def build_kpis(df: pd.DataFrame) -> dict:
    start = df["ts"].min()
    end = df["ts"].max()

    if pd.notna(start) and pd.notna(end) and end >= start:
        duration_s = float((end - start).total_seconds())
    else:
        # fallback: si no hay ts, usa cantidad de muestras como referencia (no ideal, pero evita NaN “feos”)
        duration_s = float("nan")

    samples = int(len(df))
    errors = int((~df["success"]).sum())
    error_rate = (errors / samples * 100.0) if samples else 0.0

    avg = float(df["elapsed"].mean()) if samples else float("nan")
    p90 = pct(df["elapsed"], 90)
    p95 = pct(df["elapsed"], 95)
    p99 = pct(df["elapsed"], 99)
    mx = float(df["elapsed"].max()) if samples else float("nan")
    mn = float(df["elapsed"].min()) if samples else float("nan")

    throughput = (
        (samples / duration_s)
        if duration_s and not np.isnan(duration_s) and duration_s > 0
        else float("nan")
    )

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
        avg=("elapsed", "mean"),
    ).reset_index()

    g["error_rate"] = np.where(g["samples"] > 0, g["errors"] / g["samples"] * 100.0, 0.0)
    return g.sort_values(["p95", "p99"], ascending=False).head(n)


def top_errors(df: pd.DataFrame, n=10) -> pd.DataFrame:
    e = df.loc[~df["success"]].copy()
    if e.empty:
        return pd.DataFrame(columns=["label", "responseCode", "responseMessage", "count"])
    g = e.groupby(["label", "responseCode", "responseMessage"]).size().reset_index(name="count")
    return g.sort_values("count", ascending=False).head(n)


# =========================
# Charts (dashboard único)
# =========================
def _prep_charts_style():
    plt.rcParams["figure.autolayout"] = True


def save_dashboard(df: pd.DataFrame, slowest_df: pd.DataFrame, out_png: str) -> bool:
    if df.empty:
        return False

    _prep_charts_style()

    # Si no hay timestamps, el dashboard aún puede mostrar hist + top
    has_ts = not df["ts"].isna().all()
    fig = plt.figure(figsize=(10, 11))

    # 1) timeline mediana
    ax1 = fig.add_subplot(4, 1, 1)
    if has_ts:
        d = df.dropna(subset=["ts"]).copy()
        d["sec"] = d["ts"].dt.floor("s")
        g = d.groupby("sec")["elapsed"].median().reset_index()
        ax1.plot(g["sec"], g["elapsed"])
        ax1.set_title("Mediana de tiempo de respuesta (ms) por segundo")
        ax1.set_ylabel("ms")
        ax1.tick_params(axis="x", rotation=20)
    else:
        ax1.text(0.5, 0.5, "Sin timestamps en JTL (no se puede graficar timeline).",
                 ha="center", va="center")
        ax1.set_axis_off()

    # 2) throughput por segundo
    ax2 = fig.add_subplot(4, 1, 2)
    if has_ts:
        d = df.dropna(subset=["ts"]).copy()
        d["sec"] = d["ts"].dt.floor("s")
        g = d.groupby("sec").size().reset_index(name="rps")
        ax2.plot(g["sec"], g["rps"])
        ax2.set_title("Throughput (requests/seg) por segundo")
        ax2.set_ylabel("rps")
        ax2.tick_params(axis="x", rotation=20)
    else:
        ax2.text(0.5, 0.5, "Sin timestamps en JTL (no se puede graficar throughput).",
                 ha="center", va="center")
        ax2.set_axis_off()

    # 3) histograma
    ax3 = fig.add_subplot(4, 1, 3)
    ax3.hist(df["elapsed"].values, bins=35)
    ax3.set_title("Distribución de tiempos de respuesta (ms)")
    ax3.set_xlabel("ms")
    ax3.set_ylabel("Frecuencia")

    # 4) top p95
    ax4 = fig.add_subplot(4, 1, 4)
    if slowest_df is not None and not slowest_df.empty:
        labels = slowest_df["label"].astype(str).values
        vals = slowest_df["p95"].values
        ax4.bar(labels, vals)
        ax4.set_title("Top transacciones más lentas (p95 ms)")
        ax4.set_ylabel("p95 (ms)")
        ax4.tick_params(axis="x", rotation=20)
    else:
        ax4.text(0.5, 0.5, "No hay datos para Top transacciones.", ha="center", va="center")
        ax4.set_axis_off()

    plt.savefig(out_png, dpi=200)
    plt.close(fig)
    return True


# =========================
# PDF styling helpers
# =========================
ACCENT = colors.HexColor("#0B3CFF")
INK = colors.HexColor("#0B1220")
MUTED = colors.HexColor("#667085")
LINE = colors.HexColor("#E6E8EC")
CARD_BG = colors.HexColor("#F7F8FA")


def fmt_ms(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:.0f} ms"


def fmt_num(x, nd=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:.{nd}f}"


def calc_status(kpis: dict, sla_p95_ms: float, sla_err_pct: float):
    err = float(kpis.get("error_rate", 0.0) or 0.0)
    p95 = kpis.get("p95_ms", float("nan"))

    if (err >= sla_err_pct) or (not np.isnan(p95) and p95 >= sla_p95_ms * 1.5):
        return "FAIL", colors.HexColor("#D32F2F")
    if (err > 0.0) or (not np.isnan(p95) and p95 > sla_p95_ms):
        return "UNSTABLE", colors.HexColor("#F57C00")
    return "PASS", colors.HexColor("#2E7D32")


def _header_footer(canvas, doc, meta_line: str):
    canvas.saveState()
    w, _ = A4

    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.6)
    canvas.line(1.6 * cm, 1.25 * cm, w - 1.6 * cm, 1.25 * cm)

    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(1.6 * cm, 0.75 * cm, meta_line)
    canvas.drawRightString(w - 1.6 * cm, 0.75 * cm, f"Página {doc.page}")

    canvas.restoreState()


def styled_table(data, col_widths, font_size=9, header=True, zebra=True, valign="MIDDLE"):
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.25, LINE),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), valign),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    if zebra and len(data) > (2 if header else 1):
        start = 1 if header else 0
        for r in range(start, len(data)):
            if (r - start) % 2 == 1:
                style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#FAFAFB")))
    t.setStyle(TableStyle(style))
    return t


def kpi_cards(kpis: dict, p_style: ParagraphStyle):
    cards = [
        ("Muestras", str(kpis["samples"])),
        ("Errores", str(kpis["errors"])),
        ("Error rate", f'{fmt_num(kpis["error_rate"], 2)} %'),
        ("Throughput", f'{fmt_num(kpis["throughput_rps"], 2)} rps'),
        ("p95", fmt_ms(kpis["p95_ms"])),
        ("p99", fmt_ms(kpis["p99_ms"])),
    ]

    rows = []
    for i in range(0, len(cards), 2):
        l = cards[i]
        r = cards[i + 1]
        left = Paragraph(f"<b>{l[0]}</b><br/>{l[1]}", p_style)
        right = Paragraph(f"<b>{r[0]}</b><br/>{r[1]}", p_style)
        rows.append([left, right])

    t = Table(rows, colWidths=[8.15 * cm, 8.15 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CARD_BG),
        ("BOX", (0, 0), (-1, -1), 0.8, LINE),
        ("INNERGRID", (0, 0), (-1, -1), 0.8, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    return t


def build_pdf(
    out_pdf: str,
    kpis: dict,
    slowest: pd.DataFrame,
    errors: pd.DataFrame,
    dashboard_png: str | None,
    job_name: str = "N/A",
    build_number: str = "N/A",
    build_url: str = "N/A",
    sla_p95_ms: float = 800.0,
    sla_err_pct: float = 1.0,
):
    meta_line = f"Generado: {kpis.get('generated_at','')} | Job: {job_name} | Build: {build_number}"

    def on_page(canvas, doc):
        _header_footer(canvas, doc, meta_line)

    doc = SimpleDocTemplate(
        out_pdf, pagesize=A4,
        leftMargin=1.6 * cm, rightMargin=1.6 * cm,
        topMargin=1.8 * cm, bottomMargin=1.6 * cm
    )

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], textColor=colors.white, spaceAfter=0)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=INK, spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("body", parent=styles["BodyText"], textColor=INK, leading=13)
    small = ParagraphStyle("small", parent=styles["BodyText"], textColor=MUTED, fontSize=9, leading=11)

    # Para que URLs hagan wrap bonito dentro de celdas
    wrap = ParagraphStyle("wrap", parent=body, wordWrap="CJK")  # CJK wrap funciona bien para strings largos

    story = []

    status_text, status_color = calc_status(kpis, sla_p95_ms, sla_err_pct)

    # ===== Cover band
    cover = Table(
        [[
            Paragraph("<b>Reporte de Pruebas de Performance</b>", h1),
            Paragraph(f"<b>Estado:</b> {status_text}", ParagraphStyle("s", parent=styles["BodyText"], textColor=colors.white))
        ]],
        colWidths=[12.0 * cm, 4.3 * cm]
    )
    cover.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), ACCENT),
        ("BACKGROUND", (1, 0), (1, 0), status_color),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("BOX", (0, 0), (-1, -1), 0, colors.white),
    ]))
    story.append(cover)
    story.append(Spacer(1, 10))

    # ===== Meta info (con wrap)
    meta_data = [
        ["Generado", Paragraph(kpis["generated_at"], wrap)],
        ["Ventana", Paragraph(f'{kpis["start"]} → {kpis["end"]}', wrap)],
        ["Duración", Paragraph(f'{fmt_num(kpis["duration_s"], 2)} s', wrap)],
        ["Job / Build", Paragraph(f"{job_name} / {build_number}", wrap)],
        ["Build URL", Paragraph(build_url, wrap)],
        ["SLA", Paragraph(f"p95 ≤ {sla_p95_ms:.0f} ms | error ≤ {sla_err_pct:.2f}%", wrap)],
    ]
    meta_tbl = styled_table(meta_data, col_widths=[4.0 * cm, 12.3 * cm], font_size=9, header=False, zebra=True)
    story.append(meta_tbl)
    story.append(Spacer(1, 10))

    # ===== KPI cards
    story.append(Paragraph("Resumen Ejecutivo", h2))
    story.append(kpi_cards(kpis, body))
    story.append(Spacer(1, 8))

    extra = [[
        "Avg", fmt_ms(kpis["avg_ms"]),
        "p90", fmt_ms(kpis["p90_ms"]),
        "Min", fmt_ms(kpis["min_ms"]),
        "Max", fmt_ms(kpis["max_ms"]),
    ]]
    extra_tbl = Table(extra, colWidths=[1.2*cm, 2.2*cm, 1.2*cm, 2.2*cm, 1.2*cm, 2.2*cm, 1.2*cm, 2.2*cm])
    extra_tbl.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.25, LINE),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(extra_tbl)

    # ===== Dashboard charts (1 sola imagen => no hay “huecos”)
    if dashboard_png and os.path.exists(dashboard_png):
        story.append(Spacer(1, 12))
        story.append(Paragraph("Gráficos", h2))
        story.append(Paragraph("Vista consolidada para identificar variación temporal, distribución y endpoints críticos.", small))
        story.append(Spacer(1, 6))
        story.append(Image(dashboard_png, width=16.5 * cm, height=18.0 * cm))

    story.append(PageBreak())

    # ===== Slowest table
    story.append(Paragraph("Top 10 transacciones más lentas (por p95)", h2))
    if not slowest.empty:
        data = [["Transacción", "Muestras", "Errores", "Error %", "Avg", "p95", "p99"]]
        for _, r in slowest.iterrows():
            data.append([
                Paragraph(str(r["label"]), wrap),
                str(int(r["samples"])),
                str(int(r["errors"])),
                f'{r["error_rate"]:.2f}%',
                fmt_ms(r["avg"]),
                fmt_ms(r["p95"]),
                fmt_ms(r["p99"]),
            ])
        tbl = styled_table(
            data,
            col_widths=[5.6 * cm, 2.0 * cm, 1.8 * cm, 2.0 * cm, 1.9 * cm, 1.9 * cm, 1.9 * cm],
            font_size=9,
            header=True,
            zebra=True
        )
        story.append(tbl)
    else:
        story.append(Paragraph("No hay datos para mostrar.", body))

    story.append(Spacer(1, 12))

    # ===== Errors table
    story.append(Paragraph("Top errores", h2))
    if errors.empty:
        story.append(Paragraph("No se registraron errores.", body))
    else:
        data = [["Transacción", "Código", "Mensaje", "Cantidad"]]
        for _, r in errors.iterrows():
            msg = (r["responseMessage"] or "")[:150]
            data.append([
                Paragraph(str(r["label"]), wrap),
                str(r["responseCode"]),
                Paragraph(msg, wrap),
                str(int(r["count"])),
            ])
        tbl = styled_table(
            data,
            col_widths=[4.5 * cm, 1.8 * cm, 8.0 * cm, 2.0 * cm],
            font_size=8,
            header=True,
            zebra=True,
            valign="TOP"
        )
        story.append(tbl)

    story.append(PageBreak())

    # ===== Conclusions
    story.append(Paragraph("Conclusiones y Recomendaciones", h2))

    conclusions = []
    if kpis["error_rate"] >= sla_err_pct:
        conclusions.append(
            f"• Error rate <b>{kpis['error_rate']:.2f}%</b> supera SLA (<b>{sla_err_pct:.2f}%</b>). Prioridad: estabilidad/errores."
        )
    else:
        conclusions.append(f"• Error rate <b>{kpis['error_rate']:.2f}%</b> dentro del SLA.")

    if not np.isnan(kpis["p95_ms"]) and kpis["p95_ms"] > sla_p95_ms:
        conclusions.append(
            f"• p95 = <b>{kpis['p95_ms']:.0f} ms</b> supera SLA (<b>{sla_p95_ms:.0f} ms</b>). Revisar cuellos de botella."
        )
    else:
        conclusions.append(f"• p95 = <b>{kpis['p95_ms']:.0f} ms</b> dentro del SLA.")

    if not np.isnan(kpis["throughput_rps"]):
        conclusions.append(f"• Throughput observado: <b>{kpis['throughput_rps']:.2f} rps</b> (interpretar según concurrencia/objetivo).")

    story.append(Paragraph("<br/>".join(conclusions), body))
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        "<b>Acciones recomendadas:</b><br/>"
        "1) Definir SLA por endpoint (p95, error rate) y comparar contra baseline.<br/>"
        "2) Ejecutar 3 corridas (misma carga) y medir variación (consistencia).<br/>"
        "3) Correlacionar con métricas del servidor (CPU/Mem/DB/GC).<br/>"
        "4) Separar escenarios: smoke (rápido), carga (estable), stress (límite).",
        body
    ))

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jtl", required=True)
    ap.add_argument("--out", required=True)

    # Metadata Jenkins (opcionales)
    ap.add_argument("--job", default=os.getenv("JOB_NAME", "N/A"))
    ap.add_argument("--build", default=os.getenv("BUILD_NUMBER", "N/A"))
    ap.add_argument("--url", default=os.getenv("BUILD_URL", "N/A"))

    # SLAs (opcionales)
    ap.add_argument("--sla_p95", type=float, default=800.0)
    ap.add_argument("--sla_err", type=float, default=1.0)

    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = read_jtl(args.jtl)
    kpis = build_kpis(df)
    slow = top_slowest(df, 10)
    err = top_errors(df, 10)

    # Guardar KPIs
    kpis_path = os.path.join(args.out, "kpis.json")
    with open(kpis_path, "w", encoding="utf-8") as f:
        json.dump(kpis, f, ensure_ascii=False, indent=2)

    # Dashboard chart
    dashboard_png = os.path.join(args.out, "chart_dashboard.png")
    if not save_dashboard(df, slow, dashboard_png):
        dashboard_png = None

    out_html = os.path.join(args.out, "performance_report_custom.html")
    out_pdf = os.path.join(args.out, "performance_report_custom.pdf")

    # HTML simple (compat) - mínimo pero ordenado
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Reporte Performance</title>
</head>
<body style="font-family:Arial; max-width:900px; margin:24px auto;">
  <h1>Reporte de Pruebas de Performance</h1>
  <p><b>Generado:</b> {kpis["generated_at"]}</p>
  <p><b>Job:</b> {args.job} &nbsp; <b>Build:</b> {args.build}</p>
  <p><b>Build URL:</b> <a href="{args.url}">{args.url}</a></p>
  <h3>KPIs</h3>
  <ul>
    <li>Muestras: {kpis["samples"]}</li>
    <li>Throughput: {kpis["throughput_rps"]:.2f} rps</li>
    <li>Error rate: {kpis["error_rate"]:.2f}%</li>
    <li>Avg: {kpis["avg_ms"]:.0f} ms</li>
    <li>p95: {kpis["p95_ms"]:.0f} ms</li>
    <li>p99: {kpis["p99_ms"]:.0f} ms</li>
  </ul>
</body>
</html>""")

    build_pdf(
        out_pdf, kpis, slow, err, dashboard_png,
        job_name=args.job,
        build_number=args.build,
        build_url=args.url,
        sla_p95_ms=args.sla_p95,
        sla_err_pct=args.sla_err
    )

    print("OK")
    print("HTML:", os.path.abspath(out_html))
    print("PDF :", os.path.abspath(out_pdf))
    print("KPIs:", os.path.abspath(kpis_path))


if __name__ == "__main__":
    main()
