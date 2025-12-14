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
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
)

import matplotlib.pyplot as plt


# ============================================================
# Helpers (robustos)
# ============================================================
def safe_float(x):
    try:
        if x is None:
            return float("nan")
        v = float(x)
        return v
    except Exception:
        return float("nan")

def is_nan(x):
    return isinstance(x, float) and np.isnan(x)

def fmt_num(x, nd=2, suffix=""):
    x = safe_float(x)
    if is_nan(x):
        return "-"
    return f"{x:.{nd}f}{suffix}"

def fmt_ms(x):
    x = safe_float(x)
    if is_nan(x):
        return "-"
    return f"{x:.0f} ms"

def clamp_str(s: str, max_len: int = 110) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\n", " ").replace("\r", " ")
    return (s[: max_len - 1] + "…") if len(s) > max_len else s

def pct(series: pd.Series, q: float) -> float:
    if series is None or series.empty:
        return float("nan")
    arr = pd.to_numeric(series, errors="coerce").dropna().values
    if len(arr) == 0:
        return float("nan")
    return float(np.percentile(arr, q))


# ============================================================
# Data
# ============================================================
def read_jtl(jtl_path: str) -> pd.DataFrame:
    """
    Lee JTL CSV de JMeter (tolerante a variaciones de columnas).
    """
    df = pd.read_csv(jtl_path)

    # Map case-insensitive
    colmap = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            c = colmap.get(n.lower())
            if c:
                return c
        return None

    # timestamp
    ts_col = pick("timeStamp", "timestamp", "time")
    if ts_col:
        df["ts"] = pd.to_datetime(df[ts_col], unit="ms", errors="coerce")
    else:
        df["ts"] = pd.NaT

    # elapsed
    el_col = pick("elapsed")
    if not el_col:
        raise ValueError("No encuentro columna 'elapsed' en el JTL.")
    df["elapsed"] = pd.to_numeric(df[el_col], errors="coerce")

    # label
    lab_col = pick("label", "samplerlabel", "request")
    df["label"] = df[lab_col].astype(str) if lab_col else "ALL"

    # success
    suc_col = pick("success")
    if suc_col:
        df["success"] = df[suc_col].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        df["success"] = True

    # response code/message
    rc_col = pick("responseCode", "responsecode")
    rm_col = pick("responseMessage", "responsemessage")
    df["responseCode"] = df[rc_col].astype(str) if rc_col else ""
    df["responseMessage"] = df[rm_col].astype(str) if rm_col else ""

    # opcionales útiles
    lat_col = pick("Latency", "latency")
    con_col = pick("Connect", "connect")
    df["latency"] = pd.to_numeric(df[lat_col], errors="coerce") if lat_col else np.nan
    df["connect"] = pd.to_numeric(df[con_col], errors="coerce") if con_col else np.nan

    df = df.dropna(subset=["elapsed"])
    return df


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

    throughput = (samples / duration_s) if (not is_nan(duration_s) and duration_s > 0) else float("nan")

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
    def _p(series, q):
        return pct(series, q)

    g = df.groupby("label").agg(
        samples=("elapsed", "size"),
        errors=("success", lambda s: int((~s).sum())),
        avg=("elapsed", "mean"),
        p95=("elapsed", lambda s: _p(s, 95)),
        p99=("elapsed", lambda s: _p(s, 99)),
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


# ============================================================
# Charts (estilo consistente)
# ============================================================
def _prep_charts_style():
    plt.rcParams.update({
        "figure.autolayout": True,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

def save_plot_timeline_median(df: pd.DataFrame, out_png: str) -> bool:
    if df["ts"].isna().all():
        return False

    d = df.dropna(subset=["ts"]).copy()
    d["sec"] = d["ts"].dt.floor("s")
    g = d.groupby("sec")["elapsed"].median().reset_index()

    _prep_charts_style()
    plt.figure(figsize=(10, 3.3))
    plt.plot(g["sec"], g["elapsed"])
    plt.title("Mediana de tiempo de respuesta (ms) por segundo")
    plt.xlabel("Tiempo")
    plt.ylabel("ms")
    plt.xticks(rotation=20, ha="right")
    plt.savefig(out_png, dpi=180)
    plt.close()
    return True

def save_plot_throughput(df: pd.DataFrame, out_png: str) -> bool:
    if df["ts"].isna().all():
        return False

    d = df.dropna(subset=["ts"]).copy()
    d["sec"] = d["ts"].dt.floor("s")
    g = d.groupby("sec").size().reset_index(name="rps")

    _prep_charts_style()
    plt.figure(figsize=(10, 3.3))
    plt.plot(g["sec"], g["rps"])
    plt.title("Throughput (requests/seg) por segundo")
    plt.xlabel("Tiempo")
    plt.ylabel("rps")
    plt.xticks(rotation=20, ha="right")
    plt.savefig(out_png, dpi=180)
    plt.close()
    return True

def save_plot_hist(df: pd.DataFrame, out_png: str) -> bool:
    _prep_charts_style()
    plt.figure(figsize=(10, 3.3))
    plt.hist(df["elapsed"].values, bins=35)
    plt.title("Distribución de tiempos de respuesta (ms)")
    plt.xlabel("ms")
    plt.ylabel("Frecuencia")
    plt.savefig(out_png, dpi=180)
    plt.close()
    return True

def save_plot_top_slowest(slowest_df: pd.DataFrame, out_png: str) -> bool:
    if slowest_df.empty:
        return False

    _prep_charts_style()
    plt.figure(figsize=(10, 3.3))
    plt.bar(slowest_df["label"].astype(str).values, slowest_df["p95"].values)
    plt.title("Top transacciones más lentas (p95 ms)")
    plt.xlabel("Transacción")
    plt.ylabel("p95 (ms)")
    plt.xticks(rotation=20, ha="right")
    plt.savefig(out_png, dpi=180)
    plt.close()
    return True


# ============================================================
# PDF styling
# ============================================================
ACCENT = colors.HexColor("#0B3CFF")
INK = colors.HexColor("#0B1220")
MUTED = colors.HexColor("#667085")
LINE = colors.HexColor("#E6E8EC")
CARD_BG = colors.HexColor("#F7F8FA")


def calc_status(kpis: dict, sla_p95_ms: float, sla_err_pct: float) -> tuple[str, colors.Color]:
    """
    - FAIL: error_rate >= sla_err_pct  OR p95 >= sla_p95_ms * 1.5
    - UNSTABLE: error_rate > 0 OR p95 > sla_p95_ms
    - PASS: ok
    """
    err = safe_float(kpis.get("error_rate", 0.0))
    p95 = safe_float(kpis.get("p95_ms", float("nan")))

    if (not is_nan(err) and err >= sla_err_pct) or (not is_nan(p95) and p95 >= sla_p95_ms * 1.5):
        return "FAIL", colors.HexColor("#D32F2F")
    if (not is_nan(err) and err > 0.0) or (not is_nan(p95) and p95 > sla_p95_ms):
        return "UNSTABLE", colors.HexColor("#F57C00")
    return "PASS", colors.HexColor("#2E7D32")


def _header_footer(canvas, doc, meta_line: str):
    canvas.saveState()
    w, _ = A4

    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.6)
    canvas.line(1.6*cm, 1.25*cm, w-1.6*cm, 1.25*cm)

    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(1.6*cm, 0.75*cm, meta_line)
    canvas.drawRightString(w-1.6*cm, 0.75*cm, f"Página {doc.page}")

    canvas.restoreState()


def kpi_cards(kpis: dict):
    cards = [
        ("Muestras", str(kpis["samples"])),
        ("Errores", str(kpis["errors"])),
        ("Error rate", f'{fmt_num(kpis["error_rate"], 2, " %")}'),
        ("Throughput", f'{fmt_num(kpis["throughput_rps"], 2, " rps")}'),
        ("p95", fmt_ms(kpis["p95_ms"])),
        ("p99", fmt_ms(kpis["p99_ms"])),
    ]

    rows = []
    for i in range(0, len(cards), 2):
        left = cards[i]
        right = cards[i+1]
        rows.append([
            Paragraph(f"<b>{left[0]}</b><br/>{left[1]}", None),
            Paragraph(f"<b>{right[0]}</b><br/>{right[1]}", None),
        ])

    t = Table(rows, colWidths=[8.15*cm, 8.15*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), CARD_BG),
        ("BOX", (0,0), (-1,-1), 0.8, LINE),
        ("INNERGRID", (0,0), (-1,-1), 0.8, LINE),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 12),
        ("TEXTCOLOR", (0,0), (-1,-1), INK),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 12),
        ("RIGHTPADDING", (0,0), (-1,-1), 12),
        ("TOPPADDING", (0,0), (-1,-1), 12),
        ("BOTTOMPADDING", (0,0), (-1,-1), 12),
    ]))
    return t


def styled_table(data, col_widths, font_size=9, header_bg=colors.whitesmoke, zebra=True, valign="MIDDLE"):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("GRID", (0,0), (-1,-1), 0.25, LINE),
        ("BACKGROUND", (0,0), (-1,0), header_bg),
        ("TEXTCOLOR", (0,0), (-1,0), INK),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), font_size),
        ("VALIGN", (0,0), (-1,-1), valign),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]
    if zebra and len(data) > 2:
        for r in range(1, len(data)):
            if r % 2 == 0:
                style.append(("BACKGROUND", (0,r), (-1,r), colors.HexColor("#FAFAFB")))
    t.setStyle(TableStyle(style))
    return t


def build_pdf(
    out_pdf: str,
    kpis: dict,
    slowest: pd.DataFrame,
    errors: pd.DataFrame,
    charts: list[str],
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
        leftMargin=1.6*cm, rightMargin=1.6*cm,
        topMargin=1.8*cm, bottomMargin=1.6*cm
    )

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], textColor=INK, spaceAfter=8)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=INK, spaceBefore=10, spaceAfter=6)
    p = ParagraphStyle("p", parent=styles["BodyText"], textColor=INK, leading=13)
    small = ParagraphStyle("small", parent=styles["BodyText"], textColor=MUTED, fontSize=9, leading=11)

    story = []

    # Header band + status
    status_text, status_color = calc_status(kpis, sla_p95_ms, sla_err_pct)

    band = Table(
        [[
            Paragraph("<b>Reporte de Pruebas de Performance</b>", ParagraphStyle("t", parent=h1, textColor=colors.white)),
            Paragraph(f"<b>Estado:</b> {status_text}", ParagraphStyle("s", parent=p, textColor=colors.white))
        ]],
        colWidths=[12.0*cm, 4.3*cm]
    )
    band.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,0), ACCENT),
        ("BACKGROUND", (1,0), (1,0), status_color),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 10),
        ("RIGHTPADDING", (0,0), (-1,-1), 10),
        ("TOPPADDING", (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ("BOX", (0,0), (-1,-1), 0, colors.white),
    ]))
    story.append(band)
    story.append(Spacer(1, 10))

    # Meta
    meta_data = [
        ["Generado", kpis["generated_at"]],
        ["Ventana", f'{kpis["start"]} → {kpis["end"]}'],
        ["Duración", f'{fmt_num(kpis["duration_s"], 2)} s'],
        ["Job / Build", f"{job_name} / {build_number}"],
        ["Build URL", build_url],
        ["SLA", f"p95 ≤ {sla_p95_ms:.0f} ms | error ≤ {sla_err_pct:.2f}%"],
    ]
    story.append(styled_table(meta_data, col_widths=[4.0*cm, 12.3*cm], font_size=9, zebra=False, valign="MIDDLE"))
    story.append(Spacer(1, 10))

    # KPI cards
    story.append(Paragraph("Resumen Ejecutivo", h2))
    story.append(kpi_cards(kpis))
    story.append(Spacer(1, 8))

    extra = [[
        "Avg", fmt_ms(kpis["avg_ms"]),
        "p90", fmt_ms(kpis["p90_ms"]),
        "Min", fmt_ms(kpis["min_ms"]),
        "Max", fmt_ms(kpis["max_ms"]),
    ]]
    extra_tbl = Table(extra, colWidths=[1.4*cm, 2.5*cm, 1.4*cm, 2.5*cm, 1.4*cm, 2.5*cm, 1.4*cm, 2.5*cm])
    extra_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.white),
        ("GRID", (0,0), (-1,-1), 0.25, LINE),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("TEXTCOLOR", (0,0), (-1,-1), INK),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(extra_tbl)

    # Charts (2 por bloque)
    usable = [c for c in charts if c and os.path.exists(c)]
    if usable:
        story.append(Spacer(1, 12))
        story.append(Paragraph("Gráficos", h2))
        story.append(Paragraph("Ayudan a ver variación temporal, distribución y transacciones críticas.", small))
        story.append(Spacer(1, 6))

        for i in range(0, len(usable), 2):
            block = []
            c1 = usable[i]
            c2 = usable[i+1] if i+1 < len(usable) else None

            block.append(Image(c1, width=16.5*cm, height=6.0*cm))
            block.append(Spacer(1, 6))
            if c2:
                block.append(Image(c2, width=16.5*cm, height=6.0*cm))
            story.append(KeepTogether(block))
            story.append(Spacer(1, 10))

    story.append(PageBreak())

    # Slowest table
    story.append(Paragraph("Top 10 transacciones más lentas (por p95)", h2))
    if not slowest.empty:
        data = [["Transacción", "Muestras", "Errores", "Error %", "Avg", "p95", "p99"]]
        for _, r in slowest.iterrows():
            data.append([
                clamp_str(r["label"], 40),
                str(int(r["samples"])),
                str(int(r["errors"])),
                f'{safe_float(r["error_rate"]):.2f}%',
                fmt_ms(r["avg"]),
                fmt_ms(r["p95"]),
                fmt_ms(r["p99"]),
            ])
        story.append(styled_table(
            data,
            col_widths=[5.6*cm, 2.0*cm, 1.8*cm, 2.0*cm, 1.9*cm, 1.9*cm, 1.9*cm],
            font_size=9,
            zebra=True,
            valign="MIDDLE"
        ))
    else:
        story.append(Paragraph("No hay datos para mostrar.", p))

    story.append(Spacer(1, 12))

    # Errors table
    story.append(Paragraph("Top errores", h2))
    if errors.empty:
        story.append(Paragraph("No se registraron errores.", p))
    else:
        data = [["Transacción", "Código", "Mensaje", "Cantidad"]]
        for _, r in errors.iterrows():
            data.append([
                clamp_str(r["label"], 35),
                clamp_str(r["responseCode"], 10),
                clamp_str(r["responseMessage"], 100),
                str(int(r["count"])),
            ])
        story.append(styled_table(
            data,
            col_widths=[5.0*cm, 2.0*cm, 7.3*cm, 2.0*cm],
            font_size=8,
            zebra=True,
            valign="TOP"
        ))

    story.append(PageBreak())

    # Conclusions
    story.append(Paragraph("Conclusiones y Recomendaciones", h2))

    conclusions = []
    if safe_float(kpis["error_rate"]) >= sla_err_pct:
        conclusions.append(f"• Error rate <b>{safe_float(kpis['error_rate']):.2f}%</b> supera SLA (<b>{sla_err_pct:.2f}%</b>). Prioridad: estabilidad.")
    else:
        conclusions.append(f"• Error rate <b>{safe_float(kpis['error_rate']):.2f}%</b> dentro del SLA.")

    p95v = safe_float(kpis["p95_ms"])
    if (not is_nan(p95v)) and p95v > sla_p95_ms:
        conclusions.append(f"• p95 = <b>{p95v:.0f} ms</b> supera SLA (<b>{sla_p95_ms:.0f} ms</b>). Revisar cuellos de botella.")
    else:
        conclusions.append(f"• p95 = <b>{fmt_ms(kpis['p95_ms'])}</b> dentro del SLA.")

    thr = safe_float(kpis["throughput_rps"])
    if not is_nan(thr):
        conclusions.append(f"• Throughput observado: <b>{thr:.2f} rps</b> (interpretar según concurrencia/objetivo).")

    story.append(Paragraph("<br/>".join(conclusions), p))
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        "<b>Acciones recomendadas:</b><br/>"
        "1) Definir SLA por endpoint (p95, error rate) y comparar contra baseline.<br/>"
        "2) Ejecutar 3 corridas y medir variación (consistencia).<br/>"
        "3) Correlacionar con métricas del servidor (CPU/Mem/DB/GC).<br/>"
        "4) Separar escenarios: smoke, carga y stress.",
        p
    ))

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)


# ============================================================
# Main
# ============================================================
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

    # Charts
    charts = []
    p1 = os.path.join(args.out, "chart_timeline_median.png")
    if save_plot_timeline_median(df, p1):
        charts.append(p1)

    p2 = os.path.join(args.out, "chart_throughput.png")
    if save_plot_throughput(df, p2):
        charts.append(p2)

    p3 = os.path.join(args.out, "chart_hist.png")
    if save_plot_hist(df, p3):
        charts.append(p3)

    p4 = os.path.join(args.out, "chart_top_slowest.png")
    if save_plot_top_slowest(slow, p4):
        charts.append(p4)

    out_html = os.path.join(args.out, "performance_report_custom.html")
    out_pdf  = os.path.join(args.out, "performance_report_custom.pdf")

    # HTML simple pero bonito (compat)
    status_text, status_color = calc_status(kpis, args.sla_p95, args.sla_err)
    badge = {
        "PASS": "#2E7D32",
        "UNSTABLE": "#F57C00",
        "FAIL": "#D32F2F",
    }.get(status_text, "#667085")

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Reporte Performance</title>
</head>
<body style="font-family:Arial; max-width:980px; margin:24px auto; color:#0B1220">
  <div style="display:flex; justify-content:space-between; align-items:center; padding:12px 14px; border:1px solid #E6E8EC; border-radius:10px;">
    <div>
      <h2 style="margin:0;">Reporte de Pruebas de Performance</h2>
      <div style="color:#667085; font-size:13px;">Generado: {kpis["generated_at"]} · Job: {args.job} · Build: {args.build}</div>
      <div style="color:#667085; font-size:13px;">SLA: p95 ≤ {args.sla_p95:.0f} ms · error ≤ {args.sla_err:.2f}%</div>
      <div style="color:#667085; font-size:13px;">Build URL: <a href="{args.url}">{args.url}</a></div>
    </div>
    <div style="padding:8px 12px; border-radius:999px; background:{badge}; color:white; font-weight:bold;">
      {status_text}
    </div>
  </div>

  <h3 style="margin-top:18px;">KPIs</h3>
  <ul>
    <li>Muestras: {kpis["samples"]}</li>
    <li>Errores: {kpis["errors"]} ({kpis["error_rate"]:.2f}%)</li>
    <li>Throughput: {fmt_num(kpis["throughput_rps"], 2)} rps</li>
    <li>Avg: {fmt_ms(kpis["avg_ms"])}</li>
    <li>p95: {fmt_ms(kpis["p95_ms"])}</li>
    <li>p99: {fmt_ms(kpis["p99_ms"])}</li>
  </ul>

  <p style="color:#667085;">Tip: el PDF completo está en artefactos del build (target/performance_report_custom.pdf).</p>
</body>
</html>""")

    build_pdf(
        out_pdf, kpis, slow, err, charts,
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
