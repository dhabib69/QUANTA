"""
Build wf_sim_report_habib_khairul.pdf — Walk-Forward Simulation Results
Professional PDF companion to research_summary_habib_khairul.pdf
Run: python build_wf_report_pdf.py
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)

OUTPUT = r"C:\Users\habib\QUANTA\memory\wf_sim_report_habib_khairul.pdf"

doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=A4,
    leftMargin=2.54*cm, rightMargin=2.54*cm,
    topMargin=2.54*cm,  bottomMargin=2.54*cm,
    title="QUANTA v12.4 Walk-Forward Simulation Report",
    author="Habib Khairul",
)

W = A4[0] - 2 * 2.54*cm

# ── Styles ────────────────────────────────────────────────────────────────────
TITLE = ParagraphStyle("TITLE",
    fontName="Times-Bold", fontSize=14, leading=18,
    alignment=TA_CENTER, spaceAfter=3)

SUBTITLE = ParagraphStyle("SUBTITLE",
    fontName="Times-Roman", fontSize=11, leading=14,
    alignment=TA_CENTER, spaceAfter=2)

AUTHOR = ParagraphStyle("AUTHOR",
    fontName="Times-Italic", fontSize=10, leading=13,
    alignment=TA_CENTER, spaceAfter=10)

H1 = ParagraphStyle("H1",
    fontName="Times-Bold", fontSize=11, leading=14,
    spaceBefore=10, spaceAfter=5)

BODY = ParagraphStyle("BODY",
    fontName="Times-Roman", fontSize=10.5, leading=14,
    alignment=TA_JUSTIFY, spaceAfter=5)

CAPTION = ParagraphStyle("CAPTION",
    fontName="Times-Italic", fontSize=9.5, leading=12,
    alignment=TA_CENTER, spaceBefore=3, spaceAfter=8)

FOOTNOTE = ParagraphStyle("FOOTNOTE",
    fontName="Times-Italic", fontSize=9, leading=12,
    alignment=TA_CENTER, spaceBefore=6)

SETUP = ParagraphStyle("SETUP",
    fontName="Times-Roman", fontSize=10, leading=13,
    leftIndent=12, spaceAfter=2)

NAVY  = colors.HexColor("#1a1a2e")
DGREY = colors.HexColor("#cccccc")
ALT1  = colors.HexColor("#f0f4ff")
ALT2  = colors.white
GREEN = colors.HexColor("#d4edda")
BOLD_GREEN = colors.HexColor("#155724")

def h(t):  return Paragraph(t, H1)
def p(t):  return Paragraph(t, BODY)
def sp(n=6): return Spacer(1, n)
def rule(): return HRFlowable(width="100%", thickness=0.5,
                               color=colors.black, spaceAfter=4, spaceBefore=4)

def make_table(data, col_fracs, header_bg=NAVY, alt=True, bold_last_col=False):
    col_w = [W * f for f in col_fracs]
    t = Table(data, colWidths=col_w)
    style = [
        ("BACKGROUND",    (0, 0), (-1, 0),  header_bg),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Times-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  9.5),
        ("FONTNAME",      (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE",      (0, 1), (-1, -1), 9.5),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("ALIGN",         (0, 0), (0, -1),  "LEFT"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 7),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 7),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID",          (0, 0), (-1, -1), 0.4, DGREY),
        ("LINEBELOW",     (0, 0), (-1, 0),  1.0, header_bg),
    ]
    if alt:
        style.append(("ROWBACKGROUNDS", (0, 1), (-1, -1), [ALT1, ALT2]))
    if bold_last_col:
        for r in range(1, len(data)):
            style.append(("FONTNAME", (-1, r), (-1, r), "Times-Bold"))
    t.setStyle(TableStyle(style))
    return t

# ── Story ─────────────────────────────────────────────────────────────────────
story = []

# Header
story += [
    sp(4),
    Paragraph("QUANTA v12.4 — Walk-Forward Simulation Report", TITLE),
    Paragraph("Gompertz Dynamic Exit Timing | Out-of-Sample Validation", SUBTITLE),
    sp(4),
    Paragraph("Habib Khairul &nbsp;|&nbsp; habib.khairul32@gmail.com &nbsp;|&nbsp; Indonesia", AUTHOR),
    rule(),
    sp(4),
]

# ── Simulation Setup ──────────────────────────────────────────────────────────
story.append(h("1.  Simulation Setup"))

setup_data = [
    ["Parameter", "Value"],
    ["Run ID",                "20260415_205834"],
    ["Initial Capital",       "$10,000"],
    ["OOS Windows",           "10 \u00d7 30 days  (train 60d, purge 48 bars, step 30d)"],
    ["Total OOS Period",      "300 days"],
    ["Symbols Traded",        "137 (from 245 Binance symbols)"],
    ["Score Floor",           "68.0  (CatBoost classifier)"],
    ["Entry Gates",           "Wave strength \u2265 40%,  pre-impulse R^2 \u2264 0.70"],
    ["Exit Model",            "Gompertz dynamic: bank 4.2 ATR @ 35%  +  trail 2.0 ATR"],
    ["Stop Loss",             "3.0 ATR"],
    ["Risk Scaling",          "0.5% (score=68) \u2192 3.0% (score=100), continuous"],
    ["Commission / Slippage", "4 bps / 2 bps"],
    ["Lookahead",             "Zero — strict bar-level enforcement"],
    ["Validation Method",     "Pardo (2008) walk-forward, L\u00f3pez de Prado (2018) purge gap"],
]
story.append(make_table(setup_data, [0.42, 0.58]))
story.append(sp(10))

# ── Portfolio Result ──────────────────────────────────────────────────────────
story.append(h("2.  Portfolio Result"))

# Big headline numbers as a wide table
headline = [
    ["Final Capital", "Total Return", "Max Drawdown", "Trades Executed"],
    ["$15,033,500", "+150,235%", "9.69%", "466"],
]
hl_t = Table(headline, colWidths=[W*0.25]*4)
hl_t.setStyle(TableStyle([
    ("BACKGROUND",    (0, 0), (-1, 0), NAVY),
    ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
    ("FONTNAME",      (0, 0), (-1, 0), "Times-Bold"),
    ("FONTSIZE",      (0, 0), (-1, 0), 9.5),
    ("BACKGROUND",    (0, 1), (-1, 1), colors.HexColor("#e8f4e8")),
    ("FONTNAME",      (0, 1), (-1, 1), "Times-Bold"),
    ("FONTSIZE",      (0, 1), (-1, 1), 13),
    ("TEXTCOLOR",     (0, 1), (0, 1),  colors.HexColor("#1a5c1a")),
    ("TEXTCOLOR",     (1, 1), (1, 1),  colors.HexColor("#1a5c1a")),
    ("TEXTCOLOR",     (2, 1), (2, 1),  colors.HexColor("#8b0000")),
    ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
    ("TOPPADDING",    (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ("GRID",          (0, 0), (-1, -1), 0.5, DGREY),
    ("LINEBELOW",     (0, 0), (-1, 0),  1.0, NAVY),
]))
story.append(hl_t)
story.append(sp(10))

# ── Performance Metrics ───────────────────────────────────────────────────────
story.append(h("3.  Risk-Adjusted Performance Metrics"))

metrics = [
    ["Metric", "Value", "Benchmark / Context"],
    ["Win Rate",       "69.96%",  "Target \u2265 60% — exceeded"],
    ["Profit Factor",  "4.648",   "PF > 2.0 = excellent; 4.65 = institutional-grade"],
    ["Sharpe Ratio",   "7.275",   "Top hedge funds: 1\u20133; 7.27 = exceptional"],
    ["Sortino Ratio",  "19.246",  "Downside-only volatility — 19.2 = very low tail risk"],
    ["Calmar Ratio",   "15,497",  "Annual return / MaxDD — extremely capital-efficient"],
    ["Max Drawdown",   "9.69%",   "Kelly theoretical \u2248 11% at 2.75% f/f* \u2713"],
    ["Gross Wins",     "$19.14M", ""],
    ["Gross Losses",   "$4.12M",  ""],
    ["Avg Hold Time",  "19 bars", "\u224895 minutes at 5-min bars"],
]
story.append(make_table(metrics, [0.30, 0.22, 0.48]))
story.append(sp(10))

# ── Window Stability ──────────────────────────────────────────────────────────
story.append(h("4.  Window-by-Window Stability  (10/10 Profitable)"))

windows = [
    ["Window", "Days", "Trades", "Win Rate", "Net PnL", "Sharpe"],
    ["1",  "1\u201330",    "55", "63.6%", "+$14,744",      "6.21"],
    ["2",  "31\u201360",   "56", "67.9%", "+$20,143",      "5.49"],
    ["3",  "61\u201390",   "68", "77.9%", "+$126,544",     "9.66"],
    ["4",  "91\u2013120",  "69", "68.1%", "+$421,795",     "7.04"],
    ["5",  "121\u2013150", "50", "68.0%", "+$1,222,784",   "7.97"],
    ["6",  "151\u2013180", "41", "97.6%", "+$1,889,960",   "10.98"],
    ["7",  "181\u2013210", "39", "48.7%", "+$572,065",     "2.95"],
    ["8",  "211\u2013240", "34", "64.7%", "+$2,494,773",   "8.61"],
    ["9",  "241\u2013270", "35", "68.6%", "+$3,615,816",   "8.70"],
    ["10", "271\u2013300", "19", "73.7%", "+$4,644,877",   "10.29"],
    ["Mean \u00b1 SD", "",   "47", "69.9%", "",             "7.79 \u00b1 2.31"],
]
wt = Table(windows, colWidths=[W*f for f in [0.10, 0.14, 0.12, 0.14, 0.26, 0.14, 0.10]])
wt.setStyle(TableStyle([
    ("BACKGROUND",    (0, 0), (-1, 0),  NAVY),
    ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
    ("FONTNAME",      (0, 0), (-1, 0),  "Times-Bold"),
    ("FONTSIZE",      (0, 0), (-1, 0),  9.5),
    ("FONTNAME",      (0, 1), (-1, -2), "Times-Roman"),
    ("FONTSIZE",      (0, 1), (-1, -1), 9.5),
    ("FONTNAME",      (0, -1),(-1, -1), "Times-Bold"),   # mean row bold
    ("BACKGROUND",    (0, -1),(-1, -1), colors.HexColor("#e8e8e8")),
    ("ROWBACKGROUNDS",(0, 1), (-1, -2), [ALT1, ALT2]),
    ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
    ("ALIGN",         (0, 0), (0, -1),  "LEFT"),
    ("LEFTPADDING",   (0, 0), (-1, -1), 7),
    ("RIGHTPADDING",  (0, 0), (-1, -1), 7),
    ("TOPPADDING",    (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ("GRID",          (0, 0), (-1, -1), 0.4, DGREY),
    ("LINEBELOW",     (0, 0), (-1, 0),  1.0, NAVY),
]))
story.append(wt)
story.append(Paragraph(
    "Note: Window 7 shows lowest win rate (48.7%) yet remains profitable (+$572K, Sharpe 2.95), "
    "demonstrating robustness during adverse market regimes.",
    CAPTION))

# ── Score Distribution ────────────────────────────────────────────────────────
story.append(h("5.  Signal Quality — Score Distribution"))

scores = [
    ["Score Bucket", "Trades", "Win Rate", "Net PnL", "Avg PnL / Trade"],
    ["68 \u2013 72",  "178", "65.7%", "+$2,959,464", "+$16,626"],
    ["72 \u2013 76",  "131", "71.0%", "+$3,296,352", "+$25,163"],
    ["76 \u2013 80",   "86", "69.8%", "+$3,382,091", "+$39,327"],
    ["80 \u2013 85",   "53", "79.2%", "+$3,091,628", "+$58,333"],
    ["85 \u2013 90",   "18", "77.8%", "+$2,293,966", "+$127,443"],
]
story.append(make_table(scores, [0.20, 0.14, 0.16, 0.24, 0.26]))
story.append(Paragraph(
    "Higher CatBoost scores monotonically improve avg PnL/trade: 7.7\u00d7 increase from "
    "lowest to highest bucket. Score mean = 74.5; all 466 trades cleared the 68.0 floor.",
    CAPTION))

# ── Exit Analysis ─────────────────────────────────────────────────────────────
story.append(h("6.  Exit Analysis"))

exits = [
    ["Exit Type", "Count", "% of Trades", "Outcome"],
    ["Take Profit (Chandelier/Runner)", "265", "56.9%", "Bank hit \u2192 trail activated \u2192 runner exit"],
    ["Stop Loss",                       " 91", "19.5%", "3.0 ATR adverse excursion"],
    ["Timeout",                         "110", "23.6%", "Gompertz dynamic max holding period"],
]
story.append(make_table(exits, [0.38, 0.12, 0.16, 0.34]))
story.append(Paragraph(
    "Timeout exits replaced hardcoded 4-day (1152 bar) limit with Gompertz-derived dynamic "
    "holding periods. Fast-pumping coins are held to their natural \u03bb = n crossover; "
    "slow-drifting coins exit earlier. This contributed the 38% improvement in n<sub>daily</sub>.",
    CAPTION))

# ── Tier Breakdown ────────────────────────────────────────────────────────────
story.append(h("7.  Tier Breakdown"))

tiers = [
    ["Tier", "Trades", "Win Rate", "Net PnL", "Description"],
    ["A", "413", "69.2%", "+$14,369,425", "High-confidence signals (primary)"],
    ["B",  "43", "74.4%",   "+$636,246", "Moderate confidence"],
    ["C",  "10", "80.0%",    "+$17,830", "Lower frequency, high precision"],
]
story.append(make_table(tiers, [0.08, 0.12, 0.14, 0.24, 0.42]))
story.append(sp(8))

# ── Theoretical Consistency ───────────────────────────────────────────────────
story.append(h("8.  Theoretical Consistency Checks"))
story += [
    p("<b>Kelly Criterion:</b>  Optimal fraction f* = 0.545. System operates at "
      "f/f* = 2.75% of Kelly. Kelly-predicted MaxDD at this fraction \u2248 11%; "
      "observed MaxDD = 9.69%.  \u2713"),
    p("<b>Grinold\u2019s Fundamental Law:</b>  IC \u00d7 \u221aBR = 0.616 \u00d7 \u221a393 = 12.2 "
      "theoretical Information Ratio. Observed Sharpe = 7.27, implying "
      "\u03c1<sub>eff</sub> = (7.27/12.2)<super>2</super> = 0.355 \u2014 35.5% idiosyncratic edge after "
      "64.5% lost to cross-coin correlation.  \u2713"),
    p("<b>Pump Phase Law:</b>  System growth rate n<sub>QUANTA</sub> = 0.02438 day<super>-1</super> "
      "< n<sub>pump</sub> = 0.082 day<super>-1</super> (macro), confirming the bot captures a "
      "fraction of the pump move rather than the full phase.  \u2713"),
    p("<b>Pyramid Contribution:</b>  L2 + L3 layers generated $6.12M of the $15.03M total PnL "
      "(40.7%). Averaging-in on confirmed moves is responsible for nearly half of all returns."),
    rule(),
]

# ── Footer ────────────────────────────────────────────────────────────────────
story.append(Paragraph(
    "<i>All results are strictly out-of-sample. Zero lookahead enforced at bar level. "
    "Data source: Binance public API. Simulation engine: QUANTA_WalkForward_Sim.py v12.4. "
    "Run ID: 20260415_205834. Code available upon request.</i>",
    FOOTNOTE))

# ── Build ─────────────────────────────────────────────────────────────────────
doc.build(story)
print(f"PDF saved to: {OUTPUT}")
