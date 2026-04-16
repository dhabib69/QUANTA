"""
Build research_summary_habib_khairul.pdf from the markdown source.
Run: python build_pdf.py

FIX: All Unicode sub/superscript chars replaced with ReportLab <sub>/<super> tags.
Unicode chars like \u2080 (₀), \u2091 (ₑ), \u207b\u00b9 (⁻¹) are NOT in Times-Roman
and render as black boxes. Use XML tags instead.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)

OUTPUT = r"C:\Users\habib\QUANTA\memory\research_summary_habib_khairul.pdf"

# ── Page setup ────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=A4,
    leftMargin=2.54*cm, rightMargin=2.54*cm,
    topMargin=2.54*cm,  bottomMargin=2.54*cm,
    title="Gompertz Hazard-Based Exit Timing in Cryptocurrency Momentum Strategies",
    author="Habib Khairul",
)

W = A4[0] - 2 * 2.54*cm  # usable text width

# ── Styles ────────────────────────────────────────────────────────────────────
TITLE = ParagraphStyle("TITLE",
    fontName="Times-Bold", fontSize=14, leading=18,
    alignment=TA_CENTER, spaceAfter=4)

SUBTITLE = ParagraphStyle("SUBTITLE",
    fontName="Times-Bold", fontSize=12, leading=15,
    alignment=TA_CENTER, spaceAfter=2)

AUTHOR = ParagraphStyle("AUTHOR",
    fontName="Times-Roman", fontSize=11, leading=14,
    alignment=TA_CENTER, spaceAfter=2)

AFFIL = ParagraphStyle("AFFIL",
    fontName="Times-Italic", fontSize=10, leading=13,
    alignment=TA_CENTER, spaceAfter=12)

H1 = ParagraphStyle("H1",
    fontName="Times-Bold", fontSize=11, leading=14,
    spaceBefore=10, spaceAfter=4)

H2 = ParagraphStyle("H2",
    fontName="Times-Bold", fontSize=11, leading=14,
    spaceBefore=6, spaceAfter=3)

BODY = ParagraphStyle("BODY",
    fontName="Times-Roman", fontSize=10.5, leading=14,
    alignment=TA_JUSTIFY, spaceAfter=6)

EQ = ParagraphStyle("EQ",
    fontName="Times-Italic", fontSize=10.5, leading=16,
    alignment=TA_CENTER, spaceBefore=4, spaceAfter=4)

BULLET = ParagraphStyle("BULLET",
    fontName="Times-Roman", fontSize=10.5, leading=14,
    leftIndent=18, spaceAfter=3)

REFSTYLE = ParagraphStyle("REF",
    fontName="Times-Roman", fontSize=9.5, leading=13,
    leftIndent=18, firstLineIndent=-18, spaceAfter=3)

FOOTNOTE = ParagraphStyle("FOOTNOTE",
    fontName="Times-Italic", fontSize=9, leading=12,
    alignment=TA_CENTER, spaceBefore=6)

def h(text):  return Paragraph(text, H1)
def h2(text): return Paragraph(text, H2)
def p(text):  return Paragraph(text, BODY)
def eq(text): return Paragraph(text, EQ)
def sp(n=6):  return Spacer(1, n)
def rule():   return HRFlowable(width="100%", thickness=0.5,
                                color=colors.black, spaceAfter=4, spaceBefore=4)

# Greek letters as unicode — these ARE in Times-Roman:
# lambda=\u03bb  gamma=\u03b3  rho=\u03c1  minus=\u2212  approx=\u2248
# thin-space=\u202f  times=\u00d7  sqrt=\u221a  implies=\u21d2
# NOTE: do NOT use subscript digits \u2080-\u2089 or \u2091,
#       do NOT use superscript \u207b\u00b9 — use <sub>/<super> tags instead.

lam = '\u03bb'   # λ
gam = '\u03b3'   # γ
rho = '\u03c1'   # ρ
mi  = '\u2212'   # − (minus)
ap  = '\u2248'   # ≈
imp = '\u21d2'   # ⇒
sq  = '\u221a'   # √
x   = '\u00d7'   # ×
ts  = '\u202f'   # thin space

# ── Content ───────────────────────────────────────────────────────────────────
story = []

# ── Title block ───────────────────────────────────────────────────────────────
story += [
    sp(4),
    Paragraph("Research Summary", TITLE),
    Paragraph("Gompertz Hazard-Based Exit Timing in<br/>Cryptocurrency Momentum Strategies", SUBTITLE),
    sp(6),
    Paragraph("Habib Khairul", AUTHOR),
    Paragraph("Independent Researcher | Electrical Engineering (Associate Degree)", AFFIL),
    Paragraph("Indonesia &nbsp;|&nbsp; habib.khairul32@gmail.com", AFFIL),
    rule(),
    sp(4),
]

# ── Section 1 ─────────────────────────────────────────────────────────────────
story += [
    h("1.  Research Problem"),
    p("Systematic trading strategies in cryptocurrency markets face a fundamental asymmetry: entry "
      "signals derived from machine learning models have been studied extensively, yet exit timing "
      "remains largely ad hoc. Most practitioner implementations rely on fixed ATR (Average True Range) "
      "multiples or hardcoded time windows \u2014 parameters calibrated once and never adapted to the specific "
      "velocity of an ongoing price move."),
    p("This rigidity creates a measurable inefficiency. A coin experiencing a high-velocity breakout "
      "(e.g., 150% daily growth rate) and one experiencing a low-velocity drift (e.g., 20% daily growth "
      "rate) are exited at the same price level despite fundamentally different collapse dynamics. The "
      "result is systematic early exit on strong moves and over-holding on weak ones."),
    p("<b>Research question:</b> Can the optimal exit time for a momentum position be derived analytically "
      "from the observed pump velocity of the underlying asset, rather than set as a fixed parameter?"),
]

# ── Section 2 ─────────────────────────────────────────────────────────────────
story += [
    h("2.  Theoretical Framework"),
    h2("2.1  Khairul's Identity \u2014 Portfolio Compound Growth"),
    p("The master growth equation of the QUANTA system is derived from first principles by combining "
      "Kelly criterion, empirical win-rate estimation, and pyramid position sizing:"),

    # C(T) = C_0 * e^(nT)
    eq("C(T)  =  C<sub>0</sub> \u00b7 e<super>nT</super>"),

    # n = lambda * [...]  — no sub/super needed here
    eq(f"n  =  {lam} \u00b7 [ P \u00b7 ln(1 + fb) + (1{mi}P) \u00b7 ln(1{mi}f) ]"),

    p(f"Where {lam} is signal frequency (trades/day), P is win probability derived from CatBoost model AUC, "
      f"f is the risk fraction per trade (continuous in model score), and "
      f"b{ts}={ts}PF{ts}{x}{ts}(1{mi}P)/P "
      f"is the win/loss ratio. This identity is over-determined by five independent constraints (Kelly, "
      f"Grinold\u2019s Fundamental Law, MAE exit geometry, pump phase dynamics, and pyramid consistency), "
      f"meaning the calibrated n value is self-consistent rather than curve-fitted."),

    h2("2.2  The Companion Equation \u2014 Pump Collapse Hazard"),
    p(f"Individual cryptocurrency price moves during momentum phases follow "
      f"P(t){ts}={ts}P<sub>0</sub>{ts}\u00b7{ts}e<super>n<sub>eff</sub>\u00b7t</super>, "
      f"where n<sub>eff</sub> is the observed pump velocity. The probability of pump continuation "
      f"declines over time as market participants take profit and liquidity exhausts. "
      f"This collapse probability is modelled as a Gompertz hazard function:"),

    # lambda(t) = lambda_0 * e^(gamma*t)
    eq(f"{lam}(t)  =  {lam}<sub>0</sub> \u00b7 e<super>{gam}t</super>"),

    p("The expected exit value peaks at t* where growth equals collapse hazard:"),

    # n_eff = lambda(t*)  =>  t* = ln(n_eff / lambda_0) / gamma
    eq(f"n<sub>eff</sub>  =  {lam}(t*)  {imp}  "
       f"t*  =  ln(n<sub>eff</sub> / {lam}<sub>0</sub>) / {gam}"),

    eq(f"Optimal exit ATR  =  ( e<super>n<sub>eff</sub> \u00b7 t*</super> {mi} 1 ) / ATR%"),

    h2("2.3  Empirical Calibration"),
    p(f"The Gompertz parameters are calibrated from Maximum Adverse Excursion (MAE) statistics "
      f"across 245 cryptocurrency symbols using two anchor points:"),

    Paragraph(
        f"\u2022  <b>Anchor 1:</b> t<sub>1</sub>{ts}={ts}0.104 days (avg bank hit, 4.20 ATR): "
        f"{lam}(t<sub>1</sub>){ts}={ts}n<sub>eff</sub>{ts}={ts}0.700 day<super>-1</super>",
        BULLET),

    Paragraph(
        f"\u2022  <b>Anchor 2:</b> t<sub>2</sub>{ts}={ts}0.226 days (avg runner exit, 6.09 ATR): "
        f"{lam}(t<sub>2</sub>){ts}={ts}1.0 day<super>-1</super>",
        BULLET),

    eq(f"{gam}  =  ln({lam}(t<sub>2</sub>) / {lam}(t<sub>1</sub>)) / "
       f"(t<sub>2</sub> {mi} t<sub>1</sub>)  =  2.92 day<super>-1</super>"),

    eq(f"{lam}<sub>0</sub>  =  0.517 day<super>-1</super>"),

    p(f"The hazard doubles every ln(2)/{gam}{ts}={ts}5.7 hours. At average pump velocity the model "
      f"recovers the empirically optimal exit (4.20 ATR), confirming internal consistency."),
]

# ── Section 3 ─────────────────────────────────────────────────────────────────
story += [
    h("3.  Methodology"),
    h2("3.1  Signal Generation"),
    p("Entry signals are generated by a CUSUM sequential change-point detector applied to 5-minute "
      "OHLCV data. Signals are filtered through a CatBoost classifier (102 features, impulse breakout "
      "domain) trained on a Triple Barrier labelling scheme (L\u00f3pez de Prado, 2018). Quality gates: "
      "CatBoost score \u2265\u202f68, taker buy flow imbalance \u2265\u202f40%, pre-impulse "
      "R\u00b2\u202f\u2264\u202f0.70 (exhaustion veto)."),

    h2("3.2  Walk-Forward Validation"),
    p("Strict Pardo (2008) walk-forward structure: Train\u202f[60d] \u2192 Purge\u202f[48 bars] "
      "\u2192 Test\u202f[30d] \u2192 Step\u202f[30d], repeated for 10 non-overlapping out-of-sample "
      "windows. Purge gap follows L\u00f3pez de Prado (2018) AFML Ch.\u202f7. Zero lookahead enforced "
      "at bar level."),

    h2("3.3  Position Sizing & Pyramid"),
    p(f"Risk is continuous in model score: f(s){ts}={ts}0.005{ts}+{ts}(s{ts}{mi}{ts}68)/32{ts}"
      f"{x}{ts}0.025, mapping [68,{ts}100] to [0.5%,{ts}3.0%]. Size is ATR-normalised: "
      f"N{ts}={ts}(C{ts}\u00b7{ts}f(s)){ts}/{ts}(3.0{ts}{x}{ts}ATR), equivalent to "
      f"implicit volatility targeting. A three-layer pyramid adds positions at +0.5{ts}ATR (L2) and "
      f"recovers at the L2 stop level with a 3.77{ts}ATR target (L3 \u2014 empirical p50 MAE runup)."),
]

# ── Section 4 — Results table ─────────────────────────────────────────────────
story += [
    h("4.  Results"),
    p("All results are out-of-sample: 300 days, 10 walk-forward windows, 245 Binance symbols."),
    sp(4),
]

tdata = [
    ["Metric", "V12.3  Fixed Exits", "V12.4  Gompertz Dynamic"],
    ["Total Return",   "+19,883%",   "+150,235%"],
    ["Win Rate",       "70.7%",      "70.0%"],
    ["Profit Factor",  "4.37",       "4.65"],
    ["Sharpe Ratio",   "7.15",       "7.27"],
    ["Sortino Ratio",  "\u2014",     "19.25"],
    ["Calmar Ratio",   "\u2014",     "15,497"],
    ["Max Drawdown",   "8.1%",       "9.7%"],
    ["n daily",        "0.01764",    "0.02438"],
]

col_w = [W * 0.40, W * 0.28, W * 0.32]
tbl = Table(tdata, colWidths=col_w)
tbl.setStyle(TableStyle([
    ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#1a1a2e")),
    ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
    ("FONTNAME",      (0, 0), (-1, 0),  "Times-Bold"),
    ("FONTSIZE",      (0, 0), (-1, 0),  10),
    ("FONTNAME",      (0, 1), (-1, -1), "Times-Roman"),
    ("FONTSIZE",      (0, 1), (-1, -1), 10),
    ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#f5f5f5"), colors.white]),
    ("FONTNAME",      (2, 1), (2, 1),   "Times-Bold"),   # +150,235%
    ("FONTNAME",      (2, 4), (2, 4),   "Times-Bold"),   # Sharpe 7.27
    ("FONTNAME",      (2, 8), (2, 8),   "Times-Bold"),   # n_daily
    ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
    ("ALIGN",         (0, 0), (0, -1),  "LEFT"),
    ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
    ("TOPPADDING",    (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
    ("LINEBELOW",     (0, 0), (-1, 0),  1.0, colors.HexColor("#1a1a2e")),
]))
story.append(tbl)
story.append(sp(8))

story += [
    p(f"The Gompertz upgrade increased n<sub>daily</sub> by 38% (0.01764 \u2192 0.02438), "
      f"attributable to fast pumps being held to their natural {lam}{ts}={ts}n crossover "
      f"rather than a fixed ATR ceiling. Pyramid layers (L2{ts}+{ts}L3) contributed "
      f"$6.12M of the $15M total PnL \u2014 41% of total returns from averaging-in alone."),

    p(f"<b>Theoretical consistency checks:</b> "
      f"Kelly f*{ts}={ts}0.545; operating at f/f*{ts}={ts}2.75% \u2192 theoretical "
      f"MaxDD{ts}\u2248{ts}11%; observed 9.7%{ts}\u2713. "
      f"Grinold IR{ts}={ts}IC{ts}\u00b7{ts}{sq}BR{ts}={ts}12.2; "
      f"observed Sharpe 7.27 \u2192 {rho}<sub>eff</sub>{ts}={ts}0.355 "
      f"(35.5% idiosyncratic edge after cross-coin correlation discount){ts}\u2713."),
]

# ── Section 5 ─────────────────────────────────────────────────────────────────
story += [
    h("5.  Proposed Research Directions"),
    Paragraph(
        f"<b>Q1 \u2014 Statistical Validation of the Gompertz Calibration.</b>  "
        f"Are {lam}<sub>0</sub>{ts}={ts}0.517 and {gam}{ts}={ts}2.92 statistically "
        f"stable across market regimes and exchanges? Requires AIC/BIC model selection "
        f"vs. exponential and Weibull alternatives, bootstrap confidence intervals, and "
        f"out-of-sample validation on non-Binance data.", BULLET),
    sp(3),
    Paragraph(
        f"<b>Q2 \u2014 Universality of the Pump Velocity Constant.</b>  "
        f"The empirical n<sub>pump</sub>{ts}{ap}{ts}0.082{ts}day<super>-1</super> "
        f"is consistent across WIF, BONK, COAI and other altcoin momentum cycles. "
        f"Is this a universal constant driven by social contagion mechanics (Sornette, 2003), "
        f"or does it vary with market capitalisation and liquidity?", BULLET),
    sp(3),
    Paragraph(
        f"<b>Q3 \u2014 Cross-Market Generalisation.</b>  "
        f"Does the Gompertz hazard model generalise to small-cap equities, commodities, or "
        f"emerging market currencies? This would establish whether pump collapse dynamics "
        f"reflect a universal speculative bubble mechanism.", BULLET),
]

# ── Section 6 ─────────────────────────────────────────────────────────────────
story += [
    h("6.  Conclusion"),
    p("This work presents an empirically calibrated, analytically derived framework for optimal "
      "exit timing in cryptocurrency momentum strategies. The central contribution \u2014 applying the "
      "Gompertz hazard function to model pump collapse probability and deriving a dynamic exit target "
      "as the growth-hazard crossover \u2014 does not appear in existing literature and is validated by "
      "a Sharpe ratio of 7.27 across 300 out-of-sample days."),
    p("The framework is self-consistent with Kelly criterion, Grinold\u2019s Fundamental Law, and "
      "empirical MAE statistics simultaneously, suggesting it captures genuine market structure "
      "rather than data artefacts. I seek supervised academic investigation to formalise the "
      "statistical tests, extend the empirical coverage, and explore cross-market generalisability."),
    rule(),
]

# ── References ────────────────────────────────────────────────────────────────
story.append(h("References"))
refs = [
    "L\u00f3pez de Prado, M. (2018). <i>Advances in Financial Machine Learning</i>. Wiley.",
    "Pardo, R. (2008). <i>The Evaluation and Optimization of Trading Strategies</i>. Wiley.",
    "Sornette, D. (2003). <i>Why Stock Markets Crash: Critical Events in Complex Financial Systems</i>. Princeton University Press.",
    "Grinold, R. & Kahn, R. (2000). <i>Active Portfolio Management</i>. McGraw-Hill.",
    "Kelly, J.L. (1956). A new interpretation of information rate. <i>Bell System Technical Journal</i>, 35(4), 917\u2013926.",
    "Easley, D., L\u00f3pez de Prado, M. & O\u2019Hara, M. (2012). Flow toxicity and liquidity in a high-frequency world. <i>Review of Financial Studies</i>, 25(5), 1457\u20131493.",
    "Kou, S.G. (2002). A jump-diffusion model for option pricing. <i>Management Science</i>, 48(8), 1086\u20131101.",
    "Gompertz, B. (1825). On the nature of the function expressive of the law of human mortality. <i>Philosophical Transactions of the Royal Society</i>, 115, 513\u2013583.",
]
for i, r in enumerate(refs, 1):
    story.append(Paragraph(f"[{i}]\u2002{r}", REFSTYLE))

story += [
    sp(8),
    rule(),
    Paragraph("<i>This research was conducted independently using live market data from Binance "
              "via public API. All simulation results are out-of-sample. Code available upon request.</i>",
              FOOTNOTE),
]

# ── Build ─────────────────────────────────────────────────────────────────────
doc.build(story)
print(f"PDF saved to: {OUTPUT}")
