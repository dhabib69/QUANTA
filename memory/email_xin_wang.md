# Email to Prof. Xin Wang — Nanyang Technological University

**To:** xin.wang@ntu.edu.sg
**Subject:** Prospective Research Student — Taker Flow Microstructure & Gompertz Exit Timing in Crypto Markets
**From:** habib.khairul32@gmail.com
**Attachments:** research_summary_habib_khairul.pdf, wf_sim_report_habib_khairul.pdf

---

Dear Professor Wang,

I am an Indonesian engineer (Associate Degree, Electrical Engineering) writing to inquire about research postgraduate opportunities at Nanyang Business School under your supervision.

Your research on how FinTech innovations alter trading dynamics in secondary markets connects directly to the system I have been building. QUANTA — a crypto momentum trading system — uses the taker buy volume imbalance as a primary entry quality gate, a signal derived from the VPIN framework of Easley, López de Prado and O'Hara. This order flow signal is precisely the kind of microstructure indicator that FinTech-driven participation — algorithmic market makers, retail flow aggregators, and now CBDC settlement mechanisms — is reshaping in real time. Your work on CBDC effects on trading raises a question my system directly touches: whether hazard parameters calibrated from current taker flow dynamics will shift as settlement infrastructure changes.

The exit side of the system extends microstructure theory further. I model the collapse of momentum positions using a Gompertz hazard function: λ(t) = λ₀·e^(γt), calibrated empirically at λ₀ = 0.517 day⁻¹ and γ = 2.92 day⁻¹ across 245 symbols. The optimal exit time t* = ln(n_eff/λ₀)/γ replaces heuristic ATR-based stops with a theoretically grounded timing rule. The universality of these parameters across symbols suggests the hazard dynamics reflect structural market mechanics — an observation I believe has implications for microstructure-informed FinTech design.

The model is validated via strict walk-forward simulation across 300 out-of-sample days: Sharpe 7.27, Profit Factor 4.65, Win Rate 70.0%, Max Drawdown 9.7%, all 10 windows profitable. I am applying for a Research Masters and have attached a 2-page research summary and the full walk-forward simulation report. I would be grateful for any feedback on whether this research direction fits within your programme at NBS.

Yours sincerely,
Habib Khairul
habib.khairul32@gmail.com
Indonesia
