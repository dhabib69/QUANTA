# Email to Prof. Frank de Jong — Tilburg University

**To:** F.deJong@tilburguniversity.edu
**Subject:** Prospective Research Student — Applying Microstructure Theory to Crypto Exit Timing via Gompertz Hazard
**From:** habib.khairul32@gmail.com
**Attachments:** research_summary_habib_khairul.pdf, wf_sim_report_habib_khairul.pdf

---

Dear Professor de Jong,

I am an Indonesian engineer (Associate Degree, Electrical Engineering) writing to inquire about research postgraduate opportunities at the Tilburg Department of Finance under your supervision.

Your co-authored textbook *The Microstructure of Financial Markets* (Cambridge University Press, 2009) has been foundational to the theoretical grounding of my independent research. The treatment of order flow, transaction costs, and liquidity dynamics in that work forms the direct conceptual base from which I have built QUANTA — a crypto momentum trading system. The entry gate uses the taker buy volume imbalance, a signal that operationalises the order flow asymmetry framework that your textbook helped define as a core pillar of microstructure theory.

The part of the work I believe warrants academic formalisation is the exit model. I model the collapse of momentum positions using a Gompertz hazard function — λ(t) = λ₀·e^(γt) — calibrated empirically at λ₀ = 0.517 day⁻¹ and γ = 2.92 day⁻¹ across 245 symbols. The optimal exit time t* = ln(n_eff/λ₀)/γ is derived from a compound growth identity linking Kelly fraction, win rate, and trade frequency. What I find striking is the universality of these parameters across symbols: the accelerating hazard appears to reflect a structural property of crypto market liquidity rather than anything coin-specific. This is, I believe, an extension of microstructure theory to exit timing — modelling the exhaustion of liquidity as a hazard process — and I am not aware of a formal academic treatment of this connection.

The system is validated via strict walk-forward simulation across 300 out-of-sample days: Sharpe 7.27, Profit Factor 4.65, Win Rate 70.0%, Max Drawdown 9.7%, all 10 windows profitable. I am applying for a Research Masters and have attached both a 2-page research summary and the full walk-forward report. I would be very grateful for any feedback on whether this direction fits within your current programme or that of your colleagues at Tilburg.

Yours sincerely,
Habib Khairul
habib.khairul32@gmail.com
Indonesia
