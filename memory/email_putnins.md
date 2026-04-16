# Email to Prof. Talis Putnins — UTS

**To:** Talis.Putnins@uts.edu.au
**Subject:** Prospective Research Student — Gompertz Exit Timing Model for Crypto Pump Cycles
**From:** habib.khairul32@gmail.com
**Attachments:** research_summary_habib_khairul.pdf

---

Dear Professor Putnins,

I am an Indonesian engineer (Associate Degree, Electrical Engineering) writing to inquire about research postgraduate opportunities at UTS under your supervision.

Your work on pump-and-dump dynamics in cryptocurrency markets — particularly the price distortion mechanics documented in "A New Wolf in Town" — directly motivated my own research. While your framework identifies and characterises the manipulation event, I have been developing a complementary model that quantifies the optimal exit timing during the pump phase using a Gompertz hazard function:

    λ(t) = λ₀ · e^(γt)

The model is calibrated empirically from Maximum Adverse Excursion statistics across 245 Binance symbols, yielding λ₀ = 0.517 day⁻¹ and γ = 2.92 day⁻¹. The optimal exit is derived as the crossover point where observed pump velocity equals collapse hazard: t* = ln(n/λ₀)/γ. This replaces ad hoc fixed exit targets with a per-position dynamic target driven by observed price action.

The framework is validated via a strict walk-forward out-of-sample simulation (Pardo 2008): 300 OOS days, 10 non-overlapping windows, zero lookahead, real CatBoost inference. Key results: Sharpe ratio 7.27, Profit Factor 4.65, Max Drawdown 9.7%.

I am applying for a Research Masters and believe this work could be formalised under supervision into a publishable study — specifically addressing the statistical robustness of the Gompertz calibration and testing whether the pump collapse dynamics you documented follow predictable hazard acceleration. I have also noted the Sydney Market Microstructure and Digital Finance Conference and would welcome any opportunity to present this work.

I have attached a 2-page research summary. I would be grateful for any guidance on whether this direction aligns with your current research agenda.

Yours sincerely,
Habib Khairul
habib.khairul32@gmail.com
Indonesia
