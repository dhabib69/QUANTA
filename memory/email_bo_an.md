# Email to Prof. Bo An — Nanyang Technological University

**To:** boan@ntu.edu.sg
**Subject:** Prospective Research Student — Analytically Derived Exit Timing as Reward Signal for RL Trading Agents
**From:** habib.khairul32@gmail.com
**Attachments:** research_summary_habib_khairul.pdf, wf_sim_report_habib_khairul.pdf

---

Dear Professor An,

I am an Indonesian engineer (Associate Degree, Electrical Engineering) writing to inquire about research postgraduate opportunities at the College of Computing and Data Science under your supervision.

Your TradeMaster platform, presented at NeurIPS, represents one of the most rigorous open frameworks for evaluating RL-based trading strategies. I have been working on a complementary approach — QUANTA — that uses supervised ML (CatBoost with Triple Barrier labelling following López de Prado 2018) for entry classification, paired with an analytically derived exit timing model rather than a learned policy. The two approaches point toward a natural research synthesis: the exit model I have derived could serve as a mathematically grounded reward shaping signal for RL agents within a framework like TradeMaster.

The exit model is a Gompertz hazard function calibrated from empirical crypto price data: λ(t) = λ₀·e^(γt), with λ₀ = 0.517 day⁻¹ and γ = 2.92 day⁻¹. The optimal exit time t* = ln(n_eff/λ₀)/γ is derived analytically from a compound growth identity I term Khairul's Identity — C(T) = C₀·e^(nT), where n encodes the Kelly-weighted trade frequency. Unlike heuristic ATR stops, t* provides a principled intermediate reward signal: an RL agent that exits near t* is structurally rewarded for avoiding the accelerating collapse hazard rather than simply chasing terminal PnL. I believe this could address the sparse reward and delayed feedback problems that make training RL trading agents difficult in practice.

The system is validated via strict walk-forward out-of-sample simulation across 300 days: Sharpe 7.27, Profit Factor 4.65, Win Rate 70.0%, Max Drawdown 9.7%, all 10 windows profitable. I am applying for a Research Masters and have attached a 2-page research summary and the full simulation report. I would welcome any feedback on whether this direction fits within your current research programme.

Yours sincerely,
Habib Khairul
habib.khairul32@gmail.com
Indonesia
