# QUANTA Memory Index

- [QUANTA Project Overview](project_quanta_overview.md) - Architecture, 278-dim feature vector layout, agent list, core design
- [QUANTA Key Files Map](project_quanta_files.md) - What each file does, which are highest-risk to edit
- [GPT QUANTA Workflow Memory](gpt_quanta_workflow_memory.md) - Practical operating memory for future GPT fixes: live path ownership, Nike ownership map, BS/Kou wiring, benchmark rules, and debugging order
- [QUANTA Completed Work](project_quanta_completed_work.md) - All phases + BS integration + BS execution power + HMM overhaul + dashboard + GitHub + Optuna NaN fix + Tier 3 (Kou jump-diffusion, regime routing automation, unified HMM)
- [QUANTA Technical Decisions](project_quanta_technical_decisions.md) - Critical constants, HMM sort-order convention, NaN gotcha, thread safety rules
- [Conditional Kou First-Passage Note](kou_first_passage_conditional_note.md) - Finite-horizon TP-before-SL math for Nike-style post-jump entries; includes trigger-jump conditioning, one-sided Kou-Wang approximation, and PIDE solver
- [Nike Cache Performance Report](../NIKE_CACHE_PERFORMANCE_REPORT.md) - Full-cache benchmark over 230 local 5m feather files; separates anomaly-event recall from realized Nike TP/SL outcomes
- [Nike V2 Cache Performance Report](../NIKE_V2_CACHE_PERFORMANCE_REPORT.md) - Tiered Nike v2 benchmark with same/+1/+2 recall, per-tier trade stats, rollout gates, and the conservative live decision after full-cache tuning
- [Habib User Profile](user_habib.md) - UTC+8, MX130 GPU, strong dev, wants concise execution
- [Working Feedback](feedback_quanta.md) - Be concise, no preamble, read before editing, continue = resume immediately
- [Academic References](reference_journals.md) - All papers/books used in QUANTA: Lopez de Prado, Lim TFT, Kyle, Amihud, VPIN, PPO, HMM, MF-DFA, Conformal
- [Future: C/C++ Refactoring Plan](project_quanta_future_cpp_plan.md) - Blueprint for porting the Python core engine to C/C++ for ultra-low latency
