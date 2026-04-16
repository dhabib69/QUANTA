import os
from dataclasses import dataclass, field
from pathlib import Path

# Centralized API Configuration
try:
    from apis.quanta_api import (
        QUANTA_PROXY_PORT,
        AI_BASE_URL,
        AI_API_KEY,
        AI_MODEL_NAME
    )
except ImportError:
    QUANTA_PROXY_PORT = os.getenv("QUANTA_PROXY_PORT", "52681")
    AI_BASE_URL = os.getenv("AI_BASE_URL", "https://api.openai.com/v1")
    AI_API_KEY = os.getenv("AI_API_KEY", "")
    AI_MODEL_NAME = os.getenv("AI_MODEL_NAME", "claude-3-5-sonnet-20241022")

@dataclass
class NetworkConfig:
    """Network-level configurations for API routing and proxies."""
    proxy_port: str = QUANTA_PROXY_PORT  # Centralized in apis/quanta_api.py
    binance_weight_limit_1m: int = 1200
    binance_weight_soft_limit: float = 0.90
    max_retries: int = 3
    retry_delay_base: float = 2.0

    def __post_init__(self):
        assert self.binance_weight_limit_1m > 0, "binance_weight_limit_1m must be > 0"
        assert 0.0 < self.binance_weight_soft_limit <= 1.0, "binance_weight_soft_limit must be in (0, 1]"
        assert self.max_retries >= 0, "max_retries must be >= 0"
        assert self.retry_delay_base > 0, "retry_delay_base must be > 0"

    @property
    def proxy_url(self) -> str:
        return f"http://127.0.0.1:{self.proxy_port}" if self.proxy_port else ""

@dataclass
class MarketConfig:
    """Market scanning and threshold variables."""
    historical_days: int = 180
    scan_interval: int = 90
    stats_interval: int = 100
    whale_threshold: float = 500000.0

    def __post_init__(self):
        assert self.historical_days > 0, "historical_days must be > 0"
        assert self.scan_interval > 0, "scan_interval must be > 0"
        assert self.stats_interval > 0, "stats_interval must be > 0"
        assert self.whale_threshold > 0, "whale_threshold must be > 0"

@dataclass
class IndicatorConfig:
    """Technical Indicator calculation periods."""
    rsi_period: int = 7
    macd_fast: int = 8
    macd_slow: int = 17
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    adx_period: int = 14
    stoch_period: int = 14
    ma_short: int = 20
    ma_long: int = 50

    def __post_init__(self):
        assert self.rsi_period > 1, "rsi_period must be > 1"
        assert self.macd_fast > 1, "macd_fast must be > 1"
        assert self.macd_slow > self.macd_fast, "macd_slow must be > macd_fast"
        assert self.macd_signal > 0, "macd_signal must be > 0"
        assert self.bb_period > 1, "bb_period must be > 1"
        assert self.bb_std > 0, "bb_std must be > 0"
        assert self.atr_period > 1, "atr_period must be > 1"
        assert self.adx_period > 1, "adx_period must be > 1"
        assert self.ma_short > 0, "ma_short must be > 0"
        assert self.ma_long > self.ma_short, "ma_long must be > ma_short"

@dataclass
class ModelConfig:
    """Hyperparameters and Architecture definitions for ML."""
    use_gpu: bool = True
    direction_threshold: float = 0.08
    cat_ensemble_weight: float = 1.0
    tft_ensemble_weight: float = 0.15  # TFT v2 enabled: temporal seq complements CatBoost snapshots

    # Feature Dimensions
    spike_dump_feature_count: int = 20
    timeframe_count: int = 7  # v11.5b: 7 TFs (5m, 15m, 1h, 4h, 6h, 12h, 1d)
    base_feature_count: int = 278  # AUTHORITATIVE COUNT — 275 base + 3 BS barrier features (2026-04-02)
    
    # TFT Architecture
    tft_hidden_size: int = 64
    tft_num_heads: int = 4
    tft_dropout: float = 0.1
    tft_seq_length: int = 5
    tft_train_epochs: int = 10
    
    # Specialist Weights
    weight_foundation: float = 0.5
    weight_hunter: float = 0.3
    weight_anchor: float = 0.2

    def __post_init__(self):
        assert 0.0 < self.direction_threshold < 1.0, "direction_threshold must be in (0, 1)"
        assert self.base_feature_count > 0, "base_feature_count must be > 0"
        assert self.spike_dump_feature_count > 0, "spike_dump_feature_count must be > 0"
        assert self.timeframe_count > 0, "timeframe_count must be > 0"
        assert self.tft_hidden_size > 0, "tft_hidden_size must be > 0"
        assert self.tft_num_heads > 0, "tft_num_heads must be > 0"
        assert 0.0 <= self.tft_dropout < 1.0, "tft_dropout must be in [0, 1)"
        assert self.tft_seq_length > 0, "tft_seq_length must be > 0"
        assert self.tft_train_epochs > 0, "tft_train_epochs must be > 0"
        assert self.weight_foundation > 0, "weight_foundation must be > 0"
        assert self.weight_hunter > 0, "weight_hunter must be > 0"
        assert self.weight_anchor > 0, "weight_anchor must be > 0"

@dataclass
class RLConfig:
    """Reinforcement Learning / PPO Agent constraints."""
    min_confidence_rl: float = 60.0
    min_confidence_alert: float = 70.0
    rl_retrain_threshold: int = 500
    rl_outcome_window: int = 3600
    rl_add_cooldown: int = 1800
    catastrophic_forgetting_buffer_ratio: float = 0.8
    memory_max_size: int = 10000
    
    # PPO Heimdall is dormant in V12 Thor Architecture
    rl_enabled: bool = False
    
    # Network Arch
    hidden_dim: int = 256
    lr: float = 2.5e-5
    epochs: int = 4
    batch_size: int = 1024
    clip: float = 0.08
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.015
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Differential Sharpe
    dsr_adaptation_rate: float = 0.02
    dsr_eps: float = 1e-8

    def __post_init__(self):
        assert 0.0 < self.min_confidence_rl <= 100.0, "min_confidence_rl must be in (0, 100]"
        assert 0.0 < self.min_confidence_alert <= 100.0, "min_confidence_alert must be in (0, 100]"
        assert self.min_confidence_alert >= self.min_confidence_rl, \
            "min_confidence_alert must be >= min_confidence_rl"
        assert self.rl_retrain_threshold > 0, "rl_retrain_threshold must be > 0"
        assert self.memory_max_size > 0, "memory_max_size must be > 0"
        assert self.batch_size > 0, "batch_size must be > 0"
        assert 0.0 < self.gamma <= 1.0, "gamma must be in (0, 1]"
        assert 0.0 < self.gae_lambda <= 1.0, "gae_lambda must be in (0, 1]"
        assert self.entropy_coef >= 0, "entropy_coef must be >= 0"
        assert self.dsr_eps > 0, "dsr_eps must be > 0"

@dataclass
class StrategyConfig:
    """Triple Barrier strategy coefficients."""
    max_streak: int = 20
    max_magnitude: float = 10.0
    cusum_threshold: float = 0.01
    
    # Stop Loss & Take Profit ATR Ratios
    sl_ratio: float = 1.5
    tp1_ratio: float = 1.5
    tp2_ratio: float = 3.0
    tp3_ratio: float = 4.5
    
    # Reward Weighting
    tp1_weight: float = 1.0
    tp2_weight: float = 2.5
    tp3_weight: float = 5.0
    sl_weight: float = 0.5

    def __post_init__(self):
        assert self.sl_ratio > 0, "sl_ratio must be > 0"
        assert self.tp1_ratio > 0, "tp1_ratio must be > 0"
        assert self.tp2_ratio > self.tp1_ratio, "tp2_ratio must be > tp1_ratio"
        assert self.tp3_ratio > self.tp2_ratio, "tp3_ratio must be > tp2_ratio"
        assert self.tp1_weight > 0, "tp1_weight must be > 0"
        assert self.tp2_weight > 0, "tp2_weight must be > 0"
        assert self.tp3_weight > 0, "tp3_weight must be > 0"
        assert self.sl_weight > 0, "sl_weight must be > 0"

@dataclass
class EventExtractionConfig:
    """
    Event extraction parameters for Numba-optimized specialist extractors.
    All parameters are research-backed with academic citations.
    """
    # ── Shared Parameters ──
    # MIN_GAP: Minimum candle gap between events to prevent autocorrelation.
    # López de Prado (2018) AFML Ch.4 §4.1: "Sampling must ensure statistical
    # independence. For 5m bars, 12 bars (1 hour) decorrelates serial momentum."
    min_gap: int = 12

    # LOOKBACK_WINDOW: Window for structural pattern detection (50 bars = ~4 hours).
    # Cont (2001) "Empirical properties of asset returns": autocorrelation in
    # crypto returns decays to noise after ~50 lags at 5m frequency.
    lookback_window: int = 50

    # ── Triple Barrier Method (López de Prado 2018 Ch.3 §3.4) ──
    # TP_WEIGHT / SL_WEIGHT: Asymmetric weighting for class imbalance.
    # Lin et al. (2017) "Focal Loss": penalize false negatives more heavily.
    tp_weight: float = 1.0
    sl_weight: float = 1.5       # SL samples weighted 1.5× to combat imbalanced label skew

    # ── Feature-aware Norse simulation controls ──
    pump_material_drawdown_atr: float = 1.0
    stop_market_penetration_factor: float = 0.20
    stop_market_slip_atr_cap: float = 0.25
    norse_drawdown_penalty_weight: float = 1.5

    # ── Compound Growth Engine (Asymmetric Target Mode - 2026-04-12) ──
    # Re-engineered: Target exactly 10% equity gain on confirmed pump wins,
    # hard-capped at 2-3% equity loss on stops. Compounding happens naturally
    # because 10% of a glowing equity base is a larger notional size.
    compound_mode: str = "asymmetric_target"  # "flat" | "geometric" | "asymmetric_target"
    compound_target_win_pct: float = 10.0     # Target: 10% equity gain per win
    compound_max_loss_pct: float = 3.0        # Hard cap: 3% equity loss per trade
    compound_activation_score: float = 70.0   # Unleashed: Optuna proved 70+ scores yield a 75.7% win rate.
    compound_pump_tp_atr_min: float = 5.0     # Minimum ATR for TP (to reach 10% win properly)
    compound_pump_sl_atr_max: float = 1.5     # Tighter SL to keep losses <= 3%

    # -- Liquidation Cascade Targeting (2026-04-12) --
    # Identifies short-squeeze liquidation clusters above entry to extend TP targets.
    # Derived from open interest accumulation + funding rate extremes (Makarov & Schoar 2020).
    liq_enabled: bool = True
    liq_oi_surge_threshold: float = 0.12     # OI increase >12% in 4h = dense cluster
    liq_funding_extreme: float = -0.025      # Funding < -0.025% = heavy short buildup
    liq_cascade_size_boost: float = 1.25     # Position boost when cascade target exists
    liq_max_target_dist_pct: float = 15.0   # Don't target clusters >15% away
    liq_min_oi_vol_ratio: float = 1.5        # OI/volume ratio for meaningful cluster

    # ── Thor Signal Detection Parameters (V12 — single active specialist) ──
    thor_body_min: float = 0.55              # Min body efficiency (body/range)
    thor_body_ratio_mult: float = 1.8        # Body must be 1.8x avg body
    thor_body_lookback: int = 20             # Bars to compute avg body
    thor_quiet_body_pct: float = 1.5         # Max avg body as % of price (quiet market filter)
    thor_vol_mult: float = 1.5               # Volume must be 1.5x avg volume
    thor_immediate_body_min: float = 0.65    # Tier A: stricter body efficiency
    thor_immediate_body_ratio_mult: float = 2.2  # Tier A: stricter body ratio
    thor_immediate_vol_mult: float = 2.0     # Tier A: stricter volume
    thor_continuation_vol_mult: float = 1.2  # Tier C: continuation volume

    # ── Thor Exit / Position Parameters ──
    thor_tp_atr: float = 5.4                 # Bank trigger (early partial TP)
    thor_sl_atr: float = 2.0                 # Stop-loss distance in ATR
    thor_max_bars: int = 288                 # Max bars to hold (24h at 5m)
    thor_max_bars_post_bank: int = 288       # Max bars after bank hit
    thor_bank_atr: float = 5.4              # ATR distance to bank price
    thor_bank_fraction: float = 0.5         # Fraction to close at bank
    thor_runner_trail_atr: float = 6.0      # Chandelier trail width (ATR)
    thor_trail_activate_atr: float = 5.4    # Activate trail after this run-up
    thor_max_bars_pre_bank: int = 288       # Timeout before bank hit

    # ── Thor Scoring Weights (signal quality decomposition) ──
    thor_score_body_ratio_weight: float = 0.30  # Body/avg-body ratio contribution
    thor_score_vol_ratio_weight:  float = 0.25  # Volume spike contribution
    thor_score_body_eff_weight:   float = 0.20  # Body efficiency (body/range)
    thor_score_quiet_weight:      float = 0.10  # Quiet-market filter contribution
    thor_score_confirm_weight:    float = 0.15  # Confirmation bar contribution

    # ── Thor Tier Thresholds (Tier A = immediate impulse, B = standard, C = continuation) ──
    thor_tier_a_confidence: float = 84.0        # Score >= this → Tier A
    thor_tier_b_confidence: float = 78.0        # Score >= this → Tier B
    thor_tier_c_confidence: float = 72.0        # Score >= this → Tier C
    thor_tier_a_size_mult:  float = 1.15        # Position size multiplier (Tier A)
    thor_tier_b_size_mult:  float = 1.00        # Position size multiplier (Tier B)
    thor_tier_c_size_mult:  float = 0.75        # Position size multiplier (Tier C)
    thor_tier_b_bs_floor:   float = 0.30        # Tier B requires BS prob >= this
    thor_tier_c_bs_floor:   float = 0.35        # Tier C requires BS prob >= this

    # ── Thor Entry Quality Gates (wave_strength + pre_impulse_r2) ──
    thor_wave_strength_min: float = 40.0    # Skip if wave_strength < this
    thor_pre_impulse_r2_max: float = 0.70   # Skip if pre_impulse_r2 > this (linear moves fail)

    # ── Thor Asymmetric Compounding ──
    thor_compound_mode: str = "asymmetric_target"  # "flat" | "asymmetric_target"
    thor_compound_activation_score: float = 85.0   # Score threshold to unlock 3% risk
    thor_compound_max_loss_pct: float = 3.0         # Max risk % of equity at high conviction

    # ── Thor MAE-Calibrated Exit Params (WF sim V12.3, overrides stale defaults above) ──
    thor_sl_atr_calibrated: float = 3.00             # SL: P90 winner MAE (WF sim)
    thor_bank_atr_calibrated: float = 4.20           # Bank: top MAE decision-matrix combo
    thor_bank_fraction_calibrated: float = 0.35      # Bank fraction: top MAE combo
    thor_runner_trail_atr_calibrated: float = 2.00   # Trail: tighter runner (WF sim)
    thor_trail_activate_atr_calibrated: float = 1.50 # Trail only after 1.5 ATR above bank
    thor_max_bars_pre_calibrated: int = 48           # Pre-bank base: 4h (overridden by Gompertz)
    thor_max_bars_post_calibrated: int = 96          # Post-bank base: 8h (overridden by Gompertz)
    thor_mae_veto_bars: int = 5                      # MAE veto window: first N bars
    thor_mae_veto_atr: float = 3.62                  # MAE veto threshold: exit if adverse > this early

    # ── Gompertz Hazard Model — Dynamic Exit Timing (Khairul's Identity λ companion) ──
    # λ(t) = λ₀·e^(γt).  Optimal bank: t* = ln(n/λ₀)/γ  where n = λ(t*).
    # Calibrated from WF micro-pump anchors (5-min bars, 245 symbols):
    #   t₁=0.104d (30 bars, avg bank 4.20 ATR): n=λ(t₁)=0.700 → λ₀=0.517, γ=2.92
    # Ref: memory/reference_journals.md — Khairul's Identity λ companion equation
    thor_gompertz_n_prior: float = 0.700     # Prior micro-pump velocity day⁻¹
    thor_gompertz_lambda0: float = 0.517     # λ₀: baseline collapse hazard day⁻¹
    thor_gompertz_gamma: float = 2.92        # γ: hazard acceleration day⁻¹
    thor_gompertz_k_pre: float = 1.5         # Pre-bank abort: λ(t)=k_pre·n
    thor_gompertz_k_runner: float = 2.5      # Runner exit: λ(t)=k_runner·n
    thor_gompertz_bank_min_atr: float = 2.0  # Floor: never bank below this ATR
    thor_gompertz_bank_max_atr: float = 10.0 # Ceiling: cap dynamic bank ATR

    # ── Backward-compatibility aliases (old nike_* names map here) ──
    @property
    def nike_body_min(self): return self.thor_body_min
    @property
    def nike_body_ratio_mult(self): return self.thor_body_ratio_mult
    @property
    def nike_body_lookback(self): return self.thor_body_lookback
    @property
    def nike_quiet_body_pct(self): return self.thor_quiet_body_pct
    @property
    def nike_vol_mult(self): return self.thor_vol_mult
    @property
    def nike_immediate_body_min(self): return self.thor_immediate_body_min
    @property
    def nike_immediate_body_ratio_mult(self): return self.thor_immediate_body_ratio_mult
    @property
    def nike_immediate_vol_mult(self): return self.thor_immediate_vol_mult
    @property
    def nike_continuation_vol_mult(self): return self.thor_continuation_vol_mult
    @property
    def nike_tp_atr(self): return self.thor_tp_atr
    @property
    def nike_sl_atr(self): return self.thor_sl_atr
    @property
    def nike_max_bars(self): return self.thor_max_bars
    @property
    def nike_max_bars_post_bank(self): return self.thor_max_bars_post_bank
    @property
    def nike_bank_atr(self): return self.thor_bank_atr
    @property
    def nike_bank_fraction(self): return self.thor_bank_fraction
    @property
    def nike_runner_trail_atr(self): return self.thor_runner_trail_atr
    @property
    def nike_score_body_ratio_weight(self): return self.thor_score_body_ratio_weight
    @property
    def nike_score_vol_ratio_weight(self): return self.thor_score_vol_ratio_weight
    @property
    def nike_score_body_eff_weight(self): return self.thor_score_body_eff_weight
    @property
    def nike_score_quiet_weight(self): return self.thor_score_quiet_weight
    @property
    def nike_score_confirm_weight(self): return self.thor_score_confirm_weight
    @property
    def nike_tier_a_confidence(self): return self.thor_tier_a_confidence
    @property
    def nike_tier_b_confidence(self): return self.thor_tier_b_confidence
    @property
    def nike_tier_c_confidence(self): return self.thor_tier_c_confidence
    @property
    def nike_tier_a_size_mult(self): return self.thor_tier_a_size_mult
    @property
    def nike_tier_b_size_mult(self): return self.thor_tier_b_size_mult
    @property
    def nike_tier_c_size_mult(self): return self.thor_tier_c_size_mult
    @property
    def nike_tier_b_bs_floor(self): return self.thor_tier_b_bs_floor
    @property
    def nike_tier_c_bs_floor(self): return self.thor_tier_c_bs_floor


@dataclass
class TFTConfig:
    """
    Temporal Fusion Transformer v2 architecture (Lim et al. 2021).
    Real TFT with VSN, GRN, and Interpretable Multi-Head Attention.
    """
    hidden_size: int = 64               # Hidden dimension (Lim 2021, MX130 optimized)
    num_heads: int = 4                  # Attention heads (Lim 2021)
    num_variables: int = 16             # VSN variable groups for feature selection
    dropout: float = 0.1               # Dropout rate (Lim 2021)
    num_lstm_layers: int = 2            # LSTM encoder layers
    temperature_init: float = 1.0       # Logit temperature scaling
    attention_mask_fill: float = -1e9   # Masked attention fill value
    seq_length: int = 5                 # Input sequence length (timesteps)
    train_epochs: int = 10              # Training epochs per cycle
    early_stop_patience: int = 3        # Early stopping patience


@dataclass
class GNNConfig:
    """
    Cross-Asset Graph Neural Network (Velickovic et al. 2018).
    GAT-based correlation learning for cross-coin alpha.
    """
    hidden_dim: int = 16                # GAT hidden dimension
    learning_rate: float = 0.005        # Adam optimizer LR
    train_epochs: int = 5               # SGD epochs per graph update
    leaky_relu_alpha: float = 0.2       # LeakyReLU negative slope
    min_return_length: int = 5          # Minimum return array length for graph build
    attention_mask_fill: float = -1e12  # Zero-out non-adjacent attention


@dataclass
class ExplainerConfig:
    """SHAP TreeExplainer settings (Lundberg & Lee 2017)."""
    top_n: int = 3                      # Top features to show per prediction


@dataclass
class MonitorConfig:
    """
    Real-time model monitoring thresholds.
    Based on: Bayram et al. (2023) Concept Drift Detection Methods Survey.
    """
    window_size: int = 100              # Rolling accuracy window
    accuracy_alert_threshold: float = 0.55   # Alert when accuracy < this
    calibration_alert_threshold: float = 0.15  # Alert when ECE > this
    alert_cooldown: int = 3600          # Seconds between alerts (anti-spam)
    feature_buffer_size: int = 200      # Recent feature vectors to keep
    baseline_min_samples: int = 50      # Observations before setting drift baseline
    drift_check_interval: int = 20      # Drift recalculation frequency
    drift_recent_window: int = 50       # Recent observations for drift calc
    drift_zscore_threshold: float = 2.0 # Z-score threshold for drift alert
    calibration_bins: int = 5           # ECE calibration bins
    calibration_prob_min: float = 0.5   # ECE bin lower bound
    calibration_prob_max: float = 1.0   # ECE bin upper bound
    alert_check_interval: int = 10      # Check alerts every N predictions
    alert_min_outcomes: int = 30        # Min outcomes before checking
    metrics_save_interval: int = 300    # Seconds between metrics saves
    metrics_batch_save: int = 50        # Save to disk every N snapshots


@dataclass
class OnChainConfig:
    """On-chain whale analytics polling settings."""
    fetch_interval: int = 300           # Seconds between whale data polls (5 min)
    thread_join_timeout: float = 2.0    # Thread join timeout on stop


@dataclass
class MultiExchangeConfig:
    """Multi-exchange connectivity settings."""
    api_timeout: int = 5                # REST API timeout (seconds)
    order_timeout: int = 10             # Order placement timeout
    recv_window: str = "5000"           # Bybit API recv window
    price_cache_ttl: float = 2.0        # Price cache time-to-live (seconds)


@dataclass
class FundingArbConfig:
    """
    Funding rate arbitrage engine settings.
    Based on: Makarov & Schoar (2020), Shleifer & Vishny (1997).
    """
    min_spread_bps: float = 5.0         # Minimum funding spread threshold (bps)
    scan_interval: int = 60             # Seconds between scans
    max_positions: int = 3              # Maximum concurrent arb positions
    rate_history_limit: int = 100       # Max rate snapshots per symbol
    opportunity_log_limit: int = 500    # Max logged opportunities
    thread_join_timeout: float = 3.0    # Thread join timeout
    scan_symbols: tuple = (             # Symbols to scan for funding arb
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT',
        'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'MATICUSDT',
        'DOTUSDT', 'UNIUSDT', 'ATOMUSDT', 'NEARUSDT', 'APTUSDT',
        'ARBUSDT', 'OPUSDT', 'SUIUSDT', 'INJUSDT', 'AAVEUSDT'
    )


@dataclass
class SmartExecConfig:
    """
    TWAP/VWAP/Iceberg smart execution settings.
    Based on: Almgren & Chriss (2001), Berkowitz et al. (1988).
    """
    twap_default_duration_min: int = 10     # Default TWAP duration
    twap_default_slices: int = 5            # Default TWAP slice count
    vwap_default_duration_min: int = 30     # Default VWAP duration
    vwap_default_slices: int = 10           # Default VWAP slice count
    iceberg_default_interval_sec: int = 5   # Default iceberg inter-slice wait
    max_completed_plans: int = 100          # Max completed plans to keep in memory
    schedule_poll_sec: float = 0.5          # Sleep granularity during schedule wait
    # Slice retry / resilience (Phase 3b)
    slice_max_retries: int = 3              # Max re-attempts per slice on transient error
    slice_retry_backoff_sec: float = 2.0    # Base backoff; doubles each retry (2s, 4s, 8s)
    slice_order_timeout_sec: float = 10.0   # Per-slice wall-clock timeout before marking failed
    partial_fill_threshold: float = 0.95    # Warn if filled_qty / requested_qty < this
    # Empirical BTC/ETH hourly volume profile (Binance 2024 data, 24 values)
    # Peak: 14:00-16:00 UTC, Trough: 04:00-06:00 UTC
    hourly_volume_profile: tuple = (
        0.032, 0.028, 0.025, 0.022, 0.020, 0.020,  # 00-05 UTC
        0.025, 0.030, 0.038, 0.045, 0.048, 0.050,  # 06-11 UTC
        0.052, 0.055, 0.060, 0.058, 0.055, 0.050,  # 12-17 UTC
        0.048, 0.045, 0.042, 0.040, 0.038, 0.034   # 18-23 UTC
    )


@dataclass
class ConformalConfig:
    """
    Adaptive Conformal Inference calibration (Romano et al. 2020, Gibbs & Candes 2021).
    Replaces broken VennAbers with distribution-free coverage guarantees.
    """
    alpha_target: float = 0.10          # Miscoverage rate (0.10 = 90% coverage)
    aci_gamma: float = 0.01             # ACI adaptation rate (Gibbs & Candes 2021)
    min_cal_samples: int = 50           # Minimum calibration set size
    score_type: str = "lac"             # "lac" = Least Ambiguous set-valued Classifier


@dataclass
class DriftConfig:
    """
    Multi-stream ADWIN drift detection (Bifet & Gavalda 2007, Gama et al. 2014).
    Monitors accuracy, calibration, and feature drift simultaneously.
    """
    accuracy_delta: float = 0.002       # ADWIN delta for accuracy stream
    calibration_delta: float = 0.005    # ADWIN delta for calibration error stream
    feature_delta: float = 0.01         # ADWIN delta for feature drift stream
    max_window: int = 2000              # Maximum ADWIN window size
    min_retrain_interval: int = 3600    # Cooldown between drift-triggered retrains (seconds)
    require_dual_confirmation: bool = True  # Require 2+ streams to fire before confirming drift


@dataclass
class TrainingPipelineConfig:
    """
    Training pipeline parameters for QUANTA_ml_engine.py.
    All parameters are research-backed with academic citations.
    """
    # ── Data Window ──
    # 4000 days of 5m data captures full history of all existing coins up to their listing dates.
    train_days: int = 730
    oos_cutoff_days: int = 650       # CatBoost trains on first 650 days
    candles_per_day: int = 288       # 24h × 60m / 5m = 288

    # ── Event Count Thresholds ──
    # MIN_EVENTS: Minimum total events before a specialist is trained.
    # Rule of thumb: need at least 5×features for tree models (Hastie et al. 2009 Ch.7).
    # With feature subspace masking, effective features ≈ 50, so min = 50 events.
    min_events_per_specialist: int = 50

    # MIN_CLASS_BALANCE: Minimum samples per class after balancing.
    # CatBoost with warm start can learn from ≥10 per class.
    # 10 per class × 80/20 split = 8 train, 2 val per class = minimum for AUC.
    min_class_balance: int = 10

    # ── Memory Limits ──
    # MAX_EVENTS_PER_COIN: Cap per-coin events to prevent RAM explosion.
    # Isele & Cosgun (2018) "Selective Experience Replay": diminishing returns
    # beyond 1K-2K samples per distribution source.
    max_events_per_coin: int = 1500
    max_agent_events: int = 180000   # 1500 × 120 coins

    # ── Time Decay ──
    # López de Prado AFML Ch.5: "Sample weights should reflect information decay."
    # Linear decay from 1.0 (oldest) to 3.0 (newest).
    time_decay_min: float = 1.0
    time_decay_max: float = 3.0

    # ── Parallel Processing ──
    max_workers: int = 4

    # ── Event Cache ──
    # Cache validity: 24 hours. After that, re-extract to capture new market data.
    cache_max_age_days: float = 1.0

    # ── CatBoost Parameters ──
    # Prokhorenkova et al. (2018): optimal GPU quantization bins for CatBoost
    catboost_border_count: int = 254
    
    # Prechelt (1998): Early stopping window to prevent overfitting
    catboost_early_stopping: int = 50
    
    # Freund & Schapire (1997): AdaBoost weighting for misclassified samples
    catboost_hard_neg_boost: float = 2.0


@dataclass
class RiskManagerConfig:
    """
    Capital protection and circuit breaker settings (v11.4).
    Based on: Bouchaud & Potters (2003), López de Prado (2018) AFML Ch.10.
    """
    max_daily_drawdown_pct: float = 8.0       # Compound mode: pause if daily loss >8% (was 3%)
    max_open_positions: int = 50              # Max concurrent open positions
    max_single_coin_pct: float = 50.0         # Max exposure to any single coin (% of balance)
    max_total_exposure_pct: float = 5000.0    # Max total portfolio exposure (% of balance)
    consecutive_loss_throttle: int = 3        # Reduce size after N consecutive losses
    throttle_size_factor: float = 0.5         # Size multiplier per excess loss (0.5 = half)
    cooldown_after_breaker_min: int = 60      # Minutes to pause after circuit breaker triggers
    max_correlation_exposure: int = 5         # Max same-direction positions
    max_risk_per_trade_pct: float = 33.0      # Hard cap: single trade cannot risk more than 33% of balance

    # -- Compound Growth Engine Guardrails (2026-04-12) --
    # Prevents ruin during aggressive compound sizing. Inspired by Taleb (2012) barbell strategy.
    compound_weekly_dd_limit: float = 20.0    # Hard pause if weekly rolling DD > 20%
    compound_ruin_threshold: float = 0.35     # Emergency flat if equity drops to 35% of peak
    compound_recovery_mode_mult: float = 0.30 # After ruin trigger, size at 30% until 60% of peak recovered
    compound_win_streak_boost: float = 0.15   # +15% position per streak tier above trigger
    compound_streak_trigger: int = 3          # Wins before streak boost kicks in

    def __post_init__(self):
        assert self.max_daily_drawdown_pct > 0, \
            "max_daily_drawdown_pct must be > 0 (it's a percentage magnitude, e.g. 3.0 = 3%)"
        assert self.max_daily_drawdown_pct <= 100.0, "max_daily_drawdown_pct must be <= 100"
        assert self.max_open_positions > 0, "max_open_positions must be > 0"
        assert 0.0 < self.max_single_coin_pct <= 100.0, "max_single_coin_pct must be in (0, 100]"
        assert 0.0 < self.max_total_exposure_pct <= 5000.0, "max_total_exposure_pct must be in (0, 5000]"
        assert self.max_single_coin_pct <= self.max_total_exposure_pct, \
            "max_single_coin_pct cannot exceed max_total_exposure_pct"
        assert self.consecutive_loss_throttle > 0, "consecutive_loss_throttle must be > 0"
        assert 0.0 < self.throttle_size_factor <= 1.0, "throttle_size_factor must be in (0, 1]"
        assert self.cooldown_after_breaker_min >= 0, "cooldown_after_breaker_min must be >= 0"
        assert self.max_correlation_exposure > 0, "max_correlation_exposure must be > 0"
        assert 0.0 < self.max_risk_per_trade_pct <= 100.0, "max_risk_per_trade_pct must be in (0, 100]"


@dataclass
class BacktestConfig:
    """
    Walk-Forward Optimization backtester settings (v11.4).
    Based on: Pardo (2008), Bailey et al. (2014), López de Prado (2018) AFML Ch.12.
    """
    train_window_days: int = 180              # Training window per WFO fold
    test_window_days: int = 30                # OOS test window per fold
    step_days: int = 30                       # Step size between folds
    min_confidence: float = 60.0              # Min confidence to trigger a backtest trade
    commission_bps: float = 4.0               # Commission per side (Binance default: 0.04%)
    slippage_bps: float = 2.0                 # Estimated slippage per side
    max_concurrent_positions: int = 10        # Max positions open simultaneously
    risk_per_trade_pct: float = 1.0           # Risk per trade as % of balance

    def __post_init__(self):
        assert self.train_window_days > 0, "train_window_days must be > 0"
        assert self.test_window_days > 0, "test_window_days must be > 0"
        assert self.step_days > 0, "step_days must be > 0"
        assert 0.0 < self.min_confidence <= 100.0, "min_confidence must be in (0, 100]"
        assert self.commission_bps >= 0, "commission_bps must be >= 0"
        assert self.slippage_bps >= 0, "slippage_bps must be >= 0"
        assert self.max_concurrent_positions > 0, "max_concurrent_positions must be > 0"
        assert 0.0 < self.risk_per_trade_pct <= 100.0, "risk_per_trade_pct must be in (0, 100]"


@dataclass
class PaperTradingConfig:
    """
    Paper trading decision logger settings (v11.4).
    Based on: Tharp (2006), Schwager (1993).
    """
    snapshot_interval_min: int = 15           # Equity snapshot interval (minutes)
    max_decisions_in_memory: int = 5000       # Rolling decision buffer size
    max_trades_in_memory: int = 2000          # Rolling trade buffer size

    def __post_init__(self):
        assert self.snapshot_interval_min > 0, "snapshot_interval_min must be > 0"
        assert self.max_decisions_in_memory > 0, "max_decisions_in_memory must be > 0"
        assert self.max_trades_in_memory > 0, "max_trades_in_memory must be > 0"


@dataclass
class ZeusConfig:
    """ZEUS.ai Universal Orchestrator settings and Hard Guardrails."""
    ai_base_url: str = AI_BASE_URL
    ai_api_key: str = AI_API_KEY
    ai_model_name: str = AI_MODEL_NAME
    
    # CatBoost Guardrails
    max_catboost_iter: int = 3000
    min_catboost_iter: int = 200
    max_learning_rate: float = 0.5
    min_learning_rate: float = 0.01
    max_depth: int = 10
    min_depth: int = 4
    min_feature_mask_size: int = 50
    
    # PPO Guardrails
    max_ppo_lr: float = 5e-3
    min_ppo_lr: float = 1e-6
    max_ppo_entropy: float = 0.10
    min_ppo_entropy: float = 0.001
    max_ppo_clip: float = 0.40
    min_ppo_clip: float = 0.05
    max_ppo_batch_size: int = 1024
    min_ppo_batch_size: int = 64


@dataclass
class SystemConfig:
    """Global system paths and orchestrator configurations."""
    base_dir: Path = Path("C:/Users/habib/QUANTA")
    model_dir: Path = base_dir / "models"
    data_dir: Path = base_dir / "data"
    feature_dir: Path = base_dir / "features"
    rl_memory_file: Path = base_dir / "rl_memory.json"
    
    network: NetworkConfig = field(default_factory=NetworkConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    events: EventExtractionConfig = field(default_factory=EventExtractionConfig)
    training: TrainingPipelineConfig = field(default_factory=TrainingPipelineConfig)
    conformal: ConformalConfig = field(default_factory=ConformalConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    tft: TFTConfig = field(default_factory=TFTConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)
    explainer: ExplainerConfig = field(default_factory=ExplainerConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    onchain: OnChainConfig = field(default_factory=OnChainConfig)
    multi_exchange: MultiExchangeConfig = field(default_factory=MultiExchangeConfig)
    funding_arb: FundingArbConfig = field(default_factory=FundingArbConfig)
    smart_exec: SmartExecConfig = field(default_factory=SmartExecConfig)
    risk_manager: RiskManagerConfig = field(default_factory=RiskManagerConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    paper_trading: PaperTradingConfig = field(default_factory=PaperTradingConfig)
    zeus: ZeusConfig = field(default_factory=ZeusConfig)

    def __post_init__(self):
        """Ensure directories exist and sub-config fields are valid."""
        for d in [self.model_dir, self.data_dir, self.feature_dir]:
            d.mkdir(parents=True, exist_ok=True)
        # Trigger validation on all nested dataclass fields
        for attr_name in ('network', 'market', 'indicators', 'model', 'rl', 'strategy',
                          'risk_manager', 'backtest', 'paper_trading', 'zeus'):
            obj = getattr(self, attr_name, None)
            if obj is not None and hasattr(obj, '__post_init__'):
                obj.__post_init__()

# Instantiate the global singleton configuration
Config = SystemConfig()

def update_config_from_overrides():
    import json
    import dataclasses
    override_file = Config.base_dir / "quanta_config_overrides.json"
    if override_file.exists():
        try:
            with open(override_file, 'r') as f:
                overrides = json.load(f)
            
            for category, fields_dict in overrides.items():
                if not hasattr(Config, category):
                    continue
                category_obj = getattr(Config, category)
                if not dataclasses.is_dataclass(category_obj):
                    continue
                
                for key, val in fields_dict.items():
                    if hasattr(category_obj, key):
                        old_val = getattr(category_obj, key)
                        if isinstance(old_val, int):
                            setattr(category_obj, key, int(val))
                        elif isinstance(old_val, float):
                            setattr(category_obj, key, float(val))
                        elif isinstance(old_val, bool):
                            setattr(category_obj, key, bool(val))
                        else:
                            setattr(category_obj, key, val)
            print(f"\u2699\ufe0f Loaded JSON config overrides from dashboard".encode('utf-8', errors='replace').decode('utf-8'))
        except Exception as e:
            pass  # Silently skip if overrides cannot be loaded

def export_config_to_dict():
    import dataclasses
    config_dict = {}
    for field_obj in dataclasses.fields(Config):
        if field_obj.name.endswith('_dir') or field_obj.name == 'rl_memory_file' or field_obj.name == 'base_dir':
            continue
        val = getattr(Config, field_obj.name)
        if dataclasses.is_dataclass(val):
            config_dict[field_obj.name] = dataclasses.asdict(val)
    return config_dict

# Inject helper functions onto the Config object dynamically
Config.update_from_dict = update_config_from_overrides
Config.export_to_dict = export_config_to_dict

Config.update_from_dict()  # Load overrides immediately
