"""
QUANTA CUDA Indicator Kernels — Batched GPU computation for 200+ coins.

Uses Numba CUDA JIT to compute RSI, EMA, MACD, ATR, ADX, Bollinger
across ALL coins simultaneously (1 GPU thread per coin).

Auto-fallback: If no CUDA GPU available, uses CPU pre-computation path.

Requirements:
    - Numba with CUDA support
    - CUDA Toolkit 12.x
    - GPU with sufficient VRAM (RTX 4090 24GB recommended)
"""
import numpy as np
import time
import logging

# =================== CUDA AVAILABILITY CHECK ===================
try:
    from numba import cuda
    import math as cuda_math
    CUDA_AVAILABLE = cuda.is_available()
    if CUDA_AVAILABLE:
        GPU_NAME = cuda.get_current_device().name
        GPU_MEM = cuda.current_context().get_memory_info()
        GPU_MEM_GB = GPU_MEM[1] / (1024**3)
        print(f"⚡ CUDA Indicators: GPU detected — {GPU_NAME} ({GPU_MEM_GB:.1f} GB)")
    else:
        print("⚠️ CUDA Indicators: No GPU — will use CPU fallback")
except ImportError:
    CUDA_AVAILABLE = False
    print("⚠️ CUDA Indicators: Numba CUDA not available — will use CPU fallback")


# =================== CUDA KERNELS ===================
# Each kernel: 1 thread = 1 coin. All coins computed in parallel.
# Grid: (num_coins,), Block: (1,)

if CUDA_AVAILABLE:

    @cuda.jit
    def _cuda_ema_batch(data, output, lengths, period):
        """Batched EMA series. Each thread computes EMA for one coin.
        
        Args:
            data: (num_coins, max_len) float64 input
            output: (num_coins, max_len) float64 output
            lengths: (num_coins,) int64 actual lengths
            period: int EMA period
        """
        coin = cuda.grid(1)
        if coin >= data.shape[0]:
            return
        n = lengths[coin]
        if n == 0:
            return
        
        alpha = 2.0 / (period + 1.0)
        output[coin, 0] = data[coin, 0]
        for i in range(1, n):
            output[coin, i] = data[coin, i] * alpha + output[coin, i-1] * (1.0 - alpha)

    @cuda.jit
    def _cuda_rsi_batch(closes, rsi_out, lengths, period):
        """Batched RSI series. Each thread = one coin."""
        coin = cuda.grid(1)
        if coin >= closes.shape[0]:
            return
        n = lengths[coin]
        if n <= period:
            for i in range(n):
                rsi_out[coin, i] = 50.0
            return
        
        # Initialize
        for i in range(n):
            rsi_out[coin, i] = 50.0
        
        avg_gain = 0.0
        avg_loss = 0.0
        for i in range(1, period + 1):
            diff = closes[coin, i] - closes[coin, i - 1]
            if diff > 0:
                avg_gain += diff
            else:
                avg_loss -= diff
        avg_gain /= period
        avg_loss /= period
        
        if avg_loss == 0:
            rsi_out[coin, period] = 100.0
        else:
            rsi_out[coin, period] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
        
        for i in range(period + 1, n):
            diff = closes[coin, i] - closes[coin, i - 1]
            gain = diff if diff > 0 else 0.0
            loss = -diff if diff < 0 else 0.0
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period
            if avg_loss == 0:
                rsi_out[coin, i] = 100.0
            else:
                rsi_out[coin, i] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))

    @cuda.jit
    def _cuda_atr_batch(highs, lows, closes, tr_out, atr_out, lengths, period):
        """Batched ATR series. Computes True Range then EMA-smooths it."""
        coin = cuda.grid(1)
        if coin >= closes.shape[0]:
            return
        n = lengths[coin]
        if n == 0:
            return
        
        # True Range
        tr_out[coin, 0] = highs[coin, 0] - lows[coin, 0]
        for i in range(1, n):
            hl = highs[coin, i] - lows[coin, i]
            hc = abs(highs[coin, i] - closes[coin, i-1])
            lc = abs(lows[coin, i] - closes[coin, i-1])
            tr_out[coin, i] = hl
            if hc > tr_out[coin, i]:
                tr_out[coin, i] = hc
            if lc > tr_out[coin, i]:
                tr_out[coin, i] = lc
        
        # EMA of TR = ATR
        alpha = 2.0 / (period + 1.0)
        atr_out[coin, 0] = tr_out[coin, 0]
        for i in range(1, n):
            atr_out[coin, i] = tr_out[coin, i] * alpha + atr_out[coin, i-1] * (1.0 - alpha)

    @cuda.jit
    def _cuda_adx_batch(highs, lows, closes, adx_out, lengths, period):
        """Batched ADX series. Each thread = one coin."""
        coin = cuda.grid(1)
        if coin >= closes.shape[0]:
            return
        n = lengths[coin]
        if n <= period:
            for i in range(n):
                adx_out[coin, i] = 25.0
            return
        
        # Initialize all to 25
        for i in range(n):
            adx_out[coin, i] = 25.0
        
        # Compute TR, +DM, -DM
        # Use local arrays via shared memory approach — but since each thread is independent,
        # we'll use sequential computation within the thread.
        
        # EMA accumulators
        alpha = 2.0 / (period + 1.0)
        
        # First pass: compute smoothed TR, +DM, -DM via running EMA
        smooth_tr = highs[coin, 0] - lows[coin, 0]
        smooth_pdm = 0.0
        smooth_mdm = 0.0
        
        for i in range(1, n):
            hl = highs[coin, i] - lows[coin, i]
            hc = abs(highs[coin, i] - closes[coin, i-1])
            lc = abs(lows[coin, i] - closes[coin, i-1])
            tr = hl
            if hc > tr: tr = hc
            if lc > tr: tr = lc
            
            up = highs[coin, i] - highs[coin, i-1]
            dn = lows[coin, i-1] - lows[coin, i]
            
            pdm = 0.0
            mdm = 0.0
            if up > dn and up > 0:
                pdm = up
            if dn > up and dn > 0:
                mdm = dn
            
            smooth_tr = tr * alpha + smooth_tr * (1.0 - alpha)
            smooth_pdm = pdm * alpha + smooth_pdm * (1.0 - alpha)
            smooth_mdm = mdm * alpha + smooth_mdm * (1.0 - alpha)
            
            if smooth_tr > 0:
                pdi = 100.0 * smooth_pdm / smooth_tr
                mdi = 100.0 * smooth_mdm / smooth_tr
                den = pdi + mdi
                if den > 0:
                    dx = 100.0 * abs(pdi - mdi) / den
                else:
                    dx = 0.0
            else:
                dx = 0.0
            
            # EMA of DX = ADX
            if i == 1:
                adx_out[coin, i] = dx
            else:
                adx_out[coin, i] = dx * alpha + adx_out[coin, i-1] * (1.0 - alpha)

    @cuda.jit
    def _cuda_rolling_mean_batch(data, output, lengths, period):
        """Batched rolling mean (SMA). Each thread = one coin."""
        coin = cuda.grid(1)
        if coin >= data.shape[0]:
            return
        n = lengths[coin]
        if n == 0:
            return
        
        # Expanding mean for warmup
        s = 0.0
        lim = min(period, n)
        for i in range(lim):
            s += data[coin, i]
            output[coin, i] = s / (i + 1)
        
        # Rolling window
        if n > period:
            window_sum = s
            for i in range(period, n):
                window_sum += data[coin, i] - data[coin, i - period]
                output[coin, i] = window_sum / period

    @cuda.jit
    def _cuda_rolling_std_batch(data, output, lengths, period):
        """Batched rolling standard deviation. Each thread = one coin."""
        coin = cuda.grid(1)
        if coin >= data.shape[0]:
            return
        n = lengths[coin]
        
        for i in range(period - 1, n):
            start = i - period + 1
            if start < 0:
                start = 0
            wn = i - start + 1
            
            mean = 0.0
            for j in range(start, i + 1):
                mean += data[coin, j]
            mean /= wn
            
            var = 0.0
            for j in range(start, i + 1):
                diff = data[coin, j] - mean
                var += diff * diff
            
            output[coin, i] = (var / wn) ** 0.5


# =================== ORCHESTRATOR (CHUNKED — fits any GPU) ===================

# VRAM budget: use 3/4 of total VRAM, leave 1/4 free
VRAM_BUDGET_FRACTION = 0.75

def _estimate_vram_per_coin(max_len):
    """Estimate GPU VRAM needed per coin in bytes.
    
    Per coin per TF: 4 input arrays + 12 output arrays = 16 arrays × max_len × 8 bytes
    Plus lengths array (negligible).
    """
    arrays_per_coin = 16  # closes, highs, lows, vols + rsi, atr, tr, adx, ema_fast, ema_slow, ma20, ma50, bb_std, vol_ma, returns, vol_rolling
    return arrays_per_coin * max_len * 8  # bytes


def compute_all_indicators_gpu(all_klines_dict, tf_windows=None):
    """
    Batch-compute ALL indicator series for ALL coins on GPU.
    Uses CHUNKED batching to fit within VRAM budget (safe for MX130 2GB).
    
    Args:
        all_klines_dict: dict {symbol: klines_np} where klines_np is (N, 6+) float64
        tf_windows: timeframe aggregation windows (default: 5m/15m/1h/4h/1d)
        
    Returns:
        dict {symbol: precomputed} — same format as _precompute_coin_indicators()
    """
    if not CUDA_AVAILABLE or len(all_klines_dict) == 0:
        return None
    
    if tf_windows is None:
        tf_windows = {'5m': 1, '15m': 3, '1h': 12, '4h': 48, '1d': 288}
    
    symbols = list(all_klines_dict.keys())
    num_coins = len(symbols)
    
    # Determine VRAM budget
    try:
        mem_info = cuda.current_context().get_memory_info()
        free_vram = mem_info[0]
        total_vram = mem_info[1]
        budget = int(total_vram * VRAM_BUDGET_FRACTION)
        budget = min(budget, free_vram - 50 * 1024 * 1024)  # Keep 50MB safety margin
        if budget < 10 * 1024 * 1024:  # Less than 10MB usable
            print(f"  ⚠️ GPU VRAM too low ({free_vram/(1024**2):.0f}MB free) — skipping")
            return None
    except Exception:
        return None
    
    # Max candle length across all coins
    max_candle_len = max(len(kl) for kl in all_klines_dict.values())
    
    # Calculate chunk size (how many coins fit in VRAM budget)
    vram_per_coin = _estimate_vram_per_coin(max_candle_len)
    chunk_size = max(1, budget // vram_per_coin)
    num_chunks = (num_coins + chunk_size - 1) // chunk_size
    
    budget_mb = budget / (1024**2)
    per_coin_mb = vram_per_coin / (1024**2)
    print(f"\n⚡ GPU Batch: {num_coins} coins, VRAM budget={budget_mb:.0f}MB, "
          f"{per_coin_mb:.1f}MB/coin → {chunk_size} coins/chunk × {num_chunks} chunks")
    
    t0 = time.time()
    result = {}
    
    for tf, w in tf_windows.items():
        # 1. Aggregate candles to this TF for ALL coins (CPU — fast)
        all_aggregated = []
        all_lengths = []
        
        for sym in symbols:
            klines = all_klines_dict[sym]
            n = len(klines)
            closes = klines[:, 4]
            highs = klines[:, 2]
            lows = klines[:, 3]
            volumes = klines[:, 5]
            
            if w == 1:
                c, h, l, v = closes, highs, lows, volumes
                rem = 0
            else:
                rem = n % w
                if n < w * 20:
                    all_aggregated.append(None)
                    all_lengths.append(0)
                    continue
                c = closes[rem:].reshape(-1, w)[:, -1]
                h = np.max(highs[rem:].reshape(-1, w), axis=1)
                l = np.min(lows[rem:].reshape(-1, w), axis=1)
                v = np.sum(volumes[rem:].reshape(-1, w), axis=1)
            
            all_aggregated.append({'c': c, 'h': h, 'l': l, 'v': v, 'rem': rem})
            all_lengths.append(len(c))
        
        valid_coins = [i for i, ln in enumerate(all_lengths) if ln >= 20]
        if not valid_coins:
            continue
        
        # 2. Process in CHUNKS to stay within VRAM budget
        tf_max_len = max(all_lengths[i] for i in valid_coins)
        
        # Recalculate chunk size for this TF's max_len
        tf_vram_per_coin = _estimate_vram_per_coin(tf_max_len)
        tf_chunk_size = max(1, budget // tf_vram_per_coin)
        
        for chunk_start in range(0, num_coins, tf_chunk_size):
            chunk_end = min(chunk_start + tf_chunk_size, num_coins)
            chunk_indices = list(range(chunk_start, chunk_end))
            chunk_n = len(chunk_indices)
            
            # Find max length in this chunk
            chunk_max_len = max(all_lengths[i] if all_lengths[i] > 0 else 1 for i in chunk_indices)
            chunk_lengths = [all_lengths[i] for i in chunk_indices]
            
            # Skip if no valid coins in chunk
            if max(chunk_lengths) < 20:
                continue
            
            # 3. Pad and stack chunk into GPU tensors
            closes_batch = np.zeros((chunk_n, chunk_max_len), dtype=np.float64)
            highs_batch = np.zeros((chunk_n, chunk_max_len), dtype=np.float64)
            lows_batch = np.zeros((chunk_n, chunk_max_len), dtype=np.float64)
            vols_batch = np.zeros((chunk_n, chunk_max_len), dtype=np.float64)
            lengths_arr = np.array(chunk_lengths, dtype=np.int64)
            
            for local_i, global_i in enumerate(chunk_indices):
                agg = all_aggregated[global_i]
                if agg is not None:
                    ln = chunk_lengths[local_i]
                    closes_batch[local_i, :ln] = agg['c']
                    highs_batch[local_i, :ln] = agg['h']
                    lows_batch[local_i, :ln] = agg['l']
                    vols_batch[local_i, :ln] = agg['v']
            
            # 4. Transfer to GPU
            d_closes = cuda.to_device(closes_batch)
            d_highs = cuda.to_device(highs_batch)
            d_lows = cuda.to_device(lows_batch)
            d_vols = cuda.to_device(vols_batch)
            d_lengths = cuda.to_device(lengths_arr)
            
            d_rsi = cuda.device_array((chunk_n, chunk_max_len), dtype=np.float64)
            d_atr = cuda.device_array((chunk_n, chunk_max_len), dtype=np.float64)
            d_tr = cuda.device_array((chunk_n, chunk_max_len), dtype=np.float64)
            d_adx = cuda.device_array((chunk_n, chunk_max_len), dtype=np.float64)
            d_ema_fast = cuda.device_array((chunk_n, chunk_max_len), dtype=np.float64)
            d_ema_slow = cuda.device_array((chunk_n, chunk_max_len), dtype=np.float64)
            d_ma20 = cuda.device_array((chunk_n, chunk_max_len), dtype=np.float64)
            d_ma50 = cuda.device_array((chunk_n, chunk_max_len), dtype=np.float64)
            d_bb_std = cuda.device_array((chunk_n, chunk_max_len), dtype=np.float64)
            d_vol_ma = cuda.device_array((chunk_n, chunk_max_len), dtype=np.float64)
            
            # 5. Launch kernels
            tpb = min(256, chunk_n)
            blocks = (chunk_n + tpb - 1) // tpb
            
            _cuda_rsi_batch[blocks, tpb](d_closes, d_rsi, d_lengths, 14)
            _cuda_ema_batch[blocks, tpb](d_closes, d_ema_fast, d_lengths, 12)
            _cuda_ema_batch[blocks, tpb](d_closes, d_ema_slow, d_lengths, 26)
            _cuda_atr_batch[blocks, tpb](d_highs, d_lows, d_closes, d_tr, d_atr, d_lengths, 14)
            _cuda_adx_batch[blocks, tpb](d_highs, d_lows, d_closes, d_adx, d_lengths, 14)
            _cuda_rolling_mean_batch[blocks, tpb](d_closes, d_ma20, d_lengths, 20)
            _cuda_rolling_mean_batch[blocks, tpb](d_closes, d_ma50, d_lengths, 50)
            _cuda_rolling_std_batch[blocks, tpb](d_closes, d_bb_std, d_lengths, 20)
            _cuda_rolling_mean_batch[blocks, tpb](d_vols, d_vol_ma, d_lengths, 20)
            
            cuda.synchronize()
            
            # 6. Copy results back
            rsi_all = d_rsi.copy_to_host()
            ema_fast_all = d_ema_fast.copy_to_host()
            ema_slow_all = d_ema_slow.copy_to_host()
            atr_all = d_atr.copy_to_host()
            adx_all = d_adx.copy_to_host()
            ma20_all = d_ma20.copy_to_host()
            ma50_all = d_ma50.copy_to_host()
            bb_std_all = d_bb_std.copy_to_host()
            vol_ma_all = d_vol_ma.copy_to_host()
            
            # 7. FREE GPU MEMORY immediately
            del d_closes, d_highs, d_lows, d_vols, d_lengths
            del d_rsi, d_atr, d_tr, d_adx, d_ema_fast, d_ema_slow
            del d_ma20, d_ma50, d_bb_std, d_vol_ma
            cuda.current_context().deallocations.clear()
            
            # 8. Package results per coin
            from quanta_features import _jit_rolling_std
            for local_i, global_i in enumerate(chunk_indices):
                ln = chunk_lengths[local_i]
                if ln < 20:
                    continue
                agg = all_aggregated[global_i]
                if agg is None:
                    continue
                
                sym = symbols[global_i]
                c = agg['c']
                
                returns_arr = np.zeros(ln, dtype=np.float64)
                if ln > 1:
                    returns_arr[1:] = np.diff(c) / (c[:-1] + 1e-8)
                vol_rolling = _jit_rolling_std(returns_arr, 20)
                
                macd_line = ema_fast_all[local_i, :ln] - ema_slow_all[local_i, :ln]
                
                tf_data = {
                    'closes': c, 'highs': agg['h'], 'lows': agg['l'], 'volumes': agg['v'],
                    'tf_len': ln, 'w': w, 'rem': agg['rem'],
                    'rsi': rsi_all[local_i, :ln],
                    'macd_line': macd_line,
                    'atr': atr_all[local_i, :ln],
                    'adx': adx_all[local_i, :ln],
                    'ma20': ma20_all[local_i, :ln],
                    'ma50': ma50_all[local_i, :ln],
                    'bb_std': bb_std_all[local_i, :ln],
                    'vol_ma': vol_ma_all[local_i, :ln],
                    'returns': returns_arr,
                    'vol_rolling': vol_rolling,
                    'atr_sampled': atr_all[local_i, 14:ln:5],
                }
                
                if sym not in result:
                    klines = all_klines_dict[sym]
                    result[sym] = {
                        'open_times': klines[:, 0].astype(np.int64),
                        'raw_closes': klines[:, 4].copy(),
                        'raw_highs': klines[:, 2].copy(),
                        'raw_lows': klines[:, 3].copy(),
                        'raw_volumes': klines[:, 5].copy(),
                    }
                result[sym][tf] = tf_data
    
    elapsed = time.time() - t0
    print(f"⚡ GPU Batch: {num_coins} coins × {len(tf_windows)} TFs done in {elapsed:.2f}s "
          f"({num_chunks} chunks of ≤{chunk_size})")
    
    return result


def is_gpu_available():
    """Check if GPU batch computation is available."""
    if not CUDA_AVAILABLE:
        return False
    try:
        mem_info = cuda.current_context().get_memory_info()
        total_gb = mem_info[1] / (1024**3)
        budget = total_gb * VRAM_BUDGET_FRACTION
        return budget > 0.05  # Need at least 50MB budget (= 200MB total VRAM)
    except Exception:
        return False

