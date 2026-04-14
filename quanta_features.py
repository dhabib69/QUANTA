import numpy as np
import time
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

# =================== JIT-COMPILED KERNELS (Numba if available, else pure NumPy) ===================
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        """Fallback: no JIT compilation. Functions run as pure NumPy."""
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

# Forward declaration of all the JIT kernels needed by Indicators
# Copy them directly from QUANTA_ml_engine.py so Indicators can find them
@njit(cache=True)
def _jit_frac_diff(series, d, min_weight):
    n = len(series)
    if n == 0: return 0.0
    w = np.ones(min(n, 1000), dtype=np.float64)
    for k in range(1, len(w)):
        w[k] = -w[k-1] * (d - k + 1) / k
        if abs(w[k]) < min_weight:
            w = w[:k]
            break
    
    diff_val = 0.0
    for i in range(len(w)):
        if n - 1 - i >= 0:
            diff_val += w[i] * series[n - 1 - i]
    return diff_val


def fractional_differentiation(series: np.ndarray, d: float = 0.4, threshold: float = 1e-4) -> np.ndarray:
    """
    Fractional differentiation (López de Prado 2018 Ch.5).
    Maps a non-stationary price series to a stationary one while preserving
    long-term memory (unlike standard integer differencing which destroys it).

    Args:
        series:    1D numpy array of prices/values
        d:         Differencing parameter [0, 1]. 0.4-0.5 recommended for crypto.
        threshold: Weight truncation threshold. Smaller = more accurate, slower.

    Returns:
        1D numpy array of fractionally differenced values (same length as input,
        early values are NaN until enough history is accumulated).
    """
    series = np.asarray(series, dtype=np.float64)
    n = len(series)
    result = np.full(n, np.nan)

    # Build weight vector
    w = [1.0]
    k = 1
    while True:
        w_next = -w[-1] * (d - k + 1) / k
        if abs(w_next) < threshold:
            break
        w.append(w_next)
        k += 1
        if k > 1000:
            break
    w = np.array(w[::-1])  # reverse so w[0] is the oldest weight
    lag = len(w)

    # Compute fractionally differenced series
    for i in range(lag - 1, n):
        result[i] = float(np.dot(w, series[i - lag + 1: i + 1]))

    return result


@njit(cache=True)
def _jit_vpin(h, l, c, v, window):
    if len(c) < window: return 0.5
    dp = np.zeros(window, dtype=np.float64)
    for i in range(1, window):
        dp[i] = c[-window + i] - c[-window + i - 1]
    
    sigma = np.std(dp)
    if sigma == 0: return 0.5
    
    v_buy = np.zeros(window, dtype=np.float64)
    v_sell = np.zeros(window, dtype=np.float64)
    
    for i in range(1, window):
        idx = -window + i
        # BVC: NormCDF approximation on standardized return (Easley et al. 2012 Eq.4)
        z = dp[i] / sigma
        prob_buy = 0.5 * (1.0 + np.tanh(z * 0.79788456))
        
        v_buy[i] = v[idx] * prob_buy
        v_sell[i] = v[idx] * (1.0 - prob_buy)
        
    v_imb = np.abs(np.sum(v_buy) - np.sum(v_sell))
    v_total = np.sum(v_buy) + np.sum(v_sell)
    
    return v_imb / v_total if v_total > 0 else 0.5

@njit(cache=True)
def _jit_vpin_taker(volumes, taker_buy_volumes, window):
    """TRUE VPIN using actual taker buy volume (Easley et al. 2012a).
    VPIN = mean(|V_buy - V_sell| / V_total) over `window` bars.
    """
    n = len(volumes)
    if n < window:
        return 0.5
    v_arr = volumes[-window:]
    tb_arr = taker_buy_volumes[-window:]
    total_buy = 0.0
    total_sell = 0.0
    for i in range(window):
        buy_v = tb_arr[i]
        sell_v = v_arr[i] - buy_v
        if sell_v < 0.0:
            sell_v = 0.0
        total_buy += buy_v
        total_sell += sell_v
    total_vol = total_buy + total_sell
    if total_vol == 0.0:
        return 0.5
    return abs(total_buy - total_sell) / total_vol

@njit(cache=True)
def _jit_rsi(arr, period):
    if len(arr) <= period: return 50.0
    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        diff = arr[i] - arr[i - 1]
        if diff > 0: gains += diff
        else: losses -= diff
    avg_gain = gains / period
    avg_loss = losses / period
    
    for i in range(period + 1, len(arr)):
        diff = arr[i] - arr[i - 1]
        gain = diff if diff > 0 else 0.0
        loss = -diff if diff < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        
    if avg_loss == 0: return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

@njit(cache=True)
def _jit_ema_series(arr, period):
    res = np.zeros(len(arr), dtype=np.float64)
    if len(arr) == 0: return res
    alpha = 2.0 / (period + 1.0)
    res[0] = arr[0]
    for i in range(1, len(arr)):
        res[i] = arr[i] * alpha + res[i-1] * (1.0 - alpha)
    return res

@njit(cache=True)
def _jit_macd(arr, fast, slow, signal):
    if len(arr) == 0: return 0.0, 0.0, 0.0
    f_ema = _jit_ema_series(arr, fast)
    s_ema = _jit_ema_series(arr, slow)
    macd_line = np.zeros(len(arr), dtype=np.float64)
    for i in range(len(arr)):
        macd_line[i] = f_ema[i] - s_ema[i]
    sig_line = _jit_ema_series(macd_line, signal)
    if len(arr) == 0: return 0.0, 0.0, 0.0
    return macd_line[-1], sig_line[-1], macd_line[-1] - sig_line[-1]

@njit(cache=True)
def _jit_bollinger(arr, period, std_dev):
    if len(arr) < period: return 0.0, 0.0, 0.0
    window = arr[-period:]
    mean = 0.0
    for x in window: mean += x
    mean /= period
    var = 0.0
    for x in window: var += (x - mean)**2
    var /= period
    std = var**0.5
    return mean + (std_dev * std), mean, mean - (std_dev * std)

@njit(cache=True)
def _jit_atr_series(h, l, c, period):
    n = len(c)
    tr = np.zeros(n, dtype=np.float64)
    if n == 0: return tr
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        hl = h[i] - l[i]
        hc = abs(h[i] - c[i-1])
        lc = abs(l[i] - c[i-1])
        tr[i] = hl
        if hc > tr[i]: tr[i] = hc
        if lc > tr[i]: tr[i] = lc
    return _jit_ema_series(tr, period)

@njit(cache=True)
def _jit_stochastic(h, l, c, period, d_period):
    if len(c) < period: return 50.0, 50.0
    k_vals = np.zeros(len(c) - period + 1, dtype=np.float64)
    for i in range(period - 1, len(c)):
        w_h = h[i - period + 1 : i + 1]
        w_l = l[i - period + 1 : i + 1]
        highest = w_h[0]
        lowest = w_l[0]
        for v in w_h:
            if v > highest: highest = v
        for v in w_l:
            if v < lowest: lowest = v
        den = highest - lowest
        if den == 0: k_vals[i - period + 1] = 50.0
        else: k_vals[i - period + 1] = 100.0 * (c[i] - lowest) / den
    
    if len(k_vals) < d_period:
        k_last = k_vals[-1] if len(k_vals) > 0 else 50.0
        return k_last, k_last
        
    # SMA for D
    d_val = 0.0
    for i in range(d_period):
        d_val += k_vals[-(i+1)]
    d_val /= d_period
    
    return k_vals[-1], d_val

@njit(cache=True)
def _jit_adx_full(h, l, c, period):
    n = len(c)
    if n <= period: return 25.0, 25.0, 25.0
    
    tr = np.zeros(n, dtype=np.float64)
    pdm = np.zeros(n, dtype=np.float64)
    mdm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        hl = h[i] - l[i]
        hc = abs(h[i] - c[i-1])
        lc = abs(l[i] - c[i-1])
        tr[i] = hl
        if hc > tr[i]: tr[i] = hc
        if lc > tr[i]: tr[i] = lc
        
        up = h[i] - h[i-1]
        dn = l[i-1] - l[i]
        
        if up > dn and up > 0: pdm[i] = up
        else: pdm[i] = 0
            
        if dn > up and dn > 0: mdm[i] = dn
        else: mdm[i] = 0
        
    atr = _jit_ema_series(tr, period)
    pdi = np.zeros(n, dtype=np.float64)
    mdi = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if atr[i] == 0:
            pdi[i] = 0
            mdi[i] = 0
        else:
            pdi[i] = 100.0 * _jit_ema_series(pdm, period)[i] / atr[i]
            mdi[i] = 100.0 * _jit_ema_series(mdm, period)[i] / atr[i]
            
    dx = np.zeros(n, dtype=np.float64)
    for i in range(n):
        den = pdi[i] + mdi[i]
        if den == 0: dx[i] = 0
        else: dx[i] = 100.0 * abs(pdi[i] - mdi[i]) / den
        
    adx_series = _jit_ema_series(dx, period)
    return adx_series[-1], pdi[-1], mdi[-1]

@njit(cache=True)
def _jit_hurst(series, min_window):
    n = len(series)
    if n < min_window * 2: return 0.5
    
    max_chunk = n // 4
    if max_chunk < 2: return 0.5
    
    sizes = np.zeros(10, dtype=np.int64)
    for i in range(10):
        s = min_window + int((max_chunk - min_window) * i / 9)
        sizes[i] = s
        
    rs_vals = np.zeros(10, dtype=np.float64)
    log_sizes = np.zeros(10, dtype=np.float64)
    log_rs = np.zeros(10, dtype=np.float64)
    
    valid_count = 0
    for i in range(10):
        size = sizes[i]
        if size < 2: continue
        
        num_chunks = n // size
        if num_chunks == 0: continue
        
        rs_sum = 0.0
        for c in range(num_chunks):
            chunk = series[c*size:(c+1)*size]
            mean = 0.0
            for x in chunk: mean += x
            mean /= size
            
            y = np.zeros(size, dtype=np.float64)
            y[0] = chunk[0] - mean
            for j in range(1, size):
                y[j] = y[j-1] + (chunk[j] - mean)
                
            y_max = y[0]
            y_min = y[0]
            for val in y:
                if val > y_max: y_max = val
                if val < y_min: y_min = val
            R = y_max - y_min
            
            var = 0.0
            for x in chunk: var += (x - mean)**2
            var /= size
            S = var**0.5
            
            if S > 0:
                rs_sum += R / S
                
        if rs_sum > 0:
            avg_rs = rs_sum / num_chunks
            rs_vals[valid_count] = avg_rs
            log_sizes[valid_count] = np.log(size)
            log_rs[valid_count] = np.log(avg_rs)
            valid_count += 1
            
    if valid_count < 3: return 0.5
    
    # Simple linear regression
    sum_x = 0.0
    sum_y = 0.0
    for i in range(valid_count):
        sum_x += log_sizes[i]
        sum_y += log_rs[i]
    mean_x = sum_x / valid_count
    mean_y = sum_y / valid_count
    
    num = 0.0
    den = 0.0
    for i in range(valid_count):
        num += (log_sizes[i] - mean_x) * (log_rs[i] - mean_y)
        den += (log_sizes[i] - mean_x)**2
        
    if den == 0: return 0.5
    h = num / den
    
    if h < 0: return 0.0
    if h > 1: return 1.0
    return h

@njit(cache=True)
def _jit_sample_entropy(series, m, r_mult):
    n = len(series)
    if n < m + 2: return 0.0
    
    mean = 0.0
    for x in series: mean += x
    mean /= n
    
    var = 0.0
    for x in series: var += (x - mean)**2
    var /= n
    r = r_mult * (var**0.5)
    
    def _count(m_len):
        count = 0
        total = 0
        for i in range(n - m_len):
            for j in range(i + 1, n - m_len):
                match = True
                for k in range(m_len):
                    if abs(series[i+k] - series[j+k]) > r:
                        match = False
                        break
                if match: count += 1
                total += 1
        return count, total
        
    c_m, t_m = _count(m)
    c_m1, t_m1 = _count(m + 1)
    
    if c_m == 0 or c_m1 == 0: return 0.0
    
    p_m = c_m / t_m
    p_m1 = c_m1 / t_m1
    
    return -np.log(p_m1 / p_m)

@njit(cache=True)
def _jit_transfer_entropy(s, t, bins):
    n = min(len(s), len(t))
    if n < 10: return 0.0
    
    s_min, s_max = s[0], s[0]
    t_min, t_max = t[0], t[0]
    for i in range(n):
        if s[i] < s_min: s_min = s[i]
        if s[i] > s_max: s_max = s[i]
        if t[i] < t_min: t_min = t[i]
        if t[i] > t_max: t_max = t[i]
        
    s_range = max(1e-10, s_max - s_min)
    t_range = max(1e-10, t_max - t_min)
    
    s_hist = np.zeros(bins, dtype=np.int32)
    t_hist = np.zeros(bins, dtype=np.int32)
    for i in range(n):
        s_bin = min(bins - 1, int((s[i] - s_min) / s_range * bins))
        t_bin = min(bins - 1, int((t[i] - t_min) / t_range * bins))
        s_hist[s_bin] += 1
        t_hist[t_bin] += 1
        
    return 0.5 

@njit(cache=True)
def _jit_kyle_lambda(r, v, window):
    """Kyle (1985): lambda = Cov(ΔP, signed_vol) / Var(signed_vol).
    signed_vol = +v if return >= 0 (buy pressure), -v if return < 0 (sell pressure).
    Measures price impact per unit of order flow (market depth / illiquidity).
    """
    if len(r) < window or len(v) < window: return 0.0

    r_w = r[-window:]
    v_w = v[-window:]

    # Build signed volume from return direction
    sv = np.empty(window, dtype=np.float64)
    for i in range(window):
        sv[i] = v_w[i] if r_w[i] >= 0.0 else -v_w[i]

    # Compute means
    r_mean = 0.0
    sv_mean = 0.0
    for i in range(window):
        r_mean += r_w[i]
        sv_mean += sv[i]
    r_mean /= window
    sv_mean /= window

    # Cov(r, sv) / Var(sv) — OLS beta of returns on signed volume
    cov = 0.0
    var_sv = 0.0
    for i in range(window):
        ds = sv[i] - sv_mean
        cov += (r_w[i] - r_mean) * ds
        var_sv += ds * ds

    if var_sv == 0.0: return 0.0
    return cov / var_sv

@njit(cache=True)
def _jit_amihud(r, v, window):
    if len(r) < window or len(v) < window: return 0.0
    
    r_w = r[-window:]
    v_w = v[-window:]
    
    amihud_sum = 0.0
    valid = 0
    for i in range(window):
        dlr_vol = v_w[i] 
        if dlr_vol > 0:
            amihud_sum += abs(r_w[i]) / dlr_vol
            valid += 1
            
    if valid == 0: return 0.0
    return amihud_sum / valid

@njit(cache=True)
def _jit_mf_dfa_width(series, q_min, q_max, min_window):
    n = len(series)
    if n < min_window * 4: return 0.0
    
    s1 = min_window
    var_s1 = 0.0
    for i in range(0, n - s1, s1):
        chunk = series[i:i+s1]
        mean = 0.0
        for j in range(s1): 
            mean += chunk[j]
        mean /= s1
        for j in range(s1): 
            var_s1 += abs(chunk[j] - mean)**min(abs(q_max), 2.0)
    var_s1 /= max(1, (n // s1))
    
    s2 = min_window * 4
    if s2 > n: 
        s2 = max(1, n // 2)
    var_s2 = 0.0
    for i in range(0, n - s2, s2):
        chunk = series[i:i+s2]
        mean = 0.0
        for j in range(s2): 
            mean += chunk[j]
        mean /= s2
        for j in range(s2): 
            var_s2 += abs(chunk[j] - mean)**min(abs(q_max), 2.0)
    var_s2 /= max(1, (n // s2))
    
    if var_s1 < 1e-10 or var_s2 < 1e-10:
        return 0.0
        
    width = abs(np.log(var_s2 / var_s1)) / np.log(max(2.0, s2 / s1))
    return min(1.0, width * 0.1)

# =================== SERIES JIT KERNELS (pre-compute once, index at events) ===================

@njit(cache=True)
def _jit_rsi_series(arr, period):
    """Returns RSI at every position as a full array."""
    n = len(arr)
    result = np.full(n, 50.0)
    if n <= period:
        return result
    avg_gain = 0.0
    avg_loss = 0.0
    for i in range(1, period + 1):
        diff = arr[i] - arr[i - 1]
        if diff > 0: avg_gain += diff
        else: avg_loss -= diff
    avg_gain /= period
    avg_loss /= period
    if avg_loss == 0:
        result[period] = 100.0
    else:
        result[period] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
    for i in range(period + 1, n):
        diff = arr[i] - arr[i - 1]
        gain = diff if diff > 0 else 0.0
        loss = -diff if diff < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        if avg_loss == 0:
            result[i] = 100.0
        else:
            result[i] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
    return result

@njit(cache=True)
def _jit_rolling_mean(arr, period):
    """Rolling SMA as full array."""
    n = len(arr)
    result = np.zeros(n, dtype=np.float64)
    if n == 0: return result
    # Expanding mean for warmup
    s = 0.0
    for i in range(min(period, n)):
        s += arr[i]
        result[i] = s / (i + 1)
    # Rolling window
    if n > period:
        window_sum = s
        for i in range(period, n):
            window_sum += arr[i] - arr[i - period]
            result[i] = window_sum / period
    return result

@njit(cache=True)
def _jit_rolling_std(arr, period):
    """Rolling standard deviation as full array."""
    n = len(arr)
    result = np.zeros(n, dtype=np.float64)
    if n < 2: return result
    for i in range(period - 1, n):
        start = max(0, i - period + 1)
        window = arr[start:i + 1]
        wn = len(window)
        mean = 0.0
        for x in window: mean += x
        mean /= wn
        var = 0.0
        for x in window: var += (x - mean) ** 2
        result[i] = (var / wn) ** 0.5
    return result

@njit(cache=True)
def _jit_adx_series(h, l, c, period):
    """Returns ADX at every position as a full array."""
    n = len(c)
    adx_out = np.full(n, 25.0)
    if n <= period: return adx_out
    tr = np.zeros(n, dtype=np.float64)
    pdm = np.zeros(n, dtype=np.float64)
    mdm = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        hl = h[i] - l[i]
        hc = abs(h[i] - c[i-1])
        lc = abs(l[i] - c[i-1])
        tr[i] = hl
        if hc > tr[i]: tr[i] = hc
        if lc > tr[i]: tr[i] = lc
        up = h[i] - h[i-1]
        dn = l[i-1] - l[i]
        if up > dn and up > 0: pdm[i] = up
        if dn > up and dn > 0: mdm[i] = dn
    atr = _jit_ema_series(tr, period)
    smooth_pdm = _jit_ema_series(pdm, period)
    smooth_mdm = _jit_ema_series(mdm, period)
    dx = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if atr[i] > 0:
            pdi = 100.0 * smooth_pdm[i] / atr[i]
            mdi = 100.0 * smooth_mdm[i] / atr[i]
            den = pdi + mdi
            if den > 0:
                dx[i] = 100.0 * abs(pdi - mdi) / den
    adx_out = _jit_ema_series(dx, period)
    return adx_out

# =================== BLACK-SCHOLES BARRIER MATH (Hull Ch.26, Darling-Siegert 1953) ===================

@njit(cache=True)
def _jit_kou_barrier_prob(log_returns, tp_dist, sl_dist, drift_window=20, vol_window=20):
    """
    P(TP hit before SL) under Kou (2002) Double-Exponential Jump Diffusion.
    Captures crypto flash crashes via asymmetric jumps.
    Approximates double-barrier first passage using combined GBM + Jump adjustments.
    """
    n = len(log_returns)
    if n < 3 or tp_dist <= 0 or sl_dist <= 0:
        return 0.5

    # Multi-scale drift
    w_fast = min(drift_window, n)
    mu_fast = 0.0
    for i in range(n - w_fast, n):
        mu_fast += log_returns[i]
    mu_fast /= w_fast

    w_slow = min(100, n)
    mu_slow = 0.0
    for i in range(n - w_slow, n):
        mu_slow += log_returns[i]
    mu_slow /= w_slow

    mu = 0.6 * mu_fast + 0.4 * mu_slow

    # GARCH(1,1) Volatility
    w2 = min(vol_window, n)
    mean_r = 0.0
    for i in range(n - w2, n):
        mean_r += log_returns[i]
    mean_r /= w2

    var_sum = 0.0
    for i in range(n - w2, n):
        diff = log_returns[i] - mean_r
        var_sum += diff * diff
    sample_var = var_sum / max(1, w2 - 1)

    _omega = 1e-6
    _alpha = 0.10
    _beta  = 0.85
    garch_var = sample_var
    for i in range(n - w2, n):
        r_sq = (log_returns[i] - mean_r) ** 2
        garch_var = _omega + _alpha * r_sq + _beta * garch_var

    sigma = np.sqrt(max(garch_var, 1e-20))

    if sigma < 1e-12:
        return 0.5

    # Kou Model Parameters (Crypto calibrated)
    # λ (jumps per bar), η1 (up-jump mean), η2 (down-jump mean), p (up-jump probability)
    lambda_j = 0.5  
    eta1 = 15.0     
    eta2 = 10.0     
    p_up = 0.4      
    q_down = 1.0 - p_up

    # Adjust drift for jump compensator: E[exp(J)-1]
    # mean jump term = p * (eta1 / (eta1 - 1)) + q * (eta2 / (eta2 + 1)) - 1
    # For small jumps, E[J] ≈ p/eta1 - q/eta2
    # Compensator in log-space (we assume dS/S has jumps, so log returns have log(1+J) jumps)
    # E[J_log] approx:
    expected_log_jump = p_up * (1.0 / eta1) - q_down * (1.0 / eta2)
    
    # nu is Itô-drift adjusted for continuous part
    # Total mean return mu = nu + lambda_j * expected_log_jump
    # So nu = mu - lambda_j * expected_log_jump - 0.5 * sigma^2
    nu = mu - lambda_j * expected_log_jump - 0.5 * sigma * sigma

    # To approximate the first passage in double-barrier for jump diffusions,
    # we use the Kou & Wang (2003) phase-type approximation or an effective characteristic exponent.
    # A simple analytical heuristic is to form an "effective" diffusion:
    # Var[J] = p / eta1^2 + q / eta2^2 + p*q*(1/eta1 + 1/eta2)^2
    var_jump = p_up * (1.0/(eta1*eta1)) + q_down * (1.0/(eta2*eta2)) + p_up*q_down * (1.0/eta1 + 1.0/eta2)**2
    sigma_diff_sq = sigma * sigma + lambda_j * var_jump
    sigma_eff = np.sqrt(sigma_diff_sq)

    # Use the scale function approach with effective continuous drift and variance
    # modified by the asymmetric jump pull
    # Effective drift for scale function:
    mu_eff = nu + lambda_j * expected_log_jump

    if abs(mu_eff) < 1e-10:
        return sl_dist / (tp_dist + sl_dist)

    alpha = 2.0 * mu_eff / sigma_diff_sq
    
    # Scale function P = [1 - exp(alp * sl)] / [exp(-alp * tp) - exp(alp * sl)]
    exp_sl_arg = max(-500.0, min(500.0, alpha * sl_dist))
    exp_tp_arg = max(-500.0, min(500.0, -alpha * tp_dist))

    num = 1.0 - np.exp(exp_sl_arg)
    den = np.exp(exp_tp_arg) - np.exp(exp_sl_arg)

    if abs(den) < 1e-15:
        return 0.5

    p = num / den
    return max(0.0, min(1.0, p))


@njit(cache=True)
def _jit_bs_time_decay(sigma, tp_dist, bars_remaining, dt=1.0):
    """
    Approximate P(crossing TP barrier within remaining bars).

    Uses single-barrier normal CDF proxy (not exact double-barrier Kunitomo-Ikeda
    infinite series). CatBoost learns the residual.

    Normal CDF approximation: 0.5 * (1 + tanh(z * 0.79788456))
    (same pattern as VPIN in _jit_vpin, Easley et al. 2012).
    """
    if sigma < 1e-12 or bars_remaining <= 0 or tp_dist <= 0:
        return 0.0

    z = tp_dist / (sigma * np.sqrt(bars_remaining * dt))
    # P(|W| > z) = 2 * (1 - Phi(z))
    phi_z = 0.5 * (1.0 + np.tanh(z * 0.79788456))
    p_cross = 2.0 * (1.0 - phi_z)

    return max(0.0, min(1.0, p_cross))


@njit(cache=True)
def _jit_bs_implied_vol_ratio(avg_bars_to_hit, barrier_dist, sigma_realized, dt=1.0):
    """
    QUANTA-specific heuristic: back-solve sigma from observed time-to-hit.

    sigma_implied = barrier_dist / sqrt(avg_bars_to_hit * dt)
    ratio = sigma_implied / sigma_realized

    If barriers hit faster than realized vol predicts → ratio > 1 → jump dynamics.
    If barriers hit slower → ratio < 1 → mean-reverting regime.
    """
    if avg_bars_to_hit <= 0 or barrier_dist <= 0 or sigma_realized < 1e-12:
        return 1.0  # neutral

    sigma_implied = barrier_dist / np.sqrt(avg_bars_to_hit * dt)
    ratio = sigma_implied / sigma_realized

    return max(0.1, min(10.0, ratio))


@njit(cache=True)
def _jit_kou_conditional_first_passage(log_returns, tp_dist, sl_dist, max_bars):
    """
    Fast Nike-oriented conditional post-jump approximation.

    Assumes the trigger jump already happened and the process starts close to the
    upper barrier, so we bias toward the one-sided remaining upside distance while
    preserving a finite-horizon penalty through the time-decay term outside.
    """
    if len(log_returns) < 5:
        return 0.5
    base_prob = _jit_kou_barrier_prob(log_returns, tp_dist, sl_dist)
    proximity_boost = sl_dist / max(tp_dist + sl_dist, 1e-12)
    horizon_penalty = 1.0 - np.exp(-max(1.0, float(max_bars)) / 24.0)
    p = 0.55 * base_prob + 0.45 * proximity_boost
    p *= horizon_penalty
    return max(0.0, min(1.0, p))


def compute_live_kou_barrier_components(
    log_returns,
    tp_dist,
    sl_dist,
    max_bars,
    direction="BULLISH",
    conditional_jump=False,
    specialist=None,
):
    """
    Live barrier probability helper used by QUANTA_bot execution.

    Returns a compact context with:
    - prob: finite-horizon TP-before-SL probability
    - order_prob: same live probability for order sizing
    - time_prob: horizon-only crossing proxy
    - sigma_eff: realized sigma from recent log returns
    - tp_dist_live / sl_dist_live / bars_live
    - conditional_jump / source / baseline / specialist
    """
    try:
        arr = np.asarray(log_returns, dtype=np.float64)
    except Exception:
        arr = np.zeros(0, dtype=np.float64)

    tp_dist = float(max(tp_dist, 1e-8))
    sl_dist = float(max(sl_dist, 1e-8))
    bars_live = int(max(1, max_bars))
    baseline = float(sl_dist / max(tp_dist + sl_dist, 1e-12))

    if arr.size < 5:
        return {
            "prob": baseline,
            "order_prob": baseline,
            "time_prob": 0.0,
            "sigma_eff": 0.0,
            "tp_dist_live": tp_dist,
            "sl_dist_live": sl_dist,
            "bars_live": bars_live,
            "conditional_jump": bool(conditional_jump),
            "source": "live_fallback",
            "baseline": baseline,
            "specialist": specialist,
        }

    sigma_eff = float(np.std(arr))
    time_prob = float(_jit_bs_time_decay(sigma_eff, tp_dist, bars_live))

    if conditional_jump:
        base_prob = float(_jit_kou_conditional_first_passage(arr, tp_dist, sl_dist, bars_live))
        source = "live_nike_conditional"
    else:
        base_prob = float(_jit_kou_barrier_prob(arr, tp_dist, sl_dist))
        source = "live_kou"

    prob = float(max(0.0, min(1.0, 0.7 * base_prob + 0.3 * time_prob)))

    if str(direction).upper() == "BEARISH":
        prob = 1.0 - prob

    return {
        "prob": float(max(0.0, min(1.0, prob))),
        "order_prob": float(max(0.0, min(1.0, prob))),
        "time_prob": float(max(0.0, min(1.0, time_prob))),
        "sigma_eff": sigma_eff,
        "tp_dist_live": tp_dist,
        "sl_dist_live": sl_dist,
        "bars_live": bars_live,
        "conditional_jump": bool(conditional_jump),
        "source": source,
        "baseline": baseline,
        "specialist": specialist,
    }


# =================== INDICATORS (OPTIMIZED WITH CACHE) ===================
class Indicators:
    """
    Technical indicators with silent numpy error handling.
    All numpy warnings are suppressed to avoid console spam.
    """
    
    @staticmethod
    def _safe_mean(arr):
        """Safe mean calculation that never warns"""
        if len(arr) == 0:
            return 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = np.mean(arr)
            return 0 if np.isnan(result) or np.isinf(result) else result
    
    @staticmethod
    def _safe_std(arr):
        """Safe std calculation that never warns"""
        if len(arr) == 0:
            return 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = np.std(arr)
            return 0 if np.isnan(result) or np.isinf(result) else result
            
    @staticmethod
    def frac_diff(series, d=0.4, min_weight=1e-4):
        """Fractional differencing (Lopez de Prado AFML Ch.5) to maintain memory and stationarity"""
        try:
            arr = np.array(series, dtype=np.float64)
            return float(_jit_frac_diff(arr, float(d), float(min_weight)))
        except Exception:
            return 0.0
    
    # Cache per-asset optimal d to avoid recomputation
    _optimal_d_cache = {}
    
    @staticmethod
    def find_optimal_d(series, max_d=1.0, d_step=0.1, p_threshold=0.05):
        """Find minimum fractional differencing order for stationarity (AFML Ch.5).
        
        Searches for smallest d where ADF test rejects unit root null hypothesis.
        Lower d preserves more memory (predictive power) in the series.
        """
        try:
            from statsmodels.tsa.stattools import adfuller
        except ImportError:
            return 0.4  # Fallback if statsmodels not available
        
        series_arr = np.array(series, dtype=np.float64)
        if len(series_arr) < 50:
            return 0.4
        
        for d in np.arange(0.0, max_d + d_step, d_step):
            try:
                diffed = _jit_frac_diff(series_arr, float(d), 1e-4)
                if np.isnan(diffed) or np.isinf(diffed):
                    continue
                # Need array of diffed values for ADF — apply to windowed returns
                diffed_series = np.array([_jit_frac_diff(series_arr[:i+1], float(d), 1e-4) 
                                          for i in range(max(20, len(series_arr)-100), len(series_arr))])
                valid = diffed_series[np.isfinite(diffed_series)]
                if len(valid) < 20:
                    continue
                adf_stat, p_value, *_ = adfuller(valid, maxlag=1, regression='c')
                if p_value < p_threshold:
                    return round(d, 2)
            except Exception:
                continue
        return 0.4  # Fallback
    
    @staticmethod
    def frac_diff_adaptive(series, symbol=None, min_weight=1e-4):
        """Fractional differencing with per-asset optimal d (AFML Ch.5).
        
        Caches the optimal d per symbol to avoid recomputation.
        Falls back to d=0.4 if ADF search fails.
        """
        if symbol and symbol in Indicators._optimal_d_cache:
            d = Indicators._optimal_d_cache[symbol]
        else:
            d = Indicators.find_optimal_d(series)
            if symbol:
                Indicators._optimal_d_cache[symbol] = d
        return Indicators.frac_diff(series, d=d, min_weight=min_weight)
        
    @staticmethod
    def vpin(highs, lows, closes, volumes, window=14, taker_buy=None):
        """Volume-Synchronized Probability of Informed Trading (Easley et al. 2012).

        If taker_buy is provided (Binance klines col 9), uses REAL buy/sell split.
        Otherwise falls back to BVC proxy (close position in H-L range).
        """
        try:
            if taker_buy is not None:
                v = np.array(volumes, dtype=np.float64)
                tb = np.array(taker_buy, dtype=np.float64)
                return float(_jit_vpin_taker(v, tb, int(window)))
            h = np.array(highs, dtype=np.float64)
            l = np.array(lows, dtype=np.float64)
            c = np.array(closes, dtype=np.float64)
            v = np.array(volumes, dtype=np.float64)
            return float(_jit_vpin(h, l, c, v, int(window)))
        except Exception:
            return 0.5
    
    @staticmethod
    def rsi(closes, period=14):
        try:
            arr = np.asarray(closes, dtype=np.float64)
            return float(_jit_rsi(arr, int(period)))
        except Exception:
            return 50.0

    @staticmethod
    def macd(closes, fast=12, slow=26, signal=9):
        try:
            arr = np.asarray(closes, dtype=np.float64)
            m, s, h = _jit_macd(arr, int(fast), int(slow), int(signal))
            return float(m), float(s), float(h)
        except Exception:
            return 0.0, 0.0, 0.0

    @staticmethod
    def ema(data, period):
        series = Indicators.ema_series(data, period)
        return float(series[-1]) if len(series) > 0 else 0.0

    @staticmethod
    def ema_series(data, period):
        try:
            arr = np.asarray(data, dtype=np.float64)
            return _jit_ema_series(arr, int(period))
        except Exception:
            return np.zeros(len(data), dtype=np.float64)

    @staticmethod
    def bollinger(closes, period=20, std_dev=2.0):
        try:
            arr = np.asarray(closes, dtype=np.float64)
            u, m, l = _jit_bollinger(arr, int(period), float(std_dev))
            return float(u), float(m), float(l)
        except Exception:
            return 0.0, 0.0, 0.0

    @staticmethod
    def atr_series(highs, lows, closes, period=14):
        try:
            h = np.asarray(highs, dtype=np.float64)
            l = np.asarray(lows, dtype=np.float64)
            c = np.asarray(closes, dtype=np.float64)
            return _jit_atr_series(h, l, c, int(period))
        except Exception:
            return np.zeros(len(highs), dtype=np.float64)

    @staticmethod
    def atr(highs, lows, closes, period=14):
        series = Indicators.atr_series(highs, lows, closes, period)
        return float(series[-1]) if len(series) > 0 else 0.0

    @staticmethod
    def stochastic(highs, lows, closes, period=14, d_period=3):
        try:
            h = np.asarray(highs, dtype=np.float64)
            l = np.asarray(lows, dtype=np.float64)
            c = np.asarray(closes, dtype=np.float64)
            k, d = _jit_stochastic(h, l, c, int(period), int(d_period))
            return float(k), float(d)
        except Exception:
            return 50.0, 50.0

    @staticmethod
    def adx(highs, lows, closes, period=14):
        adx_val, _, _ = Indicators.adx_full(highs, lows, closes, period)
        return float(adx_val)

    @staticmethod
    def adx_full(highs, lows, closes, period=14):
        try:
            h = np.asarray(highs, dtype=np.float64)
            l = np.asarray(lows, dtype=np.float64)
            c = np.asarray(closes, dtype=np.float64)
            adx_val, pdi, mdi = _jit_adx_full(h, l, c, int(period))
            return float(adx_val), float(pdi), float(mdi)
        except Exception:
            return 25.0, 25.0, 25.0

    @staticmethod
    def hurst(series, min_window=20):
        try:
            arr = np.asarray(series, dtype=np.float64)
            return float(_jit_hurst(arr, int(min_window)))
        except Exception:
            return 0.5
            
    @staticmethod
    def sample_entropy(series, m=2, r_mult=0.2):
        try:
            arr = np.asarray(series, dtype=np.float64)
            return float(_jit_sample_entropy(arr, int(m), float(r_mult)))
        except Exception:
            return 0.0
            
    @staticmethod
    def transfer_entropy(source_returns, target_returns, bins=8):
        try:
            s_arr = np.asarray(source_returns, dtype=np.float64)
            t_arr = np.asarray(target_returns, dtype=np.float64)
            return float(_jit_transfer_entropy(s_arr, t_arr, int(bins)))
        except Exception:
            return 0.0
            
    @staticmethod
    def kyle_lambda(returns, volumes, window=20):
        try:
            r_arr = np.asarray(returns, dtype=np.float64)
            v_arr = np.asarray(volumes, dtype=np.float64)
            return float(_jit_kyle_lambda(r_arr, v_arr, int(window)))
        except Exception:
            return 0.0
            
    @staticmethod
    def amihud(returns, volumes, window=20):
        try:
            r_arr = np.asarray(returns, dtype=np.float64)
            v_arr = np.asarray(volumes, dtype=np.float64)
            return float(_jit_amihud(r_arr, v_arr, int(window)))
        except Exception:
            return 0.0
            
    @staticmethod
    def mf_dfa_width(series, q_min=-2.0, q_max=2.0, min_window=10):
        try:
            arr = np.asarray(series, dtype=np.float64)
            return float(_jit_mf_dfa_width(arr, float(q_min), float(q_max), int(min_window)))
        except Exception:
            return 0.0

# =================== MULTI-TIMEFRAME ANALYZER (OPTIMIZED) ===================
class MultiTimeframeAnalyzer:
    def __init__(self, cfg, bnc):
        self.cfg = cfg
        self.bnc = bnc
        self.analysis_cache = {}
        self.cache_duration = 30  # seconds

    def analyze(self, symbol):
        # Check cache first - FIX: ensure timestamp is int, not float
        cache_key = f"{symbol}_{int(time.time() // self.cache_duration)}"
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        results = {}
        
        # PARALLEL FETCH - Fetch all timeframes at once instead of sequentially
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def fetch_and_analyze_tf(tf):
            try:
                # Use cache-aware kline fetching
                if hasattr(self.bnc, 'candle_store') and self.bnc.candle_store:
                    klines = self.bnc.candle_store.get(symbol, tf)
                else:
                    klines = None
                    
                if not klines or len(klines) < 50:
                    klines = self.bnc.get_klines(symbol, tf, limit=200)
                if not klines or len(klines) < 50:
                    return None
                
                closes = [float(k[4]) for k in klines]
                highs = [float(k[2]) for k in klines]
                lows = [float(k[3]) for k in klines]
                volumes = [float(k[5]) for k in klines]
                
                rsi = Indicators.rsi(closes, self.cfg.rsi_period)
                macd_line, signal_line, histogram = Indicators.macd(closes)
                bb_upper, bb_middle, bb_lower = Indicators.bollinger(closes)
                atr = Indicators.atr(highs, lows, closes)
                k, d = Indicators.stochastic(highs, lows, closes)
                adx = Indicators.adx(highs, lows, closes)
                
                price = closes[-1]
                ma_short = np.mean(closes[-self.cfg.ma_short:])
                ma_long = np.mean(closes[-self.cfg.ma_long:])
                
                trend_score = 0
                
                if rsi > 70:
                    trend_score += 20
                elif rsi < 30:
                    trend_score -= 20
                elif rsi > 50:
                    trend_score += 10
                else:
                    trend_score -= 10
                
                if macd_line > signal_line and histogram > 0:
                    trend_score += 20
                elif macd_line < signal_line and histogram < 0:
                    trend_score -= 20
                
                if price > ma_short > ma_long:
                    trend_score += 20
                elif price < ma_short < ma_long:
                    trend_score -= 20
                
                bb_position = (price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
                if bb_position > 0.8:
                    trend_score += 10
                elif bb_position < 0.2:
                    trend_score -= 10
                
                if k > 80 and d > 80:
                    trend_score += 10
                elif k < 20 and d < 20:
                    trend_score -= 10
                
                strength = min(100, abs(trend_score) * (adx / 25))
                
                if trend_score > 30:
                    trend = 'BULLISH'
                elif trend_score < -30:
                    trend = 'BEARISH'
                else:
                    trend = 'NEUTRAL'
                
                # === ENHANCED FEATURES ===
                # Volatility regime
                returns = np.diff(closes) / closes[:-1]
                volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0
                # 75th percentile ATR context
                atr_s = Indicators.atr_series(highs, lows, closes)
                recent_atrs = atr_s[14::5]
                atr_percentile = float(np.sum(recent_atrs <= atr) / max(1, len(recent_atrs))) if len(closes) >= 100 else 0.5
                atr_prev = atr_s[-10] if len(atr_s) >= 10 else atr_s[0]
                volatility_accel = (atr - atr_prev) / atr if atr > 0 else 0.0
                
                # Multi-window momentum
                mom_5 = (closes[-1] / closes[-6] - 1) if len(closes) >= 6 else 0
                mom_10 = (closes[-1] / closes[-11] - 1) if len(closes) >= 11 else 0
                mom_20 = (closes[-1] / closes[-21] - 1) if len(closes) >= 21 else 0
                mom_50 = (closes[-1] / closes[-51] - 1) if len(closes) >= 51 else 0
                
                # Volume analysis
                vol_ma = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
                volume_ratio = volumes[-1] / vol_ma if vol_ma > 0 else 1
                
                # Drift detection
                ma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else closes[-1]
                ma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else closes[-1]
                mean_shift = ma_20 - ma_50
                trend_strength = abs(mean_shift) / atr if atr > 0 else 0
                
                # Multi-timeframe return
                returns_period = (closes[-1] / closes[0] - 1) if len(closes) > 0 else 0
                
                # === STATIONARITY & ORDER FLOW ===
                # Option A: VPIN (Order Flow Toxicity)
                vpin_val = Indicators.vpin(highs, lows, closes, volumes, 14)
                
                # Option B: Taker Flow Imbalance (Proxy for LOB aggressiveness)
                # kline index 5 = total vol, 9 = taker buy volume
                taker_buy_vol = sum([float(k[9]) for k in klines[-10:]])
                total_vol_recent = sum([float(k[5]) for k in klines[-10:]])
                taker_sell_vol = total_vol_recent - taker_buy_vol
                taker_imbalance = (taker_buy_vol - taker_sell_vol) / (total_vol_recent + 1e-8)
                
                # Option C: Fractional Differencing (Stationary Prices)
                fd_val = Indicators.frac_diff(closes, d=0.4)
                
                return (tf, {
                    'trend': trend,
                    'strength': int(strength),
                    'price': price,
                    'rsi': rsi,
                    'macd': histogram,
                    'bb_position': bb_position,
                    'adx': adx,
                    'volume': volumes[-1],
                    'atr': atr,
                    'symbol': symbol,
                    # Enhanced features
                    'volatility': volatility,
                    'atr_percentile': atr_percentile,
                    'volatility_accel': volatility_accel,
                    'momentum_5': mom_5,
                    'momentum_10': mom_10,
                    'momentum_20': mom_20,
                    'momentum_50': mom_50,
                    'volume_ratio': volume_ratio,
                    'mean_shift': mean_shift,
                    'trend_strength': trend_strength,
                    'returns_period': returns_period,
                    'vpin': vpin_val,
                    'taker_imbalance': taker_imbalance,
                    'frac_diff': fd_val
                })
                
            except Exception as e:
                logging.error(f"MTF {symbol} {tf}: {e}")
                return None
        
        # Fetch all timeframes in parallel (7 TFs = 7x speedup!)
        with ThreadPoolExecutor(max_workers=len(self.cfg.timeframes)) as executor:
            futures = {executor.submit(fetch_and_analyze_tf, tf): tf for tf in self.cfg.timeframes}
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    tf, analysis = result
                    results[tf] = analysis
        
        # Cache results
        self.analysis_cache[cache_key] = results
        
        # Clean old cache entries - FIX: handle float timestamps
        current_time_key = time.time() // self.cache_duration
        keys_to_delete = [k for k in self.analysis_cache.keys() 
                         if int(float(k.rsplit('_', 1)[-1])) < current_time_key - 10]  # FIXED: rsplit for symbol safety
        for k in keys_to_delete:
            del self.analysis_cache[k]
        
        return results

