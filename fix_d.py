import os

with open('quanta_features.py', 'rb') as f:
    feat = f.read().decode('utf-8')

KOU_FUNC = r'''
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
'''

# Find the start and end of _jit_bs_barrier_prob
start_idx = feat.find('@njit(cache=True)\r\ndef _jit_bs_barrier_prob(')
if start_idx == -1:
    start_idx = feat.find('@njit(cache=True)\ndef _jit_bs_barrier_prob(')
    
if start_idx != -1:
    # Find the next function definition
    next_func_idx = feat.find('@njit(', start_idx + 10)
    
    # Replace it
    feat = feat[:start_idx] + KOU_FUNC.strip() + '\n\n\n' + feat[next_func_idx:]
    print("Replaced _jit_bs_barrier_prob with _jit_kou_barrier_prob")
else:
    print("Could not find _jit_bs_barrier_prob in quanta_features.py")

with open('quanta_features.py', 'wb') as f:
    f.write(feat.encode('utf-8'))

# Now update QUANTA_ml_engine.py
with open('QUANTA_ml_engine.py', 'rb') as f:
    ml = f.read().decode('utf-8')

# We need to replace all '_jit_bs_barrier_prob' with '_jit_kou_barrier_prob'
if '_jit_bs_barrier_prob' in ml:
    ml = ml.replace('_jit_bs_barrier_prob', '_jit_kou_barrier_prob')
    print("Updated _jit_bs_barrier_prob to _jit_kou_barrier_prob in ML engine")
    with open('QUANTA_ml_engine.py', 'wb') as f:
        f.write(ml.encode('utf-8'))
else:
    print("Could not find _jit_bs_barrier_prob in QUANTA_ml_engine.py")
