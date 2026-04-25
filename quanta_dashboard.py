"""
QUANTA Dashboard v2 - Local web interface on http://localhost:5000
Tabbed layout: Overview | Predictions | Models | System
"""

import json
import time
import os
import threading
import logging
from pathlib import Path
from collections import deque
from flask import Flask, render_template, jsonify, request
from dataclasses import asdict
from quanta_norse_agents import display_agent_name

app = Flask(__name__)
app.config['SECRET_KEY'] = 'quanta-dashboard-local'
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True

_bot = None
_log_buffer = deque(maxlen=100)  # Rolling log buffer


def _safe(obj, attr, default=None):
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default


def _get_overview():
    """Overview tab data: equity, drawdown, key metrics, active positions."""
    bot = _bot
    if bot is None:
        return {'status': 'offline'}

    now = time.time()
    uptime_s = now - getattr(bot, 'session_start_time', now)
    hours, rem = divmod(int(uptime_s), 3600)
    mins, secs = divmod(rem, 60)

    ml = _safe(bot, 'ml')
    generation = ml.model_generation if ml else 0
    is_trained = ml.is_trained if ml else False
    is_training = getattr(bot, '_is_training', False) or getattr(bot, 'is_retraining', threading.Event()).is_set()

    # RL stats
    rl_stats = {}
    rl_mem = _safe(bot, 'rl_memory')
    if rl_mem:
        try:
            rl_stats = rl_mem.get_stats()
        except Exception:
            pass

    # Performance
    perf = _safe(bot, 'perf_monitor')
    perf_data = {}
    if perf:
        try:
            stats = perf.get_stats()
            perf_data = {
                'coins_per_min': round(stats.get('coins_per_min', 0), 1),
                'total_processed': stats.get('total_coins', 0),
                'avg_fetch_ms': round(stats.get('avg_fetch_time', 0) * 1000, 1),
                'avg_compute_ms': round(stats.get('avg_compute_time', 0) * 1000, 1),
            }
        except Exception:
            pass

    # Fear & Greed
    sentiment = _safe(bot, 'sentiment')
    fg_data = {'value': '?', 'label': '?'}
    if sentiment:
        try:
            fg = sentiment.get_fear_greed()
            fg_data = {'value': fg.get('value', '?'), 'label': fg.get('value_classification', '?')}
        except Exception:
            pass

    # Paper trading - balance, PnL, positions
    paper = _safe(bot, 'paper')
    balance = 0
    initial_balance = 0
    total_pnl = 0
    active_positions = []
    legacy_active_positions = []
    trade_history = []
    if paper:
        balance = getattr(paper, 'balance', 0)
        initial_balance = getattr(paper, 'initial_balance', 0)
        total_pnl = getattr(paper, 'total_pnl', 0)
        total_trades = getattr(paper, 'total_trades', 0)
        total_wins = getattr(paper, 'total_wins', 0)
        total_losses = getattr(paper, 'total_losses', 0)
        # Pre-map daily picks to symbols to easily find TP/SL
        daily_picks = _safe(bot, 'daily_picks', {})
        symbol_to_pick = {}
        if daily_picks:
            for pid, p in daily_picks.items():
                symbol_to_pick[p.get('symbol')] = p

        positions = getattr(paper, 'positions', {})
        for sym, pos in positions.items():
            entry_price = pos.get('entry', 0)
            size = pos.get('size', 0)
            direction = pos.get('direction', '?')
            specialist = str(pos.get('specialist') or '').lower()
            display_agent = pos.get('display_agent') or display_agent_name(specialist or 'unknown')
            is_thor_executed = specialist == 'thor' and direction == 'BULLISH'
            execution_state = 'THOR EXECUTED' if is_thor_executed else 'LEGACY / NON-THOR'
            
            # For executed Thor positions, use the live paper-trading barriers
            # persisted on the position itself. daily_picks are alert-level
            # suggestions and do not reflect Thor v2's bank/trailing geometry.
            active_pick = symbol_to_pick.get(sym, {})
            tp1, tp2, tp3, sl = 0, 0, 0, 0

            if is_thor_executed:
                tp1 = pos.get('tp1_price', 0)
                tp2 = pos.get('tp2_price', 0)
                tp3 = pos.get('tp3_price', 0)
                sl = pos.get('sl_price', 0)
            elif active_pick:
                tp1 = active_pick.get('tp1', 0)
                tp2 = active_pick.get('tp2', 0)
                tp3 = active_pick.get('tp3', 0)
                sl = active_pick.get('sl', active_pick.get('stop_loss', 0))

            # Robust mathematical fallback if specific targets are missing
            atr_pct = pos.get('atr_percent') or 2.0  # Default to 2% ATR if metadata missing
            if entry_price > 0 and (tp1 == 0 or sl == 0):
                if direction == 'BULLISH':
                    tp1 = tp1 or (entry_price * (1 + (atr_pct * 1.5) / 100))
                    tp2 = tp2 or (entry_price * (1 + (atr_pct * 3.0) / 100))
                    tp3 = tp3 or (entry_price * (1 + (atr_pct * 4.5) / 100))
                    sl = sl or (entry_price * (1 - (atr_pct * 1.5) / 100))
                elif direction == 'BEARISH':
                    tp1 = tp1 or (entry_price * (1 - (atr_pct * 1.5) / 100))
                    tp2 = tp2 or (entry_price * (1 - (atr_pct * 3.0) / 100))
                    tp3 = tp3 or (entry_price * (1 - (atr_pct * 4.5) / 100))
                    sl = sl or (entry_price * (1 + (atr_pct * 1.5) / 100))

            current_price = 0
            try:
                if hasattr(bot, 'candle_store') and getattr(bot, 'candle_store') is not None:
                    current_price = getattr(bot.candle_store, 'last_price', lambda _sym: 0)(sym) or 0
            except Exception:
                current_price = 0
            if not current_price:
                try:
                    if hasattr(bot, 'bnc') and getattr(bot, 'bnc') is not None:
                        current_price = bot.bnc.get_ticker(sym) or 0
                except Exception:
                    current_price = 0

            pos_row = {
                'symbol': sym,
                'direction': direction,
                'executed_by': display_agent,
                'execution_state': execution_state,
                'is_thor_executed': is_thor_executed,
                'entry': entry_price,
                'size': size,
                'size_usd': entry_price * size,
                'confidence': round(pos.get('confidence', 0), 1),
                'specialist': specialist,
                'time': str(pos.get('time', '')),
                'tp1': tp1,
                'tp2': tp2,
                'tp3': tp3,
                'stop_loss': sl,
                'current_price': current_price,
                'thor_bank_hit': bool(pos.get('thor_bank_hit', False)),
                'thor_trail_active': bool(pos.get('thor_trail_active', False)),
                'runner_peak': pos.get('thor_runner_peak', 0),
                'bank_price': pos.get('thor_bank_price', tp1),
            }
            if is_thor_executed:
                active_positions.append(pos_row)
            else:
                legacy_active_positions.append(pos_row)
        history = getattr(paper, 'history', [])
        for t in history[-30:]:
            if isinstance(t, dict):
                trade = dict(t)
            elif hasattr(t, '__dict__'):
                try:
                    trade = asdict(t)
                except Exception:
                    continue
            else:
                continue
            hist_specialist = str(trade.get('specialist') or '').lower()
            hist_agent = trade.get('display_agent') or display_agent_name(hist_specialist or 'unknown')
            trade['executed_by'] = hist_agent
            trade['execution_state'] = 'THOR EXECUTED' if hist_specialist == 'thor' and trade.get('direction') == 'BULLISH' else 'RECORDED TRADE'
            trade_history.append(trade)

    # Equity curve from paper logger
    equity_curve = []
    paper_logger = getattr(paper, 'paper_logger', None) if paper else None
    if paper_logger:
        ec = getattr(paper_logger, '_equity_curve', [])
        for point in ec[-200:]:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                equity_curve.append({'t': point[0], 'v': point[1]})

    # Drawdown calculation
    max_balance = initial_balance
    current_dd = 0
    max_dd = 0
    if equity_curve:
        for pt in equity_curve:
            v = pt['v']
            if v > max_balance:
                max_balance = v
            dd = ((max_balance - v) / max_balance * 100) if max_balance > 0 else 0
            if dd > max_dd:
                max_dd = dd
        current_dd = ((max_balance - balance) / max_balance * 100) if max_balance > 0 else 0
    elif balance > 0 and initial_balance > 0:
        current_dd = max(0, ((initial_balance - balance) / initial_balance * 100))
        max_dd = current_dd

    # Risk manager status
    risk_status = {}
    risk_mgr = getattr(paper, 'risk_manager', None) if paper else None
    if risk_mgr:
        try:
            risk_status = risk_mgr.get_status() if hasattr(risk_mgr, 'get_status') else {}
        except Exception:
            pass
        if not risk_status:
            risk_status = {
                'trading_paused': getattr(risk_mgr, '_trading_paused', False),
                'daily_pnl': getattr(risk_mgr, '_daily_pnl', 0),
                'consecutive_losses': getattr(risk_mgr, '_consecutive_losses', 0),
                'daily_trades': getattr(risk_mgr, '_daily_trades', 0),
                'daily_wins': getattr(risk_mgr, '_daily_wins', 0),
                'daily_losses': getattr(risk_mgr, '_daily_losses', 0),
                'open_positions': len(getattr(risk_mgr, '_open_positions', {})),
                'size_multiplier': getattr(risk_mgr, 'get_size_multiplier', lambda: 1.0)(),
            }

    # PPO summary
    ppo_vetoes = getattr(bot, 'ppo_vetoes', []) if hasattr(bot, 'ppo_vetoes') else []
    ppo_summary = {'total': 0, 'saved': 0, 'missed': 0, 'pending': 0}
    if ppo_vetoes:
        ppo_summary['total'] = len(ppo_vetoes)
        ppo_summary['saved'] = sum(1 for v in ppo_vetoes if v.get('status') == 'SAVED FROM LOSS')
        ppo_summary['missed'] = sum(1 for v in ppo_vetoes if v.get('status') == 'MISSED PROFIT')
        ppo_summary['pending'] = sum(1 for v in ppo_vetoes if v.get('status') == 'PENDING')

    # PPO gate quality (CPCV-style rolling validation)
    ppo_accuracy = None
    ppo_brier = None
    ppo_gate = getattr(bot, '_ppo_gate_accuracy', deque())
    ppo_brier_q = getattr(bot, '_ppo_gate_brier', deque())
    if len(ppo_gate) >= 10:
        ppo_accuracy = round(sum(ppo_gate) / len(ppo_gate) * 100, 1)
    if len(ppo_brier_q) >= 10:
        ppo_brier = round(sum(ppo_brier_q) / len(ppo_brier_q), 3)

    # Drift alerts
    drift_alerts = []
    monitor = _safe(bot, 'monitor')
    if monitor:
        try:
            stats = monitor.get_stats() if hasattr(monitor, 'get_stats') else {}
            if stats.get('rolling_accuracy', 100) < 55:
                drift_alerts.append({'type': 'accuracy', 'msg': f"Rolling accuracy {stats['rolling_accuracy']:.1f}% < 55% threshold"})
            if stats.get('calibration_error', 0) > 0.15:
                drift_alerts.append({'type': 'calibration', 'msg': f"ECE {stats['calibration_error']:.3f} > 0.15 threshold"})
            if stats.get('feature_drift_score', 0) > 2.0:
                drift_alerts.append({'type': 'drift', 'msg': f"Feature drift z-score {stats['feature_drift_score']:.2f} > 2.0"})
        except Exception:
            pass

    # Queues
    data_q = _safe(bot, 'data_queue')

    return {
        'status': 'online',
        'uptime': f'{hours}h {mins}m {secs}s',
        'uptime_seconds': int(uptime_s),
        'generation': generation,
        'is_trained': is_trained,
        'is_training': is_training,
        'rl_stats': rl_stats,
        'performance': perf_data,
        'fear_greed': fg_data,
        'balance': round(balance, 2),
        'initial_balance': round(initial_balance, 2),
        'total_pnl': round(total_pnl, 2),
        'pnl_pct': round((total_pnl / initial_balance * 100) if initial_balance > 0 else 0, 2),
        'total_trades': total_trades,
        'total_wins': total_wins,
        'total_losses': total_losses,
        'trade_win_rate': round(total_wins / max(1, total_trades) * 100, 1),
        'current_drawdown': round(current_dd, 2),
        'max_drawdown': round(max_dd, 2),
        'active_positions': active_positions,
        'legacy_active_positions_detail': legacy_active_positions,
        'thor_active_positions': sum(1 for p in active_positions if p.get('is_thor_executed')),
        'legacy_active_positions': len(legacy_active_positions),
        'rl_enabled': bool(getattr(getattr(bot, 'cfg', None), 'rl_enabled', False)),
        'trade_history': trade_history,
        'equity_curve': equity_curve,
        'risk_status': risk_status,
        'ppo_summary': ppo_summary,
        'ppo_gate_accuracy': ppo_accuracy,
        'ppo_gate_brier': ppo_brier,
        'drift_alerts': drift_alerts,
        'queue_depth': data_q.qsize() if data_q else 0,
        'queue_max': data_q.maxsize if data_q else 0,
        'total_predictions': getattr(bot, 'total_predictions', 0),
        'pending_opportunities': len(getattr(bot, 'rl_opportunities', [])),
    }


def _get_predictions():
    """Predictions tab: daily picks + RL history."""
    bot = _bot
    if bot is None:
        return {'daily_picks': [], 'recent_predictions': []}

    daily_picks = _safe(bot, 'daily_picks', {})
    picks_list = []
    if daily_picks:
        for pid, p in sorted(daily_picks.items(), key=lambda x: x[1].get('timestamp', 0), reverse=True):
            sp = p.get('specialist_probs', [])
            if hasattr(sp, 'tolist'): sp = sp.tolist()
            elif isinstance(sp, list): sp = [float(x) for x in sp]
            
            pa = p.get('ppo_action')
            try:
                if pa is not None: pa = int(pa)
            except (ValueError, TypeError):
                pa = None

            pm = p.get('ppo_size_mult')
            try:
                if pm is not None: pm = float(pm)
            except (ValueError, TypeError):
                pm = None

            picks_list.append({
                'symbol': p.get('symbol', '?'),
                'direction': p.get('direction', '?'),
                'confidence': round(p.get('confidence', 0), 1),
                'entry_price': p.get('entry_price', 0),
                'tp1': p.get('tp1', 0),
                'tp2': p.get('tp2', 0),
                'tp3': p.get('tp3', 0),
                'stop_loss': p.get('stop_loss', 0),
                'status': p.get('status', 'PENDING'),
                'max_favorable': round(p.get('max_favorable', 0), 2),
                'timestamp': p.get('timestamp', 0),
                'specialist_probs': sp,
                'ppo_action': pa,
                'ppo_size_mult': pm,
                'shap_summary': p.get('shap_summary'),
            })

    recent_preds = []
    rl_mem = _safe(bot, 'rl_memory')
    if rl_mem and hasattr(rl_mem, 'memory'):
        completed = [p for p in rl_mem.memory if p.get('outcome') in ('correct', 'wrong')]
        for p in completed[-50:]:
            sp = p.get('specialist_probs', [])
            if hasattr(sp, 'tolist'): sp = sp.tolist()
            elif isinstance(sp, list): sp = [float(x) for x in sp]
            
            pa = p.get('ppo_action')
            try:
                if pa is not None: pa = int(pa)
            except (ValueError, TypeError):
                pa = None

            pm = p.get('ppo_size_mult')
            try:
                if pm is not None: pm = float(pm)
            except (ValueError, TypeError):
                pm = None
            
            recent_preds.append({
                'symbol': p.get('symbol', '?'),
                'direction': p.get('direction', '?'),
                'confidence': round(p.get('confidence', 0), 1),
                'outcome': p.get('outcome', '?'),
                'actual_move': round(p.get('actual_move', 0), 2) if p.get('actual_move') else 0,
                'outcome_tier': p.get('outcome_tier', '?'),
                'specialist_probs': sp,
                'ppo_action': pa,
                'ppo_size_mult': pm,
                'shap_summary': p.get('shap_summary'),
            })
        recent_preds.reverse()

    return {'daily_picks': picks_list[:20], 'recent_predictions': recent_preds}


def _get_models():
    """Models tab: specialist details, Brier scores, drift status."""
    bot = _bot
    if bot is None:
        return {'specialists': [], 'brier_scores': {}, 'monitor_stats': {}}

    ml = _safe(bot, 'ml')
    specialists = []
    brier = {}
    if ml:
        if hasattr(ml, 'specialist_models'):
            for name, spec in ml.specialist_models.items():
                specialists.append({
                    'name': display_agent_name(name),
                    'internal_name': name,
                    'weight': round(spec.get('weight', 0), 4),
                    'description': spec.get('description', ''),
                    'has_model': spec.get('model') is not None,
                    'generation': spec.get('generation', 0),
                    'feature_mask': spec.get('feature_mask', ''),
                    'recency_boost': spec.get('recency_boost', 1.0),
                })
        if hasattr(ml, '_brier_scores'):
            for name, bs in ml._brier_scores.items():
                count = bs.get('count', 0)
                brier[name] = {
                    'score': round(bs['sum'] / count, 4) if count > 0 else None,
                    'count': count,
                    'rolling_avg': round(sum(bs.get('rolling', [])) / max(1, len(bs.get('rolling', []))), 4) if bs.get('rolling') else None,
                }

    monitor_stats = {}
    monitor = _safe(bot, 'monitor')
    if monitor and hasattr(monitor, 'get_stats'):
        try:
            monitor_stats = monitor.get_stats()
        except Exception:
            pass

    return {
        'specialists': specialists,
        'brier_scores': brier,
        'monitor_stats': monitor_stats,
        'generation': ml.model_generation if ml else 0,
    }


def _get_system():
    """System tab: queues, latency, risk events, logs."""
    bot = _bot
    data_q = _safe(bot, 'data_queue') if bot else None
    result_q = _safe(bot, 'result_queue') if bot else None
    perf = _safe(bot, 'perf_monitor') if bot else None

    # Read last N lines of log file
    log_lines = list(_log_buffer) if _log_buffer else []
    if not log_lines:
        log_path = Path('C:/Users/habib/QUANTA/quanta_runtime.log')
        if log_path.exists():
            try:
                with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
                    lines = f.readlines()
                    log_lines = [l.rstrip() for l in lines[-50:]]
            except Exception:
                pass

    # Risk events
    risk_events = []
    paper = _safe(bot, 'paper') if bot else None
    risk_mgr = getattr(paper, 'risk_manager', None) if paper else None
    if risk_mgr and hasattr(risk_mgr, '_risk_events'):
        for ev in list(risk_mgr._risk_events)[-20:]:
            if hasattr(ev, '__dict__'):
                risk_events.append({
                    'time': getattr(ev, 'timestamp', 0),
                    'type': getattr(ev, 'event_type', '?'),
                    'details': getattr(ev, 'details', ''),
                    'action': getattr(ev, 'action_taken', ''),
                })
            elif isinstance(ev, dict):
                risk_events.append(ev)
        risk_events.reverse()

    stats = {}
    if perf:
        try:
            stats = perf.get_stats()
        except Exception:
            pass

    # ZEUS AI Audit Log
    zeus_logs = []
    zeus_path = Path('C:/Users/habib/QUANTA/models/zeus_audit_log.jsonl')
    if zeus_path.exists():
        try:
            with open(zeus_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for l in lines[-15:]:
                    import json
                    zeus_logs.append(json.loads(l.strip()))
            zeus_logs.reverse() # Newest first
        except Exception:
            pass
            
            
    return {
        'queue_depth': data_q.qsize() if data_q else 0,
        'queue_max': data_q.maxsize if data_q else 0,
        'result_queue': result_q.qsize() if result_q else 0,
        'coins_per_min': round(stats.get('coins_per_min', 0), 1),
        'total_processed': stats.get('total_coins', 0),
        'avg_fetch_ms': round(stats.get('avg_fetch_time', 0) * 1000, 1),
        'avg_compute_ms': round(stats.get('avg_compute_time', 0) * 1000, 1),
        'log_lines': log_lines,
        'risk_events': risk_events,
        'zeus_logs': zeus_logs,
    }


def _get_thor():
    """Thor model status, live params, WF sim results, Nike screener health."""
    bot = _bot

    # Live config params - source of truth is the active Thor v2 exit profile,
    # not the stale legacy thor_* aliases.
    cfg = _safe(bot, 'cfg')
    params = {}
    exit_profile = {}
    paper = _safe(bot, 'paper') if bot else None
    if paper and hasattr(paper, '_build_thor_exit_profile'):
        try:
            exit_profile = paper._build_thor_exit_profile() or {}
        except Exception:
            exit_profile = {}
    if cfg or exit_profile:
        params = {
            'sl_atr':            exit_profile.get('sl_atr', 3.00),
            'bank_atr':          exit_profile.get('bank_atr', 4.20),
            'runner_trail_atr':  exit_profile.get('runner_trail_atr', 2.00),
            'bank_fraction':     exit_profile.get('bank_fraction', 0.35),
            'trail_activate_atr':exit_profile.get('trail_activate_atr', 1.50),
            'mae_veto_bars':     exit_profile.get('mae_veto_bars', 5),
            'mae_veto_atr':      exit_profile.get('mae_veto_atr', 3.62),
            'min_score':         getattr(cfg, 'thor_min_score_trade', 68.0) if cfg else 68.0,
            'context_min_score': getattr(cfg, 'thor_context_min_score', 72.0) if cfg else 72.0,
            'max_bars_pre_bank': exit_profile.get('max_bars_pre_bank', getattr(cfg, 'thor_max_bars_pre_bank', 48) if cfg else 48),
            'max_bars_post_bank':exit_profile.get('max_bars_post_bank', getattr(cfg, 'thor_max_bars_post_bank', 96) if cfg else 96),
            'compound_mode':     getattr(cfg, 'thor_compound_mode', 'asymmetric_target') if cfg else 'asymmetric_target',
        }

    # Thor model generation (count .cbm files)
    import glob as _glob
    cbm_files = sorted(_glob.glob('C:/Users/habib/QUANTA/models/thor_gen*.cbm'))
    thor_gen = len(cbm_files)
    last_trained_ts = None
    if cbm_files:
        try:
            last_trained_ts = os.path.getmtime(cbm_files[-1])
        except Exception:
            pass

    # Brier score for Thor if available
    ml = _safe(bot, 'ml')
    thor_brier = None
    if ml and hasattr(ml, '_brier_scores'):
        bs = ml._brier_scores.get('thor', {})
        cnt = bs.get('count', 0)
        if cnt > 0:
            thor_brier = round(bs['sum'] / cnt, 4)

    # Nike screener health
    nike = getattr(bot, 'nike_screener', None) if bot else None
    nike_info = {
        'active': nike is not None,
        'symbols_watched': getattr(nike, '_n_streams', 0) if nike else 0,
        'signals_today': getattr(nike, '_signals_today', 0) if nike else 0,
        'total_signals': getattr(nike, '_total_signals', 0) if nike else 0,
    }

    # WF sim results if available
    wf_results = {}
    wf_path = Path('C:/Users/habib/QUANTA/wf_sim_results.json')
    if wf_path.exists():
        try:
            with open(wf_path, 'r', encoding='utf-8') as _f:
                wf_results = json.load(_f)
        except Exception:
            pass

    # Latest WF run folder
    wf_runs = sorted(_glob.glob('C:/Users/habib/QUANTA/wf_runs/*/'), reverse=True)
    latest_wf_run = os.path.basename(wf_runs[0].rstrip('/')) if wf_runs else None

    # Paper perf snapshot (same source as overview)
    paper_perf = {
        'balance': round(getattr(paper, 'balance', 0), 2),
        'pnl': round(getattr(paper, 'total_pnl', 0), 2),
        'trades': getattr(paper, 'total_trades', 0),
        'wins': getattr(paper, 'total_wins', 0),
        'losses': getattr(paper, 'total_losses', 0),
    }

    return {
        'generation': thor_gen,
        'last_trained_ts': last_trained_ts,
        'params': params,
        'brier': thor_brier,
        'baldur_live': False,
        'freya_live': False,
        'nike': nike_info,
        'wf_results': wf_results,
        'latest_wf_run': latest_wf_run,
        'paper_perf': paper_perf,
    }


# ─── Routes ───────────────────────────────────────────────────────────

@app.route('/api/settings', methods=['GET', 'POST'])
def api_settings():
    from quanta_config import Config
    import json
    
    if request.method == 'POST':
        try:
            payload = request.get_json()
            if payload:
                # Update memory
                Config.update_from_dict(payload)
                
                # Write to json reliably
                override_file = Config.base_dir / "quanta_config_overrides.json"
                
                # Merge existing overrides with new payload to not overwrite other sections
                existing = {}
                if override_file.exists():
                    with open(override_file, 'r') as f:
                        existing = json.load(f)
                        
                for k, v in payload.items():
                    if k not in existing:
                        existing[k] = {}
                    existing[k].update(v)
                    
                with open(override_file, 'w') as f:
                    json.dump(existing, f, indent=4)
                    
                return jsonify({'status': 'success'})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 400
            
    # GET method returns current active config dictionary
    return jsonify(Config.export_to_dict())



@app.after_request
def add_no_cache(response):
    if 'text/html' in response.content_type:
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
    return response

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/overview')
def api_overview():
    return jsonify(_get_overview())

@app.route('/api/predictions')
def api_predictions():
    return jsonify(_get_predictions())

@app.route('/api/models')
def api_models():
    return jsonify(_get_models())

@app.route('/api/system')
def api_system():
    return jsonify(_get_system())

@app.route('/api/thor')
def api_thor():
    return jsonify(_get_thor())

# Keep old endpoint for backward compat
@app.route('/api/status')
def api_status():
    return jsonify(_get_overview())


# ─── Launcher ─────────────────────────────────────────────────────────

def start_dashboard(bot_instance, host='0.0.0.0', port=5000):
    global _bot
    _bot = bot_instance
    logging.getLogger('werkzeug').setLevel(logging.WARNING)

    def _run():
        app.run(host=host, port=port, debug=False, use_reloader=False)

    t = threading.Thread(target=_run, daemon=True, name='quanta-dashboard')
    t.start()
    print(f"  QUANTA Dashboard live at http://localhost:{port}")
    return t


if __name__ == '__main__':
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    print("QUANTA Dashboard (standalone) at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
