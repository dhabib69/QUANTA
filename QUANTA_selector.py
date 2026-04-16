import os
import json
import logging
import math
import numpy as np
import pandas as pd
import time
import concurrent.futures
from datetime import datetime, timedelta

from QUANTA_network import NetworkHelper
from quanta_features import Indicators

class QuantaSelector:
    """
    QUANTA Selector Orchestrator
    
    This script is the engine for all Data Preparation. It completely separates 
    Coin Selection & Event Extraction from the ML training loop.
    """
    def __init__(self, data_dir="ml_models_pytorch/matrices", base_url="https://api.binance.com/api/v3", cache=None):
        self.data_dir = data_dir
        self.base_url = base_url
        self.futures_url = "https://fapi.binance.com/fapi/v1"
        self.cache = cache  # FeatherCache instance (shared with exchange)
        self.universe_file = os.path.join(data_dir, '..', 'coin_universe.json')
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        cache_status = "with FeatherCache" if cache else "no cache"
        print(f"QUANTA Selector & Event Matrix Orchestrator Initialized ({cache_status})")

    # =========================================================================
    # CORE UTILITIES
    # =========================================================================
    def get_klines(self, symbol, interval, limit=100):
        """Fetch klines specifically for the selection engine."""
        url = f"{self.base_url}/klines?symbol={symbol}&interval={interval}&limit={limit}"
        response = NetworkHelper.get(url, timeout=10, adaptive_timeout=True)
        if response:
            return response.json()
        return []

    def _log_request(self, success=True):
        pass

    # =========================================================================
    # OPTIMIZATION HELPERS
    # =========================================================================

    def _detect_btc_regime(self):
        """Opt 4: Detect BTC market regime for dynamic weight adjustment.
        Returns: 'bull', 'bear', or 'range'
        """
        try:
            klines = self.get_klines('BTCUSDT', '1d', limit=30)
            if not klines or len(klines) < 20:
                return 'range'
            closes = [float(k[4]) for k in klines]
            pct_30d = ((closes[-1] - closes[0]) / closes[0]) * 100
            if pct_30d > 15:
                return 'bull'
            elif pct_30d < -15:
                return 'bear'
            return 'range'
        except Exception:
            return 'range'

    def _get_regime_weights(self, regime):
        """Opt 4: Return scoring weights based on detected regime."""
        if regime == 'bull':
            return {'volume': 0.20, 'momentum': 0.30, 'atr': 0.20, 'adx': 0.15, 'consistency': 0.05, 'funding': 0.10}
        elif regime == 'bear':
            return {'volume': 0.30, 'momentum': 0.15, 'atr': 0.15, 'adx': 0.20, 'consistency': 0.05, 'funding': 0.15}
        else:  # range
            return {'volume': 0.20, 'momentum': 0.20, 'atr': 0.25, 'adx': 0.10, 'consistency': 0.10, 'funding': 0.15}

    def _get_funding_rates(self):
        """Opt 3: Fetch funding rates from Binance Futures (single batch call).
        Returns: dict {symbol: lastFundingRate}
        """
        try:
            resp = NetworkHelper.get(f'{self.futures_url}/premiumIndex', timeout=10, adaptive_timeout=True)
            if not resp:
                return {}
            data = resp.json()
            rates = {}
            for item in data:
                sym = item.get('symbol', '')
                rate = float(item.get('lastFundingRate', 0))
                rates[sym] = rate
            return rates
        except Exception:
            return {}

    def _get_open_interest_map(self):
        """Opt 6: Fetch open interest data from Binance Futures.
        Returns: dict {symbol: openInterest_USDT}
        """
        try:
            resp = NetworkHelper.get(f'{self.futures_url}/ticker/24hr', timeout=10, adaptive_timeout=True)
            if not resp:
                return {}
            data = resp.json()
            oi_map = {}
            for item in data:
                sym = item.get('symbol', '')
                # baseVolume * weightedAvgPrice approximates open interest in USDT
                try:
                    oi_map[sym] = float(item.get('openInterest', 0)) * float(item.get('lastPrice', 1))
                except Exception:
                    pass
            return oi_map
        except Exception:
            return {}

    def _deduplicate_by_correlation(self, scored_coins, max_coins=None):
        """Opt 1: Remove highly correlated coins (r > 0.85).
        Keeps the higher-scored coin from each correlated pair.
        Uses 30d daily returns fetched during scoring.
        """
        if len(scored_coins) < 3:
            return scored_coins

        # Fetch 30d daily closes for correlation
        symbols = [c['symbol'] for c in scored_coins]
        returns_map = {}

        def fetch_returns(sym):
            try:
                time.sleep(0.05)
                kl = self.get_klines(sym, '1d', limit=30)
                if kl and len(kl) >= 20:
                    closes = np.array([float(k[4]) for k in kl])
                    rets = np.diff(closes) / closes[:-1]
                    return sym, rets
            except Exception:
                pass
            return sym, None

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            for sym, rets in ex.map(lambda s: fetch_returns(s), symbols[:80]):
                if rets is not None and len(rets) >= 15:
                    returns_map[sym] = rets

        # Build correlation matrix and mark duplicates
        syms_with_data = [c for c in scored_coins if c['symbol'] in returns_map]
        remove_set = set()

        for i in range(len(syms_with_data)):
            if syms_with_data[i]['symbol'] in remove_set:
                continue
            for j in range(i + 1, len(syms_with_data)):
                if syms_with_data[j]['symbol'] in remove_set:
                    continue
                s_i = syms_with_data[i]['symbol']
                s_j = syms_with_data[j]['symbol']
                r_i = returns_map[s_i]
                r_j = returns_map[s_j]
                min_len = min(len(r_i), len(r_j))
                if min_len < 15:
                    continue
                corr = np.corrcoef(r_i[:min_len], r_j[:min_len])[0, 1]
                if abs(corr) > 0.85:
                    # Drop the lower-scored one
                    if syms_with_data[i].get('score', 0) >= syms_with_data[j].get('score', 0):
                        remove_set.add(s_j)
                    else:
                        remove_set.add(s_i)

        deduped = [c for c in scored_coins if c['symbol'] not in remove_set]
        if len(remove_set) > 0:
            print(f'   🔗 Correlation filter: removed {len(remove_set)} redundant coins (r>0.85)')
        return deduped

    def _update_survivorship_tracker(self, symbols):
        """Opt 7: Track all coins ever seen. Detect potential delistings."""
        try:
            universe = {}
            if os.path.exists(self.universe_file):
                with open(self.universe_file, 'r') as f:
                    universe = json.load(f)

            today = datetime.now().strftime('%Y-%m-%d')
            current_set = set(symbols)

            # Update seen coins
            for sym in symbols:
                if sym not in universe:
                    universe[sym] = {'first_seen': today, 'last_seen': today, 'delisted': False}
                else:
                    universe[sym]['last_seen'] = today
                    universe[sym]['delisted'] = False

            # Check for potential delistings (seen before but missing now for 7+ days)
            delisted_count = 0
            for sym, info in universe.items():
                if sym not in current_set and not info.get('delisted', False):
                    last_seen = datetime.strptime(info['last_seen'], '%Y-%m-%d')
                    if (datetime.now() - last_seen).days >= 7:
                        universe[sym]['delisted'] = True
                        delisted_count += 1

            # Save
            os.makedirs(os.path.dirname(self.universe_file), exist_ok=True)
            with open(self.universe_file, 'w') as f:
                json.dump(universe, f, indent=2)

            total_tracked = len(universe)
            total_delisted = sum(1 for v in universe.values() if v.get('delisted'))
            if total_delisted > 0:
                print(f'   📊 Universe tracker: {total_tracked} coins seen historically, {total_delisted} potentially delisted')

        except Exception as e:
            logging.debug(f'Survivorship tracker error: {e}')

    # =========================================================================
    # 1. COIN SELECTION ENGINE 
    # =========================================================================
    def _get_fallback_top_movers(self, limit=50):
        """Fallback with REGIME DIVERSITY."""
        fallback = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 
            'AVAXUSDT', 'MATICUSDT', 'LINKUSDT', 'DOTUSDT', 'UNIUSDT', 'ATOMUSDT', 'LTCUSDT', 
            'NEARUSDT', 'APTUSDT', 'ARBUSDT', 'OPUSDT', 'SUIUSDT', 'INJUSDT', 'RNDRUSDT', 
            'TAOUSDT', 'WLDUSDT', 'SEIUSDT', 'TIAUSDT', 'SHIBUSDT', 'PEPEUSDT', 'FILUSDT', 
            'ICPUSDT', 'ALGOUSDT', 'AAVEUSDT', 'MKRUSDT', 'COMPUSDT', 'SANDUSDT', 'MANAUSDT', 
            'AXSUSDT', 'GMTUSDT', 'FTMUSDT', 'VETUSDT', 'APEUSDT', 'GALAUSDT', 'CHZUSDT', 'ENJUSDT'
        ]
        logging.info(f'📋 Using fallback selection: {len(fallback[:limit])} pairs (diversity ensured)')
        return fallback[:limit]

    def get_top_movers(self, limit=50):
        """🔥 RESEARCH-BACKED COIN SELECTION (v2 — 7 optimizations)"""
        try:
            print('\n🔍 Fetching market data for coin selection...')
            response = NetworkHelper.get(f'{self.base_url}/ticker/24hr', timeout=15, adaptive_timeout=True)
            if not response:
                return self._get_fallback_top_movers(limit)
                
            tickers = response.json()

            # Opt 4: Detect BTC regime for dynamic weights
            regime = self._detect_btc_regime()
            weights = self._get_regime_weights(regime)
            print(f'📊 BTC Regime: {regime.upper()} → weights: {weights}')

            # Opt 3: Fetch funding rates (single batch call)
            funding_rates = self._get_funding_rates()

            # Opt 6: Fetch open interest data
            oi_map = self._get_open_interest_map()

            # Opt 5: Build volume baseline for relative volume
            volume_map = {}
            for t in tickers:
                sym = t.get('symbol', '')
                vol = float(t.get('quoteVolume', 0))
                volume_map[sym] = vol

            candidates = []
            for t in tickers:
                symbol = t.get('symbol', '')
                if not symbol.endswith('USDT'): continue
                volume_usd = float(t.get('quoteVolume', 0))
                if volume_usd < 10000000: continue
                if any(stable in symbol for stable in ['BUSD', 'USDC', 'TUSD', 'DAI', 'FDUSD']): continue
                    
                candidates.append({
                    'symbol': symbol, 'volume_24h': volume_usd, 
                    'price_change_pct': float(t.get('priceChangePercent', 0)), 
                    'price': float(t.get('lastPrice', 1)), 
                    'high': float(t.get('highPrice', 1)), 'low': float(t.get('lowPrice', 1))
                })

            if len(candidates) < 20:
                return self._get_fallback_top_movers(limit)

            print(f'✅ {len(candidates)} coins passed liquidity filter ($10M+ volume)')
            scored_coins = []
            
            for coin in candidates:
                try:
                    klines_30d = self.get_klines(coin['symbol'], '1d', limit=30)
                    if not klines_30d or len(klines_30d) < 20: continue
                        
                    closes = np.array([float(k[4]) for k in klines_30d])
                    highs = np.array([float(k[2]) for k in klines_30d])
                    lows = np.array([float(k[3]) for k in klines_30d])
                    volumes = np.array([float(k[5]) for k in klines_30d])
                    
                    pct_change_30d = (closes[-1] / closes[0] - 1) * 100
                    abs_momentum = abs(pct_change_30d)
                    atr = Indicators.atr(highs[-14:], lows[-14:], closes[-14:])
                    atr_pct = atr / closes[-1] * 100 if closes[-1] > 0 else 0
                    adx = Indicators.adx(highs, lows, closes)
                    volume_stddev = np.std(volumes) / np.mean(volumes) if np.mean(volumes) > 0 else 999
                    volume_consistency = 1 / (1 + volume_stddev)

                    # Opt 5: Relative volume — how much today vs 30d avg
                    avg_30d_vol = np.mean(volumes) if len(volumes) > 0 else 1
                    vol_ratio = coin['volume_24h'] / max(avg_30d_vol * coin['price'], 1)  # normalize to USDT
                    rel_volume_score = min(math.log10(max(vol_ratio, 1) + 1) * 5, 10)  # cap at 10

                    # Opt 3: Funding rate signal — extreme = crowded positioning
                    fr = funding_rates.get(coin['symbol'], 0)
                    funding_signal = min(abs(fr) * 10000, 10)  # scale up, cap at 10
                    
                    # Opt 6: Open interest conviction
                    oi_val = oi_map.get(coin['symbol'], 0)
                    oi_signal = min(math.log10(max(oi_val, 1)) * 0.5, 5) if oi_val > 0 else 0
                    
                    # Opt 4: Dynamic regime-weighted scoring
                    score = (
                        (np.log10(coin['volume_24h']) + rel_volume_score) * weights['volume'] +
                        abs_momentum * weights['momentum'] + 
                        min(atr_pct, 10) * weights['atr'] +
                        adx * weights['adx'] +
                        volume_consistency * 10 * weights['consistency'] +
                        (funding_signal + oi_signal) * weights['funding']
                    )
                             
                    scored_coins.append({
                        'symbol': coin['symbol'], 'score': score, 'pct_change_30d': pct_change_30d,
                        'abs_momentum': abs_momentum, 'atr_pct': atr_pct, 'adx': adx,
                        'volume_24h': coin['volume_24h'], 'funding_rate': fr, 'oi_usdt': oi_val,
                        'vol_ratio': vol_ratio,  # current 24h vol / 30d avg (discovery signal)
                    })
                except Exception:
                    continue

            scored_coins.sort(key=lambda x: x['score'], reverse=True)

            # Opt 1: Correlation de-duplication
            print('🔗 Running correlation de-duplication...')
            scored_coins = self._deduplicate_by_correlation(scored_coins)

            print('🎯 Enforcing regime diversity...')
            
            selections = []
            used_symbols = set()
            
            # Slot A: Gainers
            gainers = [c for c in scored_coins if c['pct_change_30d'] > 15 and c['symbol'] not in used_symbols]
            gainers.sort(key=lambda x: x['score'], reverse=True)
            selections.extend(gainers[:5])
            used_symbols.update(c['symbol'] for c in gainers[:5])
            
            # Slot B: Losers
            losers = [c for c in scored_coins if c['pct_change_30d'] < -15 and c['symbol'] not in used_symbols]
            losers.sort(key=lambda x: x['score'], reverse=True)
            selections.extend(losers[:5])
            used_symbols.update(c['symbol'] for c in losers[:5])
            
            # Slot C: Ranging
            ranging = [c for c in scored_coins if c['adx'] < 20 and c['symbol'] not in used_symbols]
            ranging.sort(key=lambda x: x['score'], reverse=True)
            selections.extend(ranging[:5])
            used_symbols.update(c['symbol'] for c in ranging[:5])
            
            # Slot D: Volatile
            volatile = [c for c in scored_coins if c['symbol'] not in used_symbols]
            volatile.sort(key=lambda x: x['atr_pct'], reverse=True)
            selections.extend(volatile[:5])
            used_symbols.update(c['symbol'] for c in volatile[:5])
            
            # Slot E: Anchors
            anchors = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
            for anchor in anchors:
                if anchor not in used_symbols:
                    anchor_data = next((c for c in scored_coins if c['symbol'] == anchor), None)
                    if anchor_data:
                        selections.append(anchor_data)
                        used_symbols.add(anchor)
                        
            # Slot G: Discovery — sudden volume spike + small-mid cap (pre-crowd alpha)
            # Coins with vol_ratio > 3× their 30d avg but volume_24h < $500M (not yet mainstream)
            # Inspired by: Lo (2004) AMH — price discovery follows order flow, not market cap
            discovery = [
                c for c in scored_coins
                if c['symbol'] not in used_symbols
                and c.get('vol_ratio', 0) >= 3.0
                and c['volume_24h'] < 500_000_000
            ]
            discovery.sort(key=lambda x: x.get('vol_ratio', 0), reverse=True)
            selections.extend(discovery[:3])
            used_symbols.update(c['symbol'] for c in discovery[:3])

            # Slot F: Remaining (fill to limit after discovery slot)
            remaining_needed = limit - len(selections)
            remaining = [c for c in scored_coins if c['symbol'] not in used_symbols]
            selections.extend(remaining[:remaining_needed])

            final_symbols = [c['symbol'] for c in selections[:limit]]
            print(f'\n✅ SELECTED {len(final_symbols)} COINS (regime={regime}, corr_filtered)')
            self._log_request(success=True)
            return final_symbols
            
        except Exception as e:
            logging.error(f'get_top_movers error: {e}')
            return self._get_fallback_top_movers(limit)

    def get_live_prediction_feed(self):
        """🎯 LIVE PREDICTION: TOP 100 MOVERS (v2 — confidence-gated + corr-filtered)
        
        Opt 2: Pre-filters flat coins (<1% move) to reduce noise predictions.
        Opt 1: Correlation de-duplication to maximize diversity.
        
        Returns: list of symbols (top 50 gainers + top 50 losers by 24h %)
        """
        try:
            response = NetworkHelper.get(f'{self.base_url}/ticker/24hr', timeout=15, adaptive_timeout=True)
            if not response: return []
            tickers = response.json()
            
            STABLES = {'BUSD', 'USDC', 'TUSD', 'DAI', 'FDUSD', 'WBTC', 'STETH'}
            MIN_VOL = 5000000  # $5M daily volume minimum
            MIN_ABS_PCT = 1.0  # Opt 2: Confidence gate — ignore <1% movers
            
            eligible = []
            flat_skipped = 0
            for t in tickers:
                sym = t.get('symbol', '')
                if not sym.endswith('USDT'): continue
                if any(s in sym for s in STABLES): continue
                vol = float(t.get('quoteVolume', 0))
                if vol < MIN_VOL: continue
                
                pct = float(t.get('priceChangePercent', 0))
                # Opt 2: Skip flat coins — these generate noise predictions
                if abs(pct) < MIN_ABS_PCT:
                    flat_skipped += 1
                    continue
                eligible.append({'symbol': sym, 'pct': pct, 'volume': vol, 'score': abs(pct) * math.log10(max(vol, 1))})
            
            if len(eligible) < 20:
                # Fallback: relax the flat filter if too few coins pass
                eligible = []
                for t in tickers:
                    sym = t.get('symbol', '')
                    if not sym.endswith('USDT'): continue
                    if any(s in sym for s in STABLES): continue
                    vol = float(t.get('quoteVolume', 0))
                    if vol < MIN_VOL: continue
                    pct = float(t.get('priceChangePercent', 0))
                    eligible.append({'symbol': sym, 'pct': pct, 'volume': vol, 'score': abs(pct) * math.log10(max(vol, 1))})
            
            # Top 50 gainers + Top 50 losers
            gainers = sorted([e for e in eligible if e['pct'] > 0], key=lambda x: x['pct'], reverse=True)[:50]
            losers = sorted([e for e in eligible if e['pct'] < 0], key=lambda x: x['pct'])[:50]
            
            # Merge and deduplicate
            merged = gainers + losers
            
            # Opt 1: Correlation de-duplication on the merged list
            if len(merged) > 10:
                merged = self._deduplicate_by_correlation(merged)

            seen = set()
            result = []
            for coin in merged:
                if coin['symbol'] not in seen:
                    seen.add(coin['symbol'])
                    result.append(coin['symbol'])
            
            # Always include anchors
            for a in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']:
                if a not in seen:
                    result.append(a)
            
            print(f"🎯 Live Feed: {len(result)} coins ({len(gainers)} gainers + {len(losers)} losers, {flat_skipped} flat skipped)")
            return result
            
        except Exception:
            return []

    def get_research_backed_coins_for_training(self, limit=50):
        """🔬 INITIAL TRAINING: EXTREME GAINERS & LOSERS SELECTION (v2 — corr-filtered + tracked)
        We only want the absolute extremes to train the models on the full anatomy
        of pumps and dumps without human bias (no ADX/ATR pre-filtering).
        Opt 1: Correlation de-duplication applied after selection.
        Opt 7: Survivorship tracker updated.
        """
        import time
        try:
            response = NetworkHelper.get(f'{self.base_url}/ticker/24hr', timeout=20, adaptive_timeout=True)
            if not response: return self._get_fallback_top_movers(limit)
            tickers = response.json()
            
            usdt_pairs = [t for t in tickers if t.get('symbol', '').endswith('USDT')]
            eligible = []
            
            # Strict Garbage Filter (Fiats, Stablecoins, Pegged, Wrapped)
            garbage_bases = {
                'BUSD', 'USDC', 'TUSD', 'DAI', 'USDP', 'USDD', 'FDUSD', 'PYUSD', 
                'EUR', 'GBP', 'USD', 'USD1', 'TRY', 'BRL', 'ZAR', 'RUB', 'IDR', 'UAH', 'NGN',
                'PAXG', 'WBTC', 'WETH', 'STETH', 'AEUR', 'VAI', 'FRAX'
            }
            
            for t in usdt_pairs:
                base_coin = t['symbol'].replace('USDT', '')
                if float(t.get('quoteVolume', 0)) >= 500000 and base_coin not in garbage_bases:
                    eligible.append({'symbol': t['symbol']})
            
            if len(eligible) < 20: return self._get_fallback_top_movers(limit)

            # Opt 7: Update survivorship tracker with all eligible symbols
            self._update_survivorship_tracker([e['symbol'] for e in eligible])
            
            target_coins = eligible[:300]
            scored_coins = []
            
            def score_coin(coin):
                try:
                    time.sleep(0.05) # Prevent Binance IP ban from multi-threading 300 requests
                    klines = self.get_klines(coin['symbol'], '1d', limit=365)
                    if not klines or len(klines) < 180: return None
                    closes = [float(k[4]) for k in klines]
                    pct_change_365d = ((closes[-1] - closes[0]) / closes[0]) * 100
                    return {**coin, 'pct_change_365d': pct_change_365d, 'score': abs(pct_change_365d)}
                except Exception as e:
                    logging.debug(f"Score coin {coin.get('symbol', '?')} failed: {e}")
                    return None
                
            # Lower max workers to 4 to protect against silent API rate limits dropping coins
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                for res in executor.map(score_coin, target_coins):
                    if res: scored_coins.append(res)
                    
            if not scored_coins: return self._get_fallback_top_movers(limit)
            
            # STRICT EXTREME THRESHOLDS
            # Exclude anything that just mildly chopped around
            true_gainers = [c for c in scored_coins if c['pct_change_365d'] >= 100.0]
            true_losers = [c for c in scored_coins if c['pct_change_365d'] <= -40.0]
            
            true_gainers.sort(key=lambda x: x['pct_change_365d'], reverse=True)
            true_losers.sort(key=lambda x: x['pct_change_365d'], reverse=False) # Most negative first
            
            # Opt 1: Correlation de-duplication within gainers and losers separately
            if len(true_gainers) > 5:
                true_gainers = self._deduplicate_by_correlation(true_gainers)
            if len(true_losers) > 5:
                true_losers = self._deduplicate_by_correlation(true_losers)
            
            # We want a healthy mix of both pumps and dumps for the models
            merged = []
            max_len = max(len(true_gainers), len(true_losers))
            for i in range(max_len):
                if i < len(true_gainers):
                    merged.append(true_gainers[i]['symbol'])
                if i < len(true_losers):
                    merged.append(true_losers[i]['symbol'])
            
            merged = list(dict.fromkeys(merged)) # deduplicate
            
            # Supplement with the highest volume volatile coins to meet the requested limit
            if len(merged) < limit:
                # 1. First, try to pad with non-extreme coins that successfully fetched their 365d data
                remaining = [c for c in scored_coins if c['symbol'] not in merged]
                remaining.sort(key=lambda x: abs(x['pct_change_365d']), reverse=True)
                needed = limit - len(merged)
                merged.extend([c['symbol'] for c in remaining[:needed]])
                
                # 2. If STILL short (Binance proxy banned threads, limiting scored_coins), 
                # pad with raw highest-volume eligible coins directly from the 24h ticker
                if len(merged) < limit:
                    raw_needed = limit - len(merged)
                    raw_remaining = [c['symbol'] for c in target_coins if c['symbol'] not in merged]
                    merged.extend(raw_remaining[:raw_needed])
                
            return merged[:limit]
            
        except Exception as e:
            return self._get_fallback_top_movers(limit)

    def get_cached_coins_for_training(self, limit=200):
        """🗂️ CACHE-FIRST COIN SELECTION — Zero API calls.
        
        Scans feather_cache/ directory for all cached 5m .feather files.
        Returns symbols sorted by data size (most candles first).
        
        This enables 200+ coin training with no network dependency,
        using pre-downloaded Binance Vision archive data.
        """
        from pathlib import Path
        import pyarrow.feather as feather
        
        cache_dir = Path(self.cache.cache_dir) if self.cache else Path("feather_cache")
        if not cache_dir.exists():
            print("❌ Cache directory not found")
            return []
        
        # Scan for all 5m feather files
        candidates = []
        for f in cache_dir.glob("*_5m.feather"):
            symbol = f.stem.replace("_5m", "").upper()
            try:
                # Read row count without loading full data
                tbl = feather.read_table(str(f))
                row_count = tbl.num_rows
                del tbl
                if row_count >= 10000:  # ~35 days minimum
                    candidates.append((symbol, row_count))
            except Exception:
                continue
        
        if not candidates:
            print("⚠️ No cached coins with sufficient data found")
            return []
        
        # Sort by data size descending (most data = best for training)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        symbols = [c[0] for c in candidates[:limit]]
        print(f"🗂️ Cache-first selection: {len(symbols)} coins from feather cache")
        for i, (sym, cnt) in enumerate(candidates[:limit]):
            if i < 5 or i == len(candidates[:limit]) - 1:
                print(f"   {sym}: {cnt:,} candles")
            elif i == 5:
                print(f"   ... ({len(candidates[:limit]) - 6} more) ...")
        
        return symbols

    def categorize_coins_for_specialists(self, symbols):
        """🧬 CATEGORIZE COINS FOR 3 SPECIALIST MODELS"""
        if not symbols: return {'foundation': [], 'hunter': [], 'anchor': []}
        try:
            ANCHORS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT', 'DOTUSDT', 'AVAXUSDT', 'MATICUSDT', 'LINKUSDT']
            anchor_coins = [s for s in symbols if s in ANCHORS]
            tickers = NetworkHelper.get(f'{self.base_url}/ticker/24hr', timeout=10)
            
            if not tickers:
                foundation_coins = [s for s in symbols if s not in anchor_coins][:30]
                hunter_coins = [s for s in symbols if s not in anchor_coins and s not in foundation_coins]
                return {'foundation': foundation_coins, 'hunter': hunter_coins, 'anchor': anchor_coins}
                
            ticker_data = tickers.json()
            volume_map = {t['symbol']: float(t['quoteVolume']) for t in ticker_data if t['symbol'] in symbols}
            remaining = [s for s in symbols if s not in anchor_coins]
            
            sorted_by_volume = sorted(remaining, key=lambda x: volume_map.get(x, 0), reverse=True)
            split_point = max(10, int(len(sorted_by_volume) * 0.4))
            
            return {
                'quality': sorted_by_volume[:split_point], 
                'volatile': sorted_by_volume[split_point:], 
                'anchor': anchor_coins
            }
        except Exception:
            return {'quality': [s for s in symbols if s not in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']], 'volatile': [], 'anchor': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']}


    # =========================================================================
    # 2. EVENT EXTRACTION ENGINE 
    # =========================================================================
    
    def get_historical_klines_paginated(self, symbol, interval='5m', days=1000):
        """Fetch up to 1000 days of klines via backward pagination.
        
        Uses FeatherCache if available:
        - Check cache first, skip API if sufficient data exists
        - Save new fetches to cache for future runs
        """
        minutes_per_candle = {'1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440}.get(interval, 5)
        total_needed = (days * 1440) // minutes_per_candle
        
        # === CACHE CHECK ===
        if self.cache:
            cached = self.cache.get(symbol, interval, limit=total_needed)
            if cached and len(cached) >= min(total_needed * 0.8, 500):
                return cached
            
            # Check disk even if memory cache missed
            cached_length = self.cache.get_length(symbol, interval)
            if cached_length >= total_needed * 0.8:
                disk_data = self.cache.get(symbol, interval, limit=total_needed)
                if disk_data and len(disk_data) >= 500:
                    return disk_data
        
        # === API FETCH ===
        batch_size = 1000
        all_klines = []
        current_end = int(time.time() * 1000)
        
        num_batches = (total_needed // batch_size) + 1
        for batch_num in range(num_batches):
            try:
                end_time = current_end - (batch_num * batch_size * minutes_per_candle * 60 * 1000)
                resp = NetworkHelper.get(
                    f"{self.base_url}/klines",
                    params={'symbol': symbol, 'interval': interval, 'endTime': end_time, 'limit': batch_size},
                    timeout=20, adaptive_timeout=True
                )
                if resp:
                    data = resp.json()
                    if isinstance(data, list) and len(data) > 0:
                        all_klines = data + all_klines
                        # Early stop if we hit listing date
                        if len(data) < batch_size:
                            break
                    else:
                        break
                else:
                    break
                time.sleep(0.15)  # Rate limit safety
            except Exception:
                break
                
        # Deduplicate by open_time and sort chronologically
        seen = set()
        unique = []
        for k in all_klines:
            ot = k[0]
            if ot not in seen:
                seen.add(ot)
                unique.append(k)
        unique.sort(key=lambda x: x[0])
        
        # === SAVE TO CACHE ===
        if self.cache and len(unique) > 0:
            self.cache.set(symbol, interval, unique)
        
        return unique
    
    def _klines_to_df(self, klines):
        """Convert raw klines list to a pandas DataFrame."""
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df

    def _calculate_base_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Minimal price-only preprocessing + CUSUM structural break detection.
        
        Computes:
        - ATR / ATR_pct: for triple-barrier labeling + volatility-normalized thresholds
        - Vol_Avg_50: for volume surge detection (Artemis)
        - CUSUM pos/neg: structural break accumulator (López de Prado AFML Ch.2)
        """
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        
        # ATR — for triple-barrier + volatility-normalized event thresholds
        tr = np.maximum(highs - lows, np.maximum(
            np.abs(highs - np.roll(closes, 1)), 
            np.abs(lows - np.roll(closes, 1))
        ))
        df['ATR'] = pd.Series(tr).rolling(14).mean().values
        df['ATR_pct'] = df['ATR'] / (df['close'] + 1e-8)  # Volatility as % of price
        
        # Volume average — for volume surge detection (Artemis + Nike)
        df['Vol_Avg_50'] = df['volume'].rolling(50).mean()
        df['Vol_Avg_20'] = df['volume'].rolling(20).mean()
        
        # CUSUM filter (López de Prado 2018, AFML Ch.2)
        # Accumulates signed returns; resets on structural break
        returns = df['close'].pct_change().fillna(0).values
        cusum_pos = np.zeros(len(returns))
        cusum_neg = np.zeros(len(returns))
        for i in range(1, len(returns)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + returns[i])
            cusum_neg[i] = min(0, cusum_neg[i-1] + returns[i])
        df['CUSUM_pos'] = cusum_pos
        df['CUSUM_neg'] = cusum_neg
        
        # CRITICAL: Preserve original klines index before dropna removes rows
        df['orig_idx'] = df.index.copy()
        
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def extract_events_from_klines(self, klines, max_candle_idx=None):
        """
        🧬 Run all 7 event extractors on a single coin's klines using C-Speed Numba.
        
        Returns dict[agent_name] -> [{'pos': int, 'label': int, 'weight': float}]
        NO klines stored — memory safe.
        
        Args:
            klines: raw klines list
            max_candle_idx: if set, only extract events before this index (Day 700 cutoff)
        """
        import quanta_numba_extractors as qne
        df = self._klines_to_df(klines)
        df = self._calculate_base_indicators(df)
        
        if len(df) < 200:
            return {}
            
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        volumes = df['volume'].values
        atrs = df['ATR'].values
        atr_pct = df['ATR_pct'].values
        orig_idx = df['orig_idx'].values
        vol_avg = df['Vol_Avg_50'].values
        vol_avg20 = df['Vol_Avg_20'].values
        
        def to_arrays(pos_arr, label_arr, weight_arr):
            if max_candle_idx is not None:
                mask = pos_arr < max_candle_idx
                return {'pos': pos_arr[mask], 'label': label_arr[mask], 'weight': weight_arr[mask]}
            return {'pos': pos_arr, 'label': label_arr, 'weight': weight_arr}

        result = {}
        
        opens = df['open'].values
        pos, lab, wt = qne.fast_extract_nike(closes, highs, lows, opens, atrs, volumes, vol_avg20, orig_idx)
        result['thor'] = to_arrays(pos, lab, wt)
        
        return result

    def fast_online_retrain_matrix(self, audit_log: list):
        """Scans recent prediction mistakes and builds highly-weighted failure matrices."""
        print("🔄 Processing Audit Log for Retraining...")


if __name__ == "__main__":
    selector = QuantaSelector()
    print("QUANTA Selector Pipeline Active.")
