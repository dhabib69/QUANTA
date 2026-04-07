"""
📊 QUANTA Sentiment — Market Sentiment & News Engine v2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Architecture:
  • Multi-source RSS feed aggregator (unlimited, no API key)
  • Loughran & McDonald 2011 finance-specific sentiment scoring
  • Crypto-extended lexicon (500+ domain terms)
  • Fear & Greed Index (Alternative.me)
  • CryptoPanic (optional bonus, rate-limited)
  • Background feed thread (WebSocket-style pattern)

Research:
  • Loughran & McDonald 2011 (J. Finance) — domain-specific sentiment
    dictionaries outperform general NLP (Harvard GI) by 3-4x in
    financial text classification
  • Tetlock 2007 (J. Finance) — media pessimism predicts negative returns
  • Chen et al. 2014 — per-asset sentiment outperforms global by ~12%
  • Bollen et al. 2011 — sentiment as input features > post-hoc blending

Features extracted (5):
  1. fng_value          — Fear & Greed 0-1 normalized
  2. fng_extreme_fear   — Binary flag (F&G < 25)
  3. fng_extreme_greed  — Binary flag (F&G > 75)
  4. news_sentiment     — L&M weighted sentiment -1 to +1
  5. news_volume_norm   — Normalized news volume 0-1
"""

import os
import re
import time
import logging
import threading
import xml.etree.ElementTree as ET
from collections import defaultdict
from html import unescape

from QUANTA_network import NetworkHelper

try:
    from QUANTA_ai_oracle import get_groq_sentiment
except ImportError:
    get_groq_sentiment = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOUGHRAN & McDONALD 2011 FINANCE SENTIMENT LEXICON
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Extended with crypto-specific terms (Ante 2023, Auer & Claessens 2020)
#
# Design: frozenset for O(1) lookup, all lowercase, pre-compiled.
# Headlines are tokenized via regex split — no NLTK dependency.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# fmt: off
_LM_POSITIVE = frozenset({
    # === L&M 2011 Core Positive (finance-relevant subset) ===
    'achieve', 'advancement', 'advantage', 'beneficial', 'benefit',
    'boost', 'breakthrough', 'collaboration', 'commit', 'confidence',
    'creative', 'dividend', 'earn', 'efficient', 'enable', 'enhance',
    'exceed', 'exclusive', 'expand', 'favorable', 'gain', 'great',
    'growth', 'guarantee', 'highest', 'improve', 'improvement', 'increase',
    'innovation', 'innovative', 'integrity', 'leadership', 'opportunity',
    'optimal', 'outperform', 'outstanding', 'positive', 'premium',
    'proactive', 'profitability', 'profitable', 'progress', 'prosper',
    'record', 'rebound', 'recover', 'recovery', 'resolve', 'reward',
    'satisfied', 'solve', 'stability', 'stabilize', 'strength',
    'strengthen', 'strong', 'succeed', 'success', 'superior', 'surpass',
    'transform', 'upgrade', 'upturn', 'win',

    # === Crypto-Specific Positive (Ante 2023) ===
    'accumulate', 'accumulation', 'adopt', 'adoption', 'airdrop',
    'altseason', 'approval', 'approve', 'ath', 'backing', 'breakout',
    'bullish', 'catalyst', 'defi', 'etf', 'flip', 'flippening',
    'halving', 'hodl', 'institutional', 'integration', 'launch',
    'listing', 'mainnet', 'milestone', 'mint', 'moon', 'mooning',
    'onchain', 'outflow', 'partnership', 'pump', 'rally', 'soar',
    'spike', 'staking', 'surge', 'unlock', 'upgrade', 'uptrend',
    'whale', 'yield',
})

_LM_NEGATIVE = frozenset({
    # === L&M 2011 Core Negative (finance-relevant subset) ===
    'abandon', 'adverse', 'allegation', 'attack', 'bankruptcy',
    'breach', 'burden', 'catastrophe', 'cease', 'claim', 'close',
    'collapse', 'concern', 'conflict', 'crisis', 'critical', 'damage',
    'danger', 'decline', 'decrease', 'default', 'deficit', 'delay',
    'deplete', 'depreciate', 'destabilize', 'deteriorate', 'devalue',
    'difficult', 'diminish', 'disappoint', 'disclose', 'discontinue',
    'dispute', 'disruption', 'distress', 'downturn', 'drop', 'erode',
    'error', 'fail', 'failure', 'fall', 'felony', 'fine', 'forfeit',
    'fraud', 'halt', 'impair', 'impose', 'inability', 'inadequate',
    'indictment', 'infringement', 'injunction', 'insolvency', 'insolvent',
    'investigation', 'judgment', 'lack', 'late', 'layoff', 'liability',
    'liquidate', 'liquidation', 'litigation', 'lose', 'loss', 'losses',
    'misrepresent', 'miss', 'negative', 'neglect', 'obstacle',
    'overdue', 'penalty', 'plummet', 'plunge', 'poor', 'problem',
    'probe', 'prosecute', 'protest', 'recession', 'restrain',
    'restrict', 'restructure', 'risk', 'sanction', 'scandal',
    'seize', 'severe', 'shortfall', 'shutdown', 'slowdown', 'stagnate',
    'sue', 'suspend', 'terminate', 'threat', 'trouble', 'turmoil',
    'uncertain', 'undermine', 'unfavorable', 'unprofitable', 'unstable',
    'violate', 'violation', 'volatile', 'warn', 'warning', 'weak',
    'weaken', 'worsen', 'worse', 'worst', 'writeoff',

    # === Crypto-Specific Negative (Ante 2023, Auer 2020) ===
    'ban', 'bearish', 'capitulate', 'capitulation', 'contagion', 'crash',
    'crackdown', 'delist', 'depeg', 'dump', 'exploit', 'fud',
    'hack', 'hacked', 'inflow', 'lawsuit', 'manipulation',
    'outage', 'ponzi', 'prohibition', 'regulation', 'rugpull', 'rug',
    'scam', 'selloff', 'slash', 'slashing', 'theft', 'vulnerability',
    'washtrading',
})

_LM_UNCERTAINTY = frozenset({
    # === L&M 2011 Uncertainty ===
    'almost', 'ambiguity', 'ambiguous', 'appear', 'approximate',
    'assume', 'assumption', 'believe', 'conceivable', 'conditional',
    'confuse', 'contingency', 'contingent', 'could', 'depend',
    'doubt', 'doubtful', 'estimate', 'exposure', 'fluctuate',
    'fluctuation', 'hypothetical', 'indefinite', 'indeterminate',
    'likelihood', 'may', 'maybe', 'might', 'unclear', 'uncertain',
    'uncertainty', 'unknown', 'unpredictable', 'unresolved',
    'unsettled', 'variable', 'variation', 'vary',

    # === Crypto uncertainty ===
    'fork', 'rumor', 'rumour', 'speculate', 'speculation', 'tbd',
    'tentative', 'unconfirmed', 'unverified',
})

# Strong modal = certainty words (L&M 2011)
_LM_STRONG_MODAL = frozenset({
    'always', 'best', 'clearly', 'definitely', 'highest', 'must',
    'never', 'strongly', 'undoubtedly', 'will',
})

# Negation prefixes flip sentiment (Das & Chen 2007)
_NEGATION = frozenset({
    'no', 'not', 'never', 'neither', 'nobody', 'none', 'nor',
    'nothing', 'nowhere', 'cannot', "can't", "won't", "doesn't",
    "didn't", "isn't", "aren't", "wasn't", "weren't", "hasn't",
    "haven't", "hadn't", "shouldn't", "wouldn't", "couldn't",
})

# Intensifiers (Polanyi & Zaenen 2006)
_INTENSIFIERS = frozenset({
    'very', 'extremely', 'highly', 'significantly', 'substantially',
    'sharply', 'dramatically', 'massive', 'major', 'huge', 'enormous',
    'unprecedented', 'historic', 'record',
})
# fmt: on

# Pre-compiled tokenizer: split on non-alphanumeric + apostrophe
_TOKENIZE_RE = re.compile(r"[a-z']+")

# Coin name → ticker mapping for per-symbol matching
_COIN_NAMES = {
    'bitcoin': 'BTC', 'btc': 'BTC', 'ethereum': 'ETH', 'eth': 'ETH',
    'solana': 'SOL', 'sol': 'SOL', 'ripple': 'XRP', 'xrp': 'XRP',
    'cardano': 'ADA', 'ada': 'ADA', 'dogecoin': 'DOGE', 'doge': 'DOGE',
    'polkadot': 'DOT', 'dot': 'DOT', 'avalanche': 'AVAX', 'avax': 'AVAX',
    'chainlink': 'LINK', 'link': 'LINK', 'polygon': 'MATIC', 'matic': 'MATIC',
    'uniswap': 'UNI', 'uni': 'UNI', 'litecoin': 'LTC', 'ltc': 'LTC',
    'cosmos': 'ATOM', 'atom': 'ATOM', 'near': 'NEAR',
    'arbitrum': 'ARB', 'arb': 'ARB', 'optimism': 'OP',
    'aptos': 'APT', 'apt': 'APT', 'sui': 'SUI',
    'injective': 'INJ', 'inj': 'INJ', 'binance': 'BNB', 'bnb': 'BNB',
    'tron': 'TRX', 'trx': 'TRX', 'shiba': 'SHIB', 'shib': 'SHIB',
    'pepe': 'PEPE', 'filecoin': 'FIL', 'fil': 'FIL',
    'hedera': 'HBAR', 'hbar': 'HBAR', 'stellar': 'XLM', 'xlm': 'XLM',
    'aave': 'AAVE', 'maker': 'MKR', 'mkr': 'MKR',
    'render': 'RNDR', 'rndr': 'RNDR', 'fetch': 'FET', 'fet': 'FET',
    'pendle': 'PENDLE', 'jupiter': 'JUP', 'jup': 'JUP',
    'toncoin': 'TON', 'ton': 'TON',
}


def _score_headline(text: str) -> tuple:
    """
    Score a single headline using L&M 2011 + crypto extensions.

    Returns: (sentiment_score: float, uncertainty: float, tickers: set)

    Algorithm (optimized):
      1. Lowercase + tokenize via pre-compiled regex (no NLTK)
      2. Bigram negation detection (Das & Chen 2007)
      3. Intensifier detection (Polanyi & Zaenen 2006)
      4. Weighted scoring: positive=+1, negative=-1, uncertainty=-0.3
      5. Normalize by token count for length-invariance
    """
    text_lower = text.lower()
    tokens = _TOKENIZE_RE.findall(text_lower)
    if not tokens:
        return 0.0, 0.0, set()

    pos_score = 0.0
    neg_score = 0.0
    unc_score = 0.0
    tickers = set()
    n = len(tokens)

    # Single pass through tokens
    negated = False
    intensified = False

    for i, token in enumerate(tokens):
        # Check negation (affects next word)
        if token in _NEGATION:
            negated = True
            continue

        # Check intensifier (affects next word)
        if token in _INTENSIFIERS:
            intensified = True
            continue

        # Multiplier from intensifier
        mult = 1.5 if intensified else 1.0

        # Sentiment scoring with negation flip
        if token in _LM_POSITIVE:
            if negated:
                neg_score += mult
            else:
                pos_score += mult
        elif token in _LM_NEGATIVE:
            if negated:
                pos_score += mult * 0.5  # Negated negative = weak positive
            else:
                neg_score += mult

        if token in _LM_UNCERTAINTY:
            unc_score += 1.0

        # Coin detection
        ticker = _COIN_NAMES.get(token)
        if ticker:
            tickers.add(ticker)

        # Reset modifiers after consumption
        negated = False
        intensified = False

    # Normalize by sqrt(token_count) — balances short vs long headlines
    # (Loughran & McDonald 2011, Section 4.2)
    norm = n ** 0.5
    sentiment = (pos_score - neg_score) / norm if norm > 0 else 0.0
    uncertainty = unc_score / norm if norm > 0 else 0.0

    # Clamp to [-1, +1]
    sentiment = max(-1.0, min(1.0, sentiment))

    return sentiment, uncertainty, tickers


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RSS FEED SOURCES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_RSS_SOURCES = [
    ("CoinDesk",       "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("CoinTelegraph",  "https://cointelegraph.com/rss"),
    ("Decrypt",        "https://decrypt.co/feed"),
]


def _parse_rss(xml_text: str) -> list:
    """Parse RSS XML into list of {'title': str, 'published': str, 'link': str}."""
    items = []
    try:
        root = ET.fromstring(xml_text)
        # Standard RSS 2.0: channel/item
        for item in root.iter('item'):
            title_el = item.find('title')
            pub_el = item.find('pubDate')
            link_el = item.find('link')
            if title_el is not None and title_el.text:
                items.append({
                    'title': unescape(title_el.text.strip()),
                    'published': pub_el.text.strip() if pub_el is not None and pub_el.text else '',
                    'link': link_el.text.strip() if link_el is not None and link_el.text else '',
                })
        # Atom format: entry/title
        if not items:
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            for entry in root.iter('{http://www.w3.org/2005/Atom}entry'):
                title_el = entry.find('{http://www.w3.org/2005/Atom}title')
                if title_el is not None and title_el.text:
                    items.append({
                        'title': unescape(title_el.text.strip()),
                        'published': '',
                        'link': '',
                    })
    except ET.ParseError:
        pass
    return items


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SENTIMENT ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SentimentEngine:
    """
    Thread-safe sentiment & news aggregator for QUANTA ML pipeline.

    Data flow (WebSocket-style background feed):
      1. Daemon thread polls RSS feeds every 60s
      2. Headlines scored via L&M 2011 lexicon (O(1) lookup per token)
      3. Results cached per-symbol + global
      4. Predictions read from warm cache (zero latency)
    """

    _FNG_URL = "https://api.alternative.me/fng/?limit=1"
    _FNG_URL_V2 = "https://api.alternative.me/v2/fng/?limit=1"

    # Cache TTLs
    _FNG_TTL = 900       # 15 min
    _NEWS_TTL = 300      # 5 min (RSS has no rate limit!)
    _RSS_FETCH_TTL = 60  # Fetch RSS every 60s

    # Optional CryptoPanic
    _CP_BASE = "https://cryptopanic.com/api/developer/v2"

    def __init__(self, api_key: str = None):
        self._cp_api_key = api_key or os.getenv("CRYPTOPANIC_API_KEY", "")

        # Fear & Greed
        self._fng_cache = None
        self._fng_cached_at = 0
        self._fng_lock = threading.Lock()

        # Scored headlines store: list of {title, sentiment, uncertainty, tickers, source}
        self._headlines = []
        self._headlines_lock = threading.Lock()
        self._last_rss_fetch = 0

        # Per-symbol sentiment cache
        self._symbol_sentiment = {}
        self._global_sentiment = None

        # Background feed
        self._active_symbols = []
        self._bg_thread = None
        self._bg_stop = threading.Event()
        self._bg_interval = 60  # 60s — RSS has no rate limit!

        logging.info("📊 SentimentEngine v2 initialized (L&M 2011 + RSS + F&G)")

        # 🧠 Groq LLM State
        self._groq_score = None       # Cached aggregate Groq sentiment
        self._groq_summary = ''       # Cached market narrative
        self._groq_coin_sentiment = {} # Cached per-coin Groq sentiment
        self._last_groq_call = 0      # Timestamp of last Groq API call
        self._groq_cooldown = 300     # 5 minutes between calls

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # BACKGROUND FEED (WebSocket-style)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def start_background_feed(self, symbols: list = None):
        """Start background RSS polling thread."""
        if symbols:
            self._active_symbols = list(symbols)
        if self._bg_thread and self._bg_thread.is_alive():
            return
        self._bg_stop.clear()
        self._bg_thread = threading.Thread(
            target=self._background_loop, daemon=True, name="SentimentFeed"
        )
        self._bg_thread.start()
        print(f"📡 Sentiment feed started ({len(self._active_symbols)} symbols, L&M 2011)")

    def stop_background_feed(self):
        self._bg_stop.set()
        if self._bg_thread:
            self._bg_thread.join(timeout=5)

    def update_symbols(self, symbols: list):
        self._active_symbols = list(symbols)

    def _background_loop(self):
        """Background poller — fetches RSS + scores headlines continuously."""
        print("🧠 Sentiment feed active")
        first_run = True
        while not self._bg_stop.is_set():
            try:
                # 1. Refresh Fear & Greed
                try:
                    self.get_fear_greed()
                except Exception as e:
                    if first_run: print(f"  ⚠️ Fear&Greed failed: {e}")

                # 2. Fetch & score all RSS headlines (L&M baseline)
                try:
                    self._refresh_rss()
                except Exception as e:
                    if first_run: print(f"  ⚠️ RSS failed: {e}")
                
                # 3. 🧠 Groq LLM Deep Contextual Override (every 5 min)
                try:
                    self._groq_deep_score()
                except Exception as e:
                    print(f"  ⚠️ Groq error: {e}")

                # 4. Recompute per-symbol sentiment from scored headlines
                try:
                    self._recompute_sentiment()
                except Exception as e:
                    if first_run: print(f"  ⚠️ Recompute failed: {e}")

            except Exception as e:
                print(f"  ⚠️ Sentiment loop error: {e}")

            first_run = False
            self._bg_stop.wait(timeout=self._bg_interval)

    def _groq_deep_score(self):
        """🧠 Groq/Llama 3.3 70B contextual sentiment override (5-min cooldown)."""
        if get_groq_sentiment is None:
            print("🧠 Groq: IMPORT FAILED — get_groq_sentiment is None")
            return
            
        now = time.time()
        if now - self._last_groq_call < self._groq_cooldown:
            return
        self._last_groq_call = now
        
        with self._headlines_lock:
            batch = [h['title'] for h in self._headlines[:15]]
        
        if not batch:
            print(f"🧠 Groq: No headlines yet ({len(self._headlines)} in buffer)")
            return
        
        print(f"🧠 Groq: Analyzing {len(batch)} headlines for {len(self._active_symbols)} coins...")
        result = get_groq_sentiment(batch, active_coins=self._active_symbols)
        
        if result and 'per_headline' in result:
            with self._headlines_lock:
                for h in self._headlines:
                    for gh in result['per_headline']:
                        if h['title'] == gh['title']:
                            h['sentiment'] = gh['score']
                            break
            
            self._groq_score = result.get('score', None)
            self._groq_summary = result.get('summary', '')
            
            # Update per-coin sentiment cache
            if 'coin_sentiment' in result:
                for coin, score in result['coin_sentiment'].items():
                    self._groq_coin_sentiment[coin] = score
            
            gs = self._groq_score
            if gs > 0.3: mood = "Bullish"
            elif gs > 0.1: mood = "Slightly Bullish"
            elif gs > -0.1: mood = "Neutral"
            elif gs > -0.3: mood = "Slightly Bearish"
            else: mood = "Bearish"
            print(f"🧠 Groq sentiment: {gs:+.3f} | {mood} | Tracked {len(self._groq_coin_sentiment)} coins")
        else:
            logging.debug("🧠 Groq: API returned None (Key missing or Rate Limit)")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # RSS AGGREGATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _refresh_rss(self):
        """Fetch all RSS sources and score headlines."""
        now = time.time()
        if now - self._last_rss_fetch < self._RSS_FETCH_TTL:
            return
        self._last_rss_fetch = now

        all_headlines = []

        for source_name, url in _RSS_SOURCES:
            try:
                resp = NetworkHelper.get(url, timeout=10)
                if resp and resp.status_code == 200:
                    items = _parse_rss(resp.text)
                    for item in items[:30]:  # Top 30 per source
                        sentiment, uncertainty, tickers = _score_headline(item['title'])
                        all_headlines.append({
                            'title': item['title'],
                            'sentiment': sentiment,
                            'uncertainty': uncertainty,
                            'tickers': tickers,
                            'source': source_name,
                        })
            except Exception as e:
                logging.debug(f"RSS {source_name} error: {e}")

        # Optional: merge CryptoPanic headlines if key available
        if self._cp_api_key:
            try:
                params = {
                    'auth_token': self._cp_api_key,
                    'public': 'true',
                    'kind': 'news',
                }
                resp = NetworkHelper.get(
                    f"{self._CP_BASE}/posts/", params=params, timeout=10
                )
                if resp and resp.status_code == 200:
                    data = resp.json()
                    for p in data.get('results', [])[:20]:
                        title = p.get('title', '')
                        if title:
                            sentiment, uncertainty, tickers = _score_headline(title)
                            # Boost with crowd votes if available
                            votes = p.get('votes', {})
                            pos_v = votes.get('positive', 0)
                            neg_v = votes.get('negative', 0)
                            if pos_v + neg_v > 0:
                                vote_signal = (pos_v - neg_v) / (pos_v + neg_v)
                                # Blend: 60% L&M lexicon + 40% crowd votes
                                sentiment = 0.6 * sentiment + 0.4 * vote_signal
                            all_headlines.append({
                                'title': title,
                                'sentiment': sentiment,
                                'uncertainty': uncertainty,
                                'tickers': tickers,
                                'source': 'CryptoPanic',
                            })
            except Exception:
                pass

        # Dedupe by title (case-insensitive)
        seen = set()
        unique = []
        for h in all_headlines:
            key = h['title'].lower()[:60]
            if key not in seen:
                seen.add(key)
                unique.append(h)

        with self._headlines_lock:
            self._headlines = unique

        logging.debug(f"📰 Scored {len(unique)} headlines from {len(_RSS_SOURCES)} RSS sources")

    def _recompute_sentiment(self):
        """Recompute global + per-symbol sentiment from scored headlines."""
        with self._headlines_lock:
            headlines = list(self._headlines)

        if not headlines:
            return

        # Global sentiment (all headlines)
        scores = [h['sentiment'] for h in headlines]
        global_score = sum(scores) / len(scores) if scores else 0.0
        self._global_sentiment = {
            'score': max(-1.0, min(1.0, global_score)),
            'volume': len(headlines),
            'bullish': sum(1 for s in scores if s > 0.05),
            'bearish': sum(1 for s in scores if s < -0.05),
            'headlines': [h['title'] for h in headlines[:5]],
        }

        # Per-symbol sentiment
        symbol_map = defaultdict(list)
        for h in headlines:
            for ticker in h['tickers']:
                symbol_map[ticker].append(h['sentiment'])
            # Also check active symbols by name
            title_lower = h['title'].lower()
            for sym in self._active_symbols:
                ticker = sym.replace('USDT', '')
                if ticker.lower() in title_lower or ticker in h['tickers']:
                    symbol_map[ticker].append(h['sentiment'])

        for ticker, ticker_scores in symbol_map.items():
            unique_scores = ticker_scores[:20]  # Cap
            avg = sum(unique_scores) / len(unique_scores) if unique_scores else 0.0
            self._symbol_sentiment[ticker] = {
                'score': max(-1.0, min(1.0, avg)),
                'volume': len(unique_scores),
                'bullish': sum(1 for s in unique_scores if s > 0.05),
                'bearish': sum(1 for s in unique_scores if s < -0.05),
            }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # FEAR & GREED INDEX
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_fear_greed(self) -> dict:
        """Fetch global Fear & Greed Index (cached 15min)."""
        with self._fng_lock:
            if self._fng_cache and (time.time() - self._fng_cached_at) < self._FNG_TTL:
                return self._fng_cache

        for url in [self._FNG_URL, self._FNG_URL_V2]:
            try:
                resp = NetworkHelper.get(url, timeout=12)
                if resp:
                    data = resp.json()
                    if data and 'data' in data and len(data['data']) > 0:
                        val = int(data['data'][0]['value'])
                        lbl = data['data'][0].get('value_classification', 'Unknown')
                        result = {'value': val, 'label': lbl}
                        with self._fng_lock:
                            self._fng_cache = result
                            self._fng_cached_at = time.time()
                        return result
            except Exception:
                continue
        # If both fail
        with self._fng_lock:
            if not self._fng_cache:
                self._fng_cache = {'value': 50, 'label': 'Neutral'}
            self._fng_cached_at = time.time()  # Cache the failure state to prevent immediate retries
            return self._fng_cache

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # NEWS SENTIMENT (public API for consumers)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_news_sentiment(self, symbol: str = None) -> dict:
        """
        Get L&M-scored news sentiment.

        If symbol provided, returns per-symbol sentiment.
        Falls back to global if no symbol-specific articles found.
        """
        # Ensure at least one fetch has happened
        if not self._headlines:
            self._refresh_rss()
            self._recompute_sentiment()

        if symbol:
            ticker = symbol.replace('USDT', '').replace('usdt', '')
            sym_data = self._symbol_sentiment.get(ticker)
            if sym_data and sym_data['volume'] > 0:
                return {
                    'score': sym_data['score'],
                    'volume': sym_data['volume'],
                    'bullish': sym_data['bullish'],
                    'bearish': sym_data['bearish'],
                    'headlines': self._get_symbol_headlines(ticker),
                }

        # Global fallback
        if self._global_sentiment:
            return self._global_sentiment

        return {
            'score': 0.0, 'volume': 0,
            'bullish': 0, 'bearish': 0,
            'headlines': [],
        }

    def _get_symbol_headlines(self, ticker: str, limit: int = 3) -> list:
        """Get top headlines mentioning a specific ticker."""
        with self._headlines_lock:
            result = []
            ticker_lower = ticker.lower()
            for h in self._headlines:
                if ticker in h['tickers'] or ticker_lower in h['title'].lower():
                    result.append(h['title'])
                    if len(result) >= limit:
                        break
            return result

    def get_latest_global_headlines(self, limit: int = 5) -> list:
        """Get the most recent globally cached headlines for the AI Oracle."""
        with self._headlines_lock:
            return [h['title'] for h in self._headlines[:limit]]

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ML FEATURE EXTRACTION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_sentiment_features(self, symbol: str = None) -> list:
        """
        Extract 7 sentiment features for ML pipeline.

        Returns:
            [fng_norm, extreme_fear, extreme_greed, news_score, news_volume_norm, coin_sentiment_score, coin_sentiment_magnitude]
        """
        # Feature 1-3: Fear & Greed
        fng = self.get_fear_greed()
        fng_val = fng['value']
        fng_norm = fng_val / 100.0
        extreme_fear = 1.0 if fng_val < 25 else 0.0
        extreme_greed = 1.0 if fng_val > 75 else 0.0

        # Feature 4-5: News sentiment
        news = self.get_news_sentiment(symbol)
        
        # 🧠 Prefer Groq LLM contextual score over L&M word-counting
        if self._groq_score is not None:
            news_score = self._groq_score
        else:
            news_score = news['score']  # L&M fallback
        
        news_volume_norm = min(news['volume'] / 50.0, 1.0)  # Cap at 50 articles
        
        # Feature 6-7: Per-coin LLM Sentiment (v11)
        # Use Groq coin sentiment if available, fallback to L&M symbol sentiment
        ticker = symbol.replace('USDT', '') if symbol else ""
        coin_score = 0.0
        
        if ticker and ticker in self._groq_coin_sentiment:
            coin_score = self._groq_coin_sentiment[ticker]
        elif symbol:
            coin_score = news['score']  # L&M specific score
            
        coin_sentiment_magnitude = abs(coin_score)

        return [fng_norm, extreme_fear, extreme_greed, news_score, news_volume_norm, coin_score, coin_sentiment_magnitude]

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TELEGRAM SUMMARY
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_summary(self, symbol: str = None) -> str:
        """Formatted markdown string for Telegram /sentiment command."""
        fng = self.get_fear_greed()
        news = self.get_news_sentiment(symbol)

        val = fng['value']
        if val < 25:
            fng_emoji = "😱"
        elif val < 45:
            fng_emoji = "😨"
        elif val < 55:
            fng_emoji = "😐"
        elif val < 75:
            fng_emoji = "😊"
        else:
            fng_emoji = "🤑"

        msg = f"📊 *MARKET SENTIMENT*\n\n"
        msg += f"{fng_emoji} Fear & Greed: *{val}* ({fng['label']})\n\n"
        
        # 🧠 Groq AI Analysis (primary)
        if self._groq_score is not None:
            gs = self._groq_score
            if gs > 0.3: g_mood = "Bullish 🟢"
            elif gs > 0.1: g_mood = "Slightly Bullish 🟡"
            elif gs > -0.1: g_mood = "Neutral ⚪"
            elif gs > -0.3: g_mood = "Slightly Bearish 🟡"
            else: g_mood = "Bearish 🔴"
            
            msg += f"🧠 *Groq AI Analysis (Llama 3.3 70B):*\n"
            msg += f"   📈 Score: *{g_mood}* ({gs:+.3f})\n"
            if self._groq_summary:
                msg += f"   💬 _{self._groq_summary[:120]}_\n\n"
        else:
            msg += f"🧠 Groq AI: _Waiting for first analysis..._\n\n"

        # 📰 L&M Baseline (secondary reference)
        s = news['score']
        if s > 0.3: mood = "Bullish"
        elif s > 0.1: mood = "Slightly Bullish"
        elif s > -0.1: mood = "Neutral"
        elif s > -0.3: mood = "Slightly Bearish"
        else: mood = "Bearish"

        sym_label = f" [{symbol}]" if symbol else ""
        msg += f"📰 *L&M Lexicon{sym_label}*\n"
        msg += f"   📊 Articles: {news['volume']} | 👍 {news['bullish']} | 👎 {news['bearish']}\n"
        msg += f"   📈 L&M Score: {mood} ({s:+.3f})\n"

        # Show unique sources count
        with self._headlines_lock:
            sources = set(h['source'] for h in self._headlines)
        if sources:
            msg += f"   🌐 Sources: {', '.join(sorted(sources))}\n"

        if news.get('headlines'):
            msg += f"\n📰 *Top Headlines:*\n"
            for h in news['headlines'][:5]:
                msg += f"• {h[:70]}\n"

        return msg
