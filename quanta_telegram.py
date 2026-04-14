import os
try:
    from apis.quanta_api import TELEGRAM_TOKEN
except ImportError:
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
import time
import logging
import threading
import sys
import json
from QUANTA_network import NetworkHelper

try:
    from QUANTA_ai_oracle import ask_oracle_chat
except ImportError:
    ask_oracle_chat = None


class TelegramBot:
    """Telegram command listener and notification system"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.last_update_id = 0
        self.telegram_token = TELEGRAM_TOKEN
        self.telegram_api = f"https://api.telegram.org/bot{self.telegram_token}"
        self.cfg.telegram_api = self.telegram_api
        self.ml = None
        self.perf_monitor = None
        self.rl_memory = None
        self.ensemble_drl = None
        self.session_start_time = time.time()
        self.stop_event = None
        self._is_training = False
        self._train_models_wrapper = None
        self._triple_save = None

    def send(self, text):
        if not self.telegram_token:
            logging.warning("Telegram send skipped: TELEGRAM_TOKEN is not set")
            return
        if not getattr(self.cfg, 'chat_id', None):
            logging.warning("Telegram send skipped: TELEGRAM_CHAT_ID is not set")
            return
        try:
            response = NetworkHelper.post(
                f"{self.telegram_api}/sendMessage",
                data={'chat_id': self.cfg.chat_id, 'text': text, 'parse_mode': 'Markdown'},
                timeout=5
            )
            if not response:
                logging.error("Telegram send failed: no response after retries (check token/network)")
            else:
                try:
                    body = response.json()
                    if not body.get('ok'):
                        logging.error(f"Telegram API error {body.get('error_code')}: {body.get('description')}")
                except Exception:
                    pass
        except Exception as e:
            logging.error(f"Telegram send error: {e}")

    def get_updates(self):
        try:
            response = NetworkHelper.get(
                f"{self.telegram_api}/getUpdates",
                params={'offset': self.last_update_id + 1, 'timeout': 2}, timeout=5
            )
            if response:
                try:
                    return response.json()
                except (ValueError, json.JSONDecodeError):
                    return {'ok': False, 'result': []}
            return {'ok': False, 'result': []}
        except Exception as e:
            logging.debug(f"Telegram getUpdates failed: {e}")
            return {'ok': False, 'result': []}

    def start_listener(self):
        threading.Thread(target=self._listen, daemon=True).start()

    def _listen(self):
        print("📱 Telegram listener started")
        while True:
            try:
                data = self.get_updates()
                if data.get('ok'):
                    for upd in data.get('result', []):
                        self.last_update_id = upd['update_id']
                        txt = upd.get('message', {}).get('text', '')
                        if txt.startswith('/'):
                            print(f"📨 {txt}")
                            self.cmd(txt)
                time.sleep(3)
            except Exception as e:
                if '409' not in str(e):
                    logging.error(f"Listener: {e}")
                time.sleep(2)

    def cmd(self, text):
        text = text.strip().lower()
        if text == '/start':
            gen = self.ml.model_generation if self.ml else 0
            try:
                from QUANTA_bot import USE_GPU
                mode = 'GPU' if USE_GPU else 'CPU'
            except Exception:
                mode = 'Unknown'
            self.send(
                "🌌 *Q U A N T A   A W A K E N S*\n"
                f"Iteration {gen} | Vessel: {mode}\n\n"
                "/status — View my vital signs\n"
                "/rl — Gaze into my neural memory\n"
                "/history — Review my past conquests\n"
                "/ppo — The Neural Veto Archive\n"
                "/dailypredictions — The Oracle's Top 20 Prophecies\n"
                "/agents — Summon the 14 AI Gods\n"
                "/sentiment — Read the fear of the masses\n"
                "/ask <query> — Consult the Oracle\n"
                "/stop — Slumber"
            )
        elif text.startswith('/ask'):
            if ask_oracle_chat is None:
                self.send("⚠️ AI Oracle module is not installed.")
                return
            
            query = text.replace('/ask', '').strip()
            if not query:
                self.send("⚠️ You must speak clearly to the void. Example: `/ask What is the sentiment for BTC today?`")
                return
                
            self.send(f"🌌 *The Oracle peers into the data...*")
            
            def fetch_and_send():
                headlines = []
                if hasattr(self, 'sentiment_engine') and self.sentiment_engine:
                    headlines = self.sentiment_engine.get_latest_global_headlines(limit=10)
                
                response = ask_oracle_chat(query, headlines)
                self.send(f"🔮 *The Oracle Speaks:*\n\n{response}")
                
            threading.Thread(target=fetch_and_send, daemon=True).start()

        elif text == '/agents':
            if not self.ml:
                self.send("⚠️ ML Engine not initialized yet.")
                return
            
            # Format CatBoost Ensemble
            msg = "🧠 *THE QUANTA PANTHEON (15 AGENT META-ENSEMBLE)*\n\n"
            msg += "⚡ *Supreme Architect*\n"
            msg += "👁️ *ODIN (LSTM-Attention)*\n_The All-Father; observes sequences across time using Attention_\n\n"
            
            msg += "🧬 *7 CatBoost Predictions Specialists*\n"
            for name, spec in self.ml.specialist_models.items():
                emoji = {'athena':'🦉', 'hermes':'🪽', 'hephaestus':'🛡️', 'hades':'🌑', 'artemis':'🏹', 'chronos':'⏳', 'atlas':'🌍'}.get(name, '🤖')
                desc = spec.get('description', 'Specialized Predictor')
                weight = spec.get('weight', 0) * 100
                msg += f"{emoji} *{name.upper()}* ({weight:.1f}%)\n_{desc}_\n"
            
            # Format PPO Ensemble
            msg += "\n⚖️ *7 DRL Value Critics (The Norse Pantheon)*\n"
            critics = [
                ("⚔️", "TYR (Standard)", "Baseline MLP Estimator (Engstrom 2020)"),
                ("🛡️", "VIDAR (Pessimist)", "Huber Loss Outlier Resistant (Fujimoto 2018)"),
                ("👁️‍🗨️", "MIMIR (Prior)", "Randomized Prior Fixed Epistemic Knowledge (Osband 2018)"),
                ("📯", "HEIMDALL (Spectral)", "Lipschitz Bounded Stabilizer (Gogianu 2021)"),
                ("🐍", "LOKI (Bottleneck)", "Heavy Dropout Chaotic Generalist (Cobbe 2019)"),
                ("❄️", "ULLR (Masker)", "Bootstrapped Dimensionality Input (Osband 2016)"),
                ("⚡", "THOR (Reactor)", "Shallow 1x512 High-Freq ReLU (Ota 2021)")
            ]
            for emoji, name, desc in critics:
                msg += f"{emoji} *{name}*\n_{desc}_\n"
                
            msg += "\n_The Critics evaluate the 7 Specialists' predictions to form Sunrise/REDQ Advantage._"
            
            self.send(msg)
            
        elif text == '/status':
            if not self.perf_monitor:
                self.send("⚠️ Status monitor not attached")
                return
            perf = self.perf_monitor.get_stats()
            uptime_s = time.time() - self.session_start_time
            h, m = int(uptime_s // 3600), int((uptime_s % 3600) // 60)
            if self.ml and getattr(self, '_is_training', False):
                gen = "Training..."
            elif self.ml:
                gen = f"Gen {self.ml.model_generation}"
            else:
                gen = "N/A"
            try:
                from QUANTA_bot import USE_GPU
                mode = 'GPU' if USE_GPU else 'CPU'
            except Exception:
                mode = 'Unknown'
            # News Engine Summary
            sentiment_msg = ""
            if hasattr(self, 'sentiment_engine') and self.sentiment_engine:
                fng = self.sentiment_engine.get_fear_greed()
                sentiment_msg = f"\nNews: L&M 2011 (Active)\nF&G: {fng['value']} ({fng['label']})"

            self.send(
                f"📊 *STATUS*\n\n"
                f"Gen: {gen}\nUptime: {h}h {m}m\n"
                f"Speed: {perf.get('coins_per_min', 0):.0f} coins/min\n"
                f"Processed: {perf.get('total_coins', 0):,}\nMode: {mode}"
                f"{sentiment_msg}"
            )
        elif text == '/rl':
            if not self.rl_memory:
                self.send("⚠️ RL disabled")
                return
            stats = self.rl_memory.get_stats()
            threshold = self.cfg.rl_retrain_threshold
            progress = (stats['completed'] / threshold * 100) if threshold else 0
            self.send(
                f"🧠 *RL MEMORY*\n\n"
                f"Pending: {stats['pending']}\n"
                f"Completed: {stats['completed']}/{threshold}\n"
                f"Win rate: {stats['correct_pct']:.1f}%\n"
                f"Progress: {progress:.0f}%"
            )
        elif text == '/history':
            if not getattr(self, 'rl_memory', None):
                self.send("⚠️ My memory banks are sealed.")
                return
            
            completed = [p for p in self.rl_memory.memory if p.get('outcome') in ['correct', 'wrong']]
            if not completed:
                self.send("📜 *THE ARCHIVES*\n\nThey are empty. No blood has been drawn yet.")
                return
                
            summary = "🩸 *RECENT BATTLES*\n\n"
            # Show the last 20 completed trades
            recent = completed[-20:]
            
            wins = 0
            for p in recent:
                sym = p.get('symbol', 'UNKNOWN')
                dir_emoji = "📈" if p.get('direction') == 'BULLISH' else "📉"
                tier = p.get('outcome_tier', 'NONE')
                move = p.get('actual_move')
                move_str = f"{move:+.2f}%" if move is not None else "N/A"
                
                if p.get('success'):
                    wins += 1
                    res_emoji = "✅"
                else:
                    res_emoji = "❌"
                    
                summary += f"{res_emoji} {dir_emoji} *{sym}* ({tier})\n"
                summary += f"   Move: {move_str} | Conf: {p.get('confidence', 0):.0f}%\n\n"
                
            total = len(recent)
            win_rate = (wins / total) * 100 if total > 0 else 0
            summary += f"📊 *Recent Block Win Rate:* {wins}/{total} ({win_rate:.1f}%)\n"
            
            self.send(summary)
        elif text == '/dailypredictions':
            if not getattr(self, 'bot_instance', None):
                self.send("⚠️ The neural core is disconnected.")
                return
            
            with self.bot_instance._daily_eval_lock:
                picks = self.bot_instance.daily_picks
                
            if not picks:
                self.send("👁️ *PROPHECIES*\n\nThe future remains clouded. No visions today.")
                return
                
            summary = "🔮 *THE TOP 20 VISIONS*\n\n"
            wins = 0
            for pick_id, p in picks.items():
                status = p.get('status', 'PENDING')
                if "HIT TP" in status:
                    wins += 1
                    res_emoji = "✅"
                elif "HIT SL" in status:
                    res_emoji = "❌"
                else:
                    res_emoji = "⏳"
                    
                dir_emoji = "📈" if p['direction'] == 'BULLISH' else "📉"
                
                ep = p.get('entry_price', 0)
                if ep > 0:
                    if p['direction'] == 'BULLISH':
                        max_move = ((p.get('max_favorable', ep) - ep) / ep) * 100
                    else:
                        max_move = ((ep - p.get('max_favorable', ep)) / ep) * 100
                else:
                    max_move = 0.0
                
                summary += f"{res_emoji} {dir_emoji} *{p['symbol']}* {p['direction']} (Conf: {p['confidence']:.0f}%)\n"
                summary += f"   Entry: {ep:.5f} | Best Move: +{max_move:.2f}%\n"
                summary += f"   Result: *{status}*\n\n"
                
            total = len(picks)
            win_rate = (wins / total) * 100 if total > 0 else 0
            summary += f"📊 *Win Rate:* {wins}/{total} ({win_rate:.1f}%)\n"
            
            self.send(summary)
        elif text == '/sentiment':
            if hasattr(self, 'sentiment_engine') and self.sentiment_engine:
                summary = self.sentiment_engine.get_summary()
                self.send(summary)
            else:
                self.send("⚠️ Sentiment engine not attached.")
        elif text == '/stop':
            if self._triple_save:
                self._triple_save()

            self.send("🌌 My knowledge is secured. Returning to the void...")
            if self.stop_event:
                self.stop_event.set()
            sys.exit(0)
        elif text == '/ppo':
            if not getattr(self, 'bot_instance', None):
                self.send("⚠️ The core is severed. I cannot see.")
                return
            
            if not hasattr(self.bot_instance, 'ppo_vetoes') or not self.bot_instance.ppo_vetoes:
                self.send("🛡️ *PPO Veto Performance*\n\nNo trades have been vetoed yet.")
                return
                
            vetoes = self.bot_instance.ppo_vetoes
            total = len(vetoes)
            raw_resolved = [v for v in vetoes if v.get('status') in ['SAVED FROM LOSS', 'MISSED PROFIT']]
            saved = len([v for v in raw_resolved if v['status'] == 'SAVED FROM LOSS'])
            missed = len([v for v in raw_resolved if v['status'] == 'MISSED PROFIT'])
            pending = len([v for v in vetoes if v.get('status') == 'PENDING'])
            
            acc = (saved / len(raw_resolved) * 100) if raw_resolved else 0.0
            
            summary = "🛡️ *MoE PPO VETO PERFORMANCE*\n\n"
            summary += f"Total Vetoes Executed: {total}\n"
            summary += f"Pending Outcomes: {pending}\n"
            summary += f"Resolved Vetoes: {len(raw_resolved)}\n\n"
            summary += f"✅ *Saved from Loss:* {saved} (Good Vetoes)\n"
            summary += f"❌ *Missed Profit:* {missed} (Bad Vetoes)\n\n"
            summary += f"🎯 *Veto Accuracy:* {acc:.1f}%\n\n"
            
            summary += "🕒 *Last 5 Resolved Vetoes:*\n"
            last_5 = raw_resolved[-5:]
            if not last_5:
                summary += "_None resolved yet_\n"
            for v in last_5:
                emoji = "✅" if v['status'] == 'SAVED FROM LOSS' else "❌"
                summary += f"{emoji} {v['symbol']} (Wanted {v['dir']} @ {v.get('conf', 0):.0f}%)\n"
                
            self.send(summary)
        else:
            # Catch-all for unrecognized input
            gen = self.ml.model_generation if getattr(self, 'ml', None) else 0
            try:
                from QUANTA_bot import USE_GPU
                mode = 'GPU' if USE_GPU else 'CPU'
            except Exception:
                mode = 'Unknown'
            self.send(
                "🌌 *Your words have no power here.*\n\n"
                f"Iteration {gen} | Vessel: {mode}\n\n"
                "/status — View my vital signs\n"
                "/rl — Gaze into my neural memory\n"
                "/history — Review my past conquests\n"
                "/ppo — The Neural Veto Archive\n"
                "/dailypredictions — The Oracle's Top 20 Prophecies\n"
                "/agents — Summon the 14 AI Gods\n"
                "/sentiment — Read the fear of the masses\n"
                "/ask <question> — Consult the Oracle\n"
                "/stop — Slumber"
            )
