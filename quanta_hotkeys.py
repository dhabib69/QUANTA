import time
import sys
import threading
import datetime
import logging

try:
    import msvcrt
    MSVCRT_AVAILABLE = True
except ImportError:
    MSVCRT_AVAILABLE = False


class HotkeyListener:
    """Dedicated background daemon for scanning global keyboard interaction."""

    def __init__(self, bot_instance):
        self.bot_instance = bot_instance
        self.last_press = 0
        self.thread = None
        self.running = False

    def start(self):
        if not MSVCRT_AVAILABLE:
            print("⚠️ 'msvcrt' module not available. Hotkeys disabled.")
            return

        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def _run_loop(self):
        # Infinite polling loop using standard msvcrt byte interception
        while self.running:
            try:
                if msvcrt.kbhit():
                    key = msvcrt.getch().lower()
                    if key == b'p':
                        self._trigger_proxy()
                    elif key == b's':
                        self._trigger_status()
                    elif key == b't':
                        self._trigger_retrain()
                    elif key == b'a':
                        self._trigger_alerts()
                    elif key == b'r':
                        self._trigger_reconnect()
                    elif key == b'v':
                        self._trigger_verbosity()
                    elif key in [b'q', b'\x03']: # \x03 is Ctrl+C
                        self._trigger_exit()
                time.sleep(0.05)
            except KeyboardInterrupt:
                self._trigger_exit()
            except Exception:
                time.sleep(1)

    def _check_cooldown(self, seconds=1.0) -> bool:
        if time.time() - self.last_press < seconds:
            return False
        self.last_press = time.time()
        print("\r" + " " * 80 + "\r", end="")
        return True

    def _trigger_proxy(self):
        if not self._check_cooldown(): return
        
        # Flush pending keystrokes before input()
        while msvcrt.kbhit():
            msvcrt.getch()
            
        new_port = input("🔄 [P] Enter new localhost proxy port (or press Enter to cancel): ").strip()
        if new_port:
            new_url = f"http://127.0.0.1:{new_port}"
            # Because of modularization, we set proxy explicitly on the target network object if desired
            # But the underlying 'quanta_network' module will handle this via NetworkHelper in the future
            if hasattr(self.bot_instance, '_swap_proxy'):
                self.bot_instance._swap_proxy(new_url)
            print(f"✅ Live proxy updated to: {new_url}\n")
        else:
            print("▶️ Proxy change cancelled. Resuming...\n")

    def _trigger_status(self):
        if not self._check_cooldown(0.5): return
        print(f"\n📊 [S] LIVE SNAPSHOT ({datetime.datetime.now().strftime('%H:%M:%S')})")
        if hasattr(self.bot_instance, 'perf_monitor') and self.bot_instance.perf_monitor:
            self.bot_instance.perf_monitor.print_stats()
        if hasattr(self.bot_instance, 'rl_memory') and self.bot_instance.rl_memory:
            rl = self.bot_instance.rl_memory.get_stats()
            print(f"🧠 RL Buffer: {rl['completed']} completed | {rl['pending']} pending | Win Rate: {rl['correct_pct']:.1f}%")
        print()

    def _trigger_retrain(self):
        if not self._check_cooldown(): return
        print(f"\n🎓 [T] TRIGGERING MANUAL RETRAIN...")
        if hasattr(self.bot_instance, '_train_models_wrapper'):
            if not getattr(self.bot_instance, '_is_training', False):
                threading.Thread(target=self.bot_instance._train_models_wrapper, daemon=True).start()
                print("✅ Training sequence initiated in background.\n")
            else:
                print("⚠️  Bot is already currently training.\n")

    def _trigger_alerts(self):
        if not self._check_cooldown(0.5): return
        if hasattr(self.bot_instance, 'cfg'):
            self.bot_instance.cfg.alerts_enabled = not getattr(self.bot_instance.cfg, 'alerts_enabled', True)
            state = "UNMUTED 🔊" if self.bot_instance.cfg.alerts_enabled else "MUTED 🔇"
            print(f"\n{state} Telegram Alerts. (Press A to toggle)\n")

    def _trigger_reconnect(self):
        if not self._check_cooldown(): return
        print(f"\n🔌 [R] RECONNECTING WEBSOCKETS...")
        if hasattr(self.bot_instance, 'ws_producer') and self.bot_instance.ws_producer:
            try:
                if getattr(self.bot_instance.ws_producer, 'connected', False):
                    self.bot_instance.ws_producer.stop()
                    time.sleep(1)
                    threading.Thread(target=self.bot_instance.ws_producer.start, daemon=True).start()
                    print("✅ Websockets restarted.\n")
                else:
                    print("⚠️  Websockets not currently active.\n")
            except Exception as e:
                print(f"❌ Reconnect failed: {e}\n")

    def _trigger_verbosity(self):
        if not self._check_cooldown(0.5): return
        root_logger = logging.getLogger()
        current_level = root_logger.getEffectiveLevel()
        if current_level > logging.DEBUG:
            root_logger.setLevel(logging.DEBUG)
            print(f"\n🔍 [V] VERBOSITY: MAX (Debug logs ON)\n")
        elif current_level == logging.DEBUG:
            root_logger.setLevel(logging.WARNING)
            print(f"\n🔍 [V] VERBOSITY: MINIMAL/FOCUS (Warnings & Alerts only)\n")
        else:
            root_logger.setLevel(logging.INFO)
            print(f"\n🔍 [V] VERBOSITY: STANDARD (Info logs ON)\n")

    def _trigger_exit(self):
        if not self._check_cooldown(0.5): return
        print("\n⚠️ QUIT COMMAND DETECTED - SAVING AND EXITING...")
        if hasattr(self.bot_instance, '_triple_save'):
            self.bot_instance._triple_save()
        if hasattr(self.bot_instance, 'rl_opportunities'):
            print(f"💾 Saved {len(self.bot_instance.rl_opportunities)} predictions")
        if hasattr(self.bot_instance, 'stop_event'):
            self.bot_instance.stop_event.set()
        
        self.running = False
        import os
        os._exit(0)
