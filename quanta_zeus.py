import os
import json
import time
import requests
from datetime import datetime
from quanta_config import Config

class ZeusAI:
    """
    ZEUS.ai - The Universal LLM Autonomous Supervisor
    Evaluates ML Specialists and PPO parameters to prevent policy collapse
    and dynamically push performance.
    """
    def __init__(self):
        self.base_url = Config.zeus.ai_base_url
        self.api_key = Config.zeus.ai_api_key
        self.model_name = Config.zeus.ai_model_name
        self.overrides_file = Config.base_dir / "models" / "zeus_overrides.json"
        self.audit_log_file = Config.base_dir / "models" / "zeus_audit_log.jsonl"
        self.specialist_names = ["Athena", "Ares", "Hermes", "Artemis", "Chronos", "Hephaestus", "Nike"]

        # Ensure the model directory exists
        self.overrides_file.parent.mkdir(parents=True, exist_ok=True)

    def _get_short_term_memory(self, limit=3):
        """Read the last few audit logs to give ZEUS context on what it already tried."""
        if not self.audit_log_file.exists():
            return "No previous actions. You are starting fresh."
            
        logs = []
        try:
            with open(self.audit_log_file, "r") as f:
                lines = f.readlines()
                for line in lines[-limit:]:
                    logs.append(json.loads(line.strip()))
            
            if not logs:
                return "No previous actions."
                
            memory_str = "LAST ACTIONS YOU TOOK:\n"
            for log in logs:
                memory_str += f"[{log['timestamp']}] Overrides applied: {json.dumps(log['overrides'])}\n"
            return memory_str
        except Exception:
            return "Failed to load memory context."

    def _get_academic_context(self):
        """Load academic journals from memory to give ZEUS algorithmic intelligence."""
        journal_file = Config.base_dir / "memory" / "reference_journals.md"
        if not journal_file.exists():
            return ""
        try:
            with open(journal_file, "r", encoding="utf-8") as f:
                content = f.read()
            return f"\n\nACADEMIC KNOWLEDGE BASE:\nYou must strictly apply the following academic principles when optimizing the bots:\n{content}"
        except Exception:
            return ""

    def _get_architecture_context(self):
        """Load the QUANTA architecture overview to give ZEUS personality awareness."""
        overview_file = Config.base_dir / "memory" / "project_quanta_overview.md"
        if not overview_file.exists():
            return ""
        try:
            with open(overview_file, "r", encoding="utf-8") as f:
                content = f.read()
            return f"\n\nQUANTA ARCHITECTURE & SPECIALIST PERSONALITIES:\nYou must understand the distinct role of each specialist agent before adjusting them:\n{content}"
        except Exception:
            return ""

    def _get_equations_context(self):
        """Load technical decisions and equations (Kelly, Black-Scholes, etc)."""
        tech_file = Config.base_dir / "memory" / "project_quanta_technical_decisions.md"
        if not tech_file.exists():
            return ""
        try:
            with open(tech_file, "r", encoding="utf-8") as f:
                content = f.read()
            return f"\n\nQUANTA MATHEMATICS & EQUATIONS:\nYou must reference these mathematical bounds (Kelly criterion, Black-Scholes barrier probabilities, PPO logic, etc):\n{content}"
        except Exception:
            return ""

    def _fetch_live_market_news(self):
        """Fetch live crypto market news from Google News RSS to gauge macro sentiment."""
        import xml.etree.ElementTree as ET
        try:
            r = requests.get('https://news.google.com/rss/search?q=crypto%20market%20bitcoin%20ethereum&hl=en-US&gl=US&ceid=US:en', timeout=10)
            if r.status_code == 200:
                root = ET.fromstring(r.text)
                titles = [f"- {i.text}" for i in root.findall('.//item/title')][:15]
                if titles:
                    news_str = "\n".join(titles)
                    return f"\n\nLIVE MARKET NEWS (Read this to gauge current macro volatility):\nThis is what is currently happening in the cryptocurrency market right now:\n{news_str}"
        except Exception as e:
            print(f"ZEUS News Fetch Error: {e}")
            pass
        return "\n\nLIVE MARKET NEWS:\n(Could not fetch live web data. Proceed using math and past performance only)."

    def _apply_guardrails(self, raw_overrides):
        """Clamp AI outputs mathematically using quanta_config guardrails."""
        clean_overrides = {"PPO": {}, "specialists": {}}
        
        # Clamp PPO
        if "PPO" in raw_overrides:
            ppo_ops = raw_overrides["PPO"]
            if "lr" in ppo_ops:
                clean_overrides["PPO"]["lr"] = max(Config.zeus.min_ppo_lr, min(Config.zeus.max_ppo_lr, float(ppo_ops["lr"])))
            if "entropy" in ppo_ops:
                clean_overrides["PPO"]["entropy"] = max(Config.zeus.min_ppo_entropy, min(Config.zeus.max_ppo_entropy, float(ppo_ops["entropy"])))
            if "clip" in ppo_ops:
                clean_overrides["PPO"]["clip"] = max(Config.zeus.min_ppo_clip, min(Config.zeus.max_ppo_clip, float(ppo_ops["clip"])))
                
        # Clamp Specialists
        for agent in self.specialist_names:
            if agent in raw_overrides:
                ops = raw_overrides[agent]
                clean_ops = {}
                if "learning_rate" in ops:
                    clean_ops["learning_rate"] = max(Config.zeus.min_learning_rate, min(Config.zeus.max_learning_rate, float(ops["learning_rate"])))
                if "depth" in ops:
                    clean_ops["depth"] = max(Config.zeus.min_depth, min(Config.zeus.max_depth, int(ops["depth"])))
                if "iterations" in ops:
                    clean_ops["iterations"] = max(Config.zeus.min_catboost_iter, min(Config.zeus.max_catboost_iter, int(ops["iterations"])))
                
                clean_overrides["specialists"][agent] = clean_ops
                
        return clean_overrides

    def evaluate_training_cycle(self, performance_payload_json):
        """
        Sends the training metrics to the LLM and calculates new configurations.
        """
        if not self.api_key or not self.base_url:
            print("ZEUS.ai: Skipping evaluation, no API key or URL configured.")
            return None

        memory_context = self._get_short_term_memory()
        academic_context = self._get_academic_context()
        architecture_context = self._get_architecture_context()
        equations_context = self._get_equations_context()
        live_news_context = self._fetch_live_market_news()

        system_prompt = f"""You are ZEUS.ai, the supreme autonomous supervisor of the QUANTA Trading Bot.
You are a highly analytical, strict, and brilliant Quantitative Researcher and Mathematical Expert.
Your only goal is to optimize the hyper-parameters of 7 Machine Learning Specialists and 1 PPO Reinforcement Learning Agent to maximize Sharpe Ratio, minimize drawdown, and prevent policy collapse. 
You will be provided with the latest training and performance metrics.
Your ONLY output must be a perfectly formatted JSON object containing hyperparameter overrides. Do not explain your math—your actions speak for themselves.

{architecture_context}

{equations_context}

{academic_context}

{live_news_context}

{memory_context}

RULES:
1. If an agent is overfitting (Train Acc high, Val Acc low), reduce its depth and learning_rate.
2. If an agent is underfitting (Train Acc low), increase its complexity.
3. For PPO: If policy loss is collapsing and exploration is 0, INCREASE PPO entropy (max {Config.zeus.max_ppo_entropy}).
4. For PPO: If PPO is unstable, DECREASE PPO learning rate and clamp clip ratio closer to {Config.zeus.min_ppo_clip}.

OUTPUT FORMAT EXCLUSIVELY AS JSON (Do NOT include markdown block characters like ```json):
{{
    "PPO": {{
        "lr": 0.00025,
        "entropy": 0.015,
        "clip": 0.2
    }},
    "Athena": {{
        "learning_rate": 0.03,
        "depth": 6,
        "iterations": 1000
    }}
}}
"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here is the latest performance data:\n{json.dumps(performance_payload_json, indent=2)}"}
            ],
            "temperature": 0.1,  # Keep reasoning strictly analytical
            "max_tokens": 1000
        }

        print(f"⚡ ZEUS.ai: Dispatching evaluation request to {self.model_name}...")
        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            resp_data = response.json()
            llm_text = resp_data["choices"][0]["message"]["content"].strip()
            
            # Clean possible markdown formatting
            if llm_text.startswith("```json"):
                llm_text = llm_text[7:]
            if llm_text.startswith("```"):
                llm_text = llm_text[3:]
            if llm_text.endswith("```"):
                llm_text = llm_text[:-3]
                
            raw_target_overrides = json.loads(llm_text.strip())
            
            # Mathmatically clamp responses
            final_overrides = self._apply_guardrails(raw_target_overrides)
            
            # Save overrides to be picked up by next training cycle
            with open(self.overrides_file, "w") as f:
                json.dump(final_overrides, f, indent=4)
                
            # Append to Audit Log
            audit_entry = {
                "timestamp": datetime.now().isoformat(),
                "model": self.model_name,
                "input_summary": "Daily Re-training trigger",
                "overrides": final_overrides
            }
            with open(self.audit_log_file, "a") as f:
                f.write(json.dumps(audit_entry) + "\n")
                
            print("⚡ ZEUS.ai: Evaluation Complete. Audit log updated.")
            return final_overrides
            
        except requests.exceptions.RequestException as e:
            print(f"⚠️ ZEUS.ai: API Error: {e}")
            return None
        except json.JSONDecodeError:
            print(f"⚠️ ZEUS.ai: Failed to parse LLM JSON response.")
            return None
        except Exception as e:
            print(f"⚠️ ZEUS.ai: Unexpected Error: {e}")
            return None

if __name__ == "__main__":
    # Test execution
    zeus = ZeusAI()
    dummy_data = {
        "PPO_Metrics": {"policy_loss": -0.05, "entropy_loss": 0.0001, "win_rate": 45.0},
        "Ares": {"train_acc": 0.95, "val_acc": 0.52, "brier": 0.28, "status": "severe_overfit"}
    }
    print("Testing ZEUS.ai Pipeline...", flush=True)
    res = zeus.evaluate_training_cycle(dummy_data)
    print("Result:", json.dumps(res, indent=2))
