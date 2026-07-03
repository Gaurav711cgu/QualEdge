import os
import time
import logging
import numpy as np
from typing import Dict, Any, Optional

logger = logging.getLogger("Cloud-Inference-Client")

class CloudInferenceClient:
    """
    Handles cloud-based LLM execution (representing Claude 3.5 Sonnet or Gemini 1.5 Pro).
    If API keys are missing, falls back to a high-fidelity cloud mockup with typical latencies.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config["models"]["cloud"]
        self.model_name = self.config["name"]
        self.target_latency = self.config["target_latency_ms"]
        
        # Load keys
        self.gemini_key = os.environ.get("GEMINI_API_KEY")
        self.anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        
        self.active_mode = False
        if self.gemini_key or self.anthropic_key:
            self.active_mode = True
            logger.info("Cloud Inference Client initialized in ACTIVE mode.")
        else:
            logger.info("No cloud API keys found. Cloud client running in SIMULATION mode.")

    def generate(self, query: str) -> Dict[str, Any]:
        """
        Submits the query to the cloud model and returns response text, latency, and estimated pricing.
        """
        start_time = time.perf_counter()
        
        if self.active_mode:
            # Active path if keys are provided
            try:
                if self.anthropic_key:
                    import anthropic
                    client = anthropic.Anthropic(api_key=self.anthropic_key)
                    message = client.messages.create(
                        model=self.model_name,
                        max_tokens=1024,
                        messages=[{"role": "user", "content": query}]
                    )
                    text = message.content[0].text
                    input_tokens = message.usage.input_tokens
                    output_tokens = message.usage.output_tokens
                else:
                    import google.generativeai as genai
                    genai.configure(api_key=self.gemini_key)
                    model = genai.GenerativeModel(self.model_name)
                    response = model.generate_content(query)
                    text = response.text
                    # Estimate tokens for pricing
                    input_tokens = len(query.split()) * 1.3
                    output_tokens = len(text.split()) * 1.3

                latency_ms = (time.perf_counter() - start_time) * 1000.0
                
                # Pricing calculations
                input_cost = (input_tokens / 1000.0) * 0.003
                output_cost = (output_tokens / 1000.0) * 0.015
                total_cost = input_cost + output_cost
                
                return {
                    "text": text,
                    "latency_ms": round(latency_ms, 2),
                    "cost_usd": round(total_cost, 6),
                    "source": "measured_cloud"
                }
            except Exception as e:
                logger.error(f"Active cloud inference failed: {str(e)}. Falling back to simulation.")
                
        # --- SIMULATED CLOUD CLIENT ---
        # Simulate cloud round-trip delay: ~650ms to 950ms
        time.sleep(random_delay := (self.target_latency * 0.9 + np.random.uniform(50, 150)) / 1000.0)
        
        q_lower = query.lower()
        if "farmer" in q_lower or "rectangular field" in q_lower:
            text = (
                "To divide the rectangular field (200m x 100m = 20,000 m²) into three equal zones for wheat, corn, and barley, "
                "each zone must be exactly 6,666.67 m².\n\n"
                "Let's lay them out as adjacent vertical strips. Since the width is 100m, each zone strip will have width:\n"
                "Width_zone = 6,666.67m² / 100m = 66.67m.\n\n"
                "Dimensions for each zone: 66.67m (length) x 100m (width).\n\n"
                "**Irrigation Pipe Layout Optimization:**\n"
                "Since wheat requires 20% more water, we will allocate a dedicated main branch to the wheat zone with "
                "a higher diameter supply pipe (e.g. 3-inch vs 2.5-inch for corn/barley) or design lateral sprinkler lines "
                "spaced 20% closer (e.g. 8m spacing instead of 10m spacing) to achieve the 1.2x volumetric flow multiplier "
                "without increasing the pump pressure head."
            )
        elif "python" in q_lower or "scrapes" in q_lower:
            text = (
                "Here is a complete Python script using BeautifulSoup4 and smtplib for scraping and emailing news digests:\n\n"
                "```python\n"
                "import requests\n"
                "from bs4 import BeautifulSoup\n"
                "import smtplib\n"
                "from email.mime.text import MIMEText\n\n"
                "def scrape_headlines(url):\n"
                "    res = requests.get(url)\n"
                "    soup = BeautifulSoup(res.text, 'html.parser')\n"
                "    headlines = [h.text.strip() for h in soup.find_all('h2', class_='headline')]\n"
                "    return headlines[:5]\n\n"
                "def send_digest(headlines):\n"
                "    msg = MIMEText('\\n'.join(headlines))\n"
                "    msg['Subject'] = 'Daily News Digest'\n"
                "    msg['From'] = 'digest@example.com'\n"
                "    msg['To'] = 'user@example.com'\n"
                "    with smtplib.SMTP('localhost') as server:\n"
                "        server.send_message(msg)\n"
                "```"
            )
        elif "macroeconomic" in q_lower or "interest rates" in q_lower:
            text = (
                "Raising interest rates in a high-debt developing country triggers complex macroeconomic dynamics:\n"
                "1. **Debt Servicing Costs:** Sovereign debt payments rise rapidly if denominated in local currency with short maturities. If debt is in USD, a domestic rate hike doesn't directly raise costs but local currency appreciation can help service foreign debt.\n"
                "2. **Inflation Control:** Higher rates cool aggregate demand by raising the cost of borrowing, which lowers core inflation.\n"
                "3. **Capital Flight Mitigation:** Higher rates attract foreign portfolio investment, stabilizing the local currency exchange rate and dampening import-driven inflation."
            )
        else:
            text = f"Cloud Response: Processed your query: '{query}'. This query required advanced reasoning and was handled in the cloud using Claude 3.5 Sonnet. Latency reflects standard round-trip times."

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        
        # Simulated tokens: inputs ~150, output ~350
        input_tokens = len(query.split()) * 1.3
        output_tokens = len(text.split()) * 1.3
        total_cost = (input_tokens / 1000.0) * 0.003 + (output_tokens / 1000.0) * 0.015
        
        return {
            "text": text,
            "latency_ms": round(latency_ms, 2),
            "cost_usd": round(total_cost, 6),
            "source": "simulated_cloud"
        }
