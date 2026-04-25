# llm-scope 🔭

**A local-first LLM Proxy that visualizes exactly where your calls are slow. No Node.js. No Docker. No account.**

> *Like RelayPlane... but for Python devs. Zero data leaves your machine.*

Are your API calls feeling sluggish, but you don't know if it's the DNS, TTFT, or just the generation? `llm-scope` intercepts your API requests and gives you a sub-millisecond accurate waterfall breakdown right in your browser.

![Dashboard Preview](https://raw.githubusercontent.com/timchow1228/llm-scope/main/docs/screenshot.png)

## Why llm-scope?

- **Python Native & Zero Config:** `pip install llm-scope`, change one line (`base_url`), and you're done. 
- **Prompt Cache Analytics:** Wondering if you should use `deepseek-v4-flash` or `deepseek-v4-pro`? `llm-scope` visualizes the TTFT difference and accurately calculates your **Prompt Cache Hit Savings 💰**. No more guesswork on your API bill.
- **Microsecond Precision Waterfall:** Pinpoint precisely if latency is caused by TCP Handshake (`connect`), Prompt Processing (`TTFT`), or Decoding (`generation`).
- **Physical Isolation:** We never upload your prompts to a cloud service. Unlike cloud tools (Helicone is now in maintenance mode post-acquisition), all your data is stored in a plain SQLite file locally at `~/.local/share/llm-scope/calls.db`.

---

## Installation

```bash
pip install llm-scope
```

## Quick Start

1. Start the proxy and local dashboard:
```bash
llm-scope start
```
*The dashboard will automatically open at `http://localhost:7070`. Press `Ctrl+C` to stop.*

2. Route your Python code through the scope:
```bash
export OPENAI_BASE_URL=http://localhost:7070/v1
export DEEPSEEK_API_KEY=sk-...
python your_script.py
```

---

## Popular Target Workflows

### 🏎️ DeepSeek V4 Prompt Cache Savings Tracking
DeepSeek V4 introduces massive price cuts for cached contexts, but it's hard to know exactly how much you're saving. `llm-scope` automatically parses the `prompt_cache_hit_tokens` from DeepSeek's payload and shows you exactly how much your System Prompt cache is saving you, in dollars. 

**The dashboard header displays your cumulative cache savings in real time — `💰 saved: $2.45` — the kind of number worth screenshotting.**

### 💻 Track your Background Cursor Spend
Cursor is incredibly fast, but it sends huge background contexts you might not be aware of. Track exactly what tokens are being consumed and distinctly tag them to separate them from your main project codebase:
1. Open Cursor Settings.
2. Go to **Models / Advanced**.
3. Set your **OpenAI Base URL** to `http://localhost:7070/tag/cursor/v1`.

That's it! Watch the dashboard light up with every autocomplete and Chat request, all neatly labeled with the "cursor" badge.

### 🐍 OpenAI SDK Drop-in Replacement
Since `llm-scope` explicitly mirrors the OpenAI `/v1/chat/completions` specs, your existing code requires exactly zero changes other than injecting the `base_url`:

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-api-key",
    base_url="http://localhost:7070/v1"  # Or add /tag/project-name/v1
)

# Call as usual – your request will be tracked locally in the dashboard!
response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[{"role": "user", "content": "Benchmark my latency."}],
    stream=True
)
```

## License
MIT License
