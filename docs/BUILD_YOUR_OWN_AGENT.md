# Build Your Own Agent

ScreenMind's agent system lets you create custom AI automations that run on your screen data. Agents are simple files — either **Markdown** (for AI-powered analysis) or **Python** (for full programmatic control).

## Quick Start

### Option 1: Markdown Agent (No coding required)

Create a file in `~/.screenmind/agents/` with a `.md` extension:

```markdown
---
name: My Custom Agent
schedule: every 2h
description: Describe what your agent does
enabled: true
output: local
data: timeline, apps
---
Your instructions to the AI go here.

Tell the AI what to analyze, extract, or summarize from your screen data.
Be specific about the output format you want.
```

That's it. ScreenMind's built-in AI (Gemma) will run your prompt on a schedule with your screen data injected automatically.

### Option 2: Python Plugin (Full control)

Create a `.py` file in `~/.screenmind/agents/`:

```python
"""
name: My Python Agent
schedule: every 1h
description: A custom Python plugin
enabled: true
output: local
"""
from screenmind.screenmind_sdk import get_recent_activity, save_state, load_state

def run(context):
    # context["agent_name"] — your agent's sanitized name
    # context["timestamp"] — current UTC timestamp
    # context["data_dir"] — path to ~/.screenmind/
    
    activities = get_recent_activity(minutes=60)
    # ... your logic here ...
    return "Your output text"
```

## Frontmatter Reference

### Common Fields (both .md and .py)

| Field | Default | Description |
|---|---|---|
| `name` | filename | Display name in the UI |
| `schedule` | `every 6h` | Run frequency: `every 30m`, `every 1h`, `every 2h`, `every 6h`, `daily` |
| `description` | (empty) | One-line description shown in the UI |
| `enabled` | `true` | Whether the agent runs on schedule |
| `output` | `local` | Where output goes: `local`, `obsidian`, `webhook`, or comma-separated |

### Markdown-Only Fields

| Field | Default | Description |
|---|---|---|
| `data` | `timeline, apps` | Which data sections to inject (see below) |
| `model_requirement` | `0` | Minimum context window tokens needed (e.g. `8192`) |

### Data Sections

Control what screen data your markdown agent receives:

| Section | What it includes | Best for |
|---|---|---|
| `timeline` | Recent activities with timestamps, apps, summaries | General analysis |
| `apps` | App usage counts + category breakdown | Productivity tracking |
| `urls` | URLs visited (from browser address bar) | Web usage tracking |
| `meetings` | Meeting summaries and durations | Meeting intelligence |
| `mood` | Mood/sentiment breakdown from screen analysis | Wellbeing tracking |
| `ocr` | Raw screen text per capture (chronological, budget-sampled) | Ticket extraction, timesheets, exact-content tasks |

**Example combinations:**
```yaml
data: urls, timeline          # AI tool tracking, research logging
data: apps, timeline          # Time tracking, productivity reports
data: meetings, timeline      # Meeting follow-ups, action items
data: urls, apps, meetings, mood, timeline  # Everything (needs larger model)
data: ocr, timeline, urls     # Raw screen text — timesheets, ticket tracking
```

The data budget scales with your model's context window. Smaller models get fewer items per section. If your agent needs rich data, add `model_requirement: 8192` and users with smaller models will see an upgrade suggestion.

## SDK Reference (Python Plugins)

```python
from screenmind.screenmind_sdk import (
    # Data Access
    get_recent_activity,     # Recent screen activities
    get_activities_by_date,  # Activities for a specific date
    get_activities,          # Filtered query (by app, category, URL)
    get_urls_visited,        # All unique URLs visited today
    get_meetings,            # Today's meetings
    get_app_usage,           # App usage breakdown
    search,                  # Semantic search across all history
    get_summary,             # Daily AI summary
    get_stats,               # System statistics
    
    # Persistent State
    save_state,              # Save value to agent state
    load_state,              # Load value from agent state
    clear_state,             # Clear agent state
    
    # AI
    ask_gemma,               # Ask the local LLM a question
    
    # Actions
    notify,                  # Show overlay notification
    capture_now,             # Trigger screenshot capture
    write_file,              # Write to a file
    get_output_dir,          # Get agent's output directory
)
```

### Data Access Examples

```python
# Get activities filtered by app
coding = get_activities(app="Code", limit=20)

# Get activities with URLs containing "github"
github = get_activities(url_contains="github.com")

# Get all unique URLs visited today
urls = get_urls_visited()
for u in urls:
    print(f"{u['url']} — visited {u['count']}x")

# Get app usage breakdown
usage = get_app_usage()
# {"Code": {"captures": 45, "categories": ["coding"]}, "Edge": {"captures": 12, ...}}

# Search across all history
results = search("machine learning paper")
```

### Persistent State

Agents can remember things between runs. State is stored per-agent in `~/.screenmind/agents/state/{agent}.json`.

```python
# Save data
save_state("last_checked", "2026-05-21T10:00:00")
save_state("tracked_items", [{"task": "Review PR", "done": False}])

# Load data
items = load_state("tracked_items", [])  # default: empty list

# Clear
clear_state("tracked_items")  # clear one key
clear_state()                 # clear all state
```

### Ask Gemma (Local LLM)

```python
# Simple question
answer = ask_gemma("Summarize the main topics from these activities: ...")

# With recent activity context
answer = ask_gemma("What was I working on?", include_recent=True)
```

> **Note:** `ask_gemma()` waits for the GPU to be idle (up to 60 seconds). It never interrupts screen analysis. On machines with ≤4GB VRAM, it may timeout during heavy capture sessions.

## Example Agents

### Standup Report (Markdown)
```markdown
---
name: Standup Report
schedule: daily
description: Generate a daily standup report
data: apps, timeline
---
Generate a standup report with three sections:
1. **What I did yesterday** — summarize work from the timeline
2. **What I'm doing today** — infer from recent patterns
3. **Blockers** — anything that seemed stuck or repetitive

Format as bullet points. Be concise.
```

### Brand Listener (Python)
```python
"""
name: Brand Listener
schedule: every 30m
description: Alerts when brand keywords appear on screen
enabled: true
"""
from screenmind.screenmind_sdk import search, notify, save_state, load_state

KEYWORDS = ["ScreenMind", "ScreenPipe", "my-company"]

def run(context):
    last_check = load_state("last_id", 0)
    found = []
    
    for kw in KEYWORDS:
        results = search(kw, limit=5)
        for r in results:
            if r.get("id", 0) > last_check:
                found.append(f"{kw}: {r.get('summary', '?')}")
                last_check = max(last_check, r.get("id", 0))
    
    save_state("last_id", last_check)
    
    if found:
        notify("Brand Alert", f"{len(found)} mention(s) found!")
        return "## Brand Mentions\n" + "\n".join(f"- {f}" for f in found)
    return "No new brand mentions."
```

### Research Logger (Python)
```python
"""
name: Research Logger
schedule: every 2h
description: Logs research sessions with sources
enabled: true
"""
from screenmind.screenmind_sdk import get_urls_visited, get_activities, save_state, load_state, ask_gemma

def run(context):
    sessions = load_state("sessions", [])
    
    urls = get_urls_visited()
    research_urls = [u for u in urls if any(
        d in u["url"] for d in ["arxiv", "scholar.google", "stackoverflow", "docs.", "wiki"]
    )]
    
    if not research_urls:
        return format_sessions(sessions)
    
    url_text = "\n".join(f"- {u['url']} ({u['count']}x)" for u in research_urls[:8])
    
    try:
        topic = ask_gemma(
            f"What research topic connects these URLs?\n{url_text}\n"
            "Reply with a single topic name.",
            max_tokens=64
        )
    except RuntimeError:
        topic = "Unknown topic"
    
    sessions.append({
        "date": context["timestamp"][:10],
        "topic": topic.strip(),
        "sources": len(research_urls),
        "urls": [u["url"] for u in research_urls[:5]]
    })
    
    save_state("sessions", sessions[-20:])  # keep last 20
    return format_sessions(sessions)

def format_sessions(sessions):
    if not sessions:
        return "# Research Log\nNo research sessions tracked yet."
    lines = ["# Research Log\n"]
    for s in reversed(sessions[-10:]):
        lines.append(f"### {s['date']} — {s['topic']}")
        lines.append(f"{s['sources']} sources visited")
        for u in s.get('urls', [])[:3]:
            lines.append(f"- {u}")
        lines.append("")
    return "\n".join(lines)
```

## Output Destinations

| Destination | How it works |
|---|---|
| `local` | Saved to `~/.screenmind/agents/output/{agent-name}/` |
| `obsidian` | Appended to your Obsidian vault (configure vault path in Settings) |
| `webhook` | POSTed to your webhook URL (configure in Settings) |
| `local, obsidian` | Multiple destinations — comma-separated |

## Tips

1. **Start simple** — Begin with a markdown agent, upgrade to Python when you need state or API calls
2. **Be specific** — The more specific your prompt, the better the output
3. **Check outputs** — Click "📄 Outputs" on any agent card to see what it produced
4. **Test first** — Use "▶ Run Now" to test before enabling scheduled runs
5. **Data budget** — Smaller models get less data. If your agent's output is thin, upgrade your model or reduce `data:` sections
6. **State is free** — `save_state/load_state` is fast and unlimited. Use it liberally.
7. **GPU awareness** — `ask_gemma()` waits for the GPU. Avoid calling it in tight loops.

## Agent Lifecycle

```
Agent file created/edited
        │
        ▼
Discovered by agent_runner.py
        │
        ▼
Shown in Agents tab (UI)
        │
        ▼
Scheduled run OR manual "Run Now"
        │
        ├── Markdown: data injected → Gemma generates output
        │
        └── Python: run(context) called → SDK functions available
                │
                ▼
        Output saved to output destination(s)
        Run logged in agent run log
```

## Creating Agents from the UI

1. Go to **Agents** tab
2. Click **🤖 New AI Agent** or **🐍 New Python Plugin**
3. Enter a name → agent file is created
4. Click **✏️ Edit** to open the built-in code editor
5. Write your prompt or code
6. Click **▶ Run Now** to test
7. Toggle the switch to enable scheduled runs
