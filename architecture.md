# ScreenMind — System Architecture

> **Privacy-First Local Screen Activity Journal + AI Memory**  
> Powered by Gemma 4 (E2B / E4B / 12B — Vision + Audio + Reasoning) via llama.cpp

---

## 1. System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            ScreenMind                                    │
│                                                                         │
│  ┌──────────────┐   ┌────────────┐   ┌───────────────────────────────┐ │
│  │ Capture      │──▶│ Async Queue│──▶│ Analysis Worker               │ │
│  │ Worker       │   │ (max: 100) │   │                               │ │
│  │              │   └────────────┘   │ pHash Cache → OCR → Gemma 4   │ │
│  │ • mss grab   │                    │ → Layout → Embeddings → Store │ │
│  │ • pHash dedup│                    └───────────────────────────────┘ │
│  │ • A11y text  │                                                      │
│  │ • Privacy    │   ┌────────────┐   ┌───────────────────────────────┐ │
│  └──────────────┘   │ Audio      │   │ Agent Scheduler               │ │
│                     │ Worker     │   │                               │ │
│  ┌──────────────┐   │            │   │ • .md → Gemma prompt          │ │
│  │ Hotkey       │   │ • Detect   │   │ • .py → SDK execution         │ │
│  │ Listener     │   │ • Record   │   │ • Cron scheduling             │ │
│  │              │   │ • Transcr. │   └───────────────────────────────┘ │
│  │ • Bookmark   │   │ • Summary  │                                     │
│  │ • Pause      │   └────────────┘              │                      │
│  │ • Voice memo │                               ▼                      │
│  └──────────────┘                    ┌───────────────────┐             │
│                                      │ SQLite (WAL)      │             │
│                                      │ + FTS5 index      │             │
│                                      │ + Embeddings BLOB │             │
│                                      └─────────┬─────────┘             │
│                                                │                       │
│  ┌─────────────────────────────────────────────┴─────────────────────┐ │
│  │                    FastAPI REST Server (:7777)                     │ │
│  │  /timeline · /search · /chat · /stats · /agents · /mcp · /rewind │ │
│  │                                                                   │ │
│  │  ┌───────────────────────────────────────────────────────────┐   │ │
│  │  │              Web Dashboard (Vanilla JS SPA)                │   │ │
│  │  │  Timeline · Chat · Search · Analytics · Memos · Agents    │   │ │
│  │  └───────────────────────────────────────────────────────────┘   │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌───────────────────┐                                                 │
│  │ MCP Server (stdio)│ ← Claude Desktop / Cursor / VS Code            │
│  │ Separate process  │                                                 │
│  └───────────────────┘                                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

**Core Principle:** Everything runs locally. No network calls. No telemetry. Screenshots, analysis, search, and chat — all on your machine.

---

## 2. Multi-Model AI Pipeline

```
Screenshot
    │
    ├──▶ EasyOCR (text extraction, ~3-10s, CPU)
    │         │
    │         ▼
    ├──▶ Gemma 4 (understanding, 12-76s, GPU)  ◀── OCR text fed as context
    │         │
    │         ├── Structured JSON: app, category, summary, mood, scene
    │         └── Layout regions: sidebar, chat area, toolbar (accurate mode)
    │
    ├──▶ Layout Analyzer (spatial OCR clustering, ~0ms, CPU)  ← fast mode fallback
    │         │
    │         └── Organized text with [SECTION] headers
    │
    ├──▶ Text model (scene narration, optional)  ◀── organized text in, no image
    │         │
    │         └── scene_description (overwrites the vision model's field)
    │
    └──▶ MiniLM-L6-v2 (semantic embedding, ~50ms, CPU)
              │
              └── 384-dim vector for similarity search
              
    All results → SQLite + FTS5
```

Five models working in concert:
1. **EasyOCR** — extracts raw screen text (what's written)
2. **Gemma 4** — understands what you're doing (the brain)
3. **Layout Analyzer** — organizes text by screen region (spatial intelligence)
4. **Text model** (optional) — narrates the scene from extracted text, per Text Model routing
5. **MiniLM-L6-v2** — enables "search by meaning" (semantic vectors)

---

## 3. Component Deep-Dive

### 3.1 Capture Worker

| Property | Detail |
|---|---|
| **Method** | `mss` — fastest cross-platform screen capture |
| **Smart Polling** | 5s check interval, 10s minimum between saves, 40s max forced capture |
| **Deduplication** | Perceptual hash (pHash), threshold: 8 hamming distance |
| **Idle Detection** | 3+ consecutive skips → extends poll to 40s (screen unchanged) |
| **A11y Extraction** | Windows UI Automation text captured at screenshot time (correct window) |
| **Privacy Zones** | Blocked apps silently skipped. Heavy apps auto-pause capture. |
| **Output** | JPEG @ 70% quality → `~/.screenmind/screenshots/{date}/{time}.jpg` |
| **Encryption** | Optional Fernet AES encryption in-place after save |

```
Capture Loop:
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐
│ mss grab  │──▶│ pHash    │──▶│ A11y text│──▶│ Queue + DB   │
│ (5s poll) │   │ dedup    │   │ extract  │   │ insert       │
│ (40s max) │   │          │   │          │   │              │
└──────────┘    └──────────┘   └──────────┘   └──────────────┘
                    │ duplicate?
                    └──▶ skip + delete file
```

### 3.2 Analysis Worker — Per-App pHash Cache (3-Tier)

The key optimization: ~80% of screenshots are near-identical to the previous frame for the same app. The cache avoids redundant Gemma calls.

```
New screenshot → compute pHash → compare with cache[(app, title)]
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              ▼                         ▼                         ▼
         Diff 0-2                  Diff 3-7                  Diff 8+
      "IDENTICAL"               "MINOR"                   "FULL"
                                                          
    Copy everything           Re-run OCR only          Full pipeline:
    from cache                Reuse Gemma analysis     OCR → Gemma → Layout
    ~0ms                      Reuse layout regions     → Embed → Store
                              ~3-10s                   ~12-76s
```

| Tier | When | What Runs | Time |
|---|---|---|---|
| Identical (≤2) | Cursor blink, clock tick | Nothing — copy cache | ~0ms |
| Minor (3-7) | Typing, scrolling | OCR + text organization | ~3-10s |
| Full (8+) | App switch, new page | Everything | 12-76s |

Cache: LRU OrderedDict, max 30 entries, keyed by `(app_name, window_title[:100])`.

### 3.3 Gemma 4 — Three Analysis Modes

| Mode | Method | Time | Layout Source |
|---|---|---|---|
| **Accurate** (merged) | Single call WITH thinking | ~76s | Gemma detects regions |
| **Balanced** | Analysis-only WITH thinking | ~40s | OCR bounding box clustering |
| **Fast** | No-thinking prefill trick | ~12s | OCR bounding box clustering |

**Accurate mode:** One prompt asks for both layout regions (coordinates) and activity analysis. Gemma uses its thinking budget to reason about spatial boundaries.

**Balanced mode:** Same analysis-only prompt as fast, but allows Gemma to think naturally. Produces richer `scene_description` and `activity_summary` than fast, without the layout overhead. Layout computed from OCR box positions.

**Fast mode:** Pre-fills the `<think>\n</think>\n` block in the assistant message, forcing Gemma to skip reasoning and output immediately. Layout computed from OCR box positions.

**Scene description:** The vision model's `scene_description` is overwritten by a text-model call (`generate_scene_from_text`) that narrates the screen from organized/OCR text only — no image is sent. If that call fails or returns nothing, the vision-generated scene is kept.

### 3.4 Layout Analyzer — Spatial OCR Organization

Transforms raw OCR boxes into structured, section-labeled text:

```
Input: 200 OCR boxes with (x, y, width, height, text, confidence)
       + Layout regions [{"name": "nav_sidebar", "x_start": 0.0, "x_end": 0.15, ...}]

Process:
  1. Sort regions narrow-first (sidebar before main content)
  2. Classify each OCR box into a region by center-point
  3. Within chat regions: detect timestamps → attribute messages to senders
  4. Within other regions: group by Y-proximity (25px) into visual lines

Output:
  [NAV SIDEBAR]
  Home | Messages | Settings

  [CHAT MESSAGES]
  Alice: Hey, did you push the fix? | What's the status?
  Bob: Just merged it | Tests passing now

  [PROFILE PANEL]
  Alice | Online | Member since 2024
```

This organized text is what gets sent to chat as context — not raw OCR dumps.

### 3.5 Chat — Text-First RAG with Vision Fallback

```
User question
      │
      ▼
┌─────────────────────┐
│ 1. Extract keywords │  Remove stopwords, keep meaningful terms
│ 2. Check intent     │  Timeline signals? (screen, discord, yesterday...)
└──────────┬──────────┘
           │
     Has timeline intent?
     ┌─────┴─────┐
     NO          YES
     │            │
     ▼            ▼
  Casual      ┌──────────────┐
  chatbot     │ FTS5 probe   │  Keywords match any activity?
  mode        └──────┬───────┘
                     │
               ┌─────┴─────┐
               NO          YES
               │            │
               ▼            ▼
            Casual     ┌──────────────────┐
            mode       │ Score & rank     │  OCR hits, scene hits, app hits
                       │ top 5 activities │
                       └──────┬───────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │ Has text?        │
                       ├─ YES: text mode  │  Send organized_text as context
                       └─ NO: vision mode │  Send screenshot image to Gemma
                       
GPU Priority: Chat cancels in-flight analysis → GPU freed in <1s → analysis re-queued at front
```

### 3.6 Audio Worker — Meeting Transcription

```
Meeting Detection:
  CaptureWorker (every 5s) → check foreground app → meeting app keyword match?
      │
      YES → Audio probe (1s sample) → voice detected? (RMS > 0.015)
      │         │
      │    2 consecutive probes confirm → START RECORDING
      │
  Recording Loop (15s chunks):
      │
      ├── Chunk → WAV bytes → llm_client.transcribe_audio() → transcript text
      │
      └── Stop signals:
            • 3 consecutive silent chunks (45s silence)
            • Meeting process no longer alive (30s debounce)
            • 5-min hard timeout (browser-based meetings)

  On Stop:
      │
      ▼
  Map-Reduce Summarization:
      Short (≤4000 chars): Single Gemma call → structured summary
      Long (>4000 chars):  Split into ~3000 char chunks
                           → Summarize each chunk
                           → Combine into final summary
                           (TOPICS / DECISIONS / ACTION ITEMS)
```

Gemma 4's encoder-free architecture handles audio natively across all model sizes (E2B, E4B, 12B). No Whisper dependency. Voice memos are always saved to DB even if transcription fails — audio preserved for playback.

### 3.7 Agent System

```
~/.screenmind/agents/
├── daily-journal.md       ← Markdown Agent (Gemma-powered)
├── focus-report.md        ← Markdown Agent
├── meeting-actions.md     ← Markdown Agent
├── code-changelog.md      ← Markdown Agent
└── my-tracker.py          ← Python Plugin (SDK)
```

**Markdown Agents:**
```
Frontmatter → parse schedule, data selectors, output destinations
    │
    ▼
Data injection → fetch timeline/apps/urls/meetings/mood from DB
    │
    ▼
Build prompt → agent prompt + injected data (auto-scaled to context window)
    │
    ▼
Gemma 4 → generate response
    │
    ▼
Route output → local file / Obsidian vault / webhook (Slack, Discord)
```

**Python Plugins:**
```python
# Full SDK: get_activities(), get_urls_visited(), get_meetings(),
#           save_state(), load_state(), ask_gemma()
# GPU-safe: ask_gemma() polls is_inference_active() with 60s timeout
# State isolation: per-agent JSON file in agents/state/
```

### 3.8 Inference Priority & Cancellation

```
                    ┌─────────────────────┐
                    │   llama-server      │
                    │   (single slot)     │
                    │   --parallel 1      │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
         Analysis          Chat            Audio
         (background)      (user-facing)   (meeting)
              │                │
              │    User sends chat message
              │         │
              │         ▼
              │    cancel_current_inference()
              │         │
              │         ├── Set _cancel_event flag
              │         └── Close httpx.Client → llama-server frees slot
              │
              ▼
         InferenceCancelled raised
              │
              └── Re-queue at FRONT of priority deque
                  (processed before new queue items)
```

---

## 4. Storage Layer

### SQLite Schema (WAL mode, FTS5)

```sql
activities (
    id, timestamp, screenshot_path, window_title, detected_app,
    bookmarked, app_name, category, summary, details, visible_text,
    mood, confidence, embedding (BLOB), ocr_text, ocr_boxes (JSON),
    scene_description, organized_text, analyzed, analysis_error,
    analysis_method, active_url, created_at
)

dev_contexts (
    id, activity_id → activities(id), repo_name, branch,
    last_commit, changed_files (JSON), insertions, deletions
)

meetings (
    id, start_time, end_time, app_name, duration_minutes,
    transcript, summary, created_at
)

daily_summaries (
    id, date (UNIQUE), summary, standup, total_activities,
    category_breakdown (JSON), top_repos (JSON), productive_hours,
    created_at
)

-- FTS5 virtual table for keyword search
activities_fts (summary, details, ocr_text, app_name, scene_description, organized_text)

-- Indexes
idx_activities_timestamp, idx_activities_category, idx_activities_app,
idx_activities_bookmarked, idx_activities_analyzed,
idx_dev_repo, idx_dev_branch, idx_dev_activity
```

### Embedding Storage

| Property | Detail |
|---|---|
| **Model** | all-MiniLM-L6-v2 (80MB, CPU) |
| **Dimensions** | 384 floats |
| **Storage** | BLOB column in activities table (`struct.pack`) |
| **Search** | Load top 500 embeddings → numpy cosine similarity |
| **What's embedded** | summary + scene_description + details + app_name + category + visible_text |

### Search: Hybrid (Semantic + Keyword)

```
User query: "debugging auth"
      │
      ├──▶ MiniLM embed → cosine similarity vs stored vectors → ranked results
      │
      ├──▶ FTS5 MATCH "debugging OR auth" → keyword matches
      │
      └──▶ Meeting transcript LIKE search → meeting results
      
      Merge + deduplicate → sort by relevance score → return top N
```

---

## 5. Privacy & Security Layer

```
┌─────────────────────────────────────────────────────────┐
│                    Privacy Pipeline                       │
│                                                         │
│  Capture Time:                                          │
│    • App blocklist check (skip entirely)                │
│    • Heavy app auto-pause (games, video editors)        │
│    • Incognito mode (manual pause, no trace)            │
│                                                         │
│  Before AI + Storage:                                   │
│    • Sensitive data filter (regex-based redaction)       │
│      - Credit cards → [REDACTED:card]                   │
│      - SSNs → [REDACTED:ssn]                            │
│      - API keys (OpenAI, GitHub, AWS, Slack, Google)    │
│      - Passwords (key=value patterns)                   │
│                                                         │
│  At Rest:                                               │
│    • Optional Fernet AES encryption (screenshots)       │
│    • Key stored in OS keyring (Windows Credential Mgr)  │
│    • Magic header identifies encrypted files            │
│    • Transparent read: encrypted/unencrypted handled    │
│                                                         │
│  Access Control:                                        │
│    • Dashboard PIN lock (SHA-256 hashed)                │
│    • Session cookie with configurable timeout           │
│    • Auth middleware blocks all API without session      │
│                                                         │
│  Data Retention:                                        │
│    • Auto-delete activities + screenshots older than N  │
│    • Configurable: 1-365 days (default: 7)             │
└─────────────────────────────────────────────────────────┘
```

---

## 6. Platform Abstraction

```
platform_support/
├── base.py          ← PlatformAdapter ABC
├── windows.py       ← Win32 ctypes + UI Automation
├── macos.py         ← AppKit + AXUIElement (pyobjc)
└── linux.py         ← xdotool/xprop + AT-SPI

Capabilities per platform:
┌──────────────┬─────────────────┬──────────────────┬─────────────────┐
│ Feature      │ Windows         │ macOS            │ Linux           │
├──────────────┼─────────────────┼──────────────────┼─────────────────┤
│ Window title │ ctypes user32   │ AppKit+osascript │ xdotool+xprop   │
│ App name     │ ctypes kernel32 │ NSWorkspace      │ xdotool+/proc   │
│ A11y text    │ UI Automation   │ AXUIElement      │ AT-SPI          │
│ Screenshot   │ mss             │ mss              │ mss             │
│ Hotkeys      │ keyboard lib    │ keyboard lib     │ keyboard lib    │
└──────────────┴─────────────────┴──────────────────┴─────────────────┘
```

---

## 7. Integrations & Extensibility

```
┌─────────────────────────────────────────────────────────────────┐
│                        Output Channels                           │
│                                                                 │
│  ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌────────────────┐  │
│  │Obsidian │  │ Notion  │  │ Webhooks │  │ MCP Server     │  │
│  │ Export  │  │ Export  │  │ (HTTP)   │  │ (stdio)        │  │
│  │         │  │         │  │          │  │                │  │
│  │ .md to  │  │ API push│  │ Slack    │  │ Claude Desktop │  │
│  │ vault   │  │ to DB   │  │ Discord  │  │ Cursor         │  │
│  │         │  │         │  │ IFTTT    │  │ VS Code        │  │
│  └─────────┘  └─────────┘  │ Zapier   │  │                │  │
│                             │ Custom   │  │ 8 tools:       │  │
│                             │          │  │ search, recent │  │
│                             │ HMAC sig │  │ by_time, stats │  │
│                             │ Retry x1 │  │ summary, audio │  │
│                             │ Log last │  │ capture, image │  │
│                             │ 20       │  │                │  │
│                             └──────────┘  └────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   Smart Notifications                     │  │
│  │  • Distraction alert (N min on entertainment)            │  │
│  │  • Break reminder (N min continuous work)                │  │
│  │  • Auto-bookmark (keyword triggers: git push, deploy)    │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Performance Characteristics

| Metric | Value | Notes |
|---|---|---|
| Screenshot capture | ~50ms | mss grab + JPEG compress |
| pHash computation | ~15ms | imagehash.phash() |
| OCR extraction | 3-10s | EasyOCR on CPU (GPU available but not needed) |
| Gemma 4 (accurate) | ~76s | Single call with thinking, GTX 1650 4GB |
| Gemma 4 (balanced) | ~40s | Thinking, layout via OCR clustering |
| Gemma 4 (fast) | ~12s | No-thinking prefill, same GPU |
| Embedding generation | ~50ms | MiniLM on CPU |
| Cache hit (identical) | ~0ms | Copy from memory |
| Cache hit (minor) | ~3-10s | OCR only, skip Gemma |
| Chat response | ~5-15s | Text mode. Vision mode: ~15-30s |
| Disk per day | ~80-150MB | Screenshots + DB (8hr active use, 40s interval) |
| RAM usage | ~1.5-2GB | Python + EasyOCR + MiniLM (Gemma in llama-server) |
| VRAM usage | ~3-4GB | Gemma 4 E2B Q4_0 (default). E4B ~6GB, 12B ~10GB |

### Resource Management

| Setting | Effect |
|---|---|
| Performance Mode: minimal | 0 GPU layers (CPU inference, slow but frees VRAM) |
| Performance Mode: balanced | 15 GPU layers (default) |
| Performance Mode: maximum | 99 GPU layers (all on GPU) |
| Deferred Analysis | Queue captures, analyze only when 60s idle |
| Auto-Pause Heavy Apps | Skip capture when games/editors in foreground |
| KV Cache Quantization | Saves ~200MB VRAM, adds ~10s per inference |
| Flash Attention | Faster + less VRAM (enabled by default) |

---

## 9. End-to-End Data Flow

```
┌─────────┐     ┌─────────┐     ┌──────────────────────────────────────┐
│  User   │     │ Screen  │     │         Analysis Pipeline            │
│  works  │────▶│ changes │────▶│                                      │
└─────────┘     └─────────┘     │  1. A11y text (at capture time)      │
                                │  2. pHash cache check                │
                                │     ├─ identical → done (0ms)        │
                                │     ├─ minor → OCR only (3-10s)      │
                                │     └─ full → continue               │
                                │  3. EasyOCR extraction               │
                                │  4. Sensitive data redaction          │
                                │  5. URL extraction                   │
                                │  6. Gemma 4 analysis + layout        │
                                │  7. Organize text by regions         │
                                │  8. Scene narration via text model   │
                                │     (from organized text, no image)  │
                                │  9. Git context (if coding)          │
                                │ 10. MiniLM embedding                 │
                                │ 11. Store all to SQLite              │
                                │ 12. Auto-bookmark check              │
                                └──────────────────────────────────────┘
                                                 │
                                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         User Queries                                  │
│                                                                      │
│  Timeline → browse by date                                           │
│  Search → semantic + keyword hybrid                                  │
│  Chat → RAG (text-first, vision fallback, conversation history)      │
│  Analytics → category/app/hour aggregations                          │
│  Rewind → timelapse playback                                         │
│  Summary → Gemma deep reasoning over day's activities                │
│  Agents → scheduled automations on screen data                       │
│  MCP → external AI tools query screen history                        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 10. Why Gemma 4

### Supported Models

| Model | Variants | VRAM | Audio |
|---|---|---|---|
| **Gemma 4 E2B** (2B) — *default* | Q4_0 · Q8_0 · BF16 | ~4 GB | ✅ |
| **Gemma 4 E4B** (4B) | Q4_0 · Q8_0 · BF16 | ~6 GB | ✅ |
| **Gemma 4 12B** | IQ3_M · Q4_K_M · Q5_K_M · Q6_K · Q8_0 | ~10 GB | ❌ |

Models are sourced from [ggml-org](https://huggingface.co/ggml-org) (official llama.cpp-compatible GGUFs) and stored in `~/.screenmind/models/`.

E2B is the recommended default — it checks all the boxes:

| Constraint | Why It Rules Out Alternatives |
|---|---|
| Must run **continuously in background** | Rules out 12B+ models on low-VRAM GPUs |
| Must understand **screenshots natively** | Rules out text-only models |
| Must stay **100% local** for privacy | Rules out cloud APIs (Gemini, GPT-4V) |
| Must handle **audio natively** | Rules out models without audio encoder |
| Must be **fast enough** for 40s cycle | E2B: 12s (fast) to 76s (accurate) |

If you have more VRAM, E4B offers richer analysis. The 12B model gives the best text reasoning but lacks audio input. The Model Hub lets you download, switch, and delete variants from the dashboard.

---

## 11. Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **Vision + Audio AI** | Gemma 4 (E2B / E4B / 12B) via llama.cpp | Vision + audio + reasoning, runs locally on 4GB+ VRAM |
| **Inference Server** | llama-server (llama.cpp) | Direct GGUF, OpenAI-compatible API, 8-12% faster than Ollama |
| **OCR** | EasyOCR | Extracts screen text fed to Gemma as context |
| **Embeddings** | all-MiniLM-L6-v2 | 80MB, CPU, 384-dim vectors |
| **Backend** | FastAPI + Uvicorn | Async, auto-docs, serves dashboard |
| **Database** | SQLite (WAL) + FTS5 | Zero-config, concurrent reads, full-text search |
| **Capture** | mss + ctypes + UI Automation | Native capture + accessibility text |
| **Frontend** | Vanilla JS + CSS | No build step, instant load |
| **Platform** | Windows / macOS / Linux | Abstraction layer with OS-specific adapters |
| **Agents** | Custom scheduler + SDK | Markdown (Gemma) + Python (code) |
| **MCP** | mcp[cli] (stdio transport) | Claude Desktop / Cursor / VS Code |
