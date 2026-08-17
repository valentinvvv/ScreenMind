# Timesheet Agent for ScreenMind (AI agent, `mode: timesheet`)

Fills in your timesheet from a day's screen captures. Hybrid design —
code owns everything format-critical, the LLM only classifies:

- **Deterministic (code):** ticket numbers (from window titles like
  `GITS PSF - HelpDesk - Ticket 99165 Details`), time per session (capture
  spans, gaps capped at 2x capture interval, rounded to 5 min), customers
  (email domains in ticket-page OCR, domains/company names in window
  titles), grouping by customer, sorting longest-first, subtotals, and the
  exact `Ticket NNNNN | CUSTOMER | SUBJECT | TIME` line format.
- **AI (one LLM call, context window filled with screen text):** a
  customer + short subject per session, plus a one-line justification
  naming the on-screen evidence for the work (document, error, request,
  email thread), inferred from the real OCR text of that session's
  screens, budgeted proportionally to time spent. If the LLM is
  unavailable, code-derived facts still produce a valid timesheet —
  justification lines are simply omitted.

Non-ticket work is classified too — grouped by normalized window title,
never dumped into one "Other work" bucket:

- Remote-desktop windows (Citrix/Desktop Viewer) are not tasks: the LLM
  says what was done inside from the screen text; unclassifiable sessions
  fold into one `Remote administration (Citrix)` line.
- Editor windows (`File.ps1 - Project - Visual Studio Code`) use the
  active file as the fallback subject, not the whole composite title.
- Empty subjects fall back to the document/window title segment.
- OCR-typo customer variants are merged (Levenshtein).

## Install (two parts)

**Part 1 — the agent file.** Copy `timesheet.md` into ScreenMind's agents
directory:
- Windows: `%USERPROFILE%\.screenmind\agents\`
- Linux/macOS: `~/.screenmind/agents/`

**Part 2 — the engine.** This agent needs `mode: timesheet` support, which
requires the patched files in this archive. Replace the installed copies:
- `agent_runner.py` → `<install>/Lib/site-packages/screenmind/engine/agent_runner.py`
- `agents.py` → `<install>/Lib/site-packages/screenmind/api/routes/agents.py`
  (Windows installer: `%LOCALAPPDATA%\Programs\ScreenMind\Lib\site-packages\...`;
  source checkout: same relative paths)

Then restart ScreenMind. Enable **Agents** in dashboard Settings and click
**▶ Run Now** on the Timesheet card. Output is saved under
`~/.screenmind/agents/output/timesheet/`. Schedule is `daily`.

## Backfill for another day

**Dashboard:** pick a date in the date field on the Timesheet card, then
click **▶ Run Now**. Leave the field empty to report today.

The run endpoint accepts the same date override programmatically:

```
POST /api/agents/timesheet/run
{"date": "2026-08-14"}
```

(or from Python: `run_agent(agent, date="2026-08-14")`). The date reaches
the plugin as `context["date"]` and takes priority over the
`TIMESHEET_DATE` env var and the one-shot state-file key.

## Requirements

- Capture must be **running** while you work — only captured screens can be
  reported. Ticket detection needs the helpdesk tab's window title.
- Customers are only found when OCR/window titles contain an email domain
  or company name; otherwise the customer is "—".
- Works even with a dashboard PIN set (reads the SQLite DB directly).

## Output format

```
ENGELWOOD HOLDING
— | ENGELWOOD HOLDING | Email (inbox/sent) | 40 min
— | ENGELWOOD HOLDING | ELEA time/management | 15 min
Subtotal: 55 min

ATDOMCO
— | ATDOMCO | Reviewed incident report document | 30 min
    because: INCIDENT REPORT - ATDOMCO document open in Word
Ticket 98961 | ATDOMCO | Reviewed email regarding activity changes | 5 min
    because: email thread from user@atdomco.com about account changes
Subtotal: 35 min

—
— | — | Remote administration (Citrix) | 30 min
Subtotal: 30 min
```

The indented `because:` line is the justification: concrete evidence from
the screen text that backs the entry. It appears only when the LLM finds
specific evidence — never invented.

If nothing was captured: `No ticket activity found today.`
