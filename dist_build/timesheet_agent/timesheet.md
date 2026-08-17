---
name: Timesheet
schedule: daily
description: Builds copy-paste timesheet entries from helpdesk ticket screen activity
enabled: true
output: local
mode: timesheet
temperature: 0.1
---
Timesheet mode. Ticket numbers, customers, times, grouping, sorting, and
subtotals are computed deterministically from screen captures. The AI writes
a short subject and a one-line justification (the on-screen evidence for the
work) for each ticket from its screen text. No prompt body is needed — the
pipeline is fixed.
