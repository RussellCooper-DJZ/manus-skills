# AI Capability Gap Report

## 1. Target and operating budget

| Field | Recorded value |
|---|---|
| Target repository | |
| Commit or revision | |
| Selected profile | `audit` / `focused` / `full` |
| CPU, RAM and storage budget | |
| Network and model constraints | |
| Operator-approved capability scope | |

## 2. Evidence collected before changes

Describe the existing code paths, tests, deployment assumptions and measured resource baseline. Do not infer performance from dependency names or claim deployment before a runnable verification exists.

## 3. Capability decision record

| Capability | Existing evidence | Decision | Dependency impact | Disable / rollback path |
|---|---|---|---|---|
| Agent + HITL | | retain / add / defer | | |
| Vision | | retain / add / defer | | |
| Automation | | retain / add / defer | | |
| RAG | | retain / add / defer | | |

## 4. Changes made

For each selected module, record the smallest change that addressed the demonstrated gap, including configuration flags and user confirmation boundaries for side-effecting tools.

## 5. Verification

| Check | Command or procedure | Actual result | Limitation |
|---|---|---|---|
| Structural preflight | | | |
| Unit or integration test | | | |
| Resource measurement | | | |
| Security / approval behavior | | | |

## 6. Deferred work

List expensive or optional modules that were intentionally not installed, together with the data required to justify enabling them later.
