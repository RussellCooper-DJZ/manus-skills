# manus-skills

> Reusable russellcooper Skills for [OpenClaw](https://github.com/RussellCooper-DJZ/openclaw-shrimp-factory) and [RCclaw](https://github.com/RussellCooper-DJZ/RCclaw) projects.

This repository contains expert-level, production-ready AI capability modules packaged as **Manus Skills** — modular, self-contained packages that can be loaded into any Manus agent to extend its capabilities.

## Skills Index

| Skill | Description | Compatible With |
|---|---|---|
| [`ai-capability-gap-filler`](./ai-capability-gap-filler/) | Scans a codebase, evaluates gaps against 4 core AI capabilities, injects expert implementations, and generates a visual delivery report. | OpenClaw, RCclaw, any Python project |

## Usage

To use a skill, copy the skill directory into your Manus agent's `skills/` folder:

```bash
cp -r ai-capability-gap-filler /path/to/your/agent/skills/
```

Each skill contains a `SKILL.md` file that Manus reads to understand when and how to use it.

## The Four Core AI Capabilities

These skills are built around four production-grade AI capability pillars:

1. **Agent (智能体)** — LangGraph `StateGraph` with async Human-in-the-Loop (HITL) interrupt/resume, Dify Workflow API compatibility, and audit logging.
2. **Multimodal Vision (多模态视觉)** — Dual-layer UI change detection: pHash perceptual hashing for fast filtering + VLM (MiniCPM-V) semantic diffing for functional change classification.
3. **Automation (自动化执行)** — Playwright stealth automation with anti-bot injection, persistent browser context, human behavior simulation (randomized delays, mouse movement, scroll patterns).
4. **RAG Optimization (RAG优化)** — 4-layer retrieval pipeline: Semantic Chunking → HyDE Query Optimization → Hybrid Search (Vector 0.7 + BM25 0.3) → MMR Reranking → Context Compression.

## License

MIT
