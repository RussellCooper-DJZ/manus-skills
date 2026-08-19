---
name: ai-capability-gap-filler
description: 扫描代码库，评估与四项核心AI能力（智能体、多模态视觉、自动化执行、RAG优化）的差距，并自动补全缺口，生成专家级超规格实现及可视化交付报告。适用于需要将传统项目升级为具备高级AI能力系统的场景。
---

# AI Capability Gap Filler

This skill provides a standardized workflow for scanning a codebase, evaluating its gap against four core AI capabilities, injecting expert-level implementations to fill those gaps, and generating a comprehensive visual delivery report.

## Core Capabilities Defined

1. **Agent (智能体)**: LangGraph/Dify integration with Human-in-the-loop (HITL) design.
2. **Multimodal Vision (多模态视觉)**: MiniCPM-V/PaddleOCR integration for UI change detection and semantic diffing.
3. **Automation (自动化执行)**: Playwright/Selenium based execution with explicit approval gates, auditability, and respect for target-site policies.
4. **RAG Optimization (RAG优化)**: Advanced RAG with HyDE, Hybrid Search (Vector+BM25), MMR reranking, and context compression.

## Resource-first workflow

Do not install or inject every capability by default. The skill starts with a no-dependency structural preflight, then selects only the capability modules justified by the target project's evidence and resource budget.

```bash
python ai-capability-gap-filler/scripts/validate_skill.py --json
```

| Profile | Allowed scope | Dependency policy |
|---|---|---|
| `audit` (default) | Inspect code structure, validate the skill package, write a gap report. | Python standard library only; no AI model, vector DB, browser or vision dependencies. |
| `focused` | Add one selected capability with tests and an explicit rollback path. | Install only that capability's declared dependencies. |
| `full` | Combine capabilities only after the target demonstrates sufficient CPU, memory, storage and operational need. | Each dependency group must remain independently disableable. |

Filling the AI capability gaps involves these sequential steps:

1. **Scan and Evaluate**
   - Clone or read the target codebase.
   - Run the zero-dependency structural preflight before selecting a module.
   - Evaluate existing code against the four core capabilities and the project's resource budget.
   - Identify gaps (e.g., missing HITL, basic OCR without semantic diff, or basic RAG without HyDE/MMR).

2. **Inject Expert Implementations**
   - Choose the smallest module that addresses the demonstrated gap; do not inject every template as a package.
   - Keep capability modules behind explicit configuration flags so low-resource deployments can disable them.
   - The templates include:
     - `langgraph_engine.py`: LangGraph StateGraph with async HITL.
     - `vision_engine.py`: PaddleOCR + pHash + VLM semantic diff.
     - `automation_engine.py`: Playwright with stealth and smart selectors.
     - `rag_engine_enhanced.py`: 4-layer RAG optimization.
     - `core_integration.py`: Top-level integration.
     - `test_new_modules.py`: Comprehensive test suite.

3. **Run Tests**
   - Install only the selected capability's dependencies (`langgraph`, `playwright`, `paddleocr`, etc.).
   - Run structural tests first, then run the injected test suite using `pytest`; record the actual result instead of claiming a pass rate in advance.

4. **Generate Delivery Report**
   - Write a detailed Markdown report summarizing the gaps, the injected solutions, and the test results.
   - Use the template `templates/report_template.md` as a guide.

5. **Generate Visualizations**
   - Run the visualization script `scripts/visualize_report.py` to generate professional charts (Radar chart, Code lines, Test coverage, Gap matrix, RAG pipeline, LangGraph flow).
   - Ensure Chinese fonts (e.g., Noto Sans CJK SC) are available in the environment before running the script.

6. **Deliver**
   - Commit and push the changes to the target repository (e.g., via GitHub PR).
   - Deliver the final report and visualization images to the user.

## Bundled Resources

- `templates/core_modules/`: Contains the expert-level Python implementations for the four core capabilities.
- `templates/report_template.md`: Markdown template for the final delivery report.
- `scripts/visualize_report.py`: Python script using `matplotlib` to generate 6 professional visualization charts based on the delivery report data.

## Important Notes

- **Dependencies**: The bundled templates have optional dependencies. Keep them out of the default audit path and install only the selected module's dependency group.
- **Playwright**: Install a browser only when a user-approved automation capability is actually selected; preserve confirmation and audit controls for side-effecting actions.
- **Fonts**: The visualization script requires CJK fonts for proper rendering. Generate visual reports only when they are part of the requested deliverable.
- **Evidence**: Do not describe a template as deployed, tested or resource-efficient until the target project has executed its own verification path.
