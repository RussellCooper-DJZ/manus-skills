---
name: ai-capability-gap-filler
description: 扫描代码库，评估与四项核心AI能力（智能体、多模态视觉、自动化执行、RAG优化）的差距，并自动补全缺口，生成专家级超规格实现及可视化交付报告。适用于需要将传统项目升级为具备高级AI能力系统的场景。
---

# AI Capability Gap Filler

This skill provides a standardized workflow for scanning a codebase, evaluating its gap against four core AI capabilities, injecting expert-level implementations to fill those gaps, and generating a comprehensive visual delivery report.

## Core Capabilities Defined

1. **Agent (智能体)**: LangGraph/Dify integration with Human-in-the-loop (HITL) design.
2. **Multimodal Vision (多模态视觉)**: MiniCPM-V/PaddleOCR integration for UI change detection and semantic diffing.
3. **Automation (自动化执行)**: Playwright/Selenium based execution with anti-bot bypass and human behavior simulation.
4. **RAG Optimization (RAG优化)**: Advanced RAG with HyDE, Hybrid Search (Vector+BM25), MMR reranking, and context compression.

## Workflow

Filling the AI capability gaps involves these sequential steps:

1. **Scan and Evaluate**
   - Clone or read the target codebase.
   - Evaluate existing code against the four core capabilities.
   - Identify gaps (e.g., missing HITL, basic OCR without semantic diff, lack of anti-bot measures, basic RAG without HyDE/MMR).

2. **Inject Expert Implementations**
   - Use the provided templates in `templates/core_modules/` to inject expert-level implementations into the target project.
   - The templates include:
     - `langgraph_engine.py`: LangGraph StateGraph with async HITL.
     - `vision_engine.py`: PaddleOCR + pHash + VLM semantic diff.
     - `automation_engine.py`: Playwright with stealth and smart selectors.
     - `rag_engine_enhanced.py`: 4-layer RAG optimization.
     - `core_integration.py`: Top-level integration.
     - `test_new_modules.py`: Comprehensive test suite.

3. **Run Tests**
   - Install required dependencies (`langchain`, `langgraph`, `playwright`, `paddleocr`, etc.).
   - Run the injected test suite using `pytest` to ensure 100% pass rate.

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

- **Dependencies**: The injected modules require specific dependencies. Ensure you install them in the target environment (e.g., `pip install langchain langgraph langchain-openai playwright aiohttp pytest pytest-asyncio`).
- **Playwright**: Remember to run `playwright install chromium` if Playwright is used for the first time in the environment.
- **Fonts**: The visualization script requires CJK fonts for proper rendering. If missing, install them via `sudo apt-get install fonts-noto-cjk`.
