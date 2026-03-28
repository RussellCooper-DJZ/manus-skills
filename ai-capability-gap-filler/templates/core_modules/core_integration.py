"""
core_integration.py
====================
四项核心能力整合入口。
Author: RussellCooper

能力矩阵：
  1. 智能体 (LangGraph + Dify HITL)    → langgraph_engine.py
  2. 多模态视觉 (PaddleOCR + VLM)      → vision_engine.py
  3. 自动化执行 (Playwright)            → automation_engine.py
  4. RAG 优化 (混合检索 + MMR + 压缩)  → rag_engine_enhanced.py
"""
import asyncio
import logging
from typing import Optional

from automation_engine import AutomationEngine
from vision_engine import VisionEngine
from rag_engine_enhanced import EnhancedRAGEngine
from langgraph_engine import LangGraphAgent

logger = logging.getLogger(__name__)


class IntelligentAutomationSystem:
    """
    整合四项核心能力的顶层系统。
    工作流：
      1. AutomationEngine 执行浏览器操作并截图
      2. VisionEngine 分析截图，检测 UI 变化
      3. EnhancedRAGEngine 从知识库检索相关信息
      4. LangGraphAgent 综合以上信息生成决策，支持人工兜底
    """
    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        llm_api_key: str = "",
        llm_base_url: Optional[str] = None,
        embed_fn=None,
        headless: bool = True
    ):
        # 1. 智能体
        self.agent = LangGraphAgent(
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url
        )
        
        # 2. 视觉引擎
        self.vision = VisionEngine(
            vlm_api_url=llm_base_url.replace("/v1", "/v1/chat/completions") if llm_base_url else None,
            vlm_api_key=llm_api_key
        )
        
        # 3. 自动化引擎
        self.automation = AutomationEngine(headless=headless)
        
        # 4. RAG 引擎
        if embed_fn:
            self.rag = EnhancedRAGEngine(embed_fn=embed_fn)
        else:
            self.rag = None
            
    async def start(self):
        await self.automation.start()
        logger.info("IntelligentAutomationSystem started.")
        
    async def stop(self):
        await self.automation.stop()
        logger.info("IntelligentAutomationSystem stopped.")
        
    async def run_task(self, task_description: str, target_url: str, session_id: str) -> dict:
        """
        执行一个完整的自动化任务。
        """
        # Step 1: 导航
        await self.automation.navigate(target_url)
        
        # Step 2: 截图并分析
        screenshot = await self.automation.take_screenshot()
        ocr_result = self.vision.recognize_text(screenshot)
        change_event = await self.vision.analyze_ui_change(screenshot)
        
        # Step 3: 构建上下文
        context_parts = [f"任务目标：{task_description}"]
        context_parts.append(f"当前页面文字内容：{ocr_result.full_text[:500]}")
        
        if change_event:
            context_parts.append(f"UI 变化检测：{change_event.description}")
            if change_event.semantic_analysis:
                context_parts.append(f"语义分析：{change_event.semantic_analysis}")
                
        # Step 4: RAG 检索
        if self.rag:
            rag_result = self.rag.query(task_description)
            context_parts.append(f"知识库参考：{rag_result['context'][:500]}")
            
        # Step 5: 智能体决策
        full_query = "\n".join(context_parts)
        agent_result = await self.agent.run(full_query, session_id)
        
        return {
            "session_id": session_id,
            "ocr_text": ocr_result.full_text,
            "ui_change": change_event.description if change_event else None,
            "agent_result": agent_result
        }
