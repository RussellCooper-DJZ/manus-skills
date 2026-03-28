"""
test_new_modules.py
新增四项核心能力模块的单元测试
Author: RussellCooper
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── 1. 智能体测试 ──────────────────────────────────────────────
class TestLangGraphAgent:
    def test_import(self):
        """验证 LangGraph 引擎可以正常导入。"""
        from langgraph_engine import LangGraphAgent, AgentState
        assert LangGraphAgent is not None
        assert AgentState is not None

    def test_route_after_plan_direct(self):
        """单步骤规划应路由到 direct。"""
        from langgraph_engine import LangGraphAgent
        agent = LangGraphAgent.__new__(LangGraphAgent)
        state = {"plan": ["单步回答"], "error": ""}
        assert agent.route_after_plan(state) == "direct"

    def test_route_after_plan_research(self):
        """多步骤规划应路由到 research。"""
        from langgraph_engine import LangGraphAgent
        agent = LangGraphAgent.__new__(LangGraphAgent)
        state = {"plan": ["步骤1", "步骤2", "步骤3"], "error": ""}
        assert agent.route_after_plan(state) == "research"

    def test_route_after_synthesis_human(self):
        """低置信度应触发人工审核。"""
        from langgraph_engine import LangGraphAgent
        agent = LangGraphAgent.__new__(LangGraphAgent)
        state = {"hitl_required": True, "error": ""}
        assert agent.route_after_synthesis(state) == "human"

    def test_route_after_hitl_approve(self):
        """人工批准后应进入 final。"""
        from langgraph_engine import LangGraphAgent
        agent = LangGraphAgent.__new__(LangGraphAgent)
        state = {"hitl_decision": "approve"}
        assert agent.route_after_hitl(state) == "final"

    def test_route_after_hitl_revise(self):
        """人工要求修改后应重新规划。"""
        from langgraph_engine import LangGraphAgent
        agent = LangGraphAgent.__new__(LangGraphAgent)
        state = {"hitl_decision": "revise"}
        assert agent.route_after_hitl(state) == "replan"


# ── 2. 视觉引擎测试 ────────────────────────────────────────────
class TestVisionEngine:
    def test_perceptual_hash_consistency(self):
        """相同图像应产生相同哈希。"""
        from vision_engine import perceptual_hash
        from PIL import Image
        import io
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()
        
        h1 = perceptual_hash(img_bytes)
        h2 = perceptual_hash(img_bytes)
        assert h1 == h2

    def test_hamming_distance_same(self):
        """相同哈希的汉明距离应为 0。"""
        from vision_engine import hamming_distance
        h = "ff00ff00ff00ff00"
        assert hamming_distance(h, h) == 0

    def test_hamming_distance_different(self):
        """不同哈希的汉明距离应大于 0。"""
        from vision_engine import hamming_distance
        h1 = "0000000000000000"
        h2 = "ffffffffffffffff"
        assert hamming_distance(h1, h2) > 0

    @pytest.mark.asyncio
    async def test_ui_change_detection_first_frame(self):
        """第一帧不应触发变化事件。"""
        from vision_engine import VisionEngine
        from PIL import Image
        import io
        engine = VisionEngine()
        img = Image.new("RGB", (100, 100), color=(0, 255, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        event = await engine.analyze_ui_change(buf.getvalue())
        assert event is None


# ── 3. 自动化引擎测试 ──────────────────────────────────────────
class TestAutomationEngine:
    def test_import(self):
        """验证自动化引擎可以正常导入。"""
        from automation_engine import AutomationEngine
        assert AutomationEngine is not None

    def test_init(self):
        """验证初始化参数正确设置。"""
        from automation_engine import AutomationEngine
        engine = AutomationEngine(headless=True, timeout_ms=5000)
        assert engine.headless is True
        assert engine.timeout_ms == 5000


# ── 4. RAG 引擎测试 ────────────────────────────────────────────
class TestEnhancedRAGEngine:
    def _make_embed_fn(self):
        """创建一个简单的 mock embedding 函数。"""
        import random
        def embed_fn(text: str):
            random.seed(hash(text) % 1000)
            return [random.random() for _ in range(128)]
        return embed_fn

    def test_add_and_query(self):
        """验证添加文档后可以正常查询。"""
        from rag_engine_enhanced import EnhancedRAGEngine
        engine = EnhancedRAGEngine(embed_fn=self._make_embed_fn())
        engine.add_texts(["量子计算是一种利用量子力学原理的计算技术。", "机器学习是人工智能的一个子领域。"])
        result = engine.query("什么是量子计算？")
        assert "query" in result
        assert "context" in result
        assert "retrieved_docs" in result

    def test_semantic_chunker(self):
        """验证分块器正确分割文本。"""
        from rag_engine_enhanced import SemanticChunker
        chunker = SemanticChunker(chunk_size=50, chunk_overlap=10)
        text = "这是第一段。\n\n这是第二段，内容更长一些。\n\n这是第三段。"
        docs = chunker.split(text)
        assert len(docs) >= 1
        for doc in docs:
            assert len(doc.content) > 0

    def test_mmr_rerank_diversity(self):
        """验证 MMR 重排序能提高多样性。"""
        from rag_engine_enhanced import Reranker, RetrievedDoc, Document
        import random
        
        docs = []
        for i in range(10):
            d = Document.from_text(f"文档 {i}")
            d.embedding = [random.random() for _ in range(16)]
            docs.append(RetrievedDoc(document=d, final_score=random.random()))
            
        q_emb = [random.random() for _ in range(16)]
        reranked = Reranker.mmr_rerank(docs, q_emb, top_k=5)
        assert len(reranked) == 5
