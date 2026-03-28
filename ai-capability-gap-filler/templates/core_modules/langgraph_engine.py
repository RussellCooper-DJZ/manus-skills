import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, TypedDict, Annotated
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 状态定义
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    """LangGraph 状态定义"""
    messages: Annotated[List[BaseMessage], operator.add]
    query: str
    session_id: str
    plan: List[str]
    research_results: List[str]
    draft_answer: str
    confidence: float
    hitl_required: bool
    hitl_decision: str
    hitl_feedback: str
    final_answer: str
    error: str

# ---------------------------------------------------------------------------
# 节点函数
# ---------------------------------------------------------------------------
class LangGraphAgent:
    """
    基于 LangGraph 的多智能体引擎，支持人工兜底（Human-in-the-loop）。
    """
    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        llm_api_key: str = "",
        llm_base_url: Optional[str] = None,
        hitl_timeout_s: float = 120.0,
        audit_log_path: Optional[str] = "logs/agent_audit.jsonl"
    ):
        self.llm = ChatOpenAI(
            model=llm_model,
            api_key=llm_api_key,
            base_url=llm_base_url,
            temperature=0.2
        )
        self.hitl_timeout_s = hitl_timeout_s
        self.audit_log_path = audit_log_path
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # 添加节点
        workflow.add_node("planner", self.node_planner)
        workflow.add_node("researcher", self.node_researcher)
        workflow.add_node("synthesizer", self.node_synthesizer)
        workflow.add_node("human_review", self.node_human_review)
        workflow.add_node("finalizer", self.node_finalizer)

        # 设置入口
        workflow.set_entry_point("planner")

        # 添加边和条件路由
        workflow.add_conditional_edges(
            "planner",
            self.route_after_plan,
            {
                "direct": "synthesizer",
                "research": "researcher",
                "error": END
            }
        )

        workflow.add_edge("researcher", "synthesizer")

        workflow.add_conditional_edges(
            "synthesizer",
            self.route_after_synthesis,
            {
                "human": "human_review",
                "final": "finalizer",
                "error": END
            }
        )

        workflow.add_conditional_edges(
            "human_review",
            self.route_after_hitl,
            {
                "final": "finalizer",
                "replan": "planner",
                "error": END
            }
        )

        workflow.add_edge("finalizer", END)

        # 编译图，设置 human_review 为中断点（人工介入）
        return workflow.compile(
            checkpointer=self.checkpointer,
            interrupt_before=["human_review"]
        )

    # --- 节点实现 ---
    async def node_planner(self, state: AgentState) -> Dict[str, Any]:
        """规划节点：分析问题并生成执行步骤。"""
        query = state.get("query", "")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个任务规划专家。请将用户的请求拆分为具体的执行步骤。如果问题很简单，只需一步。返回 JSON 格式：{{\"plan\": [\"step1\", \"step2\"]}}"),
            ("user", "{query}")
        ])
        chain = prompt | self.llm
        try:
            response = await chain.ainvoke({"query": query})
            # 简单解析 JSON
            content = response.content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > 0:
                data = json.loads(content[start:end])
                return {"plan": data.get("plan", ["直接回答"])}
            return {"plan": ["直接回答"]}
        except Exception as e:
            logger.error(f"Planner error: {e}")
            return {"error": str(e)}

    async def node_researcher(self, state: AgentState) -> Dict[str, Any]:
        """研究节点：模拟调用工具获取信息。"""
        plan = state.get("plan", [])
        results = []
        for step in plan:
            # 这里可以集成 RAG 或 Search 工具
            results.append(f"执行步骤 [{step}] 的结果：已获取相关背景信息。")
        return {"research_results": results}

    async def node_synthesizer(self, state: AgentState) -> Dict[str, Any]:
        """综合节点：生成草稿并评估置信度。"""
        query = state.get("query", "")
        research = "\n".join(state.get("research_results", []))
        feedback = state.get("hitl_feedback", "")
        
        sys_prompt = "你是一个综合解答专家。请根据背景信息回答用户问题。"
        if feedback:
            sys_prompt += f"\n注意，之前的人工审核意见是：{feedback}。请务必根据意见修改。"
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", sys_prompt),
            ("user", "背景信息：\n{research}\n\n用户问题：{query}\n\n请给出详细回答，并在最后一行输出置信度（0.0到1.0之间的数字，如 Confidence: 0.85）。")
        ])
        chain = prompt | self.llm
        try:
            response = await chain.ainvoke({"research": research, "query": query})
            content = response.content
            
            # 解析置信度
            confidence = 0.9
            lines = content.split("\n")
            draft = content
            for line in reversed(lines):
                if "Confidence:" in line or "置信度:" in line:
                    try:
                        import re
                        nums = re.findall(r"0\.\d+|1\.0", line)
                        if nums:
                            confidence = float(nums[0])
                            draft = content.replace(line, "").strip()
                            break
                    except:
                        pass
            
            # 置信度低于 0.8 则需要人工介入
            hitl_required = confidence < 0.8
            
            return {
                "draft_answer": draft,
                "confidence": confidence,
                "hitl_required": hitl_required
            }
        except Exception as e:
            return {"error": str(e)}

    async def node_human_review(self, state: AgentState) -> Dict[str, Any]:
        """人工审核节点：此节点在执行前会被 LangGraph 中断。恢复后执行此逻辑。"""
        # 实际的决策由外部通过 update_state 注入
        decision = state.get("hitl_decision", "approve")
        feedback = state.get("hitl_feedback", "")
        
        # 记录审计日志
        if self.audit_log_path:
            import pathlib
            pathlib.Path(self.audit_log_path).parent.mkdir(parents=True, exist_ok=True)
            record = {
                "ts": time.time(),
                "session_id": state.get("session_id", ""),
                "query": state.get("query", ""),
                "draft": state.get("draft_answer", ""),
                "decision": decision,
                "feedback": feedback
            }
            with open(self.audit_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
        return {}

    async def node_finalizer(self, state: AgentState) -> Dict[str, Any]:
        """最终输出节点。"""
        return {"final_answer": state.get("draft_answer", "")}

    # --- 路由逻辑 ---
    def route_after_plan(self, state: AgentState) -> str:
        if state.get("error"): return "error"
        plan = state.get("plan", [])
        if len(plan) <= 1: return "direct"
        return "research"

    def route_after_synthesis(self, state: AgentState) -> str:
        if state.get("error"): return "error"
        if state.get("hitl_required"): return "human"
        return "final"

    def route_after_hitl(self, state: AgentState) -> str:
        decision = state.get("hitl_decision", "reject")
        if decision == "approve": return "final"
        if decision == "revise": return "replan"
        return "error"

    # --- 外部接口 ---
    async def run(self, query: str, session_id: str) -> Dict[str, Any]:
        """运行智能体，处理中断和恢复。"""
        config = {"configurable": {"thread_id": session_id}}
        
        # 初始化状态
        initial_state = {"query": query, "session_id": session_id}
        
        # 运行图直到结束或中断
        async for event in self.graph.astream(initial_state, config=config):
            pass
            
        # 检查是否被中断（等待人工审核）
        state_snapshot = self.graph.get_state(config)
        if state_snapshot.next and "human_review" in state_snapshot.next:
            logger.info(f"Session {session_id} requires human review. Draft: {state_snapshot.values.get('draft_answer')}")
            # 在实际应用中，这里会通过 API 返回给前端，等待用户操作
            # 这里模拟自动处理或超时逻辑
            return {
                "status": "waiting_for_human",
                "session_id": session_id,
                "draft": state_snapshot.values.get("draft_answer")
            }
            
        return {
            "status": "completed",
            "answer": state_snapshot.values.get("final_answer"),
            "confidence": state_snapshot.values.get("confidence")
        }
        
    async def submit_human_decision(self, session_id: str, decision: str, feedback: str = ""):
        """提交人工审核决策并恢复执行。"""
        config = {"configurable": {"thread_id": session_id}}
        
        # 更新状态
        self.graph.update_state(
            config,
            {"hitl_decision": decision, "hitl_feedback": feedback},
            as_node="human_review"
        )
        
        # 恢复执行
        async for event in self.graph.astream(None, config=config):
            pass
            
        state_snapshot = self.graph.get_state(config)
        return {
            "status": "completed",
            "answer": state_snapshot.values.get("final_answer")
        }

    async def run_dify_compatible(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Dify Workflow API 兼容接口。"""
        import uuid
        inputs = payload.get("inputs", {})
        query = inputs.get("query", inputs.get("question", ""))
        user = payload.get("user", "anonymous")
        session_id = f"dify_{user}_{int(time.time())}"
        
        result = await self.run(query, session_id)
        
        status = "succeeded" if result.get("status") == "completed" else "waiting"
        return {
            "workflow_run_id": str(uuid.uuid4()),
            "task_id": session_id,
            "data": {
                "id": session_id,
                "status": status,
                "outputs": {
                    "answer": result.get("answer", ""),
                    "confidence": result.get("confidence", 0.0),
                }
            }
        }
