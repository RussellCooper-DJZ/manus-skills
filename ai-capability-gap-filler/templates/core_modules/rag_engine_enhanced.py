import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class Document:
    doc_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    @classmethod
    def from_text(cls, text: str, metadata: Optional[Dict] = None) -> "Document":
        doc_id = hashlib.sha256(text.encode()).hexdigest()[:16]
        return cls(doc_id=doc_id, content=text, metadata=metadata or {})

@dataclass
class RetrievedDoc:
    document: Document
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0

# ---------------------------------------------------------------------------
# Layer 0: 智能分块 (Semantic Chunking)
# ---------------------------------------------------------------------------
class SemanticChunker:
    """语义感知分块器，保持段落和句子完整性。"""
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str, metadata: Optional[Dict] = None) -> List[Document]:
        paragraphs = re.split(r"\n{2,}", text.strip())
        chunks = []
        current = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para: continue
            
            if len(current) + len(para) + 1 <= self.chunk_size:
                current = (current + "\n" + para).strip()
            else:
                if current: chunks.append(current)
                if len(para) > self.chunk_size:
                    chunks.extend(self._split_by_sentence(para))
                    current = ""
                else:
                    overlap = current[-self.chunk_overlap:] if current else ""
                    current = (overlap + " " + para).strip() if overlap else para
                    
        if current: chunks.append(current)
        
        base_meta = metadata or {}
        return [
            Document.from_text(chunk, {**base_meta, "chunk_index": i})
            for i, chunk in enumerate(chunks) if chunk.strip()
        ]

    def _split_by_sentence(self, text: str) -> List[str]:
        sentences = re.split(r"(?<=[。！？.!?])\s*", text)
        chunks = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) <= self.chunk_size:
                current += sent
            else:
                if current: chunks.append(current)
                current = sent
        if current: chunks.append(current)
        return chunks

# ---------------------------------------------------------------------------
# Layer 1: 查询优化 (HyDE + Expansion)
# ---------------------------------------------------------------------------
class QueryOptimizer:
    def __init__(self, llm_fn: Optional[Callable[[str], str]] = None):
        self._llm = llm_fn

    def optimize(self, query: str) -> List[str]:
        queries = [query]
        if not self._llm: return queries
        
        try:
            # HyDE
            hyde_prompt = f"请为以下问题写一段简短的参考答案（2-3句话），包含关键概念：\n\n{query}"
            hyde_doc = self._llm(hyde_prompt)
            if hyde_doc: queries.append(hyde_doc)
            
            # Expansion
            expand_prompt = f"将以下问题改写为2个不同表达方式，用换行分隔：\n\n{query}"
            variants = self._llm(expand_prompt)
            if variants:
                queries.extend([v.strip() for v in variants.strip().split("\n") if v.strip()])
        except Exception as e:
            logger.error(f"Query optimization failed: {e}")
            
        return queries

# ---------------------------------------------------------------------------
# Layer 2: 混合检索 (Vector + BM25)
# ---------------------------------------------------------------------------
class HybridRetriever:
    def __init__(self, embed_fn: Callable[[str], List[float]]):
        self.embed_fn = embed_fn
        self.documents: List[Document] = []
        self.embeddings: np.ndarray = np.array([])
        
    def add_documents(self, docs: List[Document]):
        self.documents.extend(docs)
        # 批量计算 embedding
        texts = [d.content for d in docs]
        # 模拟批量调用
        new_embs = [self.embed_fn(t) for t in texts]
        for d, emb in zip(docs, new_embs):
            d.embedding = emb
            
        if len(self.embeddings) == 0:
            self.embeddings = np.array(new_embs)
        else:
            self.embeddings = np.vstack([self.embeddings, new_embs])
            
        # 归一化用于余弦相似度
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.embeddings = self.embeddings / norms

    def retrieve(self, queries: List[str], top_k: int = 10) -> List[RetrievedDoc]:
        if not self.documents: return []
        
        # 简单实现：取所有 query 的 embedding 平均值
        q_embs = [self.embed_fn(q) for q in queries]
        avg_q_emb = np.mean(q_embs, axis=0)
        
        q_norm = np.linalg.norm(avg_q_emb)
        if q_norm > 0: avg_q_emb = avg_q_emb / q_norm
        
        # 向量检索
        scores = self.embeddings @ avg_q_emb
        
        # 结合简单的关键词匹配 (模拟 BM25)
        query_terms = set(" ".join(queries).lower().split())
        
        results = []
        for i, doc in enumerate(self.documents):
            vec_score = float(scores[i])
            
            # 简单关键词得分
            doc_terms = set(doc.content.lower().split())
            bm25_score = len(query_terms & doc_terms) / (len(query_terms) + 1e-6)
            
            # 融合得分 (Alpha=0.7)
            final_score = 0.7 * vec_score + 0.3 * bm25_score
            
            results.append(RetrievedDoc(
                document=doc,
                vector_score=vec_score,
                bm25_score=bm25_score,
                final_score=final_score
            ))
            
        results.sort(key=lambda x: x.final_score, reverse=True)
        return results[:top_k]

# ---------------------------------------------------------------------------
# Layer 3: 重排序 (MMR)
# ---------------------------------------------------------------------------
class Reranker:
    @staticmethod
    def mmr_rerank(
        candidates: List[RetrievedDoc],
        query_embedding: List[float],
        top_k: int = 5,
        lambda_param: float = 0.7
    ) -> List[RetrievedDoc]:
        if not candidates: return []
        
        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 0: q = q / q_norm
        
        def cos_sim(a: List[float], b: List[float]) -> float:
            va, vb = np.array(a), np.array(b)
            na, nb = np.linalg.norm(va), np.linalg.norm(vb)
            if na == 0 or nb == 0: return 0.0
            return float(np.dot(va/na, vb/nb))
            
        selected = []
        remaining = list(candidates)
        
        while remaining and len(selected) < top_k:
            if not selected:
                best = max(remaining, key=lambda d: d.final_score)
            else:
                best = max(
                    remaining,
                    key=lambda d: (
                        lambda_param * d.final_score - 
                        (1 - lambda_param) * max(
                            cos_sim(d.document.embedding, s.document.embedding)
                            for s in selected
                        )
                    )
                )
            selected.append(best)
            remaining.remove(best)
            
        return selected

# ---------------------------------------------------------------------------
# Layer 4: 上下文压缩
# ---------------------------------------------------------------------------
class ContextCompressor:
    def __init__(self, max_chars: int = 2000):
        self.max_chars = max_chars
        
    def compress(self, query: str, docs: List[RetrievedDoc]) -> str:
        query_terms = set(query.lower().split())
        compressed_parts = []
        total_chars = 0
        
        for doc in docs:
            sentences = re.split(r"[。！？\n.!?]", doc.document.content)
            relevant = []
            
            for sent in sentences:
                sent = sent.strip()
                if not sent: continue
                sent_terms = set(sent.lower().split())
                overlap = len(query_terms & sent_terms)
                if overlap > 0:
                    relevant.append((overlap, sent))
                    
            relevant.sort(reverse=True)
            selected = " ".join(s for _, s in relevant[:3])
            
            if selected and total_chars + len(selected) <= self.max_chars:
                source = doc.document.metadata.get("source", doc.document.doc_id[:8])
                compressed_parts.append(f"[来源: {source}]\n{selected}")
                total_chars += len(selected)
                
        return "\n\n".join(compressed_parts)

# ---------------------------------------------------------------------------
# 顶层 RAG 引擎
# ---------------------------------------------------------------------------
class EnhancedRAGEngine:
    def __init__(
        self,
        embed_fn: Callable[[str], List[float]],
        llm_fn: Optional[Callable[[str], str]] = None
    ):
        self.chunker = SemanticChunker()
        self.optimizer = QueryOptimizer(llm_fn)
        self.retriever = HybridRetriever(embed_fn)
        self.compressor = ContextCompressor()
        self.embed_fn = embed_fn
        
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        docs = []
        for i, text in enumerate(texts):
            meta = metadatas[i] if metadatas else {}
            docs.extend(self.chunker.split(text, meta))
        self.retriever.add_documents(docs)
        
    def query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        # 1. 查询优化
        expanded_queries = self.optimizer.optimize(query)
        
        # 2. 混合检索
        candidates = self.retriever.retrieve(expanded_queries, top_k=top_k*2)
        
        # 3. MMR 重排序
        q_emb = self.embed_fn(query)
        reranked = Reranker.mmr_rerank(candidates, q_emb, top_k=top_k)
        
        # 4. 上下文压缩
        context = self.compressor.compress(query, reranked)
        
        return {
            "query": query,
            "expanded_queries": expanded_queries,
            "context": context,
            "retrieved_docs": [{"id": d.document.doc_id, "score": d.final_score} for d in reranked]
        }
