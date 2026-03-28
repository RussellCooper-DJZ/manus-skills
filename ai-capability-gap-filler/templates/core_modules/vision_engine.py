import base64
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import io

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class TextBlock:
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h

@dataclass
class OCRResult:
    blocks: List[TextBlock]
    full_text: str
    image_hash: str
    error: Optional[str] = None

@dataclass
class UIChangeEvent:
    change_type: str  # "layout_change", "content_change", "semantic_change"
    description: str
    old_hash: str
    new_hash: str
    diff_score: float
    semantic_analysis: Optional[str] = None

# ---------------------------------------------------------------------------
# 图像处理与哈希
# ---------------------------------------------------------------------------
def perceptual_hash(image_bytes: bytes, hash_size: int = 8) -> str:
    """计算图像的感知哈希 (pHash)，用于快速比对。"""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = np.array(img)
        diff = pixels[:, 1:] > pixels[:, :-1]
        return hex(int("".join(["1" if b else "0" for b in diff.flatten()]), 2))[2:]
    except Exception as e:
        logger.error(f"pHash error: {e}")
        return hashlib.md5(image_bytes).hexdigest()

def hamming_distance(hash1: str, hash2: str) -> int:
    """计算两个哈希值的汉明距离。"""
    try:
        b1 = bin(int(hash1, 16))[2:].zfill(64)
        b2 = bin(int(hash2, 16))[2:].zfill(64)
        return sum(c1 != c2 for c1, c2 in zip(b1, b2))
    except:
        return 64

# ---------------------------------------------------------------------------
# 多模态视觉引擎 (PaddleOCR + MiniCPM-V)
# ---------------------------------------------------------------------------
class VisionEngine:
    """
    专家级多模态视觉引擎。
    结合 PaddleOCR 进行精准文本定位，结合 VLM (如 MiniCPM-V) 进行语义级界面变化分析。
    """
    def __init__(
        self,
        use_gpu: bool = False,
        vlm_api_url: Optional[str] = None,
        vlm_api_key: Optional[str] = None,
        change_threshold: int = 10
    ):
        self._use_gpu = use_gpu
        self._ocr = None
        self._vlm_api_url = vlm_api_url
        self._vlm_api_key = vlm_api_key
        self._threshold = change_threshold
        self._history: List[Tuple[str, bytes, float]] = []  # (hash, image_bytes, timestamp)
        
    def _load_ocr(self):
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR
                self._ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=self._use_gpu, show_log=False)
                logger.info("PaddleOCR loaded successfully.")
            except ImportError:
                logger.warning("PaddleOCR not installed. OCR fallback to empty.")
                self._ocr = "fallback"

    def recognize_text(self, image_bytes: bytes) -> OCRResult:
        """执行 OCR 识别。"""
        img_hash = perceptual_hash(image_bytes)
        self._load_ocr()
        
        if self._ocr == "fallback" or self._ocr is None:
            return OCRResult(blocks=[], full_text="", image_hash=img_hash, error="OCR not available")
            
        try:
            img = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
            raw_results = self._ocr.ocr(img, cls=True)
            
            blocks = []
            texts = []
            for line in (raw_results or [[]]):
                for item in (line or []):
                    if not item or len(item) < 2: continue
                    bbox_raw, (text, conf) = item
                    if conf < 0.8: continue
                    
                    xs = [p[0] for p in bbox_raw]
                    ys = [p[1] for p in bbox_raw]
                    x, y = int(min(xs)), int(min(ys))
                    w, h = int(max(xs) - x), int(max(ys) - y)
                    
                    blocks.append(TextBlock(text=text, confidence=float(conf), bbox=(x, y, w, h)))
                    texts.append(text)
                    
            blocks.sort(key=lambda b: (b.bbox[1] // 20, b.bbox[0]))
            return OCRResult(blocks=blocks, full_text=" ".join(texts), image_hash=img_hash)
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return OCRResult(blocks=[], full_text="", image_hash=img_hash, error=str(e))

    async def analyze_ui_change(self, new_image_bytes: bytes) -> Optional[UIChangeEvent]:
        """
        检测界面变化。如果感知哈希差异超过阈值，则调用 VLM 进行深度语义分析。
        """
        new_hash = perceptual_hash(new_image_bytes)
        
        if not self._history:
            self._history.append((new_hash, new_image_bytes, time.time()))
            return None
            
        last_hash, last_image_bytes, _ = self._history[-1]
        dist = hamming_distance(last_hash, new_hash)
        
        if dist > self._threshold:
            diff_score = min(dist / 64.0, 1.0)
            change_type = "layout_change" if diff_score > 0.3 else "content_change"
            
            # 如果配置了 VLM，进行深度语义分析
            semantic_analysis = None
            if self._vlm_api_url:
                semantic_analysis = await self._call_vlm_for_diff(last_image_bytes, new_image_bytes)
                if semantic_analysis:
                    change_type = "semantic_change"
            
            event = UIChangeEvent(
                change_type=change_type,
                description=f"UI 变化检测：汉明距离={dist}，变化程度={diff_score:.1%}",
                old_hash=last_hash,
                new_hash=new_hash,
                diff_score=diff_score,
                semantic_analysis=semantic_analysis
            )
            
            self._history.append((new_hash, new_image_bytes, time.time()))
            # 保持历史记录不超过 5 条
            if len(self._history) > 5:
                self._history.pop(0)
                
            return event
            
        self._history.append((new_hash, new_image_bytes, time.time()))
        if len(self._history) > 5:
            self._history.pop(0)
        return None

    async def _call_vlm_for_diff(self, img1_bytes: bytes, img2_bytes: bytes) -> Optional[str]:
        """调用 MiniCPM-V 或其他 VLM 分析两张图片的语义差异。"""
        try:
            import aiohttp
            b64_1 = base64.b64encode(img1_bytes).decode('utf-8')
            b64_2 = base64.b64encode(img2_bytes).decode('utf-8')
            
            payload = {
                "model": "minicpm-v",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "请对比这两张界面截图，详细说明发生了哪些功能性或语义上的变化（例如：新增了弹窗、按钮位置改变、进入了新页面等）。"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_1}"}},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_2}"}}
                        ]
                    }
                ]
            }
            
            headers = {"Authorization": f"Bearer {self._vlm_api_key}"} if self._vlm_api_key else {}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self._vlm_api_url, json=payload, headers=headers, timeout=15) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("choices", [{}])[0].get("message", {}).get("content")
            return None
        except Exception as e:
            logger.error(f"VLM diff analysis failed: {e}")
            return None
