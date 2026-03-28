import asyncio
import logging
import random
from typing import Any, Dict, List, Optional, Union

from playwright.async_api import async_playwright, Page, Browser, BrowserContext

logger = logging.getLogger(__name__)

class AutomationEngine:
    """
    专家级自动化执行引擎（基于 Playwright）。
    支持：
    1. 隐身模式与反爬虫绕过
    2. 智能等待与重试机制
    3. 结合视觉引擎的动态元素定位
    4. 异常恢复与状态保持
    """
    def __init__(
        self,
        headless: bool = True,
        user_data_dir: Optional[str] = "./browser_data",
        proxy: Optional[Dict[str, str]] = None,
        timeout_ms: int = 30000
    ):
        self.headless = headless
        self.user_data_dir = user_data_dir
        self.proxy = proxy
        self.timeout_ms = timeout_ms
        
        self._playwright = None
        self._browser: Optional[BrowserContext] = None
        self._page: Optional[Page] = None

    async def start(self):
        """启动浏览器实例。"""
        self._playwright = await async_playwright().start()
        
        launch_args = {
            "headless": self.headless,
            "args": [
                "--disable-blink-features=AutomationControlled",
                "--disable-infobars",
                "--no-sandbox",
                "--disable-setuid-sandbox"
            ]
        }
        if self.proxy:
            launch_args["proxy"] = self.proxy

        if self.user_data_dir:
            self._browser = await self._playwright.chromium.launch_persistent_context(
                user_data_dir=self.user_data_dir,
                **launch_args,
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        else:
            browser_instance = await self._playwright.chromium.launch(**launch_args)
            self._browser = await browser_instance.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            
        # 注入反爬虫脚本
        await self._browser.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """)
        
        self._page = await self._browser.new_page()
        self._page.set_default_timeout(self.timeout_ms)
        logger.info("Automation engine started.")

    async def stop(self):
        """关闭浏览器实例。"""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        logger.info("Automation engine stopped.")

    async def navigate(self, url: str, wait_until: str = "networkidle") -> bool:
        """导航到指定 URL。"""
        try:
            await self._page.goto(url, wait_until=wait_until)
            # 模拟人类行为：随机滚动
            await self._human_scroll()
            return True
        except Exception as e:
            logger.error(f"Navigation failed: {e}")
            return False

    async def _human_scroll(self):
        """模拟人类随机滚动。"""
        scroll_steps = random.randint(2, 5)
        for _ in range(scroll_steps):
            scroll_y = random.randint(100, 500)
            await self._page.mouse.wheel(0, scroll_y)
            await asyncio.sleep(random.uniform(0.5, 1.5))

    async def smart_click(self, selector: str, fallback_selectors: List[str] = None) -> bool:
        """
        智能点击：支持多级降级选择器。
        """
        selectors = [selector] + (fallback_selectors or [])
        for sel in selectors:
            try:
                element = self._page.locator(sel).first
                await element.wait_for(state="visible", timeout=5000)
                
                # 模拟人类点击：移动到元素上，稍作停顿，然后点击
                box = await element.bounding_box()
                if box:
                    x = box["x"] + box["width"] / 2 + random.uniform(-5, 5)
                    y = box["y"] + box["height"] / 2 + random.uniform(-5, 5)
                    await self._page.mouse.move(x, y, steps=10)
                    await asyncio.sleep(random.uniform(0.1, 0.3))
                    await self._page.mouse.click(x, y)
                else:
                    await element.click()
                    
                logger.info(f"Successfully clicked element using selector: {sel}")
                return True
            except Exception as e:
                logger.debug(f"Failed to click selector {sel}: {e}")
                continue
                
        logger.error(f"All selectors failed for click operation. Primary: {selector}")
        return False

    async def smart_fill(self, selector: str, text: str, fallback_selectors: List[str] = None) -> bool:
        """
        智能输入：模拟人类打字速度。
        """
        selectors = [selector] + (fallback_selectors or [])
        for sel in selectors:
            try:
                element = self._page.locator(sel).first
                await element.wait_for(state="visible", timeout=5000)
                
                await element.click()
                await element.fill("") # 清空
                
                # 模拟人类打字
                for char in text:
                    await element.type(char, delay=random.randint(30, 100))
                    
                logger.info(f"Successfully filled text using selector: {sel}")
                return True
            except Exception as e:
                logger.debug(f"Failed to fill selector {sel}: {e}")
                continue
                
        logger.error(f"All selectors failed for fill operation. Primary: {selector}")
        return False

    async def extract_data(self, extraction_rules: Dict[str, str]) -> Dict[str, Any]:
        """
        根据规则提取页面数据。
        extraction_rules: {"field_name": "css_selector"}
        """
        result = {}
        for field, selector in extraction_rules.items():
            try:
                elements = await self._page.locator(selector).all()
                if not elements:
                    result[field] = None
                    continue
                    
                if len(elements) == 1:
                    result[field] = await elements[0].inner_text()
                else:
                    result[field] = [await el.inner_text() for el in elements]
            except Exception as e:
                logger.warning(f"Failed to extract field {field}: {e}")
                result[field] = None
                
        return result

    async def take_screenshot(self, full_page: bool = False) -> bytes:
        """获取页面截图，可用于视觉引擎分析。"""
        return await self._page.screenshot(full_page=full_page)
