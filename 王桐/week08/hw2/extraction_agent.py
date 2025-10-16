import openai
import json
import time
from typing import Optional, Dict, Any
from models import IntentDomainNerTask, DomainEnum, IntentEnum, SlotEntity
from config import settings

class ExtractionAgent:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.default_model
        self.client = openai.OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.request_timeout
        )
        
        # 系统提示词模板
        self.system_prompt = """你是一个专业的信息抽取专家，请对用户输入的文本进行意图识别、领域分类和实体抽取。

领域类别说明：
- music: 音乐相关
- app: 应用相关  
- weather: 天气相关
- bus: 公交出行
- cookbook: 菜谱烹饪
- stock: 股票查询
- news: 新闻资讯
- translation: 翻译服务
- 其他领域根据内容判断

意图类型说明：
- QUERY: 查询信息
- SEARCH: 搜索内容
- PLAY: 播放媒体
- OPEN: 打开应用
- CREATE: 创建内容
- SEND: 发送信息

请准确识别文本中的实体信息，并填充到对应的槽位中。"""

    def extract(self, text: str) -> Optional[IntentDomainNerTask]:
        """执行信息抽取"""
        start_time = time.time()
        
        user_prompt = f"请对以下文本进行信息抽取：{text}"
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        tools = [{
            "type": "function",
            "function": {
                "name": "extract_intent_domain_ner",
                "description": "抽取文本的意图、领域和实体信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "domain": {
                            "type": "string",
                            "enum": [e.value for e in DomainEnum],
                            "description": "领域类别"
                        },
                        "intent": {
                            "type": "string", 
                            "enum": [e.value for e in IntentEnum],
                            "description": "意图类型"
                        },
                        "slots": {
                            "type": "object",
                            "properties": {k: {"type": "string"} for k in SlotEntity.model_fields.keys()},
                            "description": "实体槽位"
                        }
                    },
                    "required": ["domain", "intent", "slots"]
                }
            }
        }]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "extract_intent_domain_ner"}},
                temperature=0.1
            )
            
            if response.choices[0].message.tool_calls:
                arguments = response.choices[0].message.tool_calls[0].function.arguments
                processing_time = time.time() - start_time
                result = IntentDomainNerTask.model_validate_json(arguments)
                return result, processing_time
            else:
                return None, time.time() - start_time
                
        except Exception as e:
            print(f"抽取过程中出错: {e}")
            return None, time.time() - start_time

    def extract_with_fallback(self, text: str) -> Dict[str, Any]:
        """带降级策略的信息抽取"""
        start_time = time.time()
        
        # 首先尝试工具调用
        result, processing_time = self.extract(text)
        if result:
            return {
                "success": True,
                "data": result.model_dump(),
                "error": None,
                "processing_time": processing_time
            }
        
        # 如果工具调用失败，返回错误信息
        return {
            "success": False,
            "data": None,
            "error": "信息抽取失败",
            "processing_time": processing_time
        }
