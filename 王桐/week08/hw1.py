import openai
import json
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from typing_extensions import Literal
from enum import Enum

# 定义枚举类型确保类型安全
class DomainEnum(str, Enum):
    MUSIC = "music"
    APP = "app"
    RADIO = "radio"
    LOTTERY = "lottery"
    STOCK = "stock"
    NOVEL = "novel"
    WEATHER = "weather"
    MATCH = "match"
    MAP = "map"
    WEBSITE = "website"
    NEWS = "news"
    MESSAGE = "message"
    CONTACTS = "contacts"
    TRANSLATION = "translation"
    TVCHANNEL = "tvchannel"
    CINEMAS = "cinemas"
    COOKBOOK = "cookbook"
    JOKE = "joke"
    RIDDLE = "riddle"
    TELEPHONE = "telephone"
    VIDEO = "video"
    TRAIN = "train"
    POETRY = "poetry"
    FLIGHT = "flight"
    EPG = "epg"
    HEALTH = "health"
    EMAIL = "email"
    BUS = "bus"
    STORY = "story"

class IntentEnum(str, Enum):
    OPEN = "OPEN"
    SEARCH = "SEARCH"
    REPLAY_ALL = "REPLAY_ALL"
    NUMBER_QUERY = "NUMBER_QUERY"
    DIAL = "DIAL"
    CLOSEPRICE_QUERY = "CLOSEPRICE_QUERY"
    SEND = "SEND"
    LAUNCH = "LAUNCH"
    PLAY = "PLAY"
    REPLY = "REPLY"
    RISERATE_QUERY = "RISERATE_QUERY"
    DOWNLOAD = "DOWNLOAD"
    QUERY = "QUERY"
    LOOK_BACK = "LOOK_BACK"
    CREATE = "CREATE"
    FORWARD = "FORWARD"
    DATE_QUERY = "DATE_QUERY"
    SENDCONTACTS = "SENDCONTACTS"
    DEFAULT = "DEFAULT"
    TRANSLATION = "TRANSLATION"
    VIEW = "VIEW"
    ROUTE = "ROUTE"
    POSITION = "POSITION"

class SlotEntity(BaseModel):
    """实体槽位定义"""
    code: Optional[str] = Field(None, description="代码")
    Src: Optional[str] = Field(None, description="出发地")
    startDate_dateOrig: Optional[str] = Field(None, description="开始日期原始值")
    film: Optional[str] = Field(None, description="电影")
    endLoc_city: Optional[str] = Field(None, description="目的地城市")
    artistRole: Optional[str] = Field(None, description="艺术家角色")
    location_country: Optional[str] = Field(None, description="国家")
    location_area: Optional[str] = Field(None, description="区域")
    author: Optional[str] = Field(None, description="作者")
    startLoc_city: Optional[str] = Field(None, description="出发城市")
    season: Optional[str] = Field(None, description="季节")
    dishName: Optional[str] = Field(None, description="菜名")
    media: Optional[str] = Field(None, description="媒体")
    datetime_date: Optional[str] = Field(None, description="日期")
    episode: Optional[str] = Field(None, description="剧集")
    teleOperator: Optional[str] = Field(None, description="运营商")
    questionWord: Optional[str] = Field(None, description="疑问词")
    receiver: Optional[str] = Field(None, description="接收者")
    ingredient: Optional[str] = Field(None, description="食材")
    name: Optional[str] = Field(None, description="名称")
    startDate_time: Optional[str] = Field(None, description="开始时间")
    startDate_date: Optional[str] = Field(None, description="开始日期")
    location_province: Optional[str] = Field(None, description="省份")
    endLoc_poi: Optional[str] = Field(None, description="目的地POI")
    artist: Optional[str] = Field(None, description="艺术家")
    dynasty: Optional[str] = Field(None, description="朝代")
    area: Optional[str] = Field(None, description="地区")
    location_poi: Optional[str] = Field(None, description="位置POI")
    relIssue: Optional[str] = Field(None, description="相关问题")
    Dest: Optional[List[str]] = Field(None, description="目的地列表")
    content: Optional[str] = Field(None, description="内容")
    keyword: Optional[str] = Field(None, description="关键词")
    target: Optional[str] = Field(None, description="目标")
    startLoc_area: Optional[str] = Field(None, description="出发区域")
    tvchannel: Optional[str] = Field(None, description="电视频道")
    type: Optional[str] = Field(None, description="类型")
    song: Optional[str] = Field(None, description="歌曲")
    queryField: Optional[str] = Field(None, description="查询字段")
    awayName: Optional[str] = Field(None, description="客场队伍")
    headNum: Optional[str] = Field(None, description="人数")
    homeName: Optional[str] = Field(None, description="主场队伍")
    decade: Optional[str] = Field(None, description="年代")
    payment: Optional[str] = Field(None, description="支付方式")
    popularity: Optional[str] = Field(None, description="流行度")
    tag: Optional[str] = Field(None, description="标签")
    startLoc_poi: Optional[str] = Field(None, description="出发地POI")
    date: Optional[str] = Field(None, description="日期")
    startLoc_province: Optional[str] = Field(None, description="出发省份")
    endLoc_province: Optional[str] = Field(None, description="目的省份")
    location_city: Optional[str] = Field(None, description="城市")
    absIssue: Optional[str] = Field(None, description="绝对问题")
    utensil: Optional[str] = Field(None, description="厨具")
    scoreDescr: Optional[str] = Field(None, description="分数描述")
    endLoc_area: Optional[str] = Field(None, description="目的地区域")
    resolution: Optional[str] = Field(None, description="分辨率")
    yesterday: Optional[str] = Field(None, description="昨天")
    timeDescr: Optional[str] = Field(None, description="时间描述")
    category: Optional[str] = Field(None, description="类别")
    subfocus: Optional[str] = Field(None, description="子焦点")
    theatre: Optional[str] = Field(None, description="剧院")
    datetime_time: Optional[str] = Field(None, description="时间")

class IntentDomainNerTask(BaseModel):
    """对文本抽取领域类别、意图类型、实体标签"""
    domain: DomainEnum = Field(description="领域类别")
    intent: IntentEnum = Field(description="意图类型")
    slots: SlotEntity = Field(description="实体槽位")

class ExtractionAgent:
    def __init__(self, model_name: str = "qwen-plus"):
        self.model_name = model_name
        self.client = openai.OpenAI(
            api_key="sk-78cc4e9ac8f44efdb207b7232ed8",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
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
                temperature=0.1  # 降低随机性，确保输出稳定
            )
            
            if response.choices[0].message.tool_calls:
                arguments = response.choices[0].message.tool_calls[0].function.arguments
                return IntentDomainNerTask.model_validate_json(arguments)
            else:
                print("未找到工具调用响应")
                return None
                
        except Exception as e:
            print(f"抽取过程中出错: {e}")
            return None

    def extract_with_fallback(self, text: str) -> Dict[str, Any]:
        """带降级策略的信息抽取"""
        
        # 首先尝试工具调用
        result = self.extract(text)
        if result:
            return result.model_dump()
        
        # 如果工具调用失败，使用传统提示词方法
        return self._fallback_extraction(text)
    
    def _fallback_extraction(self, text: str) -> Dict[str, Any]:
        """降级方案：使用传统提示词方法"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"请抽取以下文本的信息，以JSON格式输出：{text}"}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.1
        )
        
        # 尝试解析JSON响应
        try:
            content = response.choices[0].message.content
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            else:
                json_str = content.strip()
            
            return json.loads(json_str)
        except:
            # 最终降级：返回基础结构
            return {
                "domain": "unknown",
                "intent": "QUERY", 
                "slots": {}
            }

# 使用示例
if __name__ == "__main__":
    agent = ExtractionAgent()
    
    # 测试用例
    test_cases = [
        "糖醋鲤鱼怎么做啊？",
        "帮我查询下从北京到天津到武汉的汽车票",
        "播放周杰伦的七里香",
        "今天北京的天气怎么样",
        "查询贵州茅台的股价"
    ]
    
    for text in test_cases:
        print(f"\n输入文本: {text}")
        result = agent.extract_with_fallback(text)
        print(f"抽取结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
