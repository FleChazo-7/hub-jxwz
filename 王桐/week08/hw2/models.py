from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from typing_extensions import Literal
from enum import Enum

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

class ExtractionRequest(BaseModel):
    """信息抽取请求"""
    text: str = Field(..., description="待分析的文本", min_length=1, max_length=1000)
    model_name: Optional[str] = Field("qwen-plus", description="模型名称")

class ExtractionResponse(BaseModel):
    """信息抽取响应"""
    success: bool = Field(description="是否成功")
    data: Optional[IntentDomainNerTask] = Field(None, description="抽取结果")
    error: Optional[str] = Field(None, description="错误信息")
    processing_time: float = Field(description="处理时间(秒)")

class BatchExtractionRequest(BaseModel):
    """批量信息抽取请求"""
    texts: List[str] = Field(..., description="待分析的文本列表")
    model_name: Optional[str] = Field("qwen-plus", description="模型名称")

class BatchExtractionResponse(BaseModel):
    """批量信息抽取响应"""
    success: bool = Field(description="是否成功")
    results: List[ExtractionResponse] = Field(description="抽取结果列表")
    total_count: int = Field(description="总文本数")
    success_count: int = Field(description="成功数")
    failed_count: int = Field(description="失败数")
    total_processing_time: float = Field(description="总处理时间(秒)")
