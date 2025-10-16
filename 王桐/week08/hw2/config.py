import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """应用配置"""
    # API 配置
    api_title: str = "智能信息抽取API"
    api_description: str = "基于大模型的意图识别、领域分类和实体抽取服务"
    api_version: str = "1.0.0"
    
    # 服务器配置
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    
    # 大模型配置
    api_key: str = "sk-78cc4e9ac8f44efdb207b7232ed8"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    default_model: str = "qwen-plus"
    
    # 性能配置
    max_text_length: int = 1000
    batch_size: int = 10
    request_timeout: int = 30
    
    class Config:
        env_file = ".env"

settings = Settings()
