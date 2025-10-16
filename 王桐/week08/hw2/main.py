from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import asyncio
import time
from typing import List

from models import (
    ExtractionRequest, 
    ExtractionResponse, 
    BatchExtractionRequest, 
    BatchExtractionResponse
)
from extraction_agent import ExtractionAgent
from config import settings

# 全局变量
extraction_agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    global extraction_agent
    extraction_agent = ExtractionAgent()
    print("信息抽取服务初始化完成")
    yield
    # 关闭时清理
    print("信息抽取服务关闭")

# 创建 FastAPI 应用
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该配置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "智能信息抽取API服务",
        "version": settings.api_version,
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": time.time()
    }

@app.post("/extract", response_model=ExtractionResponse)
async def extract_intent(request: ExtractionRequest):
    """单文本信息抽取"""
    try:
        # 文本长度验证
        if len(request.text) > settings.max_text_length:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"文本长度不能超过{settings.max_text_length}个字符"
            )
        
        # 执行信息抽取
        result = extraction_agent.extract_with_fallback(request.text)
        
        if result["success"]:
            return ExtractionResponse(**result)
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=result["error"]
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"服务器内部错误: {str(e)}"
        )

@app.post("/extract/batch", response_model=BatchExtractionResponse)
async def batch_extract(request: BatchExtractionRequest, background_tasks: BackgroundTasks):
    """批量信息抽取"""
    try:
        # 验证批量请求
        if len(request.texts) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="文本列表不能为空"
            )
        
        if len(request.texts) > 100:  # 限制批量处理数量
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="单次批量处理不能超过100个文本"
            )
        
        total_start_time = time.time()
        results = []
        success_count = 0
        failed_count = 0
        
        # 批量处理
        for text in request.texts:
            try:
                if len(text) > settings.max_text_length:
                    result = ExtractionResponse(
                        success=False,
                        data=None,
                        error=f"文本长度超过限制",
                        processing_time=0
                    )
                    failed_count += 1
                else:
                    single_result = extraction_agent.extract_with_fallback(text)
                    if single_result["success"]:
                        success_count += 1
                    else:
                        failed_count += 1
                    result = ExtractionResponse(**single_result)
                
                results.append(result)
                
            except Exception as e:
                results.append(ExtractionResponse(
                    success=False,
                    data=None,
                    error=str(e),
                    processing_time=0
                ))
                failed_count += 1
        
        total_processing_time = time.time() - total_start_time
        
        return BatchExtractionResponse(
            success=True,
            results=results,
            total_count=len(request.texts),
            success_count=success_count,
            failed_count=failed_count,
            total_processing_time=total_processing_time
        )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"批量处理失败: {str(e)}"
        )

@app.get("/models")
async def list_models():
    """获取支持的模型列表"""
    return {
        "supported_models": ["qwen-plus", "qwen-turbo", "qwen-max"],
        "default_model": settings.default_model
    }

# 异常处理
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "data": None,
            "processing_time": 0
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "服务器内部错误",
            "data": None,
            "processing_time": 0
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload
    )
