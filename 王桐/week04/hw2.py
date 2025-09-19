from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.model import SentimentModel
import time
import logging
from typing import List

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="外卖评价情感分析API",
    description="基于BERT的情感分析服务",
    version="1.0.0"
)

# 初始化模型
model = SentimentModel()

class ReviewRequest(BaseModel):
    text: str

class BatchReviewRequest(BaseModel):
    texts: List[str]

class SentimentResponse(BaseModel):
    text: str
    sentiment: int
    confidence: float
    elapsed_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: ReviewRequest):
    """预测单个评价的情感倾向"""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="文本不能为空")
    
    start_time = time.time()
    
    try:
        sentiment, confidence = model.predict(request.text)
        elapsed_ms = (time.time() - start_time) * 1000
        
        logger.info(f"预测成功: '{request.text[:50]}...' -> 情感: {sentiment}, 置信度: {confidence:.4f}")
        
        return SentimentResponse(
            text=request.text,
            sentiment=sentiment,
            confidence=confidence,
            elapsed_ms=elapsed_ms
        )
    except Exception as e:
        logger.error(f"预测失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

@app.post("/predict_batch", response_model=List[SentimentResponse])
async def predict_batch_sentiment(request: BatchReviewRequest):
    """批量预测评价情感"""
    if not request.texts:
        raise HTTPException(status_code=400, detail="文本列表不能为空")
    
    results = []
    for text in request.texts:
        if text.strip():  # 跳过空文本
            start_time = time.time()
            sentiment, confidence = model.predict(text)
            elapsed_ms = (time.time() - start_time) * 1000
            
            results.append(SentimentResponse(
                text=text,
                sentiment=sentiment,
                confidence=confidence,
                elapsed_ms=elapsed_ms
            ))
    
    logger.info(f"批量预测完成: {len(results)} 条评价")
    return results

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """服务健康检查"""
    return HealthResponse(
        status="healthy",
        model_loaded=model.is_loaded,
        device=str(model.device)
    )

@app.get("/")
async def root():
    """根端点"""
    return {
        "message": "外卖评价情感分析API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict (POST)",
            "predict_batch": "/predict_batch (POST)",
            "health": "/health (GET)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
