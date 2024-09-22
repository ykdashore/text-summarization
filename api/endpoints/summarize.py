from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from models.summarizer import SummarizerModel
from trainers.trainer import start_training

class SummarizeRequest(BaseModel):
    text: str
    max_length: int = 128

router = APIRouter()
model = SummarizerModel()

@router.post("/summarize")
async def summarize_text(request: SummarizeRequest):
    summary = model.summarize(request.text, max_length=request.max_length)
    return {"summary": summary}

@router.post("/train")
async def train_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(start_training)
    return {"message": "Training started in the background. You will be notified when the model is saved."}
