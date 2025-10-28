import os
from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import init_pipeline

from dotenv import load_dotenv
load_dotenv()

app = FastAPI(
    title="RAG Demo API",
    description="Simple RAG endpoint: ask a question, get an answer.",
    version="0.1.0",
)

# Khởi tạo global pipeline khi server start
rag_chain = init_pipeline()

# Request body schema
class AskRequest(BaseModel):
    question: str

# Response schema
class AskResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    # Gọi RAG chain với câu hỏi của user
    answer = rag_chain.invoke(req.question)
    return AskResponse(answer=answer)
