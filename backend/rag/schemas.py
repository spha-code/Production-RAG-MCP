from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str
    k: int = 3


class ChatResponse(BaseModel):
    chunks: list[str]
