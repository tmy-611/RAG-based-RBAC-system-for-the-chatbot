from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    """Body of POST /chat, sent by the client"""
    message: str = Field(min_length=1, description="User's question")


class ChatResponse(BaseModel):
    """Result returned by POST /chat, sent by the server"""
    answer: str
    department: str