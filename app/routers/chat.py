from fastapi import APIRouter
from pydantic import BaseModel, Field
from app.engine import TextGenerationService
from configs.inference.settings import app_settings

text_generator = TextGenerationService(model_checkpoint_path=app_settings.api.checkpoint_path)
chat_router = APIRouter(prefix="/v1/chat", tags=["Chat"])

class ChatInput(BaseModel):
    message: str = Field(..., min_length=1, description="The recipe or culinary prompt.")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Controls randomness. 0 is deterministic, higher is more creative.")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling probability.")
    top_k: int = Field(default=100, ge=0, description="Top token cutoff.")
    max_tokens: int = Field(default=512, gt=0, le=2048, description="The maximum number of tokens to generate.")
    stop_sequences: list[str] | None = Field(default=None, description="Optional list of strings where generation should stop.")

class ChatOutput(BaseModel):
    response: str

@chat_router.post("/generate")
def generate_chat(payload: ChatInput) -> ChatOutput:
    formatted_prompt = payload.message.strip()
    raw_output = text_generator.base_generate(
        prompt=formatted_prompt,
        temperature=payload.temperature,
        top_p=payload.top_p,
        top_k=payload.top_k,
        max_tokens=payload.max_tokens,
        stop_sequences=payload.stop_sequences,
    )

    return ChatOutput(response=raw_output)