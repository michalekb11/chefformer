from fastapi import FastAPI
from app.routers.chat import chat_router

app = FastAPI()

app.include_router(chat_router)

@app.get("/")
def root():
    return {"message": "Hello World..."}

@app.get("/health")
def health_check():
    return {"status": "healthy"}
