from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
import rag, agent
from openai import OpenAI
import io, re

client = OpenAI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    count = rag.load_data()
    app.state.parts_loaded = count
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class ChatRequest(BaseModel):
    message:      str  = ""
    history:      list = []
    image_base64: str  = None
    image_mime:   str  = None

class TTSRequest(BaseModel):
    text:  str
    voice: str = "nova"

@app.get("/health")
def health():
    return {
        "status": "ok",
        "parts_loaded": getattr(app.state, "parts_loaded", 0)
    }

@app.post("/chat")
def chat(req: ChatRequest):
    return agent.run_agent(
        message      = req.message,
        history      = req.history,
        image_base64 = req.image_base64,
        image_mime   = req.image_mime
    )

@app.post("/tts")
def tts(req: TTSRequest):
    from fastapi.responses import StreamingResponse
    clean = re.sub(r'\|\|SUGGEST:.*', '', req.text)
    clean = re.sub(r'[*_`#]', '', clean).strip()
    response = client.audio.speech.create(
        model="tts-1",
        voice=req.voice,
        input=clean
    )
    audio_bytes = response.read()
    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/mpeg"
    )

@app.post("/feedback")
def feedback(data: dict):
    import json
    from datetime import datetime
    entry = {**data, "timestamp": datetime.now().isoformat()}
    try:
        with open("feedback_log.json", "a") as f:
            f.write(json.dumps(entry) + "\n")
    except:
        pass
    return {"status": "logged"}