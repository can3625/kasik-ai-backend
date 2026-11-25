import os
import json
import asyncio
from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from PIL import Image, UnidentifiedImageError

MAX_FILE_SIZE = 5 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    api_key = os.getenv("GEMINI_API_KEY")

    if api_key:
        client = genai.Client(api_key=api_key)
    else:
        print("UYARI: GEMINI_API_KEY tanımlanmamış.")

    yield

    if client is not None:
        client.close()


app = FastAPI(
    title="KaşıkAI Backend",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def home():
    return {"status": "online", "app": "KasikAI"}


@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/models")
def list_models():
    if client is None:
        return {"error": "Client yok"}
    return [m.name for m in client.models.list()]
    
@app.post("/analyze")
async def analyze_food(file: UploadFile = File(...)):
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="Gemini API anahtarı yapılandırılmamış.",
        )

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Yalnızca JPEG, PNG veya WEBP görseller kabul edilir.",
        )

    image_bytes = await file.read()

    if not image_bytes or len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="Görsel boş veya 5 MB sınırını aşıyor.",
        )

    try:
        image = Image.open(BytesIO(image_bytes))
        image.verify()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Geçersiz görsel dosyası.",
        )
    finally:
        await file.close()

    prompt = """
Sen KaşıkAI adlı yemek ve beslenme analiz asistanısın.
Görseldeki yemeği tanı ve porsiyon ile besin değerlerini tahmin et.
Eğer görselde yemek yoksa status değerini "not_food" yap.
Değerlerin fotoğraftan kesin ölçülemeyeceğini unutma.
Yalnızca şu JSON şemasına uygun cevap döndür:
{
  "status": "success",
  "food_name": "Yemeğin adı",
  "portion": {
    "amount": 1,
    "unit": "tabak"
  },
  "confidence": 0.9,
  "nutrition_estimate": {
    "energy_kcal": 300,
    "protein_g": 10,
    "carbs_g": 45,
    "fat_g": 8
  },
  "message": "Kısa açıklama ve tahmin uyarısı"
}
"""

    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model="gemini-2.5-flash",
            contents=[
                prompt,
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=file.content_type,
                ),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )

        if not response.text:
            raise ValueError("Gemini boş yanıt döndürdü.")

        return json.loads(response.text)

    except Exception as error:
        print(f"Gemini analiz hatası: {error}")
        raise HTTPException(
            status_code=500,
            detail="Yemek analiz edilemedi. Lütfen tekrar deneyin.",
        )
