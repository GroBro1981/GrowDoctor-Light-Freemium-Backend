import os
import base64
import json

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

# --------------------------------------------------
# 🔑 OpenAI-Client
# --------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY ist nicht gesetzt. Bitte als Environment Variable hinterlegen."
    )

client = OpenAI(api_key=OPENAI_API_KEY)

# --------------------------------------------------
# 🌐 FastAPI-App
# --------------------------------------------------
app = FastAPI(
    title="Canalyzer Backend",
    description="Bildbasierte Cannabis-Diagnose & Reifegrad-API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # später einschränken
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok", "message": "Canalyzer Backend läuft 😎"}


# --------------------------------------------------
# 🧠 OpenAI Helper – JETZT korrekt mit gpt-4o-mini
# --------------------------------------------------
def _call_openai_json(system_prompt: str, data_url: str, user_text: str) -> dict:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=900,
            temperature=0.1,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Fehler bei der Anfrage an OpenAI: {e}"
        )

    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="OpenAI hat kein gültiges JSON zurückgegeben."
        )


# --------------------------------------------------
# 📸 DIAGNOSE ENDPOINT
# --------------------------------------------------
DIAGNOSIS_PROMPT = """
Du bist ein sehr erfahrener Cannabis-Pflanzenarzt.
Analysiere das Bild und gib nur das JSON-Schema zurück, wie besprochen.
"""

@app.post("/diagnose")
async def diagnose(image: UploadFile = File(...)):
    if image.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=400, detail="Nur JPG oder PNG erlaubt.")

    img_bytes = await image.read()
    img_base64 = base64.b64encode(img_bytes).decode()
    data_url = f"data:{image.content_type};base64,{img_base64}"

    result = _call_openai_json(
        DIAGNOSIS_PROMPT,
        data_url,
        "Analysiere dieses Bild."
    )

    return result


# --------------------------------------------------
# 🌼 REIFEGRAD (TRICHOME) ENDPOINT
# --------------------------------------------------
RIPENESS_PROMPT = """
Du analysierst ausschließlich den Reifegrad der Trichome.
"""

@app.post("/ripeness")
async def ripeness(
    image: UploadFile = File(...),
    preference: str = Form("balanced"),
):
    if image.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=400, detail="Nur JPG oder PNG erlaubt.")

    img_bytes = await image.read()
    img_base64 = base64.b64encode(img_bytes).decode()
    data_url = f"data:{image.content_type};base64,{img_base64}"

    text = f"Nutzer-Wunschwirkung: {preference}. Analysiere den Reifegrad der Trichome."

    result = _call_openai_json(
        RIPENESS_PROMPT,
        data_url,
        text
    )

    return result
