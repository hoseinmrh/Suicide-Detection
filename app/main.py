from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from predict.inference import predict_suicidal_text_loaded

app = FastAPI()


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)

class InputData(BaseModel):
    text: str

    # Sample input
    class Config:
        json_schema_extra = {
            "example": {
                "text": "متن خود را وارد کنید"
            }
        }


@app.post("/suicide-detection")
async def detect_suicide(input_data: InputData):
    result, english_text, score, lang = predict_suicidal_text_loaded(input_data.text)
    non_suicidal_rate = round(score[0]*100, 0)
    suicidal_rate = round(score[1] * 100, 0)
    return {"result": result, "english_text": english_text, "suicidal_rate": suicidal_rate,
            "non_suicidal_rate": non_suicidal_rate, "lang": lang}

