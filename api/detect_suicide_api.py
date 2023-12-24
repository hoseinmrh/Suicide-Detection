from fastapi import FastAPI
from pydantic import BaseModel
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import inference

app = FastAPI()


class InputData(BaseModel):
    text: str

    # Sample input
    class Config:
        json_schema_extra = {
            "example": {
                "text": "متن خود را وارد کنید",
            }
        }


@app.post("/suicide-detection")
async def detect_suicide(input_data: InputData):
    result, english_text, score = inference.predict_suicidal_text_loaded(input_data.text)
    non_suicidal_rate = round(score[0]*100, 0)
    suicidal_rate = round(score[1] * 100, 0)
    return {"result": result, "english_text": english_text, "suicidal_rate": suicidal_rate,
            "non_suicidal_rate": non_suicidal_rate}

