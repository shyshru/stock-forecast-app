from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model_utils import run_forecast

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ForecastRequest(BaseModel):
    stock: str
    forecast_days: int

@app.post("/forecast")
async def forecast(req: ForecastRequest):
    forecast = run_forecast(req.stock, req.forecast_days)
    return {"forecast": forecast}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
