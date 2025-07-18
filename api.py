from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
import pandas as pd
import uvicorn
import os
import csv
from datetime import datetime
from dotenv import load_dotenv
import secrets

from main import (
    load_model,
    calculate_indicators,
    label_data,
    train_model,
    save_model,
    run_backtest,
    run_live_trading,
    get_historical_data,
    SYMBOL,
    TIMEFRAME,
    send_order
)

load_dotenv()

# --- Auto-generate API key if missing ---
if not os.getenv("API_KEY"):
    new_key = secrets.token_hex(32)
    with open(".env", "a") as f:
        f.write(f"API_KEY={new_key}\n")
    os.environ["API_KEY"] = new_key
    print(f"[INFO] New API_KEY generated and saved to .env: {new_key}")

API_KEY = os.getenv("API_KEY")

app = FastAPI(title="AI Trading Bot API", version="1.0")

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

# --- Init ---
os.makedirs("logs", exist_ok=True)
model_type = 'xgboost'
model, features, label_mapping = load_model(model_type=model_type)

# --- Input Models ---
class TrainRequest(BaseModel):
    bars: int = 5000
    genetic_optimization: bool = True

class PredictRequest(BaseModel):
    bars: int = 100

class BacktestRequest(BaseModel):
    bars: int = 2000
    initial_balance: float = 10000.0

class LiveTradingRequest(BaseModel):
    interval_seconds: int = 60

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/train", dependencies=[Depends(verify_api_key)])
def train(req: TrainRequest):
    global model, features, label_mapping
    df = get_historical_data(SYMBOL, TIMEFRAME, req.bars)
    if df is None or df.empty:
        raise HTTPException(status_code=500, detail="Failed to get historical data.")
    df = calculate_indicators(df)
    df = label_data(df)
    model, features, label_mapping = train_model(df, model_type=model_type, genetic_optimization=req.genetic_optimization)
    save_model(model, features, label_mapping, model_type=model_type)
    return {"message": "Model trained and saved successfully."}

@app.post("/predict", dependencies=[Depends(verify_api_key)])
def predict(req: PredictRequest):
    if model is None or features is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    df = get_historical_data(SYMBOL, TIMEFRAME, req.bars)
    df = calculate_indicators(df)
    latest = df.iloc[-1][features].values.reshape(1, -1)
    prediction = model.predict(latest)[0]
    proba = model.predict_proba(latest)[0]
    reverse_map = {v: k for k, v in label_mapping.items()}
    signal = reverse_map[prediction]

    # Log prediction
    with open("logs/predictions.csv", mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.utcnow().isoformat(), signal, proba.tolist()])

    return {"signal": signal, "confidence": proba.tolist()}

@app.post("/backtest", dependencies=[Depends(verify_api_key)])
def backtest(req: BacktestRequest):
    if model is None or features is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    df = get_historical_data(SYMBOL, TIMEFRAME, req.bars)
    df = calculate_indicators(df)
    run_backtest(df, model, features, initial_balance=req.initial_balance, label_mapping=label_mapping)
    return {"message": "Backtest completed. Check logs for results."}

@app.post("/start-live", dependencies=[Depends(verify_api_key)])
def start_live(req: LiveTradingRequest):
    if model is None or features is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    import threading
    t = threading.Thread(target=run_live_trading, args=(model, features, SYMBOL, TIMEFRAME, req.interval_seconds, label_mapping))
    t.start()
    return {"message": "Live trading started in background thread."}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
