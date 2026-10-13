import json
import yaml
import sqlite3
import logging
from pathlib import Path
from datetime import datetime , timezone
from typing import List, Optional
from contextlib import asynccontextmanager

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LABEL_NAMES = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


CFG = load_config()

class SentimentModel:
    def __init__(self):
        model_path = Path(CFG["training"]["output_dir"]) / "best"
        self.device = torch.device("cpu")
        logger.info("Running on CPU (Docker container)")
        logger.info(f"Loading model from {model_path} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.max_length = CFG["model"]["max_length"]
        logger.info("Model ready")

    @torch.no_grad()
    def predict(self, texts: List[str]):
        enc = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        logits = self.model(**enc).logits
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        preds  = probs.argmax(axis=-1)
        return preds.tolist(), probs.tolist()


sentiment_model = None
@asynccontextmanager
async def lifespan(app: FastAPI):
    global sentiment_model
    init_db()
    sentiment_model = SentimentModel()
    logger.info("API ready")
    yield

app = FastAPI(
    title="Sentiment Intelligence API",
    description="5-class Yelp sentiment with DistilBERT",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def serve_ui():
    return FileResponse("frontend/index.html")

def init_db():
    db_path = Path(CFG["monitoring"]["db_path"])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            text TEXT,
            predicted_label TEXT,
            predicted_class INTEGER,
            confidence REAL,
            char_len INTEGER
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            text TEXT,
            predicted_class INTEGER,
            correct_class INTEGER,
            confidence REAL,
            used_in_training INTEGER DEFAULT 0
        )
    """)
    con.commit()
    con.close()
    logger.info(f"DB ready at {CFG['monitoring']['db_path']}")


def log_prediction(text: str, pred_class: int, confidence: float):
    con = sqlite3.connect(CFG["monitoring"]["db_path"])
    con.execute(
        "INSERT INTO predictions VALUES (NULL,?,?,?,?,?,?)",
        (datetime.utcnow().isoformat(), text[:500],
         LABEL_NAMES[pred_class], pred_class, confidence, len(text))
    )
    con.commit()
    con.close()

# ── Schemas ──
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=5, max_length=5000)

class PredictResponse(BaseModel):
    text: str
    predicted_label: str
    predicted_class: int
    confidence: float
    all_probabilities: dict

class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100)

class BatchItem(BaseModel):
    text: str
    predicted_label: str
    predicted_class: int
    confidence: float

class BatchResponse(BaseModel):
    results: List[BatchItem]
    total: int

class FeedbackRequest(BaseModel):
    text: str = Field(..., min_length=1)
    predicted_class: int = Field(..., ge=0, le=4)
    correct_class: int = Field(..., ge=0, le=4)
    confidence: float = Field(..., ge=0.0, le=1.0)

class FeedbackResponse(BaseModel):
    status: str
    message: str
    corrections_total: int

@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(req: FeedbackRequest):
    con = sqlite3.connect(CFG["monitoring"]["db_path"])
    con.execute(
        """INSERT INTO corrections
           (timestamp, text, predicted_class, correct_class, confidence, used_in_training)
           VALUES (?,?,?,?,?,0)""",
        (datetime.now(timezone.utc).isoformat(),
         req.text[:500],
         req.predicted_class,
         req.correct_class,
         req.confidence)
    )
    con.commit()
    count = con.execute(
        "SELECT COUNT(*) FROM corrections WHERE used_in_training=0"
    ).fetchone()[0]
    con.close()
    logger.info(
        f"Correction logged: predicted={LABEL_NAMES[req.predicted_class]} "
        f"correct={LABEL_NAMES[req.correct_class]} conf={req.confidence:.3f}"
    )
    return FeedbackResponse(
        status="ok",
        message=f"Feedback saved. {count} corrections pending retraining.",
        corrections_total=count
    )

@app.get("/feedback/stats")
def feedback_stats():
    con = sqlite3.connect(CFG["monitoring"]["db_path"])
    total     = con.execute("SELECT COUNT(*) FROM corrections").fetchone()[0]
    pending   = con.execute("SELECT COUNT(*) FROM corrections WHERE used_in_training=0").fetchone()[0]
    by_class  = con.execute("""
        SELECT predicted_class, correct_class, COUNT(*)
        FROM corrections GROUP BY predicted_class, correct_class
    """).fetchall()
    con.close()
    return {
        "total_corrections":   total,
        "pending_retraining":  pending,
        "retraining_threshold": 20,
        "ready_to_retrain":    pending >= 20,
        "correction_patterns": [
            {"predicted": LABEL_NAMES[r[0]],
             "corrected_to": LABEL_NAMES[r[1]],
             "count": r[2]}
            for r in by_class
        ]
    }

class ModelInfo(BaseModel):
    model_name: str
    version: str
    num_labels: int
    label_names: List[str]
    evaluation_metrics: Optional[dict]
    total_predictions: int


# ── Endpoints ──
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": sentiment_model is not None}

@app.websocket("/ws/predict")
async def ws_predict(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection opened")
    try:
        while True:
            text = await websocket.receive_text()
            text = text.strip()

            if len(text) < 5:
                await websocket.send_json({
                    "status": "too_short",
                    "message": "Keep typing..."
                })
                continue

            if sentiment_model is None:
                await websocket.send_json({"status": "error", "message": "Model not loaded"})
                continue

            preds, probs = sentiment_model.predict([text])
            pred_class = preds[0]
            prob_list  = probs[0]
            confidence = round(float(prob_list[pred_class]), 4)

            log_prediction(text, pred_class, confidence)

            await websocket.send_json({
                "status":          "ok",
                "predicted_label": LABEL_NAMES[pred_class],
                "predicted_class": pred_class,
                "confidence":      confidence,
                "all_probabilities": {
                    LABEL_NAMES[i]: round(float(p), 4)
                    for i, p in enumerate(prob_list)
                }
            })

    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if sentiment_model is None:
        raise HTTPException(503, "Model not loaded")

    preds, probs = sentiment_model.predict([req.text])
    pred_class   = preds[0]
    prob_list    = probs[0]
    confidence   = round(float(prob_list[pred_class]), 4)

    log_prediction(req.text, pred_class, confidence)

    return PredictResponse(
        text=req.text,
        predicted_label=LABEL_NAMES[pred_class],
        predicted_class=pred_class,
        confidence=confidence,
        all_probabilities={
            LABEL_NAMES[i]: round(float(p), 4)
            for i, p in enumerate(prob_list)
        }
    )


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    if sentiment_model is None:
        raise HTTPException(503, "Model not loaded")

    preds, probs = sentiment_model.predict(req.texts)
    results = []
    for text, pred_class, prob_list in zip(req.texts, preds, probs):
        confidence = round(float(prob_list[pred_class]), 4)
        log_prediction(text, pred_class, confidence)
        results.append(BatchItem(
            text=text,
            predicted_label=LABEL_NAMES[pred_class],
            predicted_class=pred_class,
            confidence=confidence
        ))

    return BatchResponse(results=results, total=len(results))


@app.get("/model/info", response_model=ModelInfo)
def model_info():
    eval_path = Path("reports/evaluation/evaluation_results.json")
    eval_metrics = None
    if eval_path.exists():
        with open(eval_path) as f:
            eval_metrics = json.load(f)

    con = sqlite3.connect(CFG["monitoring"]["db_path"])
    count = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    con.close()

    return ModelInfo(
        model_name=CFG["model"]["base_model"],
        version=CFG["project"]["version"],
        num_labels=CFG["model"]["num_labels"],
        label_names=LABEL_NAMES,
        evaluation_metrics=eval_metrics,
        total_predictions=count
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False,
                app_dir="src/api")