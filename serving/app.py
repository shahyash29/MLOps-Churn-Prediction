import os, logging, json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any
import pandas as pd, numpy as np, mlflow, mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from dotenv import load_dotenv
from prometheus_client import Counter, Summary, generate_latest
from starlette.middleware.cors import CORSMiddleware

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"))
log=logging.getLogger("churn-api")

MLFLOW_TRACKING_URI=os.getenv("MLFLOW_TRACKING_URI")
if MLFLOW_TRACKING_URI: mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MODEL_NAME=os.getenv("MODEL_NAME","churn")
MODEL_STAGE=os.getenv("MODEL_STAGE")
MODEL_VER=os.getenv("MODEL_VERSION")
VALID_STAGES={"Staging","Production","Archived"}
MODEL_URI_OVERRIDE=os.getenv("MODEL_URI_OVERRIDE")

def _build_model_uri():
    if MODEL_URI_OVERRIDE: return MODEL_URI_OVERRIDE
    if MODEL_VER: return f"models:/{MODEL_NAME}/{MODEL_VER}"
    if MODEL_STAGE and MODEL_STAGE.title() in VALID_STAGES:
        return f"models:/{MODEL_NAME}/{MODEL_STAGE.title()}"
    return f"models:/{MODEL_NAME}"
MODEL_URI=_build_model_uri()

PRED_LOG_DIR=Path(os.getenv("PRED_LOG_DIR","logs/inference"));PRED_LOG_DIR.mkdir(parents=True,exist_ok=True)
PRED_LOG_FILE=PRED_LOG_DIR/"events.jsonl"
def _log_event(d): PRED_LOG_FILE.open("a").write(json.dumps(d)+"\n")

app=FastAPI(title="Churn Prediction API",version="1.0.0")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
REQS=Counter("churn_api_requests_total","Total requests",["endpoint"])
LAT=Summary("churn_api_latency_seconds","Latency (s)",["endpoint"])

_model,_err=None,None
def _load_model():
    global _model,_err
    try:
        log.info(f"Loading model {MODEL_URI}"); _model=mlflow.pyfunc.load_model(MODEL_URI);_err=None
    except Exception as e: _model=None;_err=repr(e);log.error(f"Load fail: {e}")
@app.on_event("startup")
def _startup(): _load_model()

class PredictIn(BaseModel): features: Dict[str,Any]
def _extract(p): arr=np.ravel(p.to_numpy() if hasattr(p,"to_numpy") else np.asarray(p));return float(arr[0])

@app.get("/health")
def health():
    s="ok" if _model else "model_load_failed"
    return JSONResponse({"status":s,"model_uri":MODEL_URI,"tracking_uri":mlflow.get_tracking_uri(),"error":_err})

@app.post("/predict")
@LAT.labels("predict").time()
def predict(inp:PredictIn):
    REQS.labels("predict").inc()
    if _model is None: _load_model()
    if _model is None: raise HTTPException(503,"Model unavailable")
    try:
        df=pd.DataFrame([inp.features])
        proba=_extract(_model.predict(df))
        thr=float(os.getenv("MODEL_LABEL_THRESHOLD",0.5))
        label=int(proba>=thr)
        ev={"ts":datetime.now(timezone.utc).isoformat(),"model_uri":MODEL_URI,
            "features":inp.features,"proba":proba,"label":label}
        _log_event(ev)
        return {"churn_probability":proba,"prediction":label}
    except Exception as e:
        log.exception("Prediction error");raise HTTPException(400,f"Prediction failed: {repr(e)}")

@app.get("/metrics")
def metrics(): return Response(generate_latest(),media_type="text/plain; version=0.0.4; charset=utf-8")
