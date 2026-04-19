import base64
import os
import pathlib
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .deception_mapper import DeceptionMapper
from .inference_engine import EmotionInferenceEngine


class FrameAnalyzeRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    image_base64: str
    confidence_threshold: float = Field(default=0.45, ge=0.3, le=0.9)
    ema_alpha: float = Field(default=0.3, ge=0.1, le=0.9)


class SessionResetRequest(BaseModel):
    session_id: str


@dataclass
class SessionState:
    mapper: DeceptionMapper
    last_seen: float


class SessionStore:
    def __init__(self) -> None:
        self._items: Dict[str, SessionState] = {}
        self._lock = threading.Lock()
        self._ttl_seconds = 60 * 30

    def get_or_create(self, session_id: str, alpha: float, conf_threshold: float) -> DeceptionMapper:
        now = time.time()
        with self._lock:
            self._cleanup(now)
            if session_id not in self._items:
                self._items[session_id] = SessionState(mapper=DeceptionMapper(alpha=alpha, conf_threshold=conf_threshold), last_seen=now)
            state = self._items[session_id]
            state.last_seen = now
            state.mapper.alpha = alpha
            state.mapper.conf_threshold = conf_threshold
            return state.mapper

    def reset(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._items:
                self._items[session_id].mapper.reset()
                self._items[session_id].last_seen = time.time()

    def _cleanup(self, now: float) -> None:
        expired = [k for k, v in self._items.items() if now - v.last_seen > self._ttl_seconds]
        for key in expired:
            del self._items[key]


ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / 'models' / 'best_model.pt'

app = FastAPI(title='EmotionLens API', version='1.0.0')

frontend_origin = os.getenv('FRONTEND_ORIGIN', '*')
allowed_origins: List[str] = [o.strip() for o in frontend_origin.split(',') if o.strip()]
if not allowed_origins:
    allowed_origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials='*' not in allowed_origins,
    allow_methods=['*'],
    allow_headers=['*'],
)

engine: Optional[EmotionInferenceEngine] = None
sessions = SessionStore()


@app.on_event('startup')
def startup_event() -> None:
    global engine
    engine = EmotionInferenceEngine(model_path=MODEL_PATH)


@app.get('/health')
def health() -> Dict:
    if engine is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    return {
        'status': 'ok',
        'device': engine.device_name,
        'model_path': str(MODEL_PATH),
    }


def _decode_base64_image(payload: str) -> np.ndarray:
    if ',' in payload:
        payload = payload.split(',', 1)[1]
    try:
        image_bytes = base64.b64decode(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail='Invalid base64 image payload') from exc

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail='Could not decode image')
    return frame


@app.post('/api/v1/analyze/frame')
def analyze_frame(request: FrameAnalyzeRequest) -> Dict:
    if engine is None:
        raise HTTPException(status_code=503, detail='Model not ready')

    frame = _decode_base64_image(request.image_base64)
    started = time.perf_counter()
    results, quality = engine.detect_and_predict(frame)

    mapper = sessions.get_or_create(
        session_id=request.session_id,
        alpha=request.ema_alpha,
        conf_threshold=request.confidence_threshold,
    )

    top = results[0] if results else None
    mapped = mapper.update(top['emotion'], top['confidence'], quality) if top else mapper.get_current()

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        'session_id': request.session_id,
        'latency_ms': round(elapsed_ms, 2),
        'quality': quality,
        'faces': results,
        'top_face': top,
        'mapping': mapped,
        'stats': mapper.get_session_stats(),
        'history': mapper.get_history(limit=180),
    }


@app.post('/api/v1/analyze/image')
async def analyze_image(file: UploadFile = File(...), confidence_threshold: float = 0.45, ema_alpha: float = 0.3) -> Dict:
    if engine is None:
        raise HTTPException(status_code=503, detail='Model not ready')

    raw = await file.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail='Invalid image file')

    session_id = str(uuid.uuid4())
    mapper = sessions.get_or_create(session_id, ema_alpha, confidence_threshold)

    results, quality = engine.detect_and_predict(frame)
    face_mappings = []
    for r in results:
        face_mappings.append(mapper.update(r['emotion'], r['confidence'], quality))

    return {
        'session_id': session_id,
        'quality': quality,
        'faces': results,
        'face_mappings': face_mappings,
        'stats': mapper.get_session_stats(),
    }


@app.post('/api/v1/session/reset')
def reset_session(request: SessionResetRequest) -> Dict:
    sessions.reset(request.session_id)
    return {'status': 'ok', 'session_id': request.session_id}
