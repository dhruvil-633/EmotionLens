# 🎭 EmotionLens

> **Real-time emotion recognition and deception tendency analysis** — Custom CNN trained on FER-2013, served via FastAPI, displayed in a React + Vite frontend.

[![Backend: FastAPI](https://img.shields.io/badge/backend-FastAPI-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Frontend: React + Vite](https://img.shields.io/badge/frontend-React%20%2B%20Vite-61DAFB?style=flat-square&logo=react)](https://vitejs.dev/)
[![Deploy: Render](https://img.shields.io/badge/deploy-Render-46E3B7?style=flat-square)](https://render.com)
[![Deploy: Vercel](https://img.shields.io/badge/deploy-Vercel-000?style=flat-square&logo=vercel)](https://vercel.com)

---

## ✨ Features

| Feature | Description |
|---|---|
| **Live Camera Analysis** | Webcam frames sent to inference API at configurable FPS |
| **Emotion Classification** | 7-class CNN (angry, disgust, fear, happy, neutral, sad, surprise) |
| **Deception Tendency** | EMA-smoothed truth/lie score derived from emotion → psycholinguistic mapping |
| **Bounding Box Overlay** | Color-coded face boxes drawn directly on the video feed |
| **Image Upload** | Static image analysis — returns face detections + emotion scores |
| **Live Timeline Chart** | SVG trend lines for truth, lie, and quality signals |
| **Session Stats** | Frame count, duration, dominant emotion, average quality |
| **Configurable Controls** | Confidence threshold, EMA alpha, frame interval — all live |

---

## 🏗️ Architecture

```
emotionlens/
├── backend/                    ← Python / FastAPI
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py             ← FastAPI app, session store, endpoints
│   │   ├── inference_engine.py ← CNN model + OpenCV face detector
│   │   └── deception_mapper.py ← EMA smoother + emotion→truth/lie mapping
│   ├── models/
│   │   └── best_model.pt       ← Pretrained 29 MB checkpoint (tracked in Git)
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/                   ← React + Vite
│   ├── src/
│   │   ├── main.jsx            ← React entry point
│   │   ├── App.jsx             ← All UI components
│   │   └── styles.css          ← Premium design system (CSS variables)
│   ├── index.html              ← Inter font, SEO meta tags
│   ├── package.json
│   ├── vite.config.js          ← Dev proxy → localhost:8000
│   ├── vercel.json             ← SPA rewrite rule
│   └── .env.example
│
├── render.yaml                 ← Render deployment config
└── .gitignore
```

---

## 🧠 Model — Training Results

Trained from scratch on **FER-2013** (35,887 images, 7 classes) using a custom CNN with GPU acceleration (NVIDIA RTX 4050 Laptop GPU).

| Metric | Value |
|---|---|
| **Best Validation Accuracy** | **68.99%** |
| **Test Accuracy** | **68.88%** |
| **Best Validation Loss** | 0.8672 |
| **Epochs Run** | 73 (early stopped) |
| **Training Time** | ~97 minutes |
| **Checkpoint Size** | ~29 MB |

### CNN Architecture

```
Input: 48×48 grayscale

Feature Extractor:
  Conv2d(1→64)  → BN → ReLU → Conv2d(64→64)  → BN → ReLU → MaxPool → Dropout(0.25)
  Conv2d(64→128)→ BN → ReLU → Conv2d(128→128)→ BN → ReLU → MaxPool → Dropout(0.25)
  Conv2d(128→256)→BN → ReLU → Conv2d(256→256)→ BN → ReLU → MaxPool → Dropout(0.25)
  Conv2d(256→512)→BN → ReLU → MaxPool → Dropout(0.25)

Classifier:
  Flatten → Linear(512×3×3 → 1024) → BN → ReLU → Dropout(0.5)
           → Linear(1024 → 512) → BN → ReLU → Dropout(0.4)
           → Linear(512 → 7)
```

### Deception Mapping

| Emotion | Truth % | Lie % | Rationale |
|---|---|---|---|
| 😊 Happy | 85 | 15 | Strongly associated with truthfulness |
| 😐 Neutral | 70 | 30 | Baseline honest expression |
| 😢 Sad | 60 | 40 | Correlates with genuine expression |
| 😲 Surprise | 45 | 55 | May indicate caught off-guard |
| 😠 Angry | 35 | 65 | Can accompany defensive deception |
| 🤢 Disgust | 30 | 70 | Signals suppressed discomfort |
| 😨 Fear | 25 | 75 | Often masks hidden information |

> Based on psycholinguistic research (Paul Ekman, 1969–2003). **For educational purposes only — not a scientific lie detector.**

---

## 🚀 Quick Start — Local Development

### Prerequisites

- Python 3.10+
- Node.js 18+
- A webcam (for live detector)

### 1. Clone & setup backend

```bash
git clone https://github.com/your-username/emotionlens.git
cd emotionlens

# Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # macOS/Linux

# Install backend deps
pip install -r backend/requirements.txt
```

### 2. Start the backend

```bash
# From the project root
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

API now running at `http://localhost:8000`.
Swagger docs: `http://localhost:8000/docs`

### 3. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend now running at `http://localhost:5173`.  
The Vite dev server proxies `/api` and `/health` to the backend automatically.

---

## 🔌 API Reference

Base URL (local): `http://localhost:8000`

### `GET /health`

Returns server and model status.

```json
{
  "status": "ok",
  "device": "NVIDIA GeForce RTX 4050 Laptop GPU",
  "model_path": "/path/to/backend/models/best_model.pt"
}
```

---

### `POST /api/v1/analyze/frame`

Analyze a single base64-encoded webcam frame.

**Request body:**
```json
{
  "session_id": "uuid-string",
  "image_base64": "data:image/jpeg;base64,...",
  "confidence_threshold": 0.45,
  "ema_alpha": 0.3
}
```

**Response:**
```json
{
  "session_id": "...",
  "latency_ms": 18.4,
  "quality": 82,
  "faces": [
    {
      "bbox": [120, 80, 160, 160],
      "emotion": "happy",
      "confidence": 0.91,
      "all_probs": { "angry": 0.01, "happy": 0.91, ... }
    }
  ],
  "top_face": { ... },
  "mapping": {
    "emotion": "happy",
    "smoothed_truth": 74.3,
    "smoothed_lie": 25.7,
    "reason": "Happiness strongly associated with truthfulness",
    "hex": "#639922",
    "emoji": "😊"
  },
  "stats": {
    "total_frames": 120,
    "session_duration": 18,
    "dominant_emotion": "happy",
    "avg_truth": 71.2,
    "avg_lie": 28.8,
    "avg_quality": 79.0,
    "emotion_counts": { "happy": 80, "neutral": 40 }
  },
  "history": [ ... ]
}
```

---

### `POST /api/v1/analyze/image`

Analyze an uploaded image file.

**Form data:** `file` (multipart)  
**Query params:** `confidence_threshold`, `ema_alpha`

---

### `POST /api/v1/session/reset`

Reset EMA state for a session.

```json
{ "session_id": "uuid-string" }
```

---

## ☁️ Deployment

### Backend → Render

1. Push repo to GitHub.
2. Create a new **Web Service** on [render.com](https://render.com).
3. Connect your GitHub repo — Render will auto-detect `render.yaml`.
4. Set the env variable `FRONTEND_ORIGIN` to your Vercel URL.
5. Deploy. The service URL will be something like `https://emotionlens-api.onrender.com`.

> **Note:** Render free tier spins down after inactivity. The first request may take ~30 s.

### Frontend → Vercel

1. Go to [vercel.com](https://vercel.com) → New Project → import your repo.
2. Set **Root Directory** to `frontend`.
3. Set environment variable:
   ```
   VITE_API_BASE_URL=https://emotionlens-api.onrender.com
   ```
4. Deploy. Done.

---

## ⚙️ Environment Variables

### Backend (`backend/.env.example`)

| Variable | Default | Description |
|---|---|---|
| `FRONTEND_ORIGIN` | `*` | Comma-separated allowed CORS origins |

### Frontend (`frontend/.env.example`)

| Variable | Default | Description |
|---|---|---|
| `VITE_API_BASE_URL` | _(empty = same origin)_ | Full URL of the FastAPI backend |

---

## 📦 Dependencies

### Backend

| Package | Version | Purpose |
|---|---|---|
| fastapi | 0.111.1 | REST API framework |
| uvicorn[standard] | 0.30.3 | ASGI server |
| pytorch | 2.2.2 | CNN inference |
| opencv-python-headless | 4.10.0.84 | Face detection (Haar cascade) |
| numpy | 1.26.4 | Tensor/array ops |
| python-multipart | 0.0.9 | File upload parsing |

### Frontend

| Package | Version | Purpose |
|---|---|---|
| react | 18.2.0 | UI framework |
| react-dom | 18.2.0 | DOM renderer |
| vite | 5.4.2 | Dev server + bundler |
| @vitejs/plugin-react | 4.3.1 | JSX transform |

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

> ⚠️ **Ethical Disclaimer:** Deception tendency scores are probabilistic estimates for educational and research purposes only. They are based on psycholinguistic emotion research and are **not scientifically validated lie detectors**. Do not use for real-world deception detection, legal proceedings, or any decision-making context.
