# EmotionLens (Production Web Stack)

EmotionLens is now split into:

- `backend/` - FastAPI inference API (deploy on Render)
- `frontend/` - React + Vite SPA (deploy on Vercel)

Your trained model is reused from:

- `models/best_model.pt`

No retraining is required.

## Architecture

- Browser captures continuous webcam video using `getUserMedia`
- Frontend sends compressed JPEG frames to backend at a controlled interval
- Backend runs face detection + PyTorch emotion inference + deception mapping
- Frontend renders overlays, live truth/lie bar, quality, probabilities, and timeline

## 1) Backend (Render)

### Local run

```powershell
python -m venv .venv_api
.\.venv_api\Scripts\activate
pip install -r backend\requirements.txt
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Health check:

```powershell
curl http://127.0.0.1:8000/health
```

### Deploy on Render

This repo includes `render.yaml`:

- Build command: `pip install -r backend/requirements.txt`
- Start command: `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`

Set environment variable in Render:

- `FRONTEND_ORIGIN=https://<your-vercel-domain>.vercel.app`

## 2) Frontend (Vercel)

### Local run

```powershell
cd frontend
npm install
copy .env.example .env
```

Edit `.env`:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Start frontend:

```powershell
npm run dev
```

### Deploy on Vercel

- Root directory: `frontend`
- Build command: `npm run build`
- Output directory: `dist`
- Set env var:
  - `VITE_API_BASE_URL=https://<your-render-service>.onrender.com`

`frontend/vercel.json` is included for SPA routing.

## API Endpoints

- `GET /health`
- `POST /api/v1/analyze/frame`
- `POST /api/v1/analyze/image`
- `POST /api/v1/session/reset`

## Performance Tips

- Lower frame interval in frontend for faster responsiveness (`80-180ms` range)
- Keep request-in-flight throttling enabled (already implemented)
- Prefer CUDA on backend if available; falls back to DirectML/CPU
- Use `opencv-python-headless` on server (already configured)

## Notes

- Existing Streamlit files remain in the repo but are no longer required for production deployment.
- Deception tendency output is educational and probabilistic, not a real-world lie detector.
