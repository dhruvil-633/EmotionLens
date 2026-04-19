import { useEffect, useMemo, useRef, useState } from 'react';

const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');

const EMOTION_COLORS = {
  angry: '#D85A30',
  disgust: '#EF9F27',
  fear: '#E24B4A',
  happy: '#639922',
  neutral: '#888780',
  sad: '#378ADD',
  surprise: '#7F77DD',
};

function newSessionId() {
  if (window.crypto && window.crypto.randomUUID) return window.crypto.randomUUID();
  return `sess-${Date.now()}-${Math.floor(Math.random() * 1_000_000)}`;
}

function formatMs(ms) {
  return `${Math.round(ms)} ms`;
}

function formatPct(v) {
  return `${Number(v || 0).toFixed(1)}%`;
}

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

function TrendLines({ history }) {
  const points = useMemo(() => history.slice(-45), [history]);
  if (!points.length) return <div className="empty-chart">No live data yet.</div>;

  const w = 680;
  const h = 220;
  const mapX = (i) => (i / Math.max(points.length - 1, 1)) * (w - 20) + 10;
  const mapY = (v) => h - ((clamp(v, 0, 100) / 100) * (h - 20) + 10);

  const poly = (key) => points.map((p, i) => `${mapX(i)},${mapY(p[key])}`).join(' ');

  return (
    <svg className="trend-svg" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
      <polyline points={poly('smoothed_truth')} stroke="#00C853" strokeWidth="3" fill="none" />
      <polyline points={poly('smoothed_lie')} stroke="#D50000" strokeWidth="3" fill="none" />
      <polyline points={poly('quality')} stroke="#00E5FF" strokeWidth="2" fill="none" strokeDasharray="6 5" />
    </svg>
  );
}

export default function App() {
  const [tab, setTab] = useState('live');
  const [sessionId, setSessionId] = useState(newSessionId());
  const [apiHealth, setApiHealth] = useState({ status: 'checking', device: '-' });

  const [confidenceThreshold, setConfidenceThreshold] = useState(0.45);
  const [emaAlpha, setEmaAlpha] = useState(0.3);
  const [frameIntervalMs, setFrameIntervalMs] = useState(160);

  const [isRunning, setIsRunning] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [latencyMs, setLatencyMs] = useState(0);

  const [quality, setQuality] = useState(0);
  const [topFace, setTopFace] = useState(null);
  const [mapping, setMapping] = useState(null);
  const [stats, setStats] = useState(null);
  const [history, setHistory] = useState([]);
  const [allFaces, setAllFaces] = useState([]);

  const [uploadResult, setUploadResult] = useState(null);
  const [uploadError, setUploadError] = useState('');

  const videoRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);
  const pendingRef = useRef(false);

  const api = (path) => (API_BASE ? `${API_BASE}${path}` : path);

  useEffect(() => {
    let mounted = true;
    fetch(api('/health'))
      .then((r) => r.json())
      .then((d) => {
        if (!mounted) return;
        setApiHealth({ status: d.status || 'ok', device: d.device || '-' });
      })
      .catch(() => {
        if (!mounted) return;
        setApiHealth({ status: 'down', device: '-' });
      });
    return () => {
      mounted = false;
    };
  }, []);

  const drawOverlay = (faces) => {
    const video = videoRef.current;
    const canvas = overlayCanvasRef.current;
    if (!video || !canvas) return;

    const w = video.videoWidth || 640;
    const h = video.videoHeight || 360;

    canvas.width = w;
    canvas.height = h;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, w, h);

    faces.forEach((f) => {
      const [x, y, bw, bh] = f.bbox;
      const color = EMOTION_COLORS[f.emotion] || '#00C853';
      ctx.lineWidth = 2;
      ctx.strokeStyle = color;
      ctx.strokeRect(x, y, bw, bh);

      const label = `${f.emotion.toUpperCase()} ${(f.confidence * 100).toFixed(0)}%`;
      ctx.font = '14px Segoe UI';
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = 'rgba(0,0,0,0.65)';
      ctx.fillRect(x, Math.max(0, y - 24), tw + 12, 20);
      ctx.fillStyle = '#FFFFFF';
      ctx.fillText(label, x + 6, Math.max(14, y - 9));
    });
  };

  const processFrame = async () => {
    if (pendingRef.current) return;

    const video = videoRef.current;
    const canvas = captureCanvasRef.current;
    if (!video || !canvas || video.readyState < 2) return;

    const w = video.videoWidth || 640;
    const h = video.videoHeight || 360;
    canvas.width = w;
    canvas.height = h;

    const ctx = canvas.getContext('2d', { willReadFrequently: false });
    ctx.drawImage(video, 0, 0, w, h);
    const imageBase64 = canvas.toDataURL('image/jpeg', 0.72);

    pendingRef.current = true;
    setIsSending(true);

    try {
      const res = await fetch(api('/api/v1/analyze/frame'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          image_base64: imageBase64,
          confidence_threshold: confidenceThreshold,
          ema_alpha: emaAlpha,
        }),
      });

      if (!res.ok) throw new Error('Inference request failed');
      const data = await res.json();

      setLatencyMs(data.latency_ms || 0);
      setQuality(data.quality || 0);
      setTopFace(data.top_face || null);
      setMapping(data.mapping || null);
      setStats(data.stats || null);
      setHistory(Array.isArray(data.history) ? data.history : []);
      setAllFaces(Array.isArray(data.faces) ? data.faces : []);

      drawOverlay(Array.isArray(data.faces) ? data.faces : []);
    } catch (_err) {
      setApiHealth((p) => ({ ...p, status: 'down' }));
    } finally {
      pendingRef.current = false;
      setIsSending(false);
    }
  };

  const stopCamera = () => {
    setIsRunning(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    const video = videoRef.current;
    if (video) video.srcObject = null;
  };

  const startCamera = async () => {
    try {
      if (streamRef.current) stopCamera();
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user',
        },
        audio: false,
      });

      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) return;
      video.srcObject = stream;
      await video.play();

      setIsRunning(true);
      intervalRef.current = setInterval(processFrame, frameIntervalMs);
    } catch (_err) {
      stopCamera();
      alert('Could not access camera. Please allow camera permissions and try again.');
    }
  };

  useEffect(() => () => stopCamera(), []);

  const resetSession = async () => {
    try {
      await fetch(api('/api/v1/session/reset'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId }),
      });
    } catch (_err) {
      // Ignore reset network errors; local state is reset regardless.
    }
    setSessionId(newSessionId());
    setTopFace(null);
    setMapping(null);
    setStats(null);
    setHistory([]);
    setAllFaces([]);
    setQuality(0);
    drawOverlay([]);
  };

  const uploadImage = async (file) => {
    setUploadResult(null);
    setUploadError('');
    if (!file) return;

    try {
      const form = new FormData();
      form.append('file', file);
      const url = `${api('/api/v1/analyze/image')}?confidence_threshold=${confidenceThreshold}&ema_alpha=${emaAlpha}`;
      const res = await fetch(url, { method: 'POST', body: form });
      if (!res.ok) throw new Error('Upload analysis failed');
      const data = await res.json();
      setUploadResult(data);
    } catch (_err) {
      setUploadError('Image analysis failed. Please try a clearer face photo.');
    }
  };

  const truthPct = mapping ? Math.round(mapping.smoothed_truth) : 50;
  const liePct = 100 - truthPct;

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <h1>EmotionLens</h1>
          <p>Fast real-time emotion and deception tendency analysis</p>
        </div>
        <div className="status-pill">
          API: <strong>{apiHealth.status}</strong> | Device: <strong>{apiHealth.device}</strong>
        </div>
      </header>

      <nav className="tab-row">
        <button className={tab === 'live' ? 'active' : ''} onClick={() => setTab('live')}>Live Detector</button>
        <button className={tab === 'image' ? 'active' : ''} onClick={() => setTab('image')}>Image Analyzer</button>
      </nav>

      <section className="controls-panel">
        <div>
          <label>Confidence Threshold</label>
          <input type="range" min="0.3" max="0.9" step="0.05" value={confidenceThreshold} onChange={(e) => setConfidenceThreshold(Number(e.target.value))} />
          <span>{confidenceThreshold.toFixed(2)}</span>
        </div>
        <div>
          <label>EMA Alpha</label>
          <input type="range" min="0.1" max="0.9" step="0.05" value={emaAlpha} onChange={(e) => setEmaAlpha(Number(e.target.value))} />
          <span>{emaAlpha.toFixed(2)}</span>
        </div>
        <div>
          <label>Frame Interval</label>
          <input type="range" min="80" max="450" step="10" value={frameIntervalMs} onChange={(e) => setFrameIntervalMs(Number(e.target.value))} />
          <span>{frameIntervalMs} ms</span>
        </div>
        <div className="btn-group">
          <button onClick={startCamera} disabled={tab !== 'live' || isRunning}>Start Camera</button>
          <button onClick={stopCamera} disabled={!isRunning}>Stop Camera</button>
          <button onClick={resetSession}>Reset Session</button>
        </div>
      </section>

      {tab === 'live' && (
        <main className="live-grid">
          <div className="video-card">
            <div className="video-wrap">
              <video ref={videoRef} muted playsInline className="video-feed" />
              <canvas ref={overlayCanvasRef} className="overlay" />
              <canvas ref={captureCanvasRef} className="hidden-canvas" />
            </div>
            <div className="video-meta">
              <span>Session: {sessionId.slice(0, 8)}...</span>
              <span>Latency: {formatMs(latencyMs)}</span>
              <span>{isSending ? 'Analyzing...' : isRunning ? 'Live' : 'Stopped'}</span>
            </div>
          </div>

          <div className="metrics-col">
            <div className="metric-card large">
              <h3>Truth vs Lie (EMA)</h3>
              <div className="truthbar">
                <div className="truth" style={{ width: `${truthPct}%` }} />
                <div className="lie" style={{ width: `${liePct}%` }} />
              </div>
              <div className="split-row">
                <strong>Truth {truthPct}%</strong>
                <strong>Lie {liePct}%</strong>
              </div>
            </div>

            <div className="metric-card">
              <h3>Current Detection</h3>
              <p>Emotion: <strong>{topFace?.emotion || '-'}</strong></p>
              <p>Confidence: <strong>{topFace ? formatPct(topFace.confidence * 100) : '-'}</strong></p>
              <p>Quality: <strong>{quality}%</strong></p>
            </div>

            <div className="metric-card">
              <h3>Session Stats</h3>
              <p>Frames: <strong>{stats?.total_frames || 0}</strong></p>
              <p>Dominant: <strong>{stats?.dominant_emotion || '-'}</strong></p>
              <p>Avg Truth: <strong>{stats ? formatPct(stats.avg_truth) : '-'}</strong></p>
            </div>
          </div>

          <div className="wide-card">
            <h3>Timeline (Truth, Lie, Quality)</h3>
            <TrendLines history={history} />
          </div>

          <div className="wide-card">
            <h3>Emotion Probabilities</h3>
            {topFace ? (
              <div className="prob-grid">
                {Object.entries(topFace.all_probs || {}).map(([emotion, prob]) => (
                  <div key={emotion} className="prob-row">
                    <span>{emotion}</span>
                    <div className="prob-track">
                      <div className="prob-fill" style={{ width: `${Math.round(prob * 100)}%`, background: EMOTION_COLORS[emotion] || '#00C853' }} />
                    </div>
                    <span>{(prob * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="muted">No face detected yet.</p>
            )}
          </div>
        </main>
      )}

      {tab === 'image' && (
        <main className="image-pane">
          <div className="metric-card">
            <h3>Upload Image</h3>
            <input
              type="file"
              accept="image/png,image/jpeg,image/jpg"
              onChange={(e) => uploadImage(e.target.files?.[0])}
            />
            {uploadError && <p className="error">{uploadError}</p>}
          </div>

          {uploadResult && (
            <div className="metric-card">
              <h3>Image Analysis Result</h3>
              <p>Quality: <strong>{uploadResult.quality}%</strong></p>
              <p>Faces: <strong>{uploadResult.faces?.length || 0}</strong></p>
              <div className="prob-grid">
                {(uploadResult.faces || []).map((f, idx) => (
                  <div key={`${f.emotion}-${idx}`} className="face-item">
                    <h4>Face {idx + 1}</h4>
                    <p>Emotion: <strong>{f.emotion}</strong></p>
                    <p>Confidence: <strong>{(f.confidence * 100).toFixed(1)}%</strong></p>
                    <p>Box: [{f.bbox.join(', ')}]</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </main>
      )}
    </div>
  );
}
