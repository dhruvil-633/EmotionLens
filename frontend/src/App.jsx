import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');

const EMOTION_COLORS = {
  angry:    '#D85A30',
  disgust:  '#EF9F27',
  fear:     '#E24B4A',
  happy:    '#4CAF50',
  neutral:  '#7B8FA1',
  sad:      '#378ADD',
  surprise: '#9C6BDD',
};

const EMOTION_EMOJI = {
  angry: '😠', disgust: '🤢', fear: '😨', happy: '😊',
  neutral: '😐', sad: '😢', surprise: '😲',
};

function newSessionId() {
  return window.crypto?.randomUUID?.() ?? `sess-${Date.now()}-${Math.floor(Math.random() * 1_000_000)}`;
}

function qualityClass(q) {
  if (q >= 65) return 'quality-good';
  if (q >= 35) return 'quality-mid';
  return 'quality-bad';
}

function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }

// ---- Trend Chart ----
function TrendChart({ history }) {
  const pts = useMemo(() => history.slice(-60), [history]);
  if (!pts.length) return <div className="empty-chart">No data yet — start camera to see live trend.</div>;

  const W = 700, H = 160;
  const X = (i) => (i / Math.max(pts.length - 1, 1)) * (W - 24) + 12;
  const Y = (v) => H - 8 - ((clamp(v, 0, 100) / 100) * (H - 16));
  const poly = (key) => pts.map((p, i) => `${X(i)},${Y(p[key])}`).join(' ');

  return (
    <div className="chart-wrap">
      <svg className="trend-svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
        <defs>
          <linearGradient id="gTruth" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#00c853" stopOpacity="0.18" />
            <stop offset="100%" stopColor="#00c853" stopOpacity="0" />
          </linearGradient>
          <linearGradient id="gLie" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#ff1744" stopOpacity="0.15" />
            <stop offset="100%" stopColor="#ff1744" stopOpacity="0" />
          </linearGradient>
        </defs>
        {/* Grid lines */}
        {[25, 50, 75].map(pct => (
          <line key={pct} x1={12} x2={W - 12} y1={Y(pct)} y2={Y(pct)}
            stroke="rgba(255,255,255,0.05)" strokeWidth="1" strokeDasharray="4 4" />
        ))}
        {/* Truth fill area */}
        <polygon
          points={`${X(0)},${H - 8} ${poly('smoothed_truth')} ${X(pts.length - 1)},${H - 8}`}
          fill="url(#gTruth)"
        />
        {/* Lie fill area */}
        <polygon
          points={`${X(0)},${H - 8} ${poly('smoothed_lie')} ${X(pts.length - 1)},${H - 8}`}
          fill="url(#gLie)"
        />
        <polyline points={poly('smoothed_truth')} stroke="#00c853" strokeWidth="2.5" fill="none" strokeLinejoin="round" strokeLinecap="round" />
        <polyline points={poly('smoothed_lie')} stroke="#ff1744" strokeWidth="2.5" fill="none" strokeLinejoin="round" strokeLinecap="round" />
        <polyline points={poly('quality')} stroke="#00e5ff" strokeWidth="1.5" fill="none" strokeDasharray="6 4" strokeLinecap="round" />
      </svg>
      <div className="chart-legend">
        <div className="legend-item"><div className="legend-dot" style={{ background: '#00c853' }} /> Truth (EMA)</div>
        <div className="legend-item"><div className="legend-dot" style={{ background: '#ff1744' }} /> Lie (EMA)</div>
        <div className="legend-item"><div className="legend-dot" style={{ background: '#00e5ff', opacity: 0.7 }} /> Quality</div>
      </div>
    </div>
  );
}

// ---- Upload Drop Zone ----
function UploadZone({ onFile }) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  const handle = (file) => { if (file) onFile(file); };

  return (
    <div
      className={`upload-zone ${dragging ? 'dragging' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => { e.preventDefault(); setDragging(false); handle(e.dataTransfer.files[0]); }}
      onClick={() => inputRef.current?.click()}
      role="button"
      tabIndex={0}
      aria-label="Upload face image"
      onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
    >
      <input ref={inputRef} type="file" accept="image/png,image/jpeg,image/jpg" onChange={(e) => handle(e.target.files?.[0])} />
      <div className="upload-icon">📸</div>
      <p>Drag & drop or <span>browse</span> an image</p>
      <p style={{ fontSize: '0.78rem', marginTop: 6, opacity: 0.6 }}>PNG / JPG · Max 10 MB</p>
    </div>
  );
}

// ---- Main App ----
export default function App() {
  const [tab, setTab] = useState('live');
  const [sessionId, setSessionId] = useState(newSessionId);
  const [apiHealth, setApiHealth] = useState({ status: 'checking', device: '-' });

  const [confThresh, setConfThresh] = useState(0.45);
  const [emaAlpha, setEmaAlpha] = useState(0.3);
  const [frameMs, setFrameMs] = useState(160);

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
  const [uploadLoading, setUploadLoading] = useState(false);

  const videoRef = useRef(null);
  const captureRef = useRef(null);
  const overlayRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);
  const pendingRef = useRef(false);

  const api = useCallback((path) => API_BASE ? `${API_BASE}${path}` : path, []);

  // Health check on mount
  useEffect(() => {
    let live = true;
    fetch(api('/health'))
      .then(r => r.json())
      .then(d => { if (live) setApiHealth({ status: d.status || 'ok', device: d.device || '-' }); })
      .catch(() => { if (live) setApiHealth({ status: 'down', device: '-' }); });
    return () => { live = false; };
  }, [api]);

  // Draw bounding boxes on overlay canvas
  const drawOverlay = useCallback((faces) => {
    const video = videoRef.current;
    const canvas = overlayRef.current;
    if (!video || !canvas) return;
    const w = video.videoWidth || 640;
    const h = video.videoHeight || 360;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, w, h);
    faces.forEach(f => {
      const [x, y, bw, bh] = f.bbox;
      const color = EMOTION_COLORS[f.emotion] || '#00c853';
      ctx.lineWidth = 2;
      ctx.strokeStyle = color;
      ctx.shadowColor = color;
      ctx.shadowBlur = 10;
      ctx.strokeRect(x, y, bw, bh);
      ctx.shadowBlur = 0;
      const label = `${EMOTION_EMOJI[f.emotion] || ''} ${f.emotion.toUpperCase()} ${(f.confidence * 100).toFixed(0)}%`;
      ctx.font = '600 13px Inter, system-ui';
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = 'rgba(6, 10, 20, 0.82)';
      ctx.fillRect(x, Math.max(0, y - 26), tw + 14, 22);
      ctx.fillStyle = color;
      ctx.fillText(label, x + 7, Math.max(16, y - 9));
    });
  }, []);

  // Frame capture & inference
  const processFrame = useCallback(async () => {
    if (pendingRef.current) return;
    const video = videoRef.current;
    const canvas = captureRef.current;
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
        body: JSON.stringify({ session_id: sessionId, image_base64: imageBase64, confidence_threshold: confThresh, ema_alpha: emaAlpha }),
      });
      if (!res.ok) throw new Error('inference failed');
      const data = await res.json();
      setLatencyMs(data.latency_ms || 0);
      setQuality(data.quality || 0);
      setTopFace(data.top_face || null);
      setMapping(data.mapping || null);
      setStats(data.stats || null);
      setHistory(Array.isArray(data.history) ? data.history : []);
      setAllFaces(Array.isArray(data.faces) ? data.faces : []);
      drawOverlay(Array.isArray(data.faces) ? data.faces : []);
    } catch {
      setApiHealth(p => ({ ...p, status: 'down' }));
    } finally {
      pendingRef.current = false;
      setIsSending(false);
    }
  }, [api, sessionId, confThresh, emaAlpha, drawOverlay]);

  const stopCamera = useCallback(() => {
    setIsRunning(false);
    clearInterval(intervalRef.current);
    intervalRef.current = null;
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  const startCamera = useCallback(async () => {
    try {
      if (streamRef.current) stopCamera();
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
        audio: false,
      });
      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) return;
      video.srcObject = stream;
      await video.play();
      setIsRunning(true);
      intervalRef.current = setInterval(processFrame, frameMs);
    } catch {
      stopCamera();
      alert('Could not access camera. Please allow camera permissions and try again.');
    }
  }, [stopCamera, processFrame, frameMs]);

  // Cleanup on unmount
  useEffect(() => () => stopCamera(), [stopCamera]);

  const resetSession = useCallback(async () => {
    try {
      await fetch(api('/api/v1/session/reset'), {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId }),
      });
    } catch { /* ignore */ }
    setSessionId(newSessionId());
    setTopFace(null); setMapping(null); setStats(null); setHistory([]);
    setAllFaces([]); setQuality(0); setLatencyMs(0);
    drawOverlay([]);
  }, [api, sessionId, drawOverlay]);

  const uploadImage = useCallback(async (file) => {
    setUploadResult(null); setUploadError(''); setUploadLoading(true);
    try {
      const form = new FormData();
      form.append('file', file);
      const url = `${api('/api/v1/analyze/image')}?confidence_threshold=${confThresh}&ema_alpha=${emaAlpha}`;
      const res = await fetch(url, { method: 'POST', body: form });
      if (!res.ok) throw new Error('failed');
      setUploadResult(await res.json());
    } catch {
      setUploadError('Analysis failed. Please try a clearer face photo with good lighting.');
    } finally {
      setUploadLoading(false);
    }
  }, [api, confThresh, emaAlpha]);

  const truthPct = mapping ? Math.round(mapping.smoothed_truth) : 50;
  const liePct = 100 - truthPct;
  const statusDotClass = apiHealth.status === 'ok' ? 'ok' : apiHealth.status === 'checking' ? 'checking' : 'down';

  return (
    <div className="app-shell">

      {/* ── Header ── */}
      <header className="app-header">
        <div className="header-brand">
          <h1>🎭 EmotionLens</h1>
          <p>Real-time emotion recognition &amp; deception tendency analysis</p>
        </div>
        <div className="header-right">
          <div className="status-pill">
            <span className={`status-dot ${statusDotClass}`} />
            API: <strong>{apiHealth.status}</strong>
            &nbsp;·&nbsp;
            Device: <strong>{apiHealth.device}</strong>
          </div>
        </div>
      </header>

      {/* ── Tabs ── */}
      <nav className="tab-row" role="tablist">
        <button id="tab-live" className={`tab-btn ${tab === 'live' ? 'active' : ''}`} onClick={() => setTab('live')} role="tab" aria-selected={tab === 'live'}>
          📹 Live Detector
        </button>
        <button id="tab-image" className={`tab-btn ${tab === 'image' ? 'active' : ''}`} onClick={() => setTab('image')} role="tab" aria-selected={tab === 'image'}>
          🖼️ Image Analyzer
        </button>
      </nav>

      {/* ── Controls Panel ── */}
      <section className="controls-panel" aria-label="Analysis settings">
        <div className="ctrl-group">
          <label htmlFor="ctrl-conf">Confidence Threshold</label>
          <input id="ctrl-conf" type="range" min="0.3" max="0.9" step="0.05" value={confThresh}
            onChange={e => setConfThresh(Number(e.target.value))} />
          <span className="ctrl-value">{confThresh.toFixed(2)}</span>
        </div>
        <div className="ctrl-group">
          <label htmlFor="ctrl-ema">EMA Smoothing (α)</label>
          <input id="ctrl-ema" type="range" min="0.1" max="0.9" step="0.05" value={emaAlpha}
            onChange={e => setEmaAlpha(Number(e.target.value))} />
          <span className="ctrl-value">{emaAlpha.toFixed(2)}</span>
        </div>
        <div className="ctrl-group">
          <label htmlFor="ctrl-fps">Frame Interval</label>
          <input id="ctrl-fps" type="range" min="80" max="450" step="10" value={frameMs}
            onChange={e => setFrameMs(Number(e.target.value))} />
          <span className="ctrl-value">{frameMs} ms (~{Math.round(1000 / frameMs)} fps)</span>
        </div>
        <div className="btn-group">
          <button id="btn-start" className="btn primary" onClick={startCamera} disabled={tab !== 'live' || isRunning}>
            ▶ Start
          </button>
          <button id="btn-stop" className="btn danger" onClick={stopCamera} disabled={!isRunning}>
            ■ Stop
          </button>
          <button id="btn-reset" className="btn" onClick={resetSession}>
            ↺ Reset
          </button>
        </div>
      </section>

      {/* ════════════════ LIVE DETECTOR ════════════════ */}
      {tab === 'live' && (
        <main id="live-panel" className="live-grid" role="tabpanel">

          {/* Video Card */}
          <div className="card video-card">
            <h3>📹 Camera Feed</h3>
            <div className="video-wrap">
              {!isRunning && (
                <div className="video-idle-state">
                  <svg width="52" height="52" viewBox="0 0 24 24" fill="white"><path d="M17 10.5V7c0-.55-.45-1-1-1H4c-.55 0-1 .45-1 1v10c0 .55.45 1 1 1h12c.55 0 1-.45 1-1v-3.5l4 4v-11l-4 4z"/></svg>
                  <p>Press <strong style={{ color: 'var(--accent-green)' }}>▶ Start</strong> to activate camera</p>
                </div>
              )}
              <video ref={videoRef} id="video-feed" muted playsInline className="video-feed" />
              <canvas ref={overlayRef} className="overlay-canvas" />
              <canvas ref={captureRef} className="hidden-canvas" />
            </div>
            <div className="video-meta">
              <span className="meta-tag">Session {sessionId.slice(0, 8)}…</span>
              <span className={`meta-tag ${isSending ? 'analyzing' : isRunning ? 'live' : ''}`}>
                {isSending ? '⚡ Analyzing' : isRunning ? '● Live' : '■ Stopped'}
              </span>
              <span className="meta-tag">⏱ {Math.round(latencyMs)} ms</span>
              <span className={`quality-badge ${qualityClass(quality)}`}>Q: {quality}%</span>
            </div>
          </div>

          {/* Right Column */}
          <div className="metrics-col">
            {/* Truth / Lie */}
            <div className="card truthlie-section">
              <h3>🧠 Deception Tendency</h3>
              <div className="tl-bar-wrap">
                <div className="tl-truth" style={{ width: `${truthPct}%` }} />
                <div className="tl-lie" style={{ width: `${liePct}%` }} />
              </div>
              <div className="tl-labels">
                <span className="truth-label">✓ Truth {truthPct}%</span>
                <span className="lie-label">✗ Lie {liePct}%</span>
              </div>
              {mapping && (
                <div style={{ marginTop: 12 }}>
                  <div className="metric-row">
                    <span className="label">Emotion</span>
                    <span className="value">
                      <span className="emotion-badge" style={{
                        background: `${EMOTION_COLORS[mapping.emotion] || '#888'}22`,
                        border: `1px solid ${EMOTION_COLORS[mapping.emotion] || '#888'}55`,
                        color: EMOTION_COLORS[mapping.emotion] || '#ccc',
                      }}>
                        {EMOTION_EMOJI[mapping.emotion] || ''} {mapping.emotion}
                      </span>
                    </span>
                  </div>
                  <div className="metric-row">
                    <span className="label">Reason</span>
                    <span className="value" style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', textAlign: 'right', maxWidth: '60%', lineHeight: 1.3 }}>
                      {mapping.reason}
                    </span>
                  </div>
                </div>
              )}
            </div>

            {/* Detection */}
            <div className="card">
              <h3>🎯 Current Detection</h3>
              {topFace ? (
                <>
                  <div className="metric-row">
                    <span className="label">Emotion</span>
                    <span className="value accent">{EMOTION_EMOJI[topFace.emotion]} {topFace.emotion}</span>
                  </div>
                  <div className="metric-row">
                    <span className="label">Confidence</span>
                    <span className="value">{(topFace.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="metric-row">
                    <span className="label">Faces</span>
                    <span className="value">{allFaces.length}</span>
                  </div>
                </>
              ) : (
                <p className="muted">No face detected yet.</p>
              )}
            </div>

            {/* Session Stats */}
            <div className="card">
              <h3>📊 Session Stats</h3>
              {stats ? (
                <>
                  <div className="metric-row"><span className="label">Frames</span><span className="value accent">{stats.total_frames}</span></div>
                  <div className="metric-row"><span className="label">Duration</span><span className="value">{stats.session_duration}s</span></div>
                  <div className="metric-row"><span className="label">Dominant</span>
                    <span className="value">{EMOTION_EMOJI[stats.dominant_emotion] || ''} {stats.dominant_emotion}</span></div>
                  <div className="metric-row"><span className="label">Avg Truth</span>
                    <span className="value" style={{ color: 'var(--accent-green)' }}>{stats.avg_truth.toFixed(1)}%</span></div>
                  <div className="metric-row"><span className="label">Avg Quality</span>
                    <span className={`quality-badge ${qualityClass(stats.avg_quality)}`}>{stats.avg_quality.toFixed(0)}%</span></div>
                </>
              ) : (
                <p className="muted">Start camera to collect stats.</p>
              )}
            </div>
          </div>

          {/* Trend Chart — full width */}
          <div className="card wide-row">
            <h3>📈 Live Timeline</h3>
            <TrendChart history={history} />
          </div>

          {/* Emotion Probabilities — full width */}
          <div className="card wide-row">
            <h3>🎨 Emotion Probabilities</h3>
            {topFace ? (
              <div className="prob-grid">
                {Object.entries(topFace.all_probs || {})
                  .sort(([, a], [, b]) => b - a)
                  .map(([emo, prob]) => (
                    <div key={emo} className="prob-row">
                      <span className="prob-label">{EMOTION_EMOJI[emo]} {emo}</span>
                      <div className="prob-track">
                        <div className="prob-fill" style={{
                          width: `${Math.round(prob * 100)}%`,
                          background: EMOTION_COLORS[emo] || '#888',
                        }} />
                      </div>
                      <span className="prob-pct">{(prob * 100).toFixed(1)}%</span>
                    </div>
                  ))}
              </div>
            ) : (
              <p className="muted">Start camera and point it at your face.</p>
            )}
          </div>
        </main>
      )}

      {/* ════════════════ IMAGE ANALYZER ════════════════ */}
      {tab === 'image' && (
        <main id="image-panel" className="image-pane" role="tabpanel">
          <div className="card">
            <h3>🖼️ Upload Image</h3>
            <UploadZone onFile={uploadImage} />
            {uploadLoading && <p style={{ marginTop: 12, color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)', fontSize: '0.85rem' }}>⚡ Analyzing image…</p>}
            {uploadError && <p className="error-msg" style={{ marginTop: 12 }}>{uploadError}</p>}
          </div>

          {uploadResult && (
            <div className="card">
              <h3>📋 Analysis Results</h3>
              <div className="metric-row">
                <span className="label">Image Quality</span>
                <span className={`quality-badge ${qualityClass(uploadResult.quality)}`}>{uploadResult.quality}%</span>
              </div>
              <div className="metric-row">
                <span className="label">Faces Detected</span>
                <span className="value accent">{uploadResult.faces?.length || 0}</span>
              </div>

              {uploadResult.faces?.length === 0 && (
                <p className="muted" style={{ marginTop: 12 }}>No faces detected. Try a photo with a clear, front-facing face.</p>
              )}

              <div className="face-results-grid" style={{ marginTop: 16 }}>
                {(uploadResult.faces || []).map((f, idx) => {
                  const fm = uploadResult.face_mappings?.[idx];
                  return (
                    <div key={`${f.emotion}-${idx}`} className="face-card">
                      <h4>Face {idx + 1}</h4>
                      <div className="metric-row">
                        <span className="label">Emotion</span>
                        <span className="emotion-badge" style={{
                          background: `${EMOTION_COLORS[f.emotion] || '#888'}22`,
                          border: `1px solid ${EMOTION_COLORS[f.emotion] || '#888'}55`,
                          color: EMOTION_COLORS[f.emotion] || '#ccc',
                        }}>
                          {EMOTION_EMOJI[f.emotion] || ''} {f.emotion}
                        </span>
                      </div>
                      <div className="metric-row">
                        <span className="label">Confidence</span>
                        <span className="value">{(f.confidence * 100).toFixed(1)}%</span>
                      </div>
                      {fm && (
                        <>
                          <div className="metric-row">
                            <span className="label">Truth Score</span>
                            <span className="value" style={{ color: 'var(--accent-green)' }}>{fm.raw_truth}%</span>
                          </div>
                          <div className="metric-row">
                            <span className="label">Reason</span>
                            <span className="value" style={{ fontSize: '0.76rem', color: 'var(--text-secondary)', textAlign: 'right', maxWidth: '55%', lineHeight: 1.3 }}>{fm.reason}</span>
                          </div>
                        </>
                      )}
                      <div className="metric-row">
                        <span className="label">Bbox</span>
                        <span className="value" style={{ fontFamily: 'var(--font-mono)', fontSize: '0.76rem' }}>[{f.bbox.join(', ')}]</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </main>
      )}
    </div>
  );
}
