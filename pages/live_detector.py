import threading
import time
from typing import Any, Dict, Optional

import av
import cv2
import plotly.graph_objects as go
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, WebRtcMode, webrtc_streamer


def _ensure_live_state() -> None:
    if 'latest_result' not in st.session_state:
        st.session_state.latest_result = None
    if 'latest_quality' not in st.session_state:
        st.session_state.latest_quality = 0
    if 'local_cap' not in st.session_state:
        st.session_state.local_cap = None


def _release_local_camera() -> None:
    cap = st.session_state.get('local_cap')
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    st.session_state.local_cap = None


def _render_truth_lie_bar(truth_pct: int, lie_pct: int, raw_truth: int, raw_lie: int) -> None:
    st.markdown(
        f"""
        <div class='truthlie-wrap'>
          <div style='display:flex; justify-content:space-between; margin-bottom:8px;'>
            <span style='font-size:1.1rem;'><strong>TRUTH {truth_pct}%</strong></span>
            <span style='font-size:1.1rem;'><strong>LIE {lie_pct}%</strong></span>
          </div>
          <div class='truthlie-bar'>
            <div class='truth-segment' style='width:{truth_pct}%;'></div>
            <div class='lie-segment' style='width:{lie_pct}%;'></div>
          </div>
          <div style='margin-top:8px; color:#9aa0a6; font-size:0.92rem;'>
            EMA Smoothed | Raw: T{raw_truth}% / L{raw_lie}%
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _sync_result_state(result: Optional[Dict[str, Any]], quality: int) -> None:
    st.session_state.latest_result = result
    st.session_state.latest_quality = quality
    if quality < 45:
        st.session_state.low_quality_streak += 1
    else:
        st.session_state.low_quality_streak = 0


def _render_realtime_panels(mapper, latest_result, quality: int) -> None:
    st.markdown("### Live Insights")

    row_top_left, row_top_mid, row_top_right = st.columns([1.2, 1, 1])

    with row_top_left:
        if latest_result:
            entry = mapper.get_current()
            truth_pct = int(entry['smoothed_truth'])
            lie_pct = 100 - truth_pct
            raw_truth = int(entry['raw_truth'])
            raw_lie = int(entry['raw_lie'])
        else:
            truth_pct = int(mapper.smoothed_truth)
            lie_pct = 100 - truth_pct
            raw_truth = 50
            raw_lie = 50
        _render_truth_lie_bar(truth_pct, lie_pct, raw_truth, raw_lie)

    with row_top_mid:
        if latest_result:
            entry = mapper.get_current()
            st.markdown(
                f"""
                <div class='metric-card'>
                  <div style='font-size:2rem'>{entry['emoji']}</div>
                  <div class='emotion-badge' style='background:{entry['hex']};'>{entry['emotion'].upper()}</div>
                  <div style='margin-top:8px; color:#B9C0CC;'>Confidence: {latest_result['confidence']*100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("Why this score?"):
                st.write(entry['reason'])
                st.write(f"Raw mapping: Truth {entry['raw_truth']}% / Lie {entry['raw_lie']}%")
        else:
            st.info("No face currently detected.")

    with row_top_right:
        df = mapper.get_history_df()
        delta_q = int(df['quality'].iloc[-1] - df['quality'].iloc[-2]) if len(df) > 1 else 0
        st.metric("Image Quality", f"{quality}%", delta=f"{delta_q:+d}", delta_color="normal")
        st.metric("Frames Analyzed", mapper.frame_count)

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=quality,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': 'Image Quality Meter', 'font': {'size': 13}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': '#00C853' if quality >= 75 else '#FF6D00' if quality >= 45 else '#D50000'},
                'steps': [
                    {'range': [0, 44], 'color': '#2a0a0a'},
                    {'range': [44, 74], 'color': '#2a1800'},
                    {'range': [74, 100], 'color': '#0a2a0a'},
                ],
                'threshold': {'line': {'color': 'white', 'width': 2}, 'thickness': 0.75, 'value': quality},
            },
        )
    )
    gauge.update_layout(
        height=180,
        margin=dict(t=34, b=0, l=16, r=16),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        template='plotly_dark',
    )
    st.plotly_chart(gauge, use_container_width=True, key='quality_gauge_live')

    if st.session_state.low_quality_streak >= 5:
        st.warning("Low image quality for 5+ consecutive frames. Improve lighting or move closer to camera.")

    row_bottom_left, row_bottom_right = st.columns(2)

    with row_bottom_left:
        if latest_result:
            probs_dict = latest_result['all_probs']
            detected = latest_result['emotion']
            colors = ['#00C853' if e == detected else '#444' for e in probs_dict.keys()]
            fig_probs = go.Figure(
                go.Bar(
                    x=list(probs_dict.keys()),
                    y=[v * 100 for v in probs_dict.values()],
                    marker_color=colors,
                    text=[f"{v*100:.1f}%" for v in probs_dict.values()],
                    textposition='outside',
                )
            )
            fig_probs.update_layout(
                title="All Emotion Probabilities (%)",
                yaxis_range=[0, 110],
                height=280,
                margin=dict(t=40, b=20, l=20, r=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                template='plotly_dark',
            )
            st.plotly_chart(fig_probs, use_container_width=True, key='probs_live')

    with row_bottom_right:
        df = mapper.get_history_df()
        if len(df) > 2:
            fig_tl = go.Figure()
            fig_tl.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['smoothed_truth'],
                    name='Truth %',
                    line=dict(color='#00C853', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(0,200,83,0.1)',
                )
            )
            fig_tl.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['smoothed_lie'],
                    name='Lie %',
                    line=dict(color='#D50000', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(213,0,0,0.1)',
                )
            )
            fig_tl.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['quality'],
                    name='Quality',
                    line=dict(color='#00E5FF', width=1, dash='dot'),
                )
            )
            fig_tl.update_layout(
                title="Rolling Session Timeline",
                yaxis_range=[0, 100],
                height=280,
                margin=dict(t=40, b=20, l=20, r=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                template='plotly_dark',
            )
            st.plotly_chart(fig_tl, use_container_width=True, key='timeline_live')


class EmotionVideoTransformer(VideoTransformerBase):
    def __init__(self, inference, mapper):
        self.inference = inference
        self.mapper = mapper
        self.lock = threading.Lock()
        self.latest_result = None
        self.latest_quality = 0

    def recv(self, frame):
        img = frame.to_ndarray(format='bgr24')
        t0 = time.time()
        results, quality = self.inference.detect_and_predict(img)

        with self.lock:
            self.latest_result = results[0] if results else None
            self.latest_quality = quality
            if self.latest_result:
                entry = self.mapper.update(self.latest_result['emotion'], self.latest_result['confidence'], quality)
                smoothed_truth = entry['smoothed_truth']
                smoothed_lie = entry['smoothed_lie']
            else:
                smoothed_truth = self.mapper.smoothed_truth
                smoothed_lie = self.mapper.smoothed_lie

        fps = 1.0 / max(time.time() - t0, 1e-6)
        annotated = self.inference.annotate_frame(img, results, smoothed_truth, smoothed_lie, quality, fps=fps)
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        return av.VideoFrame.from_ndarray(rgb, format='rgb24')

    def snapshot(self):
        with self.lock:
            return {'result': self.latest_result, 'quality': self.latest_quality}


def render_live_detector_page(inference, mapper) -> None:
    st.header("🎥 Live Detector")

    if inference is None:
        st.error("Model not found. Train a model first from the '🧠 Train Model' page.")
        return

    _ensure_live_state()

    st.markdown(
        """
        <div class='metric-card' style='margin-bottom:12px;'>
          <div style='font-size:1.05rem; font-weight:700;'>Real-Time Camera</div>
          <div style='color:#B9C0CC; margin-top:4px;'>Continuous video mode with live emotion, truth/lie tendency, and quality tracking.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mode = st.radio("Camera mode", ["Local OpenCV (Continuous)", "Browser WebRTC (Continuous)"], horizontal=True)

    if mode == "Local OpenCV (Continuous)":
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Start Camera", use_container_width=True):
                st.session_state.running = True
        with c2:
            if st.button("Stop Camera", use_container_width=True):
                st.session_state.running = False

        frame_placeholder = st.empty()

        if st.session_state.running:
            if st.session_state.local_cap is None:
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                st.session_state.local_cap = cap

            cap = st.session_state.local_cap
            if cap is None or not cap.isOpened():
                st.session_state.running = False
                _release_local_camera()
                st.error("Unable to open webcam 0.")
                return

            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                st.session_state.running = False
                _release_local_camera()
                st.error("Camera read failed.")
                return

            results, quality = inference.detect_and_predict(frame)
            top = results[0] if results else None
            if top:
                mapper.update(top['emotion'], top['confidence'], quality)
            _sync_result_state(top, quality)

            if top:
                entry = mapper.get_current()
                smoothed_truth = entry['smoothed_truth']
                smoothed_lie = entry['smoothed_lie']
            else:
                smoothed_truth = mapper.smoothed_truth
                smoothed_lie = mapper.smoothed_lie

            fps = 1.0 / max(time.time() - t0, 1e-6)
            annotated = inference.annotate_frame(frame, results, smoothed_truth, smoothed_lie, quality, fps=fps)
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb, channels='RGB', use_column_width=True)

            _render_realtime_panels(mapper, st.session_state.latest_result, quality)
            time.sleep(0.03)
            st.rerun()
        else:
            _release_local_camera()
            _render_realtime_panels(mapper, st.session_state.latest_result, st.session_state.latest_quality)

    else:
        st.caption("Use browser permission prompt to allow camera access.")

        ctx = webrtc_streamer(
            key="emotionlens-webrtc",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            video_transformer_factory=lambda: EmotionVideoTransformer(inference, mapper),
        )

        transformer = getattr(ctx, 'video_transformer', None) if ctx else None
        if transformer is not None:
            snap = transformer.snapshot()
            _sync_result_state(snap.get('result'), int(snap.get('quality', 0)))

        _render_realtime_panels(mapper, st.session_state.latest_result, st.session_state.latest_quality)
