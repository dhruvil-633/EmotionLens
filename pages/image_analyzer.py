import io

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from deception_mapper import EMOTION_EMOJI, EMOTION_MAP


def _render_truth_lie(truth_pct: int, lie_pct: int) -> None:
    st.markdown(
        f"""
        <div class='truthlie-wrap'>
          <div style='display:flex; justify-content:space-between; margin-bottom:6px;'>
            <span><strong>TRUTH {truth_pct}%</strong></span>
            <span><strong>LIE {lie_pct}%</strong></span>
          </div>
          <div class='truthlie-bar'>
            <div class='truth-segment' style='width:{truth_pct}%;'></div>
            <div class='lie-segment' style='width:{lie_pct}%;'></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_quality_gauge(quality: int, key: str) -> None:
    fig = go.Figure(
        go.Indicator(
            mode='gauge+number',
            value=quality,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': 'Image Quality', 'font': {'size': 13}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': '#00C853' if quality >= 75 else '#FF6D00' if quality >= 45 else '#D50000'},
                'steps': [
                    {'range': [0, 44], 'color': '#2a0a0a'},
                    {'range': [44, 74], 'color': '#2a1800'},
                    {'range': [74, 100], 'color': '#0a2a0a'},
                ],
                'threshold': {
                    'line': {'color': 'white', 'width': 2},
                    'thickness': 0.75,
                    'value': quality,
                },
            },
        )
    )
    fig.update_layout(
        height=200,
        margin=dict(t=30, b=10, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        template='plotly_dark',
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


def _face_entry(face_result, conf_threshold: float):
    emotion = face_result['emotion'] if face_result['confidence'] >= conf_threshold else 'neutral'
    mapping = EMOTION_MAP[emotion]
    return {
        'emotion': emotion,
        'emoji': EMOTION_EMOJI.get(emotion, ''),
        'confidence': face_result['confidence'],
        'raw_truth': mapping['truth'],
        'raw_lie': mapping['lie'],
        'reason': mapping['reason'],
        'hex': mapping['hex'],
    }


def render_image_analyzer_page(inference, mapper) -> None:
    st.header("🖼️ Image Analyzer")

    if inference is None:
        st.error("Model not found. Train a model first from the '🧠 Train Model' page.")
        return

    uploaded = st.file_uploader("Upload a face image", ["jpg", "jpeg", "png"])
    if not uploaded:
        st.info("Upload an image to run emotion and deception tendency analysis.")
        return

    pil_image = Image.open(uploaded).convert('RGB')
    results, quality = inference.predict_from_pil(pil_image)

    frame_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    if results:
        first = _face_entry(results[0], mapper.conf_threshold)
        annotated = inference.annotate_frame(
            frame_bgr,
            results,
            first['raw_truth'],
            first['raw_lie'],
            quality,
            fps=0,
        )
    else:
        annotated = frame_bgr.copy()

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    st.image(annotated_rgb, channels='RGB', use_column_width=True)

    png_buffer = io.BytesIO()
    Image.fromarray(annotated_rgb).save(png_buffer, format='PNG')
    st.download_button(
        "Download Annotated PNG",
        data=png_buffer.getvalue(),
        file_name='emotionlens_annotated.png',
        mime='image/png',
    )

    if not results:
        st.warning("No face detected. Use a front-facing image with a clear, well-lit face.")
        return

    st.subheader("Face Analysis")
    cols_per_row = 2
    for idx, result in enumerate(results):
        if idx % cols_per_row == 0:
            row_cols = st.columns(cols_per_row)
        col = row_cols[idx % cols_per_row]
        entry = _face_entry(result, mapper.conf_threshold)

        with col:
            st.markdown(f"### Face {idx + 1}")
            _render_truth_lie(entry['raw_truth'], entry['raw_lie'])
            st.markdown(
                f"""
                <div class='metric-card'>
                  <div style='font-size:2rem'>{entry['emoji']}</div>
                  <div class='emotion-badge' style='background:{entry['hex']};'>{entry['emotion'].upper()}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.metric("Confidence", f"{entry['confidence']*100:.1f}%")
            with st.expander("Why this score?"):
                st.write(entry['reason'])
                st.write(f"Raw mapping: Truth {entry['raw_truth']}% / Lie {entry['raw_lie']}%")

            probs_dict = result['all_probs']
            colors = ['#00C853' if e == result['emotion'] else '#444' for e in probs_dict.keys()]
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
                title="Emotion Probabilities (%)",
                yaxis_range=[0, 110],
                height=260,
                margin=dict(t=40, b=20, l=20, r=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                template='plotly_dark',
            )
            st.plotly_chart(fig_probs, use_container_width=True, key=f"img_probs_{idx}")
            _render_quality_gauge(quality, key=f"img_quality_{idx}")
