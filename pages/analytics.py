import datetime

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from deception_mapper import EMOTION_EMOJI


def _format_duration(seconds: int) -> str:
    minutes = seconds // 60
    remaining = seconds % 60
    return f"{minutes:02d}:{remaining:02d}"


def render_analytics_page(mapper) -> None:
    st.header("📊 Session Analytics")

    stats = mapper.get_session_stats()
    df = mapper.get_history_df()

    if df.empty:
        st.info("No session data yet. Start the live detector or analyze an image first.")
        return

    dominant_emotion = stats.get('dominant_emotion', 'neutral')
    dominant_label = f"{EMOTION_EMOJI.get(dominant_emotion, '')} {dominant_emotion.title()}"

    cols = st.columns(5)
    cols[0].metric("Total frames analyzed", f"{stats.get('total_frames', 0)}")
    cols[1].metric("Session duration", _format_duration(stats.get('session_duration', 0)))
    cols[2].metric("Dominant emotion", dominant_label)
    cols[3].metric("Average truth tendency", f"{stats.get('avg_truth', 0):.1f}%")
    cols[4].metric("Average image quality", f"{stats.get('avg_quality', 0):.1f}%")

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        counts = stats.get('emotion_counts', {})
        fig_pie = go.Figure(
            go.Pie(
                labels=list(counts.keys()),
                values=list(counts.values()),
                hole=0.45,
            )
        )
        fig_pie.update_layout(
            title='Emotion Distribution',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            template='plotly_dark',
            height=360,
            margin=dict(t=40, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with row2_col2:
        fig_timeline = go.Figure()
        fig_timeline.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['smoothed_truth'],
                name='Truth %',
                line=dict(color='#00C853', width=2),
            )
        )
        fig_timeline.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['smoothed_lie'],
                name='Lie %',
                line=dict(color='#D50000', width=2),
            )
        )
        fig_timeline.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['quality'],
                name='Quality',
                line=dict(color='#00E5FF', width=1, dash='dot'),
            )
        )
        fig_timeline.update_layout(
            title='Truth / Lie / Quality Timeline',
            yaxis_range=[0, 100],
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            template='plotly_dark',
            height=360,
            margin=dict(t=40, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

    row3_col1, row3_col2 = st.columns(2)

    with row3_col1:
        grouped = df.groupby('emotion', as_index=False)[['smoothed_truth', 'smoothed_lie']].mean()
        fig_grouped = go.Figure()
        fig_grouped.add_trace(
            go.Bar(x=grouped['emotion'], y=grouped['smoothed_truth'], name='Avg Truth %', marker_color='#00C853')
        )
        fig_grouped.add_trace(
            go.Bar(x=grouped['emotion'], y=grouped['smoothed_lie'], name='Avg Lie %', marker_color='#D50000')
        )
        fig_grouped.update_layout(
            title='Average Truth/Lie by Emotion',
            barmode='group',
            yaxis_range=[0, 100],
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            template='plotly_dark',
            height=340,
            margin=dict(t=40, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_grouped, use_container_width=True)

    with row3_col2:
        fig_hist = px.histogram(
            df,
            x='confidence',
            nbins=20,
            title='Confidence Score Distribution',
            color_discrete_sequence=['#00E5FF'],
            template='plotly_dark',
        )
        fig_hist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=340,
            margin=dict(t=40, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Export")
    csv_data = df.to_csv(index=False).encode('utf-8')
    summary = "\n".join(
        [
            "EmotionLens Session Summary",
            f"Generated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Frames: {stats.get('total_frames', 0)}",
            f"Session Duration (sec): {stats.get('session_duration', 0)}",
            f"Dominant Emotion: {dominant_emotion}",
            f"Average Truth %: {stats.get('avg_truth', 0):.1f}",
            f"Average Lie %: {stats.get('avg_lie', 0):.1f}",
            f"Average Quality %: {stats.get('avg_quality', 0):.1f}",
            f"Emotion Counts: {stats.get('emotion_counts', {})}",
        ]
    )

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download Session CSV",
            data=csv_data,
            file_name='emotionlens_session.csv',
            mime='text/csv',
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "Download Summary Report",
            data=summary.encode('utf-8'),
            file_name='emotionlens_summary.txt',
            mime='text/plain',
            use_container_width=True,
        )
