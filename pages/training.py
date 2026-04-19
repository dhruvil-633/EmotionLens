import json
import pathlib
import subprocess
import sys
import time

import plotly.graph_objects as go
import streamlit as st


def _read_json(path: pathlib.Path):
    if not path.exists():
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _tail_lines(path: pathlib.Path, n: int = 20) -> str:
    if not path.exists():
        return ""
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return ''.join(lines[-n:])


def _line_chart(x, y1, y2, name1, name2, color1, color2, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y1, mode='lines+markers', name=name1, line=dict(color=color1, width=2)))
    fig.add_trace(go.Scatter(x=x, y=y2, mode='lines+markers', name=name2, line=dict(color=color2, width=2)))
    fig.update_layout(
        title=title,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        template='plotly_dark',
        height=300,
        margin=dict(t=40, b=20, l=20, r=20),
    )
    return fig


def render_training_page() -> None:
    st.header("🧠 Train Model")

    if 'training_process' not in st.session_state:
        st.session_state.training_process = None
    if 'training_running' not in st.session_state:
        st.session_state.training_running = False

    root = pathlib.Path(__file__).resolve().parent.parent
    train_script = root / 'train.py'
    models_dir = root / 'models'
    live_metrics_path = models_dir / 'live_metrics.json'
    results_txt_path = models_dir / 'training_results.txt'

    dataset_path = st.text_input("Dataset path", "./Dataset")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        epochs = st.slider("Epochs", 20, 100, 80)
    with col2:
        batch_size = st.selectbox("Batch size", [32, 64, 128], index=1)
    with col3:
        lr = st.selectbox("Learning rate", [0.01, 0.001, 0.0001], index=1)
    with col4:
        patience = st.slider("Patience", 5, 20, 12)

    start_disabled = st.session_state.training_running
    if st.button("Start Training", use_container_width=True, disabled=start_disabled):
        models_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(train_script),
            '--epochs', str(epochs),
            '--batch', str(batch_size),
            '--lr', str(lr),
            '--patience', str(patience),
            '--dataset', dataset_path,
        ]

        proc = subprocess.Popen(
            cmd,
            cwd=str(root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        st.session_state.training_process = proc
        st.session_state.training_running = True
        st.success("Training started.")
        st.rerun()

    proc = st.session_state.training_process
    if st.session_state.training_running and proc is not None:
        poll_result = proc.poll()
        if poll_result is not None:
            st.session_state.training_running = False

        live = _read_json(live_metrics_path)
        if live:
            current = int(live.get('current_epoch', 0))
            total = int(live.get('total_epochs', max(1, epochs)))
            history = live.get('history', {})
            val_acc_list = history.get('val_accuracy', [])
            best_val = max(val_acc_list) if val_acc_list else 0.0

            st.progress(min(current / max(total, 1), 1.0))
            st.write(f"Epoch {current} / {total} - Best Val Acc: {best_val:.4f}")

            fig_acc = _line_chart(
                history.get('epoch', []),
                history.get('accuracy', []),
                history.get('val_accuracy', []),
                'Train Accuracy',
                'Val Accuracy',
                '#00E5FF',
                '#00C853',
                'Live Accuracy',
            )
            st.plotly_chart(fig_acc, use_container_width=True, key='live_acc')

            fig_loss = _line_chart(
                history.get('epoch', []),
                history.get('loss', []),
                history.get('val_loss', []),
                'Train Loss',
                'Val Loss',
                '#FF6D00',
                '#D50000',
                'Live Loss',
            )
            st.plotly_chart(fig_loss, use_container_width=True, key='live_loss')

            lr_list = history.get('lr', [])
            current_lr = lr_list[-1] if lr_list else lr
            st.metric("Current learning rate", f"{current_lr:.8f}")
        else:
            st.info("Waiting for live metrics file...")

        tail_content = _tail_lines(results_txt_path, n=20)
        if tail_content:
            st.subheader("Training Log Tail (last 20 lines)")
            st.code(tail_content)

        if st.session_state.training_running:
            time.sleep(2)
            st.rerun()

    live_after = _read_json(live_metrics_path)
    if live_after and live_after.get('status') == 'complete' and not st.session_state.training_running:
        st.success("Training complete!")

        confusion_path = models_dir / 'plots' / 'confusion_matrix.png'
        acc_curve_path = models_dir / 'plots' / 'accuracy_curve.png'
        loss_curve_path = models_dir / 'plots' / 'loss_curve.png'

        if confusion_path.exists():
            st.image(str(confusion_path), caption='Confusion Matrix', use_column_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            if acc_curve_path.exists():
                st.image(str(acc_curve_path), caption='Accuracy Curve', use_column_width=True)
        with col_b:
            if loss_curve_path.exists():
                st.image(str(loss_curve_path), caption='Loss Curve', use_column_width=True)

        if results_txt_path.exists():
            with st.expander("Full training_results.txt", expanded=False):
                st.code(results_txt_path.read_text(encoding='utf-8'))

        report_path = models_dir / 'plots' / 'classification_report.txt'
        if report_path.exists():
            with st.expander("Classification Report", expanded=False):
                st.code(report_path.read_text(encoding='utf-8'))
