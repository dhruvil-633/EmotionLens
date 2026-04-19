import pandas as pd
import streamlit as st

from deception_mapper import EMOTION_MAP


def render_about_page() -> None:
    st.header("ℹ️ About EmotionLens")

    st.markdown(
        """
        EmotionLens is a real-time emotion recognition and deception tendency visualization system.
        It combines webcam face detection, a PyTorch CNN-based emotion classifier, and a probabilistic
        truth-vs-lie mapping layer with exponential moving average smoothing.

        The application supports live camera inference, image uploads, session analytics,
        and in-app model training telemetry.
        """
    )

    st.subheader("How Deception Mapping Works")
    map_rows = []
    for emotion, values in EMOTION_MAP.items():
        map_rows.append(
            {
                'Emotion': emotion,
                'Truth %': values['truth'],
                'Lie %': values['lie'],
                'Color': values['hex'],
                'Reason': values['reason'],
            }
        )
    df_map = pd.DataFrame(map_rows)

    def _row_color(row):
        return [f"background-color: {row['Color']}; color: white" if c in ('Emotion', 'Color') else '' for c in row.index]

    styled = df_map.style.apply(_row_color, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.subheader("CNN Architecture Summary")
    architecture = """
class EmotionLensCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            Conv2d(1,64,3,padding=1) -> ReLU -> BatchNorm2d,
            Conv2d(64,64,3,padding=1) -> ReLU -> BatchNorm2d -> MaxPool2d -> Dropout(0.25),
            Conv2d(64,128,3,padding=1) -> ReLU -> BatchNorm2d,
            Conv2d(128,128,3,padding=1) -> ReLU -> BatchNorm2d -> MaxPool2d -> Dropout(0.25),
            Conv2d(128,256,3,padding=1) -> ReLU -> BatchNorm2d,
            Conv2d(256,256,3,padding=1) -> ReLU -> BatchNorm2d -> MaxPool2d -> Dropout(0.25),
            Conv2d(256,512,3,padding=1) -> ReLU -> BatchNorm2d -> MaxPool2d -> Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            Flatten -> Linear(4608,1024) -> ReLU -> BatchNorm1d -> Dropout(0.5),
            Linear(1024,512) -> ReLU -> BatchNorm1d -> Dropout(0.4),
            Linear(512,7)
        )
    """
    st.code(architecture, language='python')

    st.subheader("Tech Stack")
    st.markdown(
        """
        - PyTorch
        - OpenCV
        - Streamlit
        - streamlit-webrtc / aiortc
        - Plotly
        - FER-2013 dataset
        """
    )

    st.subheader("References")
    st.markdown(
        """
        - Ekman, P. (1969). Nonverbal leakage and clues to deception.
        - Goodfellow et al. FER-2013 dataset.
        - PyTorch BatchNorm and Dropout best practices.
        """
    )

    st.subheader("Ethical Disclaimer")
    st.warning(
        "For educational purposes only. Deception tendency scores are probabilistic estimates based on "
        "psycholinguistic research (Paul Ekman, 1969-2003) and are NOT scientifically absolute. "
        "Do not use for real-world lie detection."
    )
