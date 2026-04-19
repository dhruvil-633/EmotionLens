import pathlib
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def get_device() -> Tuple[torch.device, str]:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        return device, torch.cuda.get_device_name(0)

    try:
        import torch_directml  # type: ignore

        device = torch_directml.device()
        return device, 'DirectML'
    except Exception:
        return torch.device('cpu'), 'CPU'


class EmotionLensCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 7),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class EmotionInferenceEngine:
    def __init__(self, model_path: pathlib.Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f'Model not found at {model_path}')

        self.device, self.device_name = get_device()
        self.model = EmotionLensCNN().to(self.device)

        try:
            checkpoint = torch.load(str(model_path), map_location=self.device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(str(model_path), map_location=self.device)

        state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval()

        cascade_path = pathlib.Path(cv2.data.haarcascades) / 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(str(cascade_path))
        self._warmup()

    def _warmup(self) -> None:
        dummy = torch.zeros((1, 1, 48, 48), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            _ = self.model(dummy)

    @staticmethod
    def compute_quality(gray_frame: np.ndarray) -> int:
        blur = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
        blur_score = min(100, int(blur / 5))
        brightness = float(gray_frame.mean())
        bright_score = 100 - int(abs(brightness - 128) / 1.28)
        bright_score = max(0, min(100, bright_score))
        quality = int(0.6 * blur_score + 0.4 * bright_score)
        return max(0, min(100, quality))

    def _preprocess_face(self, face_roi: np.ndarray) -> torch.Tensor:
        face = cv2.resize(face_roi, (48, 48))
        if len(face.shape) == 3:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = face.astype(np.float32) / 255.0
        tensor = torch.from_numpy(face).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

    def detect_and_predict(self, frame_bgr: np.ndarray) -> Tuple[List[Dict], int]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if len(frame_bgr.shape) == 3 else frame_bgr.copy()
        quality = self.compute_quality(gray)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        results: List[Dict] = []
        with torch.no_grad():
            for (x, y, w, h) in faces:
                roi = gray[y:y + h, x:x + w]
                inp = self._preprocess_face(roi)
                logits = self.model(inp)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                idx = int(np.argmax(probs))
                results.append(
                    {
                        'bbox': [int(x), int(y), int(w), int(h)],
                        'emotion': EMOTION_LABELS[idx],
                        'confidence': float(probs[idx]),
                        'all_probs': {EMOTION_LABELS[i]: float(probs[i]) for i in range(7)},
                        'quality': quality,
                    }
                )
        return results, quality


__all__ = ['EmotionInferenceEngine', 'EMOTION_LABELS']
