import pathlib
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn


EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def get_device() -> Tuple[torch.device, str]:
    if torch.cuda.is_available():
        dev = torch.device('cuda')
        return dev, torch.cuda.get_device_name(0)

    try:
        import torch_directml  # type: ignore

        dev = torch_directml.device()
        return dev, 'DirectML'
    except Exception:
        return torch.device('cpu'), 'CPU'


class EmotionLensCNN(nn.Module):
    def __init__(self):
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

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class EmotionInference:
    def __init__(self, model_path='models/best_model.pt'):
        path = pathlib.Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found at {path}. Run train.py first.")

        self.device, self.device_name = get_device()
        self.model = EmotionLensCNN().to(self.device)

        try:
            checkpoint = torch.load(str(path), map_location=self.device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(str(path), map_location=self.device)
        state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval()

        cascade_path = pathlib.Path(cv2.data.haarcascades) / 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(str(cascade_path))
        self._warmup()

    def _warmup(self):
        dummy = torch.zeros((1, 1, 48, 48), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            _ = self.model(dummy)

    def compute_quality(self, gray_frame):
        blur = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
        blur_score = min(100, int(blur / 5))
        brightness = gray_frame.mean()
        bright_score = 100 - int(abs(brightness - 128) / 1.28)
        bright_score = max(0, min(100, bright_score))
        quality = int(0.6 * blur_score + 0.4 * bright_score)
        return max(0, min(100, quality))

    def preprocess_face(self, face_roi):
        face = cv2.resize(face_roi, (48, 48))
        if len(face.shape) == 3:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = face.astype(np.float32) / 255.0
        tensor = torch.from_numpy(face).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

    def detect_and_predict(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()
        quality = self.compute_quality(gray)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        results = []
        with torch.no_grad():
            for (x, y, w, h) in faces:
                roi = gray[y:y + h, x:x + w]
                inp = self.preprocess_face(roi)
                logits = self.model(inp)
                probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
                idx = int(np.argmax(probs))
                emotion = EMOTION_LABELS[idx]
                confidence = float(probs[idx])
                all_probs: Dict[str, float] = {EMOTION_LABELS[i]: float(probs[i]) for i in range(7)}
                results.append(
                    {
                        'bbox': (x, y, w, h),
                        'emotion': emotion,
                        'confidence': confidence,
                        'all_probs': all_probs,
                        'quality': quality,
                    }
                )
        return results, quality

    def predict_from_pil(self, pil_image):
        frame = np.array(pil_image.convert('RGB'))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return self.detect_and_predict(frame)

    def annotate_frame(self, frame, results, smoothed_truth, smoothed_lie, quality, fps=0):
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        for r in results:
            x, y, bw, bh = r['bbox']
            emotion = r['emotion']
            conf = r['confidence']

            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 200, 80), 2)
            label = f"{emotion.upper()}  {conf*100:.0f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x, y - th - 12), (x + tw + 10, y), (0, 0, 0), -1)
            cv2.putText(annotated, label, (x + 5, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            bar_y = y + bh + 6
            bar_w = bw
            bar_h = 14
            truth_w = int(bar_w * smoothed_truth / 100)
            cv2.rectangle(annotated, (x, bar_y), (x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
            cv2.rectangle(annotated, (x, bar_y), (x + truth_w, bar_y + bar_h), (0, 180, 60), -1)
            cv2.rectangle(annotated, (x + truth_w, bar_y), (x + bar_w, bar_y + bar_h), (60, 30, 200), -1)
            cv2.putText(
                annotated,
                f"T:{smoothed_truth:.0f}%  L:{smoothed_lie:.0f}%",
                (x + 2, bar_y + 11),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        q_color = (0, 200, 80) if quality >= 75 else (0, 140, 255) if quality >= 45 else (30, 30, 220)
        bar_x, bar_y2 = 10, h - 36
        bar_total_w = 160
        filled_w = int(bar_total_w * quality / 100)
        cv2.rectangle(annotated, (bar_x, bar_y2), (bar_x + bar_total_w, bar_y2 + 14), (40, 40, 40), -1)
        cv2.rectangle(annotated, (bar_x, bar_y2), (bar_x + filled_w, bar_y2 + 14), q_color, -1)
        cv2.putText(annotated, f"IMG QUALITY: {quality}%", (bar_x, bar_y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, q_color, 1, cv2.LINE_AA)
        if quality < 45:
            cv2.putText(
                annotated,
                "! Poor quality - results may be inaccurate",
                (bar_x, h - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (30, 30, 220),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(annotated, f"FPS: {fps:.0f}", (w - 80, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
        return annotated
