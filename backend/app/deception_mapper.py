import collections
import datetime
from typing import Deque, Dict, List, Optional

EMOTION_MAP = {
    'fear': {'lie': 75, 'truth': 25, 'hex': '#E24B4A', 'reason': 'Fear often masks hidden information'},
    'disgust': {'lie': 70, 'truth': 30, 'hex': '#EF9F27', 'reason': 'Disgust signals suppressed discomfort'},
    'angry': {'lie': 65, 'truth': 35, 'hex': '#D85A30', 'reason': 'Anger can accompany defensive deception'},
    'surprise': {'lie': 55, 'truth': 45, 'hex': '#7F77DD', 'reason': 'Surprise may indicate caught off-guard'},
    'neutral': {'lie': 30, 'truth': 70, 'hex': '#888780', 'reason': 'Neutral expression is typically baseline honest'},
    'sad': {'lie': 40, 'truth': 60, 'hex': '#378ADD', 'reason': 'Sadness correlates with genuine expression'},
    'happy': {'lie': 15, 'truth': 85, 'hex': '#639922', 'reason': 'Happiness strongly associated with truthfulness'},
}

EMOTION_EMOJI = {
    'angry': '😠',
    'disgust': '🤢',
    'fear': '😨',
    'happy': '😊',
    'neutral': '😐',
    'sad': '😢',
    'surprise': '😲',
}


class DeceptionMapper:
    def __init__(self, alpha: float = 0.3, conf_threshold: float = 0.45) -> None:
        self.alpha = alpha
        self.conf_threshold = conf_threshold
        self.smoothed_truth = 50.0
        self.smoothed_lie = 50.0
        self.frame_count = 0
        self.session_start = datetime.datetime.now()
        self.emotion_history: Deque[Dict] = collections.deque(maxlen=1000)

    def update(self, emotion: str, confidence: float, quality: int) -> Dict:
        self.frame_count += 1
        if confidence < self.conf_threshold:
            emotion = 'neutral'

        mapping = EMOTION_MAP[emotion]
        raw_truth = int(mapping['truth'])
        raw_lie = int(mapping['lie'])

        self.smoothed_truth = self.alpha * raw_truth + (1 - self.alpha) * self.smoothed_truth
        self.smoothed_lie = self.alpha * raw_lie + (1 - self.alpha) * self.smoothed_lie

        entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'emotion': emotion,
            'confidence': round(float(confidence), 4),
            'raw_truth': raw_truth,
            'raw_lie': raw_lie,
            'smoothed_truth': round(self.smoothed_truth, 2),
            'smoothed_lie': round(self.smoothed_lie, 2),
            'quality': int(quality),
            'reason': str(mapping['reason']),
            'hex': str(mapping['hex']),
            'emoji': EMOTION_EMOJI.get(emotion, ''),
        }
        self.emotion_history.append(entry)
        return entry

    def get_current(self) -> Optional[Dict]:
        if not self.emotion_history:
            return None
        return self.emotion_history[-1]

    def get_history(self, limit: int = 180) -> List[Dict]:
        if limit <= 0:
            return []
        return list(self.emotion_history)[-limit:]

    def get_session_stats(self) -> Dict:
        if not self.emotion_history:
            return {
                'total_frames': 0,
                'session_duration': 0,
                'dominant_emotion': 'neutral',
                'avg_truth': 50.0,
                'avg_lie': 50.0,
                'avg_quality': 0.0,
                'emotion_counts': {},
            }

        duration = int((datetime.datetime.now() - self.session_start).total_seconds())
        history = list(self.emotion_history)

        emotion_counts: Dict[str, int] = {}
        sum_truth = 0.0
        sum_lie = 0.0
        sum_quality = 0.0

        for item in history:
            emotion = item['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            sum_truth += float(item['smoothed_truth'])
            sum_lie += float(item['smoothed_lie'])
            sum_quality += float(item['quality'])

        dominant_emotion = max(emotion_counts.items(), key=lambda kv: kv[1])[0]
        n = max(len(history), 1)
        return {
            'total_frames': self.frame_count,
            'session_duration': duration,
            'dominant_emotion': dominant_emotion,
            'avg_truth': round(sum_truth / n, 1),
            'avg_lie': round(sum_lie / n, 1),
            'avg_quality': round(sum_quality / n, 1),
            'emotion_counts': emotion_counts,
        }

    def reset(self) -> None:
        self.smoothed_truth = 50.0
        self.smoothed_lie = 50.0
        self.frame_count = 0
        self.session_start = datetime.datetime.now()
        self.emotion_history.clear()
