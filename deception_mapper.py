import collections
import datetime

import pandas


EMOTION_MAP = {
    "fear": {"lie": 75, "truth": 25, "hex": "#E24B4A", "reason": "Fear often masks hidden information"},
    "disgust": {"lie": 70, "truth": 30, "hex": "#EF9F27", "reason": "Disgust signals suppressed discomfort"},
    "angry": {"lie": 65, "truth": 35, "hex": "#D85A30", "reason": "Anger can accompany defensive deception"},
    "surprise": {"lie": 55, "truth": 45, "hex": "#7F77DD", "reason": "Surprise may indicate caught off-guard"},
    "neutral": {"lie": 30, "truth": 70, "hex": "#888780", "reason": "Neutral expression is typically baseline honest"},
    "sad": {"lie": 40, "truth": 60, "hex": "#378ADD", "reason": "Sadness correlates with genuine expression"},
    "happy": {"lie": 15, "truth": 85, "hex": "#639922", "reason": "Happiness strongly associated with truthfulness"},
}

EMOTION_EMOJI = {
    "angry": "😠", "disgust": "🤢", "fear": "😨",
    "happy": "😊", "neutral": "😐", "sad": "😢", "surprise": "😲"
}


class DeceptionMapper:
    def __init__(self, alpha=0.3, window=15, conf_threshold=0.45):
        self.alpha = alpha
        self.window = window
        self.conf_threshold = conf_threshold
        self.smoothed_truth = 50.0
        self.smoothed_lie = 50.0
        self.emotion_history = collections.deque(maxlen=500)
        self.frame_count = 0
        self.session_start = datetime.datetime.now()

    def update(self, emotion, confidence, quality=100):
        self.frame_count += 1
        if confidence < self.conf_threshold:
            emotion = "neutral"
        mapping = EMOTION_MAP[emotion]
        raw_truth = mapping["truth"]
        raw_lie = mapping["lie"]

        self.smoothed_truth = (self.alpha * raw_truth + (1 - self.alpha) * self.smoothed_truth)
        self.smoothed_lie = (self.alpha * raw_lie + (1 - self.alpha) * self.smoothed_lie)

        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "emotion": emotion,
            "confidence": round(confidence, 4),
            "raw_truth": raw_truth,
            "raw_lie": raw_lie,
            "smoothed_truth": round(self.smoothed_truth, 2),
            "smoothed_lie": round(self.smoothed_lie, 2),
            "quality": quality,
            "reason": mapping["reason"],
            "hex": mapping["hex"],
            "emoji": EMOTION_EMOJI.get(emotion, ""),
        }
        self.emotion_history.append(entry)
        return entry

    def get_current(self):
        if not self.emotion_history:
            return None
        return self.emotion_history[-1]

    def get_history_df(self):
        if not self.emotion_history:
            return pandas.DataFrame()
        df = pandas.DataFrame(list(self.emotion_history))
        df['timestamp'] = pandas.to_datetime(df['timestamp'])
        return df

    def get_session_stats(self):
        df = self.get_history_df()
        if df.empty:
            return {}
        duration = (datetime.datetime.now() - self.session_start).seconds
        return {
            "total_frames": self.frame_count,
            "session_duration": duration,
            "dominant_emotion": df['emotion'].mode()[0],
            "avg_truth": round(df['smoothed_truth'].mean(), 1),
            "avg_lie": round(df['smoothed_lie'].mean(), 1),
            "avg_quality": round(df['quality'].mean(), 1),
            "emotion_counts": df['emotion'].value_counts().to_dict(),
        }

    def reset(self):
        self.smoothed_truth = 50.0
        self.smoothed_lie = 50.0
        self.frame_count = 0
        self.session_start = datetime.datetime.now()
        self.emotion_history.clear()
