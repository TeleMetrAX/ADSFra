from anomaly_detectors.online_nab_detector import OnlineAnomalyDetector
from sklearn.ensemble import IsolationForest
import numpy as np

class IsolationForestDetector(OnlineAnomalyDetector):
    
    def __init__(self, contamination=0.1, window_size=6400, *args, **kwargs):
        super(IsolationForestDetector, self).__init__(*args, **kwargs)
        self.model = IsolationForest(random_state=42, contamination=contamination)
        self.windowData = []
        self.windowSize = window_size

    def handleRecord(self, inputData):
        """
        Returns a tuple (anomalyScore).
        The anomalyScore is based on the Isolation Forest model.
        """
        inputValue = inputData["value"]
        self.windowData.append(inputValue)

        if len(self.windowData) > self.windowSize:
            self.windowData.pop(0)

        if len(self.windowData) >= self.windowSize:
            # Exclude current value to avoid data leakage
            data = np.array(self.windowData[:-1]).reshape(-1, 1)
            self.model.fit(data)
            raw_score = -self.model.decision_function([[inputValue]])[0]

            # Normalize the raw_score to [0, 1]
            anomalyScore = (raw_score - self.model.offset_) / abs(self.model.offset_)
            anomalyScore = max(0.0, min(1.0, anomalyScore))  # Ensure it's within [0, 1]
        else:
            anomalyScore = 0.0  # Not enough data to compute anomaly score

        return (anomalyScore,)
