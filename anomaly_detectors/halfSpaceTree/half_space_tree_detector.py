from anomaly_detectors.online_nab_detector import OnlineAnomalyDetector
from river.anomaly import HalfSpaceTrees
import numpy as np

class HalfSpaceTreeDetector(OnlineAnomalyDetector):
    """
    Anomaly detector using the Half-Space Trees algorithm.
    """

    def __init__(self, window_size=6400, n_trees=25, height=15, *args, **kwargs):
        super(HalfSpaceTreeDetector, self).__init__(*args, **kwargs)
        self.model = HalfSpaceTrees(n_trees=n_trees, height=height, window_size=window_size)
        self.windowData = []
        self.windowSize = window_size

    def handleRecord(self, inputData):
        """
        Returns a tuple (anomalyScore).
        The anomalyScore is based on the Half-Space Trees model.
        """
        inputValue = inputData["value"]
        self.windowData.append(inputValue)

        if len(self.windowData) > self.windowSize:
            self.windowData.pop(0)

        if len(self.windowData) >= self.windowSize:
            # Convert the input value to a dictionary for the River model
            input_dict = {"value": inputValue}
            anomalyScore = self.model.score_one(input_dict)
            self.model.learn_one(input_dict)
        else:
            anomalyScore = 0.0  # Not enough data to compute anomaly score

        return (anomalyScore,)