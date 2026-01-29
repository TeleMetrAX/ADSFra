from anomaly_detectors.online_nab_detector import OnlineAnomalyDetector
from reservoirpy.nodes import Reservoir, FORCE
import numpy as np
from collections import deque

class EchoStateNetworkDetector(OnlineAnomalyDetector):
    """
    Echo State Network anomaly detector using ReservoirPy with online learning.
    """

    def __init__(self, input_min, input_max,
                 reservoir_size=200,
                 spectral_radius=0.9,
                 input_scaling=0.5,
                 leak_rate=0.3,
                 window_size=100,
                 **kwargs):
        super().__init__(input_min, input_max)

        self.window_size = window_size
        self.error_window = deque(maxlen=window_size)

        # Initialize Reservoir and Readout with online learning capability
        self.reservoir = Reservoir(units=reservoir_size,
                                   sr=spectral_radius,
                                   input_scaling=input_scaling,
                                   lr=leak_rate,
                                   seed=42)
        self.readout = FORCE()
        self.esn = self.reservoir >> self.readout

        self.is_initialized = False
        self.last_input = None

    def initialize(self):
        self.reservoir.reset()
        self.error_window.clear()
        self.is_initialized = False
        self.last_input = None

    def handleRecord(self, inputData):
        x = np.array([[inputData["value"]]])

        if not self.is_initialized:
            # Initialize the model with the first input
            self.esn.train(x, x)
            self.is_initialized = True
            self.last_input = x
            return (0.0,)

        # Predict the next value
        y_pred = self.esn.run(self.last_input)
        error = np.abs(x - y_pred)[0][0]
        self.error_window.append(error)

        # Online update using train method
        self.esn.train(self.last_input, x)

        # Compute anomaly likelihood
        if len(self.error_window) > 10:
            mean = np.mean(self.error_window)
            std = np.std(self.error_window)
            likelihood = float(1 - np.exp(-((error - mean) ** 2) / (2 * std ** 2))) if std > 0 else 0.0
        else:
            likelihood = 0.0

        self.last_input = x
        return (likelihood,)
