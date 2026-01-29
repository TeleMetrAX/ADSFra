import numpy as np
import pandas as pd
import sys
import os

# Add the parent directory of dual_buffer (which contains OnlineDetectors) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#from anomaly_detectors.AREP.AREP_detector import AREPDetector
from anomaly_detectors.AnDePeD.AnDePeD_detector import ANDEPEDDetector
from anomaly_detectors.bayesChangePt.bayes_changept_detector import BayesChangePtDetector
from anomaly_detectors.contextOSE.context_ose_detector import ContextOSEDetector
from anomaly_detectors.earthgeckoSkyline.earthgecko_skyline_detector import EarthgeckoSkylineDetector
from anomaly_detectors.knncad.knncad_detector import KnncadDetector
from anomaly_detectors.relativeEntropy.relative_entropy_detector import RelativeEntropyDetector
from anomaly_detectors.windowedGaussian.windowedGaussian_detector import WindowedGaussianDetector
from anomaly_detectors.isolationForest.isolation_forest_detector import IsolationForestDetector
from anomaly_detectors.halfSpaceTree.half_space_tree_detector import HalfSpaceTreeDetector
from anomaly_detectors.echoStateNetwork.echo_state_network_detector import EchoStateNetworkDetector
import scaling
import vmd
import read_files
import online_buffer as obuff


class DualBufferProcedure:

    def __init__(self, algorithm: str, dataset: str, mode: str = None, data_parameters: list = None,
                 initial_data: list = None):
        """
        Initialise online pre-processing and anomaly detection.
        :param algorithm: name of anomaly detector
        :param dataset: any string to reference the data by
        :param mode: 'I' or 'II', required only for 'AnDePeD' or 'AnDePeDPro' algorithms
        :param data_parameters: [L, alpha_star, k_star, l_vmd, modes_star_path, min, max]
        :param initial_data: list of initial data to load into the buffer
        """
        self.algorithm = algorithm
        self.dataset = dataset

        # Assign default values if necessary
        self.data_min = 1
        self.data_max = 100000000

        # Initialize for AnDePeD or AnDePeDPro if mode and data_parameters are provided
        if algorithm in ['AnDePeD', 'AnDePeDPro']:
            if mode is None or data_parameters is None:
                raise ValueError(f"Mode and data_parameters must be provided for {algorithm}")
            self.mode = mode

            # Unpack data_parameters before using them
            (self.l, self.alpha_star, self.k_star, self.l_vmd, self.modes_star_path,
             self.data_min, self.data_max) = data_parameters

            # Now that self.l is set, we can initialize the buffer
            self.buffer = obuff.CircularBuffer(self._get_buffer_size())
            self.buffer.load(initial_data)

        self.detector = self.initialise_online_detector()

        self.save_data = pd.DataFrame(columns=['algorithm', 'dataset', 'timestep', 'anomaly_score'])

        self.time = 0

    def initialise_online_detector(self):
        #if self.algorithm == 'AREP':
        #    det = AREPDetector(buffer_size=100000000, algorithm='AREP',
        #                       input_min=self.data_min, input_max=self.data_max)

            #det = AREPDetector(buffer_size=100000000, algorithm='AREP',
                               #input_min=self.data_min, input_max=self.data_max)


        if self.algorithm == 'bayesChangePt':
            det = BayesChangePtDetector(input_min=self.data_min, input_max=self.data_max)

        elif self.algorithm == 'windowedGaussian':
            det = WindowedGaussianDetector(input_min=self.data_min, input_max=self.data_max)

        elif self.algorithm == 'relativeEntropy':
            det = RelativeEntropyDetector(input_min=self.data_min, input_max=self.data_max)

        elif self.algorithm == 'earthgeckoSkyline':
            det = EarthgeckoSkylineDetector(input_min=self.data_min, input_max=self.data_max)

        elif self.algorithm == 'contextOSE':
            det = ContextOSEDetector(input_min=self.data_min, input_max=self.data_max)

        elif self.algorithm == 'knncad':
            det = KnncadDetector(input_min=self.data_min, input_max=self.data_max)

        elif self.algorithm == 'AnDePeDPro':
            det = ANDEPEDDetector(buffer_size=100000, algorithm='AnDePeDPro',
                                  input_min=self.data_min, input_max=self.data_max)

        elif self.algorithm == 'AnDePeD':
            det = ANDEPEDDetector(buffer_size=100000, algorithm='AnDePeD',
                                  input_min=self.data_min, input_max=self.data_max)

        elif self.algorithm == 'isolationForest':
            det = IsolationForestDetector(input_min=self.data_min, input_max=self.data_max)

        elif self.algorithm == 'halfSpaceTree':
            det = HalfSpaceTreeDetector(input_min=self.data_min, input_max=self.data_max)

        elif self.algorithm == 'echoStateNetwork':
            det = EchoStateNetworkDetector(input_min=self.data_min, input_max=self.data_max)

        else:
            return -1

        # Call the child's initialization function
        det.initialize()

        return det

    async def process_data(self, input, label):
        for data in input:
            if self.algorithm in ['AnDePeD', 'AnDePeDPro']:

                self.buffer.add_item(data)

                values = self.buffer.get_all_items()
                scaled_values = scaling.preprocess(values)
                # (3) ...
                if self.mode == 'I':
                    remainder_values = self._mode_i_next_timestep(scaled_values)
                elif self.mode == 'II':
                    remainder_values = self._mode_ii_next_timestep(scaled_values)
                else:
                    remainder_values = [-1]

                new_remainder = remainder_values[-1]
                anom_score = self.detector.next_timestep(new_remainder)
                # (4) feed the new value to the anomaly detector

                # Save results for later analysis
                self.save_data.loc[len(self.save_data)] = [self.algorithm, self.dataset, self.time, anom_score]
                self.time += 1
            else:
                anom_score = self.detector.next_timestep(data)

                # Save results for later analysis
                self.save_data.loc[len(self.save_data)] = [self.algorithm, self.dataset, self.time, anom_score]
                self.time += 1
        pass

    def _mode_i_next_timestep(self, scaled_values: np.ndarray):
        # (3.1) VMD + mode removal using (alpha_star, K_star)
        remainder_values, _, _ = vmd.decompose(scaled_values, self.alpha_star, self.k_star)

        # vmd.decompose returns a 2D numpy array (time, data), where the second column is needed
        remainder_values = remainder_values[:, 1]

        # (3.2) cut to newest L values for anomaly detector
        remainder_values = remainder_values[-self.l:]
        return remainder_values

    def _mode_ii_next_timestep(self, scaled_values: np.ndarray):
        # (3) only mode removal using pre-computed modes
        modes_star = self.read_and_extend_modes_star_to_given_length(len(scaled_values))
        remainder_values = scaled_values - modes_star
        return remainder_values

    def export_saved_data(self, filepath: str):
        self.save_data.to_csv(filepath, index=False)

    def _get_buffer_size(self):
        if self.mode == 'I':
            return self.l_vmd + self.l_vmd % 2
        elif self.mode == 'II':
            return self.l + self.l % 2
        else:
            return -1

    def read_and_extend_modes_star_to_given_length(self, desired_length: int):
        ms_orig = read_files.read_file_pandas(self.modes_star_path, column='value', to_numpy=True)
        ms_new = np.empty(shape=0)

        while len(ms_new) + len(ms_orig) < desired_length:
            ms_new = np.append(ms_new, ms_orig)

        ms_new = np.append(ms_new, ms_orig[ : desired_length - len(ms_new)])

        return ms_new

