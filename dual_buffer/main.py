import asyncio
import pandas as pd
import optuna
import logging
import os
import numpy as np
import data_handler
import main_config_AnDePeD as conf
import offline_prep as prep
import write_files
import read_files
from csv_to_npy import algorithm_result_csv_to_npy
import joblib
import sklearn
import vmdpy
import websockets

import DualBufferProcedure

from dual_buffer_sys import DualBufferSystem
import time

from DualBufferProcedure import DualBufferProcedure

async def my_processing_function(input, label):
    pass

async def main():
    # Setup logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Configuration
    window_size = 25
    websocket_uri = "ws://127.0.0.1:8000/ws"
    algorithms = ["AnDePeD", "AnDePeDPro"]

    datasets_name = ['Yahoo_A1real_1_data',
    'Yahoo_A1real_2_data',
    'Yahoo_A1real_3_data',
    'Yahoo_A1real_4_data',
    'Yahoo_A1real_5_data',
    'Yahoo_A1real_6_data',
    'Yahoo_A1real_7_data',
    'Yahoo_A1real_8_data',
    'Yahoo_A1real_9_data',
    'Yahoo_A1real_10_data',
    'Yahoo_A1real_11_data',
    'Yahoo_A1real_12_data',
    'Yahoo_A1real_13_data',
    'Yahoo_A1real_14_data',
    'Yahoo_A1real_15_data',
    'Yahoo_A1real_16_data',
    'Yahoo_A1real_17_data',
    'Yahoo_A1real_18_data',
    'Yahoo_A1real_19_data',
    'Yahoo_A1real_20_data',
    'Yahoo_A1real_21_data',
    'Yahoo_A1real_22_data',
    'Yahoo_A1real_23_data',
    'Yahoo_A1real_24_data',
    'Yahoo_A1real_25_data',
    'Yahoo_A1real_26_data',
    'Yahoo_A1real_27_data',
    'Yahoo_A1real_28_data',
    'Yahoo_A1real_29_data',
    'Yahoo_A1real_30_data',
    'Yahoo_A1real_31_data',
    'Yahoo_A1real_32_data',
    'Yahoo_A1real_33_data',
    'Yahoo_A1real_34_data',
    'Yahoo_A1real_35_data',
    'Yahoo_A1real_36_data',
    'Yahoo_A1real_37_data',
    'Yahoo_A1real_38_data',
    'Yahoo_A1real_39_data',
    'Yahoo_A1real_40_data',
    'Yahoo_A1real_41_data',
    'Yahoo_A1real_42_data',
    'Yahoo_A1real_43_data',
    'Yahoo_A1real_44_data',
    'Yahoo_A1real_45_data',
    'Yahoo_A1real_46_data',
    'Yahoo_A1real_47_data',
    'Yahoo_A1real_48_data',
    'Yahoo_A1real_49_data',
    'Yahoo_A1real_50_data',
    'Yahoo_A1real_51_data',
    'Yahoo_A1real_52_data',
    'Yahoo_A1real_53_data',
    'Yahoo_A1real_54_data',
    'Yahoo_A1real_55_data',
    'Yahoo_A1real_56_data',
    'Yahoo_A1real_57_data',
    'Yahoo_A1real_58_data',
    'Yahoo_A1real_59_data',
    'Yahoo_A1real_60_data',
    'Yahoo_A1real_61_data',
    'Yahoo_A1real_62_data',
    'Yahoo_A1real_63_data',
    'Yahoo_A1real_64_data',
    'Yahoo_A1real_65_data',
    'Yahoo_A1real_66_data',
    'Yahoo_A1real_67_data',

    # A2 synthetic
    'Yahoo_A2synthetic_1_data',
    'Yahoo_A2synthetic_2_data',
    'Yahoo_A2synthetic_3_data',
    'Yahoo_A2synthetic_4_data',
    'Yahoo_A2synthetic_5_data',
    'Yahoo_A2synthetic_6_data',
    'Yahoo_A2synthetic_7_data',
    'Yahoo_A2synthetic_8_data',
    'Yahoo_A2synthetic_9_data',
    'Yahoo_A2synthetic_10_data',
    'Yahoo_A2synthetic_11_data',
    'Yahoo_A2synthetic_12_data',
    'Yahoo_A2synthetic_13_data',
    'Yahoo_A2synthetic_14_data',
    'Yahoo_A2synthetic_15_data',
    'Yahoo_A2synthetic_16_data',
    'Yahoo_A2synthetic_17_data',
    'Yahoo_A2synthetic_18_data',
    'Yahoo_A2synthetic_19_data',
    'Yahoo_A2synthetic_20_data',
    'Yahoo_A2synthetic_21_data',
    'Yahoo_A2synthetic_22_data',
    'Yahoo_A2synthetic_23_data',
    'Yahoo_A2synthetic_24_data',
    'Yahoo_A2synthetic_25_data',
    'Yahoo_A2synthetic_26_data',
    'Yahoo_A2synthetic_27_data',
    'Yahoo_A2synthetic_28_data',
    'Yahoo_A2synthetic_29_data',
    'Yahoo_A2synthetic_30_data',
    'Yahoo_A2synthetic_31_data',
    'Yahoo_A2synthetic_32_data',
    'Yahoo_A2synthetic_33_data',
    'Yahoo_A2synthetic_34_data',
    'Yahoo_A2synthetic_35_data',
    'Yahoo_A2synthetic_36_data',
    'Yahoo_A2synthetic_37_data',
    'Yahoo_A2synthetic_38_data',
    'Yahoo_A2synthetic_39_data',
    'Yahoo_A2synthetic_40_data',
    'Yahoo_A2synthetic_41_data',
    'Yahoo_A2synthetic_42_data',
    'Yahoo_A2synthetic_43_data',
    'Yahoo_A2synthetic_44_data',
    'Yahoo_A2synthetic_45_data',
    'Yahoo_A2synthetic_46_data',
    'Yahoo_A2synthetic_47_data',
    'Yahoo_A2synthetic_48_data',
    'Yahoo_A2synthetic_49_data',
    'Yahoo_A2synthetic_50_data',
    'Yahoo_A2synthetic_51_data',
    'Yahoo_A2synthetic_52_data',
    'Yahoo_A2synthetic_53_data',
    'Yahoo_A2synthetic_54_data',
    'Yahoo_A2synthetic_55_data',
    'Yahoo_A2synthetic_56_data',
    'Yahoo_A2synthetic_57_data',
    'Yahoo_A2synthetic_58_data',
    'Yahoo_A2synthetic_59_data',
    'Yahoo_A2synthetic_60_data',
    'Yahoo_A2synthetic_61_data',
    'Yahoo_A2synthetic_62_data',
    'Yahoo_A2synthetic_63_data',
    'Yahoo_A2synthetic_64_data',
    'Yahoo_A2synthetic_65_data',
    'Yahoo_A2synthetic_66_data',
    'Yahoo_A2synthetic_67_data',
    'Yahoo_A2synthetic_68_data',
    'Yahoo_A2synthetic_69_data',
    'Yahoo_A2synthetic_70_data',
    'Yahoo_A2synthetic_71_data',
    'Yahoo_A2synthetic_72_data',
    'Yahoo_A2synthetic_73_data',
    'Yahoo_A2synthetic_74_data',
    'Yahoo_A2synthetic_75_data',
    'Yahoo_A2synthetic_76_data',
    'Yahoo_A2synthetic_77_data',
    'Yahoo_A2synthetic_78_data',
    'Yahoo_A2synthetic_79_data',
    'Yahoo_A2synthetic_80_data',
    'Yahoo_A2synthetic_81_data',
    'Yahoo_A2synthetic_82_data',
    'Yahoo_A2synthetic_83_data',
    'Yahoo_A2synthetic_84_data',
    'Yahoo_A2synthetic_85_data',
    'Yahoo_A2synthetic_86_data',
    'Yahoo_A2synthetic_87_data',
    'Yahoo_A2synthetic_88_data',
    'Yahoo_A2synthetic_89_data',
    'Yahoo_A2synthetic_90_data',
    'Yahoo_A2synthetic_91_data',
    'Yahoo_A2synthetic_92_data',
    'Yahoo_A2synthetic_93_data',
    'Yahoo_A2synthetic_94_data',
    'Yahoo_A2synthetic_95_data',
    'Yahoo_A2synthetic_96_data',
    'Yahoo_A2synthetic_97_data',
    'Yahoo_A2synthetic_98_data',
    'Yahoo_A2synthetic_99_data',
    'Yahoo_A2synthetic_100_data',

    # A3 benchmark
    'YahooA3Benchmark-TS1_data',
    'YahooA3Benchmark-TS2_data',
    'YahooA3Benchmark-TS3_data',
    'YahooA3Benchmark-TS4_data',
    'YahooA3Benchmark-TS5_data',
    'YahooA3Benchmark-TS6_data',
    'YahooA3Benchmark-TS7_data',
    'YahooA3Benchmark-TS8_data',
    'YahooA3Benchmark-TS9_data',
    'YahooA3Benchmark-TS10_data',
    'YahooA3Benchmark-TS11_data',
    'YahooA3Benchmark-TS12_data',
    'YahooA3Benchmark-TS13_data',
    'YahooA3Benchmark-TS14_data',
    'YahooA3Benchmark-TS15_data',
    'YahooA3Benchmark-TS16_data',
    'YahooA3Benchmark-TS17_data',
    'YahooA3Benchmark-TS18_data',
    'YahooA3Benchmark-TS19_data',
    'YahooA3Benchmark-TS20_data',
    'YahooA3Benchmark-TS21_data',
    'YahooA3Benchmark-TS22_data',
    'YahooA3Benchmark-TS23_data',
    'YahooA3Benchmark-TS24_data',
    'YahooA3Benchmark-TS25_data',
    'YahooA3Benchmark-TS26_data',
    'YahooA3Benchmark-TS27_data',
    'YahooA3Benchmark-TS28_data',
    'YahooA3Benchmark-TS29_data',
    'YahooA3Benchmark-TS30_data',
    'YahooA3Benchmark-TS31_data',
    'YahooA3Benchmark-TS32_data',
    'YahooA3Benchmark-TS33_data',
    'YahooA3Benchmark-TS34_data',
    'YahooA3Benchmark-TS35_data',
    'YahooA3Benchmark-TS36_data',
    'YahooA3Benchmark-TS37_data',
    'YahooA3Benchmark-TS38_data',
    'YahooA3Benchmark-TS39_data',
    'YahooA3Benchmark-TS40_data',
    'YahooA3Benchmark-TS41_data',
    'YahooA3Benchmark-TS42_data',
    'YahooA3Benchmark-TS43_data',
    'YahooA3Benchmark-TS44_data',
    'YahooA3Benchmark-TS45_data',
    'YahooA3Benchmark-TS46_data',
    'YahooA3Benchmark-TS47_data',
    'YahooA3Benchmark-TS48_data',
    'YahooA3Benchmark-TS49_data',
    'YahooA3Benchmark-TS50_data',
    'YahooA3Benchmark-TS51_data',
    'YahooA3Benchmark-TS52_data',
    'YahooA3Benchmark-TS53_data',
    'YahooA3Benchmark-TS54_data',
    'YahooA3Benchmark-TS55_data',
    'YahooA3Benchmark-TS56_data',
    'YahooA3Benchmark-TS57_data',
    'YahooA3Benchmark-TS58_data',
    'YahooA3Benchmark-TS59_data',
    'YahooA3Benchmark-TS60_data',
    'YahooA3Benchmark-TS61_data',
    'YahooA3Benchmark-TS62_data',
    'YahooA3Benchmark-TS63_data',
    'YahooA3Benchmark-TS64_data',
    'YahooA3Benchmark-TS65_data',
    'YahooA3Benchmark-TS66_data',
    'YahooA3Benchmark-TS67_data',
    'YahooA3Benchmark-TS68_data',
    'YahooA3Benchmark-TS69_data',
    'YahooA3Benchmark-TS70_data',
    'YahooA3Benchmark-TS71_data',
    'YahooA3Benchmark-TS72_data',
    'YahooA3Benchmark-TS73_data',
    'YahooA3Benchmark-TS74_data',
    'YahooA3Benchmark-TS75_data',
    'YahooA3Benchmark-TS76_data',
    'YahooA3Benchmark-TS77_data',
    'YahooA3Benchmark-TS78_data',
    'YahooA3Benchmark-TS79_data',
    'YahooA3Benchmark-TS80_data',
    'YahooA3Benchmark-TS81_data',
    'YahooA3Benchmark-TS82_data',
    'YahooA3Benchmark-TS83_data',
    'YahooA3Benchmark-TS84_data',
    'YahooA3Benchmark-TS85_data',
    'YahooA3Benchmark-TS86_data',
    'YahooA3Benchmark-TS87_data',
    'YahooA3Benchmark-TS88_data',
    'YahooA3Benchmark-TS89_data',
    'YahooA3Benchmark-TS90_data',
    'YahooA3Benchmark-TS91_data',
    'YahooA3Benchmark-TS92_data',
    'YahooA3Benchmark-TS93_data',
    'YahooA3Benchmark-TS94_data',
    'YahooA3Benchmark-TS95_data',
    'YahooA3Benchmark-TS96_data',
    'YahooA3Benchmark-TS97_data',
    'YahooA3Benchmark-TS98_data',
    'YahooA3Benchmark-TS99_data',
    'YahooA3Benchmark-TS100_data']

    for dataset_name in datasets_name:
        for algorithm in algorithms:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            file_name = f'./results/{dataset_name}/{algorithm}_{dataset_name}_online_results.csv'
            file_path = os.path.join(script_dir, file_name)
            csv_output_dir = f"./results/{dataset_name}/"

            if algorithm in ["AnDePeD", "AnDePeDPro"]:
                if conf.MODE == 'I':
                    write_files.init_file_pandas(conf.OFFLINE_PREP_OPTIMAL_PARAMS_FILE.format('I'),
                                                 ['algorithm', 'dataset', 'alpha_star', 'k_star', 'omega_star', 'l_vmd'])
                elif conf.MODE == 'II':
                    write_files.init_file_pandas(conf.OFFLINE_PREP_OPTIMAL_PARAMS_FILE.format('II'),
                                                 ['algorithm', 'dataset', 'modes_star_path'])

                datapath_offline = f"../data/YAHOO/{dataset_name}.out"

                alpha_star, k_star, l_vmd, modes_star_path, data_min, data_max = \
                    prep.prepare_procedure(conf.MODE, algorithm, datapath_offline, conf.TEST_ID)

                data_parameters = [conf.L, alpha_star, k_star, l_vmd, modes_star_path, data_min, data_max]
                offline_dataset = np.load(f"../data/labeled_{dataset_name}_values.npy")

                procedure = DualBufferProcedure(algorithm, dataset_name, conf.MODE, data_parameters, offline_dataset)
            else:
                procedure = DualBufferProcedure(algorithm, dataset_name)

            buffer_system = DualBufferSystem(dataset_name, window_size, websocket_uri, processing_func=procedure.process_data)

            try:
                await asyncio.gather(buffer_system.ingestion_task, buffer_system.processing_task)
            except asyncio.CancelledError:
                logger.info("Tasks cancelled. Shutting down")
            finally:
                procedure.export_saved_data(file_path)
                print("DataFrame saved successfully at:", file_path)
                algorithm_result_csv_to_npy(file_path, csv_output_dir)
                await buffer_system.shutdown()

if __name__ == "__main__":
    asyncio.run(main())