TEST_ID = '00001'  # used when saving some files

MODE = 'I'  # 'I' or 'II'
# 'I':  run VMD every timestep, need labelled anomalies
# 'II': run VMD once at the beginning, normal data only

L = 100000

ALGORITHMS = ['bayesChangePt', 'windowedGaussian',
              'relativeEntropy', 'earthgeckoSkyline',
              'contextOSE', 'knncad',
              'AREP', 'Alter-ReRe']

OFFLINE_DATA_DIR = 'data_offline'
ONLINE_DATA_DIR = 'data_online'
PP_OFFL_DATA_DIR = 'datapp_offline'
PP_ONL_DATA_DIR = 'datapp_online'
RESULTS_DIR = 'results'
MODE_SUM_DIR = 'modesums'
TRUNCATED_MODE_SUM_DIR = 'modesums/truncated'

TEMP_DIR = 'temp'
OPTUNA_PARAMS_SAVE_FILE = RESULTS_DIR + '/{}_per_params_and_datasets_{}.csv'
OFFLINE_PREP_OPTIMAL_PARAMS_FILE = RESULTS_DIR + '/offline_prep_optimal_params_' + TEST_ID + '_mode_{}.csv'

SEPARATOR = '/'

# AREP related parameters
AREP_MAIN_DIR = 'AREP'
AREP_EXPORT_FILE = 'AREP/eval_results_temp.csv'
AREP_TESTRUN_LOCATION = 'AREP/testRuns/signal_testrun.csv'
AREP_TESTRUN_COLUMNS = ['B', 'THRESHOLD_STRENGTH', 'USE_WINDOW', 'WINDOW_SIZE', 'USE_AGING', 'USE_AARE_AGING',
                        'USE_THD_AGING', 'AGE_POWER', 'USE_AUTOMATIC_WS_AP', 'USE_OFFSET_COMP',
                        'ACCEPTABLE_AVG_DURATION', 'USE_AUTOMATIC_OFFSET', 'OFFSET_WINDOW_SIZE', 'OFFSET_PERCENTAGE',
                        'NUM_EPOCHS', 'NUM_NEURONS', 'FILENAME']
AREP_TESTRUN_PARAMETERS = [30, 3.0, 'T', 800, 'T', 'T', 'F', 2.5, 'T', 'T', 1, 'T', 0, 0, 30, 30]
ALTER_RERE_TESTRUN_PARAMETERS = [30, 3.0, 'T', 1000, 'T', 'T', 'F', 2.0, 'F', 'F', 1, 'F', 0, 0, 30, 30]


# NAB related parameters
NAB_MAIN_DIR = 'NAB'
NAB_ORIG_DATA_DIR = 'NAB/evaluation/data'
THRESHOLDS_FILE = 'NAB/evaluation/config/thresholds.json'
NAB_WINDOWS_FILE = 'NAB/evaluation/labels/combined_windows_decomp{}.json'
NAB_LABELS_FILE = 'NAB/evaluation/labels/combined_labels.json'

# complete list of AREP-based and NAB-based detectors
AREP_ALGOS = ['Alter-ReRe', 'AREP']
NAB_ALGOS = ['bayesChangePt', 'windowedGaussian',
             'relativeEntropy', 'earthgeckoSkyline',
             'contextOSE', 'knncad']

OFFLINE_COLUMN_NAME = 'value'
ONLINE_COLUMN_NAME = 'value'  # has to be left on 'value' for compatibility with NAB and AREP

ONLINE_FILE_STRUCTURES = ['{}_origdata.csv',  # 0
                          '{}_origlabels.csv',  # 1
                          '{}_adddata-{}.csv',  # 2
                          '{}_addlabels-{}.csv']  # 3
