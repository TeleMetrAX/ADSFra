import time
import asyncio
import os
import numpy as np
from dual_buffer.DualBufferProcedure import DualBufferProcedure


async def run_detector(detector_name: str, dataset_name: str, data: np.ndarray, results_dir: str):
    """Run a single detector on a single dataset"""
    print(f"    [..] Running process_data for {detector_name} on {dataset_name}...")
    
    # Initialize procedure
    try:
        procedure = DualBufferProcedure(
            algorithm=detector_name,
            dataset=dataset_name
        )
    except Exception as e:
        print(f"    [!] Error initializing {detector_name}: {e}")
        return

    labels = np.zeros(len(data))
    try:
        await procedure.process_data(data, labels)
        print(f"    [..] Finished process_data.")
    except Exception as e:
        print(f"    [!] Error during process_data: {e}")
        return

    result_path = f"{results_dir}/{detector_name}_{dataset_name}_results.csv"
    try:
        procedure.export_saved_data(result_path)
        print(f"[OK] Completed: {detector_name} on {dataset_name}")
    except Exception as e:
        print(f"    [!] Error exporting data: {e}")

async def main():
    # Requested detectors
    detectors = [
        'bayesChangePt', 'earthgeckoSkyline', 'windowedGaussian',
        'echoStateNetwork', 'relativeEntropy'
    ]

    # --- DATASETS ---
    
    # NAB Datasets
    nab_datasets = [
        'ambient_temperature_system_failure', 'art_daily_flatmiddle', 'art_daily_jumpsdown',
        'art_daily_jumpsup', 'art_daily_nojump', 'art_increase_spike_density', 'art_load_balancer_spikes',
        'cpu_utilization_asg_misconfiguration', 'ec2_cpu_utilization_53ea38', 'ec2_cpu_utilization_5f5533',
        'ec2_cpu_utilization_24ae8d', 'ec2_cpu_utilization_77c1ca', 'ec2_cpu_utilization_825cc2',
        'ec2_cpu_utilization_ac20cd', 'ec2_cpu_utilization_fe7f93', 'ec2_disk_write_bytes_1ef3de',
        'ec2_disk_write_bytes_c0d644', 'ec2_network_in_257a54', 'ec2_request_latency_system_failure',
        'ec2_network_in_5abac7', 'elb_request_count_8c0756', 'exchange-2_cpc_results', 'exchange-2_cpm_results',
        'exchange-3_cpc_results', 'exchange-3_cpm_results', 'exchange-4_cpc_results', 'exchange-4_cpm_results',
        'grok_asg_anomaly', 'iio_us-east-1_i-a2eb1cd9_NetworkIn', 'machine_temperature_system_failure',
        'nyc_taxi', 'occupancy_6005', 'occupancy_t4013', 'rds_cpu_utilization_cc0c53', 'rds_cpu_utilization_e47b3b',
        'rogue_agent_key_hold', 'rogue_agent_key_updown', 'speed_6005', 'speed_7578', 'speed_t4013',
        'TravelTime_387', 'TravelTime_451', 'Twitter_volume_AAPL', 'Twitter_volume_AMZN', 'Twitter_volume_CRM',
        'Twitter_volume_CVS', 'Twitter_volume_FB', 'Twitter_volume_GOOG', 'Twitter_volume_IBM',
        'Twitter_volume_KO', 'Twitter_volume_PFE', 'Twitter_volume_UPS'
    ]

    # Yahoo Datasets
    yahoo_datasets = [
        'Yahoo_A1real_1_data', 'Yahoo_A1real_2_data', 'Yahoo_A1real_3_data', 'Yahoo_A1real_4_data',
        'Yahoo_A1real_5_data', 'Yahoo_A1real_6_data', 'Yahoo_A1real_7_data', 'Yahoo_A1real_8_data',
        'Yahoo_A1real_9_data', 'Yahoo_A1real_10_data', 'Yahoo_A1real_11_data', 'Yahoo_A1real_12_data',
        'Yahoo_A1real_13_data', 'Yahoo_A1real_14_data', 'Yahoo_A1real_15_data', 'Yahoo_A1real_16_data',
        'Yahoo_A1real_17_data', 'Yahoo_A1real_18_data', 'Yahoo_A1real_19_data', 'Yahoo_A1real_20_data',
        'Yahoo_A1real_21_data', 'Yahoo_A1real_22_data', 'Yahoo_A1real_23_data', 'Yahoo_A1real_24_data',
        'Yahoo_A1real_25_data', 'Yahoo_A1real_26_data', 'Yahoo_A1real_27_data', 'Yahoo_A1real_28_data',
        'Yahoo_A1real_29_data', 'Yahoo_A1real_30_data', 'Yahoo_A1real_31_data', 'Yahoo_A1real_32_data',
        'Yahoo_A1real_33_data', 'Yahoo_A1real_34_data', 'Yahoo_A1real_35_data', 'Yahoo_A1real_36_data',
        'Yahoo_A1real_37_data', 'Yahoo_A1real_38_data', 'Yahoo_A1real_39_data', 'Yahoo_A1real_40_data',
        'Yahoo_A1real_41_data', 'Yahoo_A1real_42_data', 'Yahoo_A1real_43_data', 'Yahoo_A1real_44_data',
        'Yahoo_A1real_45_data', 'Yahoo_A1real_46_data', 'Yahoo_A1real_47_data', 'Yahoo_A1real_48_data',
        'Yahoo_A1real_49_data', 'Yahoo_A1real_50_data', 'Yahoo_A1real_51_data', 'Yahoo_A1real_52_data',
        'Yahoo_A1real_53_data', 'Yahoo_A1real_54_data', 'Yahoo_A1real_55_data', 'Yahoo_A1real_56_data',
        'Yahoo_A1real_57_data', 'Yahoo_A1real_58_data', 'Yahoo_A1real_59_data', 'Yahoo_A1real_60_data',
        'Yahoo_A1real_61_data', 'Yahoo_A1real_62_data', 'Yahoo_A1real_63_data', 'Yahoo_A1real_64_data',
        'Yahoo_A1real_65_data', 'Yahoo_A1real_66_data', 'Yahoo_A1real_67_data'
    ]

    # IOPS dataset 300-480
    iops_datasets = [
        'IOPS_300', 'IOPS_301', 'IOPS_302', 'IOPS_303', 'IOPS_304', 'IOPS_305', 'IOPS_306', 'IOPS_307', 'IOPS_308', 'IOPS_309', 
        'IOPS_310', 'IOPS_311', 'IOPS_312', 'IOPS_313', 'IOPS_314', 'IOPS_315', 'IOPS_316', 'IOPS_317', 'IOPS_318', 'IOPS_319', 
        'IOPS_320', 'IOPS_321', 'IOPS_322', 'IOPS_323', 'IOPS_324', 'IOPS_325', 'IOPS_326', 'IOPS_327', 'IOPS_328', 'IOPS_329', 
        'IOPS_330', 'IOPS_331', 'IOPS_332', 'IOPS_333', 'IOPS_334', 'IOPS_335', 'IOPS_336', 'IOPS_337', 'IOPS_338', 'IOPS_339', 
        'IOPS_340', 'IOPS_341', 'IOPS_342', 'IOPS_343', 'IOPS_344', 'IOPS_345', 'IOPS_346', 'IOPS_347', 'IOPS_348', 'IOPS_349', 
        'IOPS_350', 'IOPS_351', 'IOPS_352', 'IOPS_353', 'IOPS_354', 'IOPS_355', 'IOPS_356', 'IOPS_357', 'IOPS_358', 'IOPS_359', 
        'IOPS_360', 'IOPS_361', 'IOPS_362', 'IOPS_363', 'IOPS_364', 'IOPS_365', 'IOPS_366', 'IOPS_367', 'IOPS_368', 'IOPS_369', 
        'IOPS_370', 'IOPS_371', 'IOPS_372', 'IOPS_373', 'IOPS_374', 'IOPS_375', 'IOPS_376', 'IOPS_377', 'IOPS_378', 'IOPS_379', 
        'IOPS_380', 'IOPS_381', 'IOPS_382', 'IOPS_383', 'IOPS_384', 'IOPS_385', 'IOPS_386', 'IOPS_387', 'IOPS_388', 'IOPS_389', 
        'IOPS_390', 'IOPS_391', 'IOPS_392', 'IOPS_393', 'IOPS_394', 'IOPS_395', 'IOPS_396', 'IOPS_397', 'IOPS_398', 'IOPS_399', 
        'IOPS_400', 'IOPS_401', 'IOPS_402', 'IOPS_403', 'IOPS_404', 'IOPS_405', 'IOPS_406', 'IOPS_407', 'IOPS_408', 'IOPS_409', 
        'IOPS_410', 'IOPS_411', 'IOPS_412', 'IOPS_413', 'IOPS_414', 'IOPS_415', 'IOPS_416', 'IOPS_417', 'IOPS_418', 'IOPS_419', 
        'IOPS_420', 'IOPS_421', 'IOPS_422', 'IOPS_423', 'IOPS_424', 'IOPS_425', 'IOPS_426', 'IOPS_427', 'IOPS_428', 'IOPS_429', 
        'IOPS_430', 'IOPS_431', 'IOPS_432', 'IOPS_433', 'IOPS_434', 'IOPS_435', 'IOPS_436', 'IOPS_437', 'IOPS_438', 'IOPS_439', 
        'IOPS_440', 'IOPS_441', 'IOPS_442', 'IOPS_443', 'IOPS_444', 'IOPS_445', 'IOPS_446', 'IOPS_447', 'IOPS_448', 'IOPS_449', 
        'IOPS_450', 'IOPS_451', 'IOPS_452', 'IOPS_453', 'IOPS_454', 'IOPS_455', 'IOPS_456', 'IOPS_457', 'IOPS_458', 'IOPS_459', 
        'IOPS_460', 'IOPS_461', 'IOPS_462', 'IOPS_463', 'IOPS_464', 'IOPS_465', 'IOPS_466', 'IOPS_467', 'IOPS_468', 'IOPS_469', 
        'IOPS_470', 'IOPS_471', 'IOPS_472', 'IOPS_473', 'IOPS_474', 'IOPS_475', 'IOPS_476', 'IOPS_477', 'IOPS_478', 'IOPS_479', 
        'IOPS_480'
    ]

    dataset_groups = {
        'IOPS': iops_datasets
    }

    DATA_DIR = f"./data"
    RESULTS_DIR = f"./anomaly_detection_benchmarks"
    TEST_RESULTS_DIR = f"./test_results"

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
    
    results_file_path = f"{TEST_RESULTS_DIR}/execution_times.txt"
    
    # Initialize results file
    with open(results_file_path, "w") as f:
        f.write("Detector Execution Times\n")
        f.write("========================\n\n")

    total_datasets = sum(len(d) for d in dataset_groups.values())
    total_tasks = len(detectors) * total_datasets
    completed = 0

    print(f"Starting execution of {len(detectors)} detectors on {total_datasets} datasets.")
    print(f"Total tasks: {total_tasks}")

    for detector in detectors:
        for group_name, datasets in dataset_groups.items():
            print(f"\n[TIMER] Starting: {detector} on {group_name} datasets...")
            start_time = time.time()
            
            for dataset in datasets:
                print(f"[->] Processing: {detector} on {dataset} ({completed + 1}/{total_tasks})")
                
                data_path = f"{DATA_DIR}/labeled_{dataset}_values.npy"

                if not os.path.exists(data_path):
                    print(f"[!] Skipped (file not found): {data_path}")
                    completed += 1
                    continue

                try:
                    data = np.load(data_path).flatten()
                except Exception as e:
                    print(f"[!] Error loading {data_path}: {e}")
                    completed += 1
                    continue

                await run_detector(detector, dataset, data, RESULTS_DIR)
                completed += 1
            
            end_time = time.time()
            duration = end_time - start_time
            
            result_line = f"{detector} on {group_name}: {duration:.2f} seconds\n"
            print(f"[TIMER] {result_line.strip()}")
            
            with open(results_file_path, "a") as f:
                f.write(result_line)

    print(f"\n[OK] All tasks completed. Total processed: {completed}/{total_tasks}")
    print(f"Timing results saved to: {results_file_path}")

if __name__ == "__main__":
    asyncio.run(main())
