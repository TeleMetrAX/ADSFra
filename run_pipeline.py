import subprocess
import os
import sys
import time

def run_script(script_name):
    print(f"\n[Pipeline] Starting {script_name}...")
    start_time = time.time()
    try:
        # Run the script using the current python interpreter
        result = subprocess.run([sys.executable, script_name], check=True)
        duration = time.time() - start_time
        print(f"[Pipeline] Finished {script_name} in {duration:.2f} seconds.")
    except subprocess.CalledProcessError as e:
        print(f"[Pipeline] Error running {script_name}: {e}")
        sys.exit(1)

def main():
    print("=== Starting Anomaly Detection Pipeline in 'Mark' directory ===")
    
    # Define the scripts to run in order
    scripts = [
        # 1. Fingerprint Generation
        "PLif2.py",
        "fingerprint2.py",
        "TS2Vec.py",
        
        # 2. Dimensionality Reduction (Autoencoder)
        "autoencoder.py",
        
        # 3. Best Algorithm Selection (Requires detector results in ./results)
        "pair_best_algorithm.py"
    ]
    
    # Check if detector results exist
    # Check if detector results exist or if user wants to run them
    results_dir = "./anomaly_detection_benchmarks"
    run_detectors = False
    
    if not os.path.exists(results_dir) or not os.listdir(results_dir):
        print("\n[!] Warning: './anomaly_detection_benchmarks' directory is empty or missing.")
        print("    'pair_best_algorithm.py' requires detector results.")
        run_detectors = True
    else:
        print(f"\n[?] Detector results found in '{results_dir}'.")
        print("    If you added new datasets (e.g. NAB) or want to re-run detectors, you should run 'run_detectors_2.py'.")
        response = input("    Do you want to run 'run_detectors_2.py' now? (y/n): ")
        if response.lower() == 'y':
            run_detectors = True

    if run_detectors:
        scripts.insert(0, "run_detectors_2.py")
    
    for script in scripts:
        script_path = os.path.join(os.getcwd(), script)
        if not os.path.exists(script_path):
            print(f"[!] Script not found: {script_path}")
            continue
            
        run_script(script)

    print("\n=== Pipeline Execution Completed ===")

if __name__ == "__main__":
    main()
