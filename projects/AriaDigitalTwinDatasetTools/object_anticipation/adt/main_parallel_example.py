import argparse
import logging
import multiprocessing as mp
from itertools import product
from tqdm import tqdm
import time
import os

# ... [rest of your imports]


# ==============================================
# Parameters Settting
# ==============================================

# Parameters for the language model module
time_thresholds = [2] # [1, 2, 3]                       
avg_dot_threshold_highs = [0.7]                         
avg_dot_threshold_lows = [0.2]                          
avg_distance_threshold_highs = [3]                       
avg_distance_threshold_lows = [1]                       

high_dot_thresholds = [0.9]  # [0.7, 0.8, 0.9]               # [0.5, 0.6, 0.7, 0.8, 0.9]                 
distance_thresholds = [1.5, 2, 2.5]                        
high_dot_counters_threshold = [15, 30, 45, 60, 75, 90]  
distance_counters_threshold = [15, 30, 45, 60, 75, 90]  

variables_window_times = [3.0]                          

# Parameters for the LLM reactivation module
minimum_time_deactivated = [2.0]                        
maximum_time_deactivated = [5.0]                        
user_relative_movement = [2.0]                          
object_percentage_overlap = [0.7]                           
                   


def run_simulation(parameters):
    try:
        # Initialize a unique logger for each simulation
        logger = logging.getLogger(f"Simulation-{parameters}")
        logger.setLevel(logging.DEBUG)
        
        # Create a FileHandler for logging to a file
        log_filename = f"logs/simulation_{parameters}.log"
        fh = logging.FileHandler(log_filename)
        fh.setLevel(logging.DEBUG)
        
        # Create a StreamHandler for logging to the console
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Define a common formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Avoid adding multiple handlers to the same logger
        if not logger.handlers:
            logger.addHandler(fh)
            logger.addHandler(ch)
            logger.propagate = False  # Prevent messages from being propagated to the root logger
        
        logger.info("Starting simulation with parameters: %s", parameters)
        
        def main(parameters):
            logger.info('Executing the main algorithm')
            # ... [rest of your main algorithm]
        
        main(parameters)
        
        logger.info("Simulation completed successfully.")
    except Exception as e:
        logger.error("Simulation failed with error: %s", e)
        # Optionally, handle the exception as needed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_path", type=str, required=True, help="path to the ADT sequence")
    parser.add_argument("--device_number", type=int, default=0, help="Device_number you want to visualize, default is 0")
    parser.add_argument("--down_sampling_factor", type=int, default=4, help=argparse.SUPPRESS)
    parser.add_argument("--jpeg_quality", type=int, default=75, help=argparse.SUPPRESS)
    parser.add_argument("--rrd_output_path", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--use_llm", action='store_true', help="If included, becomes True")
    parser.add_argument("--runrr", action='store_true', help="Run the visualization part")
    parser.add_argument("--visualize_objects", action='store_true', help="Visualize the objects in rerun.io")
    return parser.parse_args()

if __name__ == "__main__":
    # Ensure the logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    # Set up the main logger to also log to console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    main_logger = logging.getLogger("Main")
    
    # ==============================================
    # Parameters Setting
    # ==============================================
    
    # [Your parameter setup as per the original script]
    
    # Generate all combinations of the parameters
    param_combinations = [
        {
            "time_threshold": t,
            "avg_dot_high": adh,
            "avg_dot_low": adl,
            "avg_distance_high": adhg,
            "avg_distance_low": adlg,
            "high_dot_threshold": hdt,
            "distance_threshold": dt,
            "high_dot_counters_threshold": hdct,
            "distance_counters_threshold": dct,
            "window_time": w, 
            "minimum_time_deactivated": mintd,                
            "maximum_time_deactivated": maxtd,             
            "user_relative_movement": urm,                 
            "object_percentage_overlap": obo,   
        }
        for t, adh, adl, adhg, adlg, hdt, dt, hdct, dct, w, mintd, maxtd, urm, obo in product(
            time_thresholds, avg_dot_threshold_highs, avg_dot_threshold_lows, 
            avg_distance_threshold_highs, avg_distance_threshold_lows,
            high_dot_thresholds, distance_thresholds,
            high_dot_counters_threshold, distance_counters_threshold, variables_window_times, 
            minimum_time_deactivated, maximum_time_deactivated, user_relative_movement, object_percentage_overlap   
        )
    ]
    
    # ==============================================
    # Run through the parameter combinations in parallel
    # ==============================================
    
    start_time = time.time()
    
    with mp.Pool(min(mp.cpu_count(), 64)) as pool:
        # Use imap_unordered with tqdm for progress monitoring
        for _ in tqdm(pool.imap_unordered(run_simulation, param_combinations), total=len(param_combinations)):
            pass  # We're not collecting results here
    
    end_time = time.time()
    main_logger.info(f"Total time taken: {end_time - start_time:.2f} seconds")
