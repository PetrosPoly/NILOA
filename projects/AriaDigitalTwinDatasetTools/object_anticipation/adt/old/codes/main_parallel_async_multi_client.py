# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import numpy as np
from itertools import product
import os
import logging
import asyncio
from multiprocessing import Process, cpu_count
from projects.AriaDigitalTwinDatasetTools.object_anticipation.adt.old.codes.algorithm_client import execute_algorithm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
main_logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_path", type=str, required=True, help="Path to the ADT sequence")
    parser.add_argument("--device_number", type=int, default=0, help="Device number to visualize, default is 0")
    parser.add_argument("--down_sampling_factor", type=int, default=4, help=argparse.SUPPRESS)
    parser.add_argument("--jpeg_quality", type=int, default=75, help=argparse.SUPPRESS)
    parser.add_argument("--rrd_output_path", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--use_llm", action='store_true', help="If included, enables LLM usage")
    parser.add_argument("--runrr", action='store_true', help="Run the visualization part if enabled")
    parser.add_argument("--visualize_objects", action='store_true', help="Visualize objects in rerun.io")
    return parser.parse_args()

def load_data_and_initialize(args):
    """Load necessary data paths and initialize directories."""
    project_path = "/home/ppolydorou/Documents/projectaria_sandbox/next_active_object_anticipation_projectaria_twin_dataset/projects/AriaDigitalTwinDatasetTools/object_anticipation/adt"
    sequence_path = args.sequence_path

    datasets_path = '/mnt/data/petros/projectaria_tools_adt_data/'
    dataset_folder = os.path.join(datasets_path, sequence_path)
    json_folder = os.path.join(project_path, 'utils', 'json')
    txt_folder = os.path.join(project_path, 'utils', 'txt_files')

    os.makedirs(dataset_folder, exist_ok=True)
    os.makedirs(json_folder, exist_ok=True)
    os.makedirs(txt_folder, exist_ok=True)

    return {
        "project_path": project_path,
        "dataset_folder": dataset_folder,
        "vrsfile": os.path.join(dataset_folder, "video.vrs"),
        "ADT_trajectory_file": os.path.join(dataset_folder, "aria_trajectory.csv"),
        "json_file": os.path.join(json_folder, 'param_combinations.json'),
        "movement_time_dict": os.path.join(project_path, 'data', 'gt', args.sequence_path, 'movement_time_dict.json'),
        "prompt": os.path.join(txt_folder, 'prompts.txt')
    }

# Define parameter sets for combinations
time_thresholds = [2]
avg_dot_threshold_highs = [0.7]
avg_dot_threshold_lows = [0.2]
avg_distance_threshold_highs = [3]
avg_distance_threshold_lows = [1]
high_dot_thresholds = [0.9] # [0.7, 0.8, 0.9]
distance_thresholds = [2] # [1.5, 2, 2.5]
high_dot_counters_threshold = [15] # [15, 30, 45, 60] # ,75, 90]
distance_counters_threshold = [30] # [15, 30, 45, 60] # ,75, 90]
variables_window_times = [3.0]
minimum_time_deactivated = [2.0]
maximum_time_deactivated = [5.0]
user_relative_movement = [2.0]
object_percentage_overlap = [0.7]

# Generate parameter combinations
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

async def execute_task(params_batch, data):
    """Execute algorithm tasks for a batch of parameters asynchronously."""
    tasks = [execute_algorithm(parameters, data) for parameters in params_batch]
    await asyncio.gather(*tasks)

def run_parameter_batches(params_batches, data):
    """Run all parameter batches in parallel using multiprocessing."""
    processes = []

    for params_batch in params_batches:
        process = Process(target=lambda: asyncio.run(execute_task(params_batch, data)))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

def main():
    args = parse_args()
    data = load_data_and_initialize(args)

    # Split parameter combinations among available CPUs
    num_processes = min(cpu_count(), len(param_combinations))
    params_batches = np.array_split(param_combinations, num_processes)

    print(f"Running on {num_processes} processes with {len(param_combinations)} parameter combinations.")
    input("Press ENTER to continue...")
    run_parameter_batches(params_batches, data)

    print("All parameter combinations processed and saved.")

if __name__ == "__main__":
    main()
