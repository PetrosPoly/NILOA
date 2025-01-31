from openai_multi_client import OpenAIMultiClient
import tiktoken
import yaml
import os
import time
import logging

import asyncio
import random
import csv

from openai import OpenAIError, APIStatusError, RateLimitError

global total_api_calls 

# Project path
project_path = "/home/ppolydorou/Documents/projectaria_sandbox/next_active_object_anticipation_projectaria_twin_dataset/projects/AriaDigitalTwinDatasetTools/object_anticipation/adt"
txt_folders = os.path.join(project_path, 'utils', 'txt_files')

# Interaction log filename
filename = 'interaction_log.txt'
filepath = os.path.join(txt_folders, filename)

# Set up logging configuration to log to a file
logging.basicConfig(filename=filepath, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Functions
def read_prompts_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        
    # Split the content based on the delimiter (e.g., '---')
    sections = content.split('---')
    prompts = {}
    for section in sections:
        if ':' in section:
            key, value = section.split(':', 1)
            prompts[key.strip()] = value.strip()
    return prompts

def append_to_history_string(time, 
                             location, 
                             filtered_names_high_dot_counts_and_distance_counts,
                             filtered_names_low_distance_counts_and_high_dot_counts, 
                             filtered_names_high_dot_counts_and_distance_values,
                             filtered_names_low_distance_counts_and_high_dot_values,
                             time_to_approach_dict, 
                             predictions_dict):
    
    log_entry = {
        'timestamp': time,
        'place': location,
        'focus_consistency_from_user_to_objects_measured_in_counts': filtered_names_high_dot_counts_and_distance_counts,
        'proximity_consistency_from_user_to_objects_measured_in_counts': filtered_names_low_distance_counts_and_high_dot_counts,
        'current_distance_from_user_to_objects_measured_in_meters': filtered_names_low_distance_counts_and_high_dot_values,
        'time_to_approach_objects_measured_in_seconds': time_to_approach_dict, 
        'past_predictions_with_timestamps': predictions_dict  
    }
    
    return log_entry

def async_retry_with_exponential_backoff(
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (APIStatusError, RateLimitError, OpenAIError, ValueError),  
    validation_func=None  
):
    def decorator(func):  # Inner decorator function that receives the function to be wrapped
        async def wrapper(*args, **kwargs):
            num_retries = 0
            delay = initial_delay

            while True:
                try:
                    # Execute the wrapped function and retrieve the result
                    result = await func(*args, **kwargs)

                    # Optional: validate result if validation function is provided
                    if validation_func and not validation_func(result):
                        raise ValueError("LLM output format is invalid or missing required content.")

                    return result

                except errors as e:
                    num_retries += 1
                    if num_retries > max_retries:
                        print(f"Maximum retries ({max_retries}) exceeded.")
                        raise

                    delay_with_jitter = delay * (1 + jitter * random.random())
                    print(f"Error: {e}. Retrying in {delay_with_jitter:.2f} seconds.")
                    await asyncio.sleep(delay_with_jitter)
                    delay *= exponential_base
                except Exception as e:
                    raise e
        return wrapper
    return decorator  # Return the decorator with the parameters set

# Define the validate function
def validate_llm_response(response):
    """Check if LLM response contains required keys and structure."""
    if not response or not isinstance(response, dict):
        return False
    required_keys = ["objects_possibility", "rationale", "predicted_objects", "goal"]
    return all(key in response for key in required_keys)

# Initialize OpenAIMultiClient
models = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]
api = OpenAIMultiClient(endpoint="chats", data_template={"model": models[0]})

@async_retry_with_exponential_backoff(validation_func= validate_llm_response)
async def activate_llm(log_content, parameters, prompt_path, timestamp_s):

    # models 
    prompts = read_prompts_from_file(prompt_path)
    prompt_instruction = prompts.get('prompt_instruction', '')
    prompt_reasoning = prompts.get('prompt_reasoning', '')
    prompt_predict = prompts.get('prompt_predict', '')
    
    full_prompt = prompt_instruction + prompt_reasoning + prompt_predict

    max_tokens = 30000  # Set your token limit
    within_limit, total_tokens = check_token_limit(full_prompt, log_content, max_tokens - 1000)  # Adjust for response tokens

    if not within_limit:
        print(f"Skipping request: Token limit exceeded ({total_tokens} > {max_tokens - 1000})")
        return None

    data = {
        "message_to_LLM": [
            {"role": "system", "content": "You are an AI assistant..."},
            {"role": "assistant", "content": "The user is performing..."},
            {"role": "user", "content": f"Spatial context information: {log_content}"},
            {"role": "user", "content": f"Thresholds: focus = {parameters['high_dot_counters_threshold']}, ..."},
            {"role": "user", "content": f"Instructions regarding the provided context: {prompt_instruction}"},
            {"role": "user", "content": f"Rationale behind the selection: {prompt_reasoning}"},
            {"role": "user", "content": f"Prediction: {prompt_predict}"}
        ]
    }

    metadata = {'timestamp': timestamp_s, 'parameters': parameters}

    # Send the request
    api.request(data=data, metadata=metadata)
    api.run_request_function(lambda: None)  # Ensure requests are executed

    # Retrieve the result
    for result in api:
        if result.response is None:
            logging.error(f"No response received for timestamp {timestamp_s}")
            return None
        try:
            raw_llm_response = result.response['choices'][0]['message']['content']
            cleaned_response = clean_llm_response(raw_llm_response)

            if validate_llm_response(cleaned_response):
                processed_response = process_llm_response(cleaned_response, parameters)
                return processed_response
            else:
                logging.error("LLM response validation failed")
                return None

        except KeyError as e:
            logging.error(f"Key error accessing response data: {e}")
            return None
        
    # # Retrieve the result
    # for result in api:
    #     if result.metadata['timestamp'] == timestamp_s:
    #         raw_llm_response = result.response['choices'][0]['message']['content']
    #         cleaned_response = clean_llm_response(raw_llm_response)
    #         if validate_llm_response(cleaned_response):
    #             processed_response = process_llm_response(cleaned_response, parameters)
    #             return processed_response
    #         else:
    #             logging.error("LLM response validation failed")
    #             # Pause execution to wait for user input to continue
    #             input("An error occurred. Press Enter to continue...")
    #             return None


def clean_llm_response(llm_response):
    # Strip leading and trailing whitespace and triple quotes
    cleaned_response = llm_response.strip('"""').strip()
    return cleaned_response

def process_llm_response(llm_response, parameters):
    cleaned_response = clean_llm_response(llm_response)
    # Clean and parse the YAML response
    try:
        data = yaml.safe_load(cleaned_response)
        most_likely_objects_to_interact_with = data['most_likely_objects_to_interact_with']
        rationale = data['rationale']
        predicted_interaction_objects = data['predicted_interaction_objects']
        goal = data['goal_of_the_user']
        return most_likely_objects_to_interact_with, rationale, predicted_interaction_objects, goal
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML: {e}")
        logging.error(f"Error parsing YAML: {e} | Parameters: {parameters}")
        input("An error occurred. Press Enter to continue...")
    except KeyError as e:
        logging.error(f"Key not found in the response: {e}")
        print(f"Key not found in the response: {e}| Parameters: {parameters}")
        input("An error occurred. Press Enter to continue...")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        logging.error(f"An unexpected error occurred: {e} | Parameters: {parameters}")
        input("An error occurred. Press Enter to continue...")
    return None, None, None, None

# Initialize the tokenizer for the OpenAI GPT-3 or GPT-4 model
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(prompt):
    """Count the number of tokens in a given prompt."""
    tokens = tokenizer.encode(prompt)
    return len(tokens)

def check_token_limit(prompt, log, max_tokens):
    """Check if the combined tokens of prompt and log are within the limit."""
    prompt_tokens = count_tokens(prompt)
    log_tokens = count_tokens(log)
    total_tokens = prompt_tokens + log_tokens
    if total_tokens > max_tokens:
        return False, total_tokens
    return True, total_tokens

def log_to_csv(timestamp_ns, obj_id, obj_name, time, csv_file):
    write_header = not os.path.exists(csv_file)
    # Ensure the CSV header is written only once
    if write_header:
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestep', 'Object ID', 'Object Name', 'Time to Contact'])
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp_ns, obj_id, obj_name, time])
            
def setup_logger(log_filename):
    logger = logging.getLogger(log_filename)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_filename)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger