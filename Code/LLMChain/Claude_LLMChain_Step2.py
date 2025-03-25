import os
import time
import json
import cv2
import base64
import numpy as np
import pandas as pd
import concurrent.futures
import httpx

from dotenv import load_dotenv
from anthropic import AnthropicVertex

# Load environment variables from .env
load_dotenv()

def extract_video_anomaly_results(filename):
    anomaly_results = {}

    # Read the JSON file line by line
    with open(filename, 'r') as file:
        for line in file:
            try:
                # Parse each line as a dictionary
                video_response = json.loads(line.strip())

                for video, response_text in video_response.items():
                    try:
                        if response_text is None:
                            raise json.JSONDecodeError("Response text is None", response_text, 0)

                        # Clean up the response text by removing unnecessary markers
                        cleaned_response_text = response_text.strip('```json').strip()
                        cleaned_response_text = cleaned_response_text.replace('\n\n', ' ').strip()

                        # Parse the cleaned response text as JSON
                        response_json = json.loads(cleaned_response_text)

                        # Extract the video_description, reasoning, and anomaly values
                        video_description = response_json.get('video_description', 'NAN')
                        reasoning = response_json.get('reasoning', 'NAN')
                        anomaly = response_json.get('anomaly', 0)

                        # Store the extracted values in the desired format
                        anomaly_results[video] = {
                            "video_description": video_description,
                            "reasoning": reasoning,
                            "anomaly": anomaly
                        }

                    except (json.JSONDecodeError, AttributeError) as e:
                        # Handle the case where response text is not valid JSON or other error
                        anomaly_results[video] = {
                            "video_description": "NAN",
                            "reasoning": "NAN",
                            "anomaly": 0  # Default to 0 for no anomaly detected
                        }

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from line: {line.strip()} - {e}")
                continue

    return anomaly_results
def load_and_format_rules(json_file_path):
    # Load the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Extract the rules list
    rules = data.get('rules', [])

    # Combine the rules into a single string
    formatted_rules = "\n".join(rules)
    
    return formatted_rules
def justify_anomaly_detection(anomaly_result, formatted_rules, client):
    # Extract video description, reasoning, and anomaly from anomaly_result
    video_description = anomaly_result.get('video_description', 'NAN')
    reasoning = anomaly_result.get('reasoning', 'NAN')
    anomaly = anomaly_result.get('anomaly', 'NAN')

    # Construct the prompt for anomaly detection with rules
    prompt = f"""
    You are an advanced smart video surveillance expert in the smart home security domain. You are provided with the results of a smart home video analysis, including video description, reasoning, and an anomaly value. Additionally, you have a set of rules for anomaly detection.
    Your task is to review the provided rules. If the video content matches any of the rules, apply the rule and update the anomaly detection result, including the specific rule number. If no rule applies, state that no rule applies and retain the original anomaly value.
    
    The video anomaly result is:
    {{
      "video_description": "{video_description}",
      "reasoning": "{reasoning}",
      "anomaly": {anomaly}
    }}
    
    The rules provided for anomaly detection are: {formatted_rules}
    
    Please think step-by-step and respond using the format below:
    {{
      "Rule_Reasoning": "If the video matches a rule, provide reasoning based on the specific rule number. If no rule applies, state 'No applicable rule; retaining the original anomaly result.'",
      "updated_anomaly": 0 or 1 // Based on the rule application, update the anomaly detection result: 0 for no anomaly detected, 1 for anomaly detected 
    }}
    """
    
    # Construct the prompt messages list
    prompt_messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ]

    # Define parameters for the Claude 3.5 Sonnet API
    params = {
        "model": "claude-3-5-sonnet@20240620",
        "messages": prompt_messages,
        "max_tokens": 8192,
    }

    try:
        # Call the Claude 3.5 Sonnet API
        message = client.messages.create(**params)
        text = message.content[0].text
        return text, None
    except Exception as e:
        print(f"Error analyzing video description: {e}")
        return None, None

# Process videos in batches
def batch_process_videos(videos_dict, batch_size, output_filename, client, formatted_rules):
    responses = []
    processing_times = []

    def process_batch(batch):
        batch_responses = {}
        start_time = time.time()

        # Process each video in the batch
        for video_id, description in batch.items():
            result = justify_anomaly_detection(description, formatted_rules, client)
            batch_responses[video_id] = result
        
        end_time = time.time()
        processing_time = end_time - start_time

        # Append the responses to the file
        with open(output_filename, "a") as f:
            for video_id, result in batch_responses.items():
                response_text = {video_id: result}
                f.write(json.dumps(response_text) + "\n")

        return batch_responses, processing_time

    # Process videos in batches
    with concurrent.futures.ThreadPoolExecutor() as executor:
        video_items = list(videos_dict.items())  # Convert dictionary to list of tuples (video_id, description)
        futures = []

        for i in range(0, len(video_items), batch_size):
            print("Processing batch:", i // batch_size + 1)
            batch = dict(video_items[i:i + batch_size])
            future = executor.submit(process_batch, batch)
            futures.append(future)
        
        for future in concurrent.futures.as_completed(futures):
            result, processing_time = future.result()
            responses.append(result)
            processing_times.append(processing_time)

    return responses, processing_times
    
# Save responses to a JSON file
def save_responses_to_file(responses, filename):
    with open(filename, 'w') as f:
        json.dump(responses, f, indent=4)

def main_part(videos_dict, formatted_rules, model_type, batch_size, client):
    print(f"Processing {len(videos_dict)} video descriptions.")

    # Create the directory if it doesn't exist
    output_dir = "rawvideo_chain"
    os.makedirs(output_dir, exist_ok=True)

    # Create file names based on the model_type
    jsonl_filename = os.path.join(output_dir, f'rule10_{model_type}.jsonl')
    json_filename = os.path.join(output_dir, f'rule10_{model_type}.json')

    # Process video descriptions in batches and save to a JSONL file
    responses, processing_times = batch_process_videos(videos_dict, batch_size, jsonl_filename, client, formatted_rules)

    # Flatten the batch responses and save to JSON
    flat_responses = {k: v for batch in responses for k, v in batch.items()}
    save_responses_to_file(flat_responses, filename=json_filename)
    print("Responses saved to JSON")

    # Create DataFrame with processing times
    df_time = pd.DataFrame(enumerate(processing_times), columns=['Batch', 'Processing Time'])

    return jsonl_filename, df_time
# Example usage
if __name__ == "__main__":
    # Set up your Claude 3.5 Sonnet API client
    LOCATION = "europe-west1"  # or "us-east5"
    client = AnthropicVertex(region=LOCATION, project_id=os.getenv("PROJECT_ID"))
    # Get the video description first:
    model_type = 'claude-3-5-sonnet'
    video_descriptions = extract_video_anomaly_results(f'rawvideo_chain/responses_{model_type}_llmchain_1203.jsonl')
    # test if the "nan" value is the same with the json
    df = pd.DataFrame({
            'Video Name': list(video_descriptions.keys()),
            'Predicted Label': list(video_descriptions.values())
        })
    df.to_csv(f'test_rawreason_{model_type}.csv', index=False)
    # load anomaly rules:
    formatted_rules = load_and_format_rules('rules_output/llmchain_rule.json')
    batch_size = 10  # Adjust batch size as needed
    jsonl_filename, df_time = main_part(video_descriptions, formatted_rules, model_type, batch_size, client)
