import os
import time	
import json
import numpy as np
import pandas as pd
import vertexai
import concurrent.futures

from dotenv import load_dotenv
from vertexai.generative_models import GenerationConfig, GenerativeModel, Part
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.cloud.aiplatform_v1beta1.types import (
    content as gapic_content_types,
)

HarmCategory = gapic_content_types.HarmCategory

# Load environment variables from .env file
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

# Initialize Vertex AI
def init_vertex_ai(project_id: str, location="us-central1"):
    vertexai.init(project=project_id, location=location)
    model = GenerativeModel(model_name="gemini-1.5-pro-001")
    return model

    
def justify_anomaly_detection(model, anomaly_result, formatted_rules):
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
    
    # Define the generation configuration
    generation_config = GenerationConfig(
        temperature=0.0001,
    )
    
    # Define safety settings
    safety_settings = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }

    try:
        # Call the Gemini model with generation configuration and safety settings
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        justification = response.candidates[0].content.parts[0]._raw_part.text
        return justification

    except Exception as e:
        print(f"Error analyzing video description: {e}")
        return str(e)
# Process videos in batches
def batch_process_videos(model, videos_dict, batch_size, output_filename, formatted_rules):
    responses = []
    processing_times = []

    def process_batch(batch):
        batch_responses = {}
        start_time = time.time()

        # Process each video in the batch
        for video_id, description in batch.items():
            result = justify_anomaly_detection(model, description, formatted_rules)
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
            time.sleep(2)
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

def main_part(model, videos_dict, formatted_rules, model_type, batch_size):
    print(f"Processing {len(videos_dict)} video descriptions.")

    # Create the directory if it doesn't exist
    output_dir = "rawvideo_chain"
    os.makedirs(output_dir, exist_ok=True)

    # Create file names based on the model_type
    jsonl_filename = os.path.join(output_dir, f'rule10_{model_type}.jsonl')
    json_filename = os.path.join(output_dir, f'rule10_{model_type}.json')

    # Process video descriptions in batches and save to a JSONL file
    responses, processing_times = batch_process_videos(model, videos_dict, batch_size, jsonl_filename, formatted_rules)

    # Flatten the batch responses and save to JSON
    flat_responses = {k: v for batch in responses for k, v in batch.items()}
    save_responses_to_file(flat_responses, filename=json_filename)
    print("Responses saved to JSON")

    # Create DataFrame with processing times
    df_time = pd.DataFrame(enumerate(processing_times), columns=['Batch', 'Processing Time'])

    return jsonl_filename, df_time
# Example usage
if __name__ == "__main__":
    # Access the project ID from environment variables
    project_id = os.getenv("PROJECT_ID")
    model = init_vertex_ai(project_id)
    # Get the video description first:
    model_type = 'pro'
    video_descriptions = extract_video_anomaly_results(f'rawvideo_chain/responses_{model_type}_llmchain_1203.jsonl')
    # test if the "nan" value is the same with the json
    df = pd.DataFrame({
            'Video Name': list(video_descriptions.keys()),
            'Predicted Label': list(video_descriptions.values())
        })
    df.to_csv(f'test_rawreason_{model_type}.csv', index=False)
    # load anomaly rules:
    formatted_rules = load_and_format_rules('rules_output/llmchain_rule.json')
    batch_size = 1  # Adjust batch size as needed
    
    jsonl_filename, df_time = main_part(model, video_descriptions, formatted_rules, model_type, batch_size)
