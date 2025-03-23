import os
import time
import json
import numpy as np
import pandas as pd
import vertexai
import concurrent.futures

from vertexai.generative_models import GenerationConfig, GenerativeModel, Part
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.cloud.aiplatform_v1beta1.types import (
    content as gapic_content_types,
)

HarmCategory = gapic_content_types.HarmCategory

# Initialize Vertex AI
def init_vertex_ai(project_id: str, location="us-central1"):
    vertexai.init(project=project_id, location=location)
    model = GenerativeModel(model_name="gemini-1.5-flash-001")
    return model

# List videos from a local 'downloads' folder
def list_local_videos(downloads_dir="downloads"):
    video_files = []
    for file in os.listdir(downloads_dir):
        if file.endswith(('.mp4', '.avi', '.mov')):
            video_files.append(file)
    return video_files

# Analyze a video using the Gemini API with the new chain-of-thought prompt
def analyze_video(model, video_uri):
    response = model.generate_content(
        [
            Part.from_uri(video_uri, mime_type="video/mp4"),
            (
                    "Please watch the video carefully and determine whether it contains any anomalies. "
                    "Reply using the following format:\n"
                    "{\n"
                    '  "result": 0  // 0 for no anomaly detected, 1 for anomaly detected\n'
                    "}\n"
            ),
        ],
        generation_config=GenerationConfig(
            temperature=0,
        ),
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
    )
    try:
        text = response.candidates[0].content.parts[0]._raw_part.text
    except Exception as e:
        print(f"Error extracting text from response: {e}")
        print(response)
        text = str(response)
        return text, 'NAN'
    return text, None

# Process videos in batches
def batch_process_videos(model, videos, batch_size, output_filename):
    responses = []
    processing_times = []

    def process_video(video):
        # Build a file URI pointing to the video in the local downloads folder
        video_uri = "file://" + os.path.abspath(os.path.join("downloads", video))
        
        start_time = time.time()
        response_text, processing_time = analyze_video(model, video_uri)
        end_time = time.time()
        
        if processing_time is None:
            processing_time = end_time - start_time

        response = {video: response_text}
        with open(output_filename, "a") as f:
            f.write(json.dumps(response) + "\n")
        
        return video, response_text, processing_time

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i in range(0, len(videos), batch_size):
            print("batch:", i)
            futures = []
            batch = videos[i:i + batch_size]
            for video in batch:
                futures.append(executor.submit(process_video, video))
                time.sleep(2)  # Delay to prevent rate-limit issues
        
            for future in concurrent.futures.as_completed(futures):
                video, response_text, processing_time = future.result()
                responses.append({video: response_text})
                processing_times.append((video, processing_time))

    return responses, processing_times

# Save responses to a JSON file
def save_responses_to_file(responses, filename):
    with open(filename, 'w') as f:
        json.dump(responses, f, indent=4)

def main_part(project_id, model_type, batch_size):
    # Initialize the model
    model = init_vertex_ai(project_id)
    
    # List all videos from local 'downloads' folder
    videos = list_local_videos("downloads")
    print(f"Found {len(videos)} videos in 'downloads' folder.")

    # Create the output directory if it doesn't exist
    output_dir = "response"
    os.makedirs(output_dir, exist_ok=True)

    # Build output filenames
    jsonl_filename = os.path.join(output_dir, f'responses_{model_type}_0shot_1203.jsonl')
    json_filename = os.path.join(output_dir, f'responses_{model_type}_0shot_1203.json')

    # Process videos in batches
    responses, processing_times = batch_process_videos(
        model, videos, batch_size, output_filename=jsonl_filename
    )

    # Print each response and save to JSON
    for response in responses:
        print(response)
    save_responses_to_file(responses, filename=json_filename)
    print("Responses saved to JSON.")

    # Create and return a DataFrame with processing times
    df_time = pd.DataFrame(processing_times, columns=['Video Name', 'Processing Time'])
    df_time['Video Name'] = df_time['Video Name'].apply(lambda x: x.replace('.mp4', ''))
    return jsonl_filename, df_time

# Example usage
if __name__ == "__main__":
    #change project_id
    project_id = 'YourProjectID'
    batch_size = 10  # Adjust batch size as needed
    model_type = 'flash'  # or 'pro'
    filename, df_time = main_part(project_id, model_type, batch_size)
