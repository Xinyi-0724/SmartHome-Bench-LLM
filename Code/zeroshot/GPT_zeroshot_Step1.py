#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This code is the step 1 designed to test GPT-4o performace on VAD (zero-shot prompting)
# In step 1, we output a json and a jsonl file to collect the response from LLM first
# Test dataset: 1203 videos in 'vad_paper_video_final' GCS bucket downloaded in the local folder named "downloads"

import os
import time
import json
import cv2
import base64
import numpy as np
import pandas as pd
import concurrent.futures

from openai import OpenAI
from google.cloud import storage
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Set up your OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# In[2]:


# # Convert video frames to base64 strings
# def convert_video_to_base64(video_path):
#     video = cv2.VideoCapture(video_path)
#     base64Frames = []
#     while video.isOpened():
#         success, frame = video.read()
#         if not success:
#             break
#         _, buffer = cv2.imencode(".jpg", frame)
#         base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
#     video.release()
#     return base64Frames

# Convert video frames to base64 strings, max frame per request for gpt is the same as claude is 20
def convert_video_to_base64(video_path):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(total_frames // 20, 1)  # Calculate the interval to extract 20 frames uniformly
    base64Frames = []
    frame_count = 0

    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        if frame_count % frame_interval == 0:
            _, buffer = cv2.imencode(".jpg", frame)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
            if len(base64Frames) >= 20:
                break
        frame_count += 1
    
    video.release()
    return base64Frames


# In[3]:


# Analyze a video using the GPT-4 API with structured prompt
def analyze_video(client, video_path):
    # Step 0: Convert video frames to base64 strings
    base64Frames = convert_video_to_base64(video_path)
    
    # Step 1: Construct the prompt message
    prompt_message = (
        "Please review the sequentially extracted frames from a video carefully and determine if they contain any anomalies. "
        "Reply using the following format:\n"
        "{\n"
        '  "result": 0  // 0 for no anomaly detected, 1 for anomaly detected\n'
        "}\n"
    )
    
    # Step 2: Prepare the parameters for API call
    prompt_messages = [
        {
            "role": "user",
            "content": [
                prompt_message,
                *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::50]),
            ],
        }
    ]
    
    params = {
        "model": "gpt-4o",
        "messages": prompt_messages,
        "max_tokens": 4096,
        "temperature": 0.0
    }
    
    try:
        # Step 3: Call the OpenAI API
        result = client.chat.completions.create(**params)
        text = result.choices[0].message.content
        return text, None
    except Exception as e:
        print(f"Error analyzing video: {e}")
        return None, None



# List videos in the local directory and subdirectories
def list_local_videos(directory):
    video_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(root, file)
                relative_path = os.path.relpath(video_path, directory)
                video_files.append(relative_path.replace(os.sep, '/'))
    return video_files


# In[4]:


# Process videos in batches
def batch_process_videos(client, videos, directory, batch_size, output_filename):
    responses = []
    processing_times = []

    def process_video(video):
        video_path = os.path.join(directory, video)
        start_time = time.time()
        response_text, processing_time = analyze_video(client, video_path)
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

            for future in concurrent.futures.as_completed(futures):
                video_name, response_text, processing_time = future.result()
                responses.append({video_name: response_text})
                processing_times.append((video_name, processing_time))
    return responses, processing_times

# Save responses to a JSON file
def save_responses_to_file(responses, filename):
    with open(filename, 'w') as f:
        json.dump(responses, f, indent=4)

def main_part(directory, model_type, batch_size):
    videos = list_local_videos(directory)
    print(f"Found {len(videos)} videos in {directory} and subdirectories.")

    # Create the directory if it doesn't exist
    output_dir = "response"
    os.makedirs(output_dir, exist_ok=True)

    # Create file name based on the model_type
    jsonl_filename = os.path.join(output_dir, f'responses_{model_type}_0shot_1203.jsonl')
    json_filename = os.path.join(output_dir, f'responses_{model_type}_0shot_1203.json')

    responses, processing_times = batch_process_videos(client, videos, directory, batch_size, output_filename=jsonl_filename)

    for response in responses:
        print(response)

    save_responses_to_file(responses, filename=json_filename)
    print("Responses saved to json")

    # Create DataFrame with processing times
    df_time = pd.DataFrame(processing_times, columns=['Video Name', 'Processing Time'])
    # Extract the video name
    df_time['Video Name'] = df_time['Video Name'].apply(lambda x: x.split('/')[-1].replace('.mp4', ''))

    return jsonl_filename, df_time


# In[5]:


# Example usage
if __name__ == "__main__":
    # Get the current directory of the script
    current_dir = os.getcwd()
    directory = os.path.join(current_dir, "/downloads")  # Directory where videos are saved
    batch_size = 10  # Adjust batch size as needed
    model_type = 'gpt4o'
    filename, df_time = main_part(directory, model_type, batch_size)


# In[6]:


df_time.to_csv(f'0shot_time_{model_type}_1203video.csv', index=False)


# In[ ]:




