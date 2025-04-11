#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import json
import cv2
import base64
import numpy as np
import pandas as pd
import concurrent.futures
import httpx

from dotenv import load_dotenvc
from anthropic import AnthropicVertex

# Load environment variables from .env
load_dotenv()

# Set up your Claude 3.5 Sonnet API client
LOCATION = "europe-west1"  # or "us-east5"
client = AnthropicVertex(region=LOCATION, project_id=os.getenv("PROJECT_ID"))


# In[2]:


def convert_video_to_base64(video_path):
    MAX_SIZE_MB = 27  # 30 MB limit
    MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024  # Convert to bytes
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(total_frames // 20, 1)  # Extract 20 frames uniformly
    base64Frames = []
    frame_count = 0
    total_size = 0

    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        if frame_count % frame_interval == 0:
            # Encode the frame as JPG and then base64
            _, buffer = cv2.imencode(".jpg", frame)
            base64_frame = base64.b64encode(buffer).decode("utf-8")
            base64Frames.append(base64_frame)
            
            # Update total size
            total_size += len(base64_frame.encode('utf-8'))
            
            # If total size exceeds 30MB, break the loop
            if len(base64Frames) >= 20 or total_size > MAX_SIZE_BYTES:
                break
        frame_count += 1
    
    video.release()
    
    # Check if size exceeds the 30MB limit
    if total_size > MAX_SIZE_BYTES:
        # Calculate resizing factor based on the size ratio
        resize_factor = (MAX_SIZE_BYTES / total_size) ** 0.5  # Square root of the ratio
        resized_base64Frames = []
        video = cv2.VideoCapture(video_path)
        frame_count = 0
        total_resized_size = 0
        
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            if frame_count % frame_interval == 0:
                # Resize the frame
                new_width = int(frame.shape[1] * resize_factor)
                new_height = int(frame.shape[0] * resize_factor)
                resized_frame = cv2.resize(frame, (new_width, new_height))
                
                # Encode the resized frame
                _, buffer = cv2.imencode(".jpg", resized_frame)
                resized_base64_frame = base64.b64encode(buffer).decode("utf-8")
                resized_base64Frames.append(resized_base64_frame)
                
                # Update total resized size
                total_resized_size += len(resized_base64_frame.encode('utf-8'))
                
                # Stop if we have 20 frames or the size is within the limit
                if len(resized_base64Frames) >= 20 or total_resized_size > MAX_SIZE_BYTES:
                    break
            frame_count += 1
        
        video.release()
        # Return the resized base64 frames
        return resized_base64Frames
    else:
        # If the size is within the limit, return the original base64 frames
        return base64Frames


# In[3]:


# Analyze a video using the Claude 3.5 Sonnet API with structured prompt
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
                {"type": "text", "text": prompt_message},
                *[
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": frame}}
                    for frame in base64Frames
                ],
            ],
        }
    ]

    params = {
        "model": "claude-3-5-sonnet@20240620",
        "messages": prompt_messages,
        "max_tokens": 8192, # 8192 is the upper bound: https://x.com/alexalbert__/status/1812921642143900036
    }

    try:
        # Step 3: Call the Claude 3.5 Sonnet API
        message = client.messages.create(**params)
        text = message.content[0].text
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
def batch_process_videos(client, videos, directory, batch_size, output_filename, sleep_time):
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
            
            # Sleep for the specified time after processing each batch
            time.sleep(sleep_time)

    return responses, processing_times

# Save responses to a JSON file
def save_responses_to_file(responses, filename):
    with open(filename, 'w') as f:
        json.dump(responses, f, indent=4)

def main_part(directory, model_type, batch_size, sleep_time):
    videos = list_local_videos(directory)
    print(f"Found {len(videos)} videos in {directory} and subdirectories.")

    # Create the directory if it doesn't exist
    output_dir = "response"
    os.makedirs(output_dir, exist_ok=True)

    # Create file name based on the model_type
    jsonl_filename = os.path.join(output_dir, f'responses_{model_type}_0shot_1203.jsonl')
    json_filename = os.path.join(output_dir, f'responses_{model_type}_0shot_1203.json')

    responses, processing_times = batch_process_videos(client, videos, directory, batch_size, output_filename=jsonl_filename, sleep_time=sleep_time)

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
    directory = os.path.join(current_dir, "downloads")  # Directory where videos are saved
    batch_size = 50  # Adjust batch size as needed
    model_type = 'claude-3-5-sonnet'
    sleep_time = 60  # Adjust sleep time (in seconds) as needed
    filename, df_time = main_part(directory, model_type, batch_size, sleep_time)
