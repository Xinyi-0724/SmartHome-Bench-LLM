#!/usr/bin/env python
# coding: utf-8

# In[1]:

# In step 1, we output a json and a jsonl file to collect the response from LLM first
# Test dataset: 1203 videos in the local folder named "downloads"

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


# Convert video frames to base64 strings
def convert_video_to_base64(video_path):
    video = cv2.VideoCapture(video_path)
    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
    video.release()
    return base64Frames


# In[3]:


# Analyze a video using the GPT-4 API with structured prompt
def analyze_video(client, video_path):
    # Step 0: Convert video frames to base64 strings
    base64Frames = convert_video_to_base64(video_path)
    
    # Step 1: Construct the prompt message
    prompt_message = (
        "You are an excellent smart video vigilance expert agent in the smart home security domain. You are given a series of sequential frames extracted from a smart home video clip, and your job is to carefully identify potentially risky, suspicious, or anomalous situations captured by the surveillance cameras. These cameras are set up by users to enhance their safety and security. Keep in mind that the people in the video may or may not be the camera owners. In this context, anomalies refer to behaviors or events that raise concerns related to security, personal safety, child safety, wildlife alerts, unusual pet behavior, senior monitoring, or any other situations that seem out of the ordinary.\n"
        "Please think step by step and respond using the format below:\n"
        "{\n"
        '  "video_description": "A concise description of the video content, including objects, movements, and environmental conditions (max 200 words)",\n'
        '  "reasoning": "Detailed reasoning for why the situation is considered abnormal or concerning, if applicable (max 100 words)",\n'
        '  "anomaly": 0, // 0 for no anomaly detected, 1 for anomaly detected\n'
        "}\n"
        "\n"
        "# Reference Examples:\n"
        "1. Example 1:\n"
        "{\n"
        '  "video_description": "The video shows a young child running towards a swimming pool. The child jumps into the pool and does not resurface. A man, who appears to be a neighbor, jumps the fence and pulls the child from the pool. The man then performs CPR on the child.",\n'
        '  "reasoning": "The situation is extremely concerning as a young child jumped into a pool and was unable to resurface. This is a life-threatening situation and requires immediate intervention, which thankfully the neighbor provided.",\n'
        '  "anomaly": 1\n'
        "}\n"
        "2. Example 2:\n"
        "{\n"
        '  "video_description": "A security camera captures a woman attempting to wrangle a small dog on a leash in a driveway. The dog breaks free and runs off-screen. The woman briefly chases after the dog before giving up and returning to the house.",\n'
        '  "reasoning": "The dog escaping its leash could be concerning for the owner as it raises concerns about the dog\'s safety and well-being. The dog running loose could lead to it getting lost or potentially encountering dangerous situations.",\n'
        '  "anomaly": 1\n'
        "}\n"
        "3. Example 3:\n"
        "{\n"
        '  "video_description": "The video, taken from a smart doorbell camera, shows a man attempting to break into a house. He is using a crowbar to pry open the front door. The man is wearing a black shirt, black shorts, a maroon beanie, and blue gloves.",\n'
        '  "reasoning": "The man\'s actions are clearly suspicious and illegal. Attempting to force entry into a home using a crowbar is a crime. His behavior suggests criminal intent, posing a serious security threat.",\n'
        '  "anomaly": 1\n'
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
        "model": "gpt-4o-mini",
        "messages": prompt_messages,
        "max_tokens": 16384,
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
    jsonl_filename = os.path.join(output_dir, f'responses_{model_type}_few_1203.jsonl')
    json_filename = os.path.join(output_dir, f'responses_{model_type}_few_1203.json')

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
    model_type = 'gpt-4o-mini'
    filename, df_time = main_part(directory, model_type, batch_size)


