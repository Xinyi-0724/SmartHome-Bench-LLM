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
# Analyze a video using the GPT-4 API with structured prompt
def analyze_video(client, video_path):
    # Step 0: Convert video frames to base64 strings
    base64Frames = convert_video_to_base64(video_path)
    
    # Step 1: Construct the prompt message
    prompt_message = (
        "You are an expert in smart video surveillance with a focus on smart home security. "
        "You are given a series of sequential frames from a smart home video clip and a set of rules (taxonomy) for identifying anomalies in various smart home scenarios. "
        "Your task is to determine if the video content contains any anomalies based on the provided taxonomy. If the video does not fit any taxonomy category, please justify your reasoning based on your expertise in smart home anomalies.\n\n"
        "Anomaly Taxonomy:\n"
        "1. Security\n"
        "   - Normal Videos:\n"
        "     - Routine activity of homeowners, known visitors, or vehicles arriving and leaving.\n"
        "     - Scheduled package deliveries or pickups without interference.\n"
        "   - Abnormal Videos:\n"
        "     - Motion or presence indicating potential break-ins, or trespassing.\n"
        "     - Loitering individuals or those wearing unusual attire that deviates from the norm.\n"
        "     - Forced entry attempts, such as fiddling with locks, tampering with doors or windows, or trying to enter a home or vehicle through unconventional means.\n"
        "     - Unauthorized removal of packages, vehicles, or other items.\n"
        "     - Acts of property damage like graffiti, broken windows, car crashes, or other forms of vandalism.\n"
        "     - Actions that might cause harm, such as kidnapping, aggressive confrontations, or any threatening behavior.\n"
        "     - Unusual or eccentric behavior by individuals that could alarm or frighten viewers.\n"
        "2. Wildlife\n"
        "   - Normal Videos:\n"
        "     - Harmless wildlife sightings, such as squirrels, birds, or rabbits, moving through the yard.\n"
        "     - Common pest activity that doesn’t pose immediate danger (e.g., bugs in the garden).\n"
        "   - Abnormal Videos:\n"
        "     - Presence of dangerous wildlife like snakes, spiders, or raccoons that may pose a health risk.\n"
        "     - Any wildlife activity that causes or potentially causes damage to property or threatens human or pet safety.\n"
        "     - Any wildlife (dangerous or not) that enters a home without clear containment.\n"
        "3. Pet Monitoring\n"
        "   - Normal Videos:\n"
        "     - Pets engaging in regular play, resting or moving around within designated safe areas.\n"
        "     - Pets interacting with known family members or other pets.\n"
        "     - Pets accompanied by their guardian without interacting with property or people in harmful ways.\n"
        "   - Abnormal Videos:\n"
        "     - Pets left outside alone for extended periods.\n"
        "     - Pets attempting to escape, leaving the designated area, or exhibiting behaviors indicating escape attempts.\n"
        "     - Pets causing property damage by actions like chewing, scratching, or digging.\n"
        "     - Behaviors that indicate illness or distress, like vomiting, excessive scratching, or erratic movements.\n"
        "     - Any interaction with others that could lead to conflict or injury.\n"
        "4. Baby Monitoring\n"
        "   - Normal Videos:\n"
        "     - Baby engaging in play or sleep within safe zones or under supervision.\n"
        "     - Harmless interactions between the baby and caregivers.\n"
        "   - Abnormal Videos:\n"
        "     - Baby nearing dangerous zones (e.g., staircases, swimming pools) without adult supervision.\n"
        "     - Baby wandering outside a crib, stroller, or designated play area without adult presence.\n"
        "     - Sudden, unexpected falls that may lead to injury.\n"
        "     - Any abusive behavior toward the baby, such as hitting, or forcing them to act against their will.\n"
        "5. Kid Monitoring\n"
        "   - Normal Videos:\n"
        "     - Kids playing or moving around indoors or outdoors within designated areas.\n"
        "     - Regular daily activities under adult supervision.\n"
        "   - Abnormal Videos:\n"
        "     - Kids found wandering outdoors or in dangerous locations without adult supervision.\n"
        "     - Dangerous actions indoors (e.g., playing with sharp objects, accessing restricted areas) or significant health/safety concerns (e.g., choking hazards).\n"
        "     - Sudden, unexpected falls that may lead to injury.\n"
        "6. Senior Care\n"
        "   - Normal Videos:\n"
        "     - Seniors engaging in routine activities like walking, resting, or interacting with caregivers or family.\n"
        "   - Abnormal Videos:\n"
        "     - Sudden, unexpected falls that may lead to injury.\n"
        "     - Signs of distress or calls for help through hand gestures or unusual body language.\n"
        "     - Any abusive or rough behavior by caregivers toward seniors, including verbal and physical abuse.\n"
        "7. Other Categories\n"
        "   - Normal Videos:\n"
        "     - Videos that do not fit any of the above categories but show harmless, everyday activities, such as trees waving, normal weather events, or background motion.\n"
        "   - Abnormal Videos:\n"
        "     - Severe weather conditions or natural disasters like fires, earthquakes, floods, or storms causing property damage or safety hazards.\n"
        "     - Unexplained phenomena of inanimate objects.\n"
        "     - Sudden, unexpected falls of inanimate objects that may cause damage or injury.\n"
        "     - Irregular activities that do not fit into other categories but may pose risks or concerns.\n"
        "Instructions:\n"
        "Please analyze the video and provide your response in the following format:\n"
        "{\n"
        '  "video_description": "A concise description of the video content, including objects, movements, and environmental conditions (max 200 words)",\n'
        '  "reasoning": "Detailed reasoning for why the situation is considered abnormal or concerning, if applicable (max 100 words)",\n'
        '  "anomaly": 0 // 0 for no anomaly detected, 1 for anomaly detected\n'
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
    jsonl_filename = os.path.join(output_dir, f'responses_{model_type}_icl_1203.jsonl')
    json_filename = os.path.join(output_dir, f'responses_{model_type}_icl_1203.json')

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
# Example usage
if __name__ == "__main__":
    # Get the current directory of the script
    current_dir = os.getcwd()
    directory = os.path.join(current_dir, "../downloads")  # Directory where videos are saved
    batch_size = 10  # Adjust batch size as needed
    model_type = 'gpt4o'
    filename, df_time = main_part(directory, model_type, batch_size)
