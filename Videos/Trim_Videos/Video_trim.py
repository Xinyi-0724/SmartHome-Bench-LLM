import os
import shutil
import pandas as pd
from moviepy.video.io.VideoFileClip import VideoFileClip

# Step 1: Read the CSV file and filter records based on "Time" column
csv_file = 'Trim_video_label.csv'
df = pd.read_csv(csv_file)

# Filter out rows where "Time" is NaN
df = df.dropna(subset=['Time'])

# Divide into three groups based on the value of "Time"
df_video_ad_end = df[df['Time'] > 0]
df_video_ad_begin = df[df['Time'] < 0]
df_video_no_trim = df[df['Time'] == 0]

# Step 2: Locate the corresponding videos in the "wyze_video" folder
input_folder = 'wyze_video'
output_folder = 'wyze_trimmed'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# Step 4: Trim the videos based on the "Time" value
def trim_video(input_path, output_path, start_time, end_time):
    with VideoFileClip(input_path) as video:
        trimmed_video = video.subclip(start_time, end_time)
        trimmed_video.write_videofile(output_path, codec="libx264")

# Function to handle trimming
def process_videos(df, folder, trim_type='end'):
    for index, row in df.iterrows():
        video_name = row['Title'] + '.mp4'
        time_value = row['Time']
        input_path = os.path.join(folder, video_name)
        output_path = os.path.join(output_folder, video_name)
        
        if os.path.exists(input_path):
            if trim_type == 'end':
                with VideoFileClip(input_path) as video:
                    duration = video.duration
                    trim_video(input_path, output_path, 0, duration - time_value)
            elif trim_type == 'begin':
                trim_video(input_path, output_path, -time_value, None)
        else:
            print(f"Video {video_name} not found in the folder {folder}.")


# Function to handle copying videos without trimming
def copy_videos(df, folder):
    for index, row in df.iterrows():
        video_name = row['Title'] + '.mp4'
        input_path = os.path.join(folder, video_name)
        output_path = os.path.join(output_folder, video_name)
        
        if os.path.exists(input_path):
            shutil.copy(input_path, output_path)
        else:
            print(f"Video {video_name} not found in the folder {folder}.")
            
# Process df_video_ad_end, df_video_ad_begin, and df_video_no_trim
process_videos(df_video_ad_end, input_folder, trim_type='end')
process_videos(df_video_ad_begin, input_folder, trim_type='begin')
copy_videos(df_video_no_trim, input_folder)

print("Video processing completed.")