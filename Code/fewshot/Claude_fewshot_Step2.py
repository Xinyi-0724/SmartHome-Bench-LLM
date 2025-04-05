#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This code is the step 2 designed to test Claude 3.5 performace on anomaly detection
# In step 2, we analyze the jsonl file generated in step 1 to obtain the accuracy of the LLM models
# In this version we analyze the accuracy of the anomaly
# Test dataset: 1203 videos in local "downloads" folder

import json
import pandas as pd
import numpy as np
import re

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


# In[2]:


def extract_json(filename):
    anomalies = {}

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

                        # Detect if there's JSON embedded in the response
                        json_match = re.search(r'{.*}', response_text, re.DOTALL)
                        
                        if json_match:
                            # Extract the JSON portion and clean comments
                            json_content = json_match.group()

                            # Remove JavaScript-style comments from the JSON content
                            json_content = re.sub(r'//.*', '', json_content)

                            # Parse the cleaned response text as JSON
                            response_json = json.loads(json_content)

                            # Extract the result value
                            anomaly_value = response_json.get('anomaly', 'NAN')
                        else:
                            anomaly_value = "NAN"  # No valid JSON found

                    except (json.JSONDecodeError, AttributeError):
                        # If response text is not a valid JSON or another error occurs, set value to "NAN"
                        anomaly_value = "NAN"

                    # Store the anomaly value
                    anomalies[video] = anomaly_value

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from line: {line.strip()} - {e}")
                continue

    return anomalies


# In[3]:


def is_equal(val1, val2):
    # Handle 'NAN' values
    if str(val1).upper() == 'NAN' or str(val2).upper() == 'NAN':
        return False
    try:
        return int(val1) == int(val2)
    except ValueError:
        return False

# record the groud truth for each video VAD
def load_ground_truth(csv_filename):
    df = pd.read_csv(csv_filename)
    label_mapping = {
        'Normal': 0,
        'Abnormal': 1,
        'Vague Abnormal': 1
    }
    df['True Label'] = df['Label'].map(label_mapping)
    return df[['Title', 'Category', 'Label', 'True Label']]


# In[4]:


def calculate_metrics(df):
    cm = confusion_matrix(df['True Label'], df['Predicted Label'])
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(df['True Label'], df['Predicted Label'])
    recall = recall_score(df['True Label'], df['Predicted Label'])
    f1 = f1_score(df['True Label'], df['Predicted Label'])

    total_abnormal_videos = df[df['True Label'] == 1].shape[0]
    total_normal_videos = df[df['True Label'] == 0].shape[0]

    accuracy_abnormal = df[df['True Label'] == 1]['Accuracy'].mean() if total_abnormal_videos > 0 else 0.0
    accuracy_normal = df[df['True Label'] == 0]['Accuracy'].mean() if total_normal_videos > 0 else 0.0

    # Calculate overall accuracy
    overall_accuracy = df['Accuracy'].mean()

    return accuracy_abnormal, accuracy_normal, overall_accuracy, precision, recall, f1, cm

# summarize vad accuracy for two difficulty levels
def summarize_anomaly_accuracy(anomalies, ground_truth_df):
    df = pd.DataFrame({
        'Video Name': list(anomalies.keys()),
        'Predicted Label': list(anomalies.values())
    })

    # Ensure Video Name column does not have the .mp4 extension for matching with ground truth titles
    df['Video Name'] = df['Video Name'].str.replace('.mp4', '')
    
    # Perform the merge with the ground truth DataFrame
    df = df.merge(ground_truth_df, left_on='Video Name', right_on='Title')

    df['Accuracy'] = df.apply(lambda row: is_equal(row['True Label'], row['Predicted Label']), axis=1).astype(int)

    df['Predicted Label'] = np.where(df['Predicted Label'].astype(str).str.upper() == 'NAN',
                                     1 - df['True Label'],
                                     df['Predicted Label']).astype(int)

    all_metrics = calculate_metrics(df)
    normal_abnormal_df = df[df['Label'].isin(['Normal', 'Abnormal'])]
    normal_abnormal_metrics = calculate_metrics(normal_abnormal_df)
    vague_df = df[df['Label'].isin(['Vague Abnormal'])]
    vague_metrics = vague_df['Accuracy'].mean()

    return df, all_metrics, normal_abnormal_metrics, vague_metrics

# summarize vad accuracy for each category
def summarize_category_accuracy(df, categories):
    metrics = {}
    
    # Iterate over each category and calculate metrics
    for category in categories:
        # Filter DataFrame based on the category
        category_df = df[df['Category'].apply(lambda x: category in x)]
        
        if not category_df.empty:
            metrics[category] = calculate_metrics(category_df)
        else:
            metrics[category] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None)  # Default metrics for empty category
    
    return metrics


# In[5]:


# Example usage
if __name__ == "__main__":
    model_type = 'claude-3-5-sonnet'
    json_filename = f'response/responses_{model_type}_few_1203.jsonl'
    csv_filename = 'Annotation_vad_1203.csv'

    anomalies = extract_json(json_filename)
    ground_truth_df = load_ground_truth(csv_filename)

    df_accuracy, all_metrics, normal_abnormal_metrics, vague_metrics = summarize_anomaly_accuracy(anomalies, ground_truth_df)
    
    categories = ["Baby Monitoring", "Wildlife", "Security", "Pet Monitoring", "Senior Care", "Kid Monitoring", "Other Category"]
    Categorical_metrics = summarize_category_accuracy(df_accuracy, categories)
    
    print("All Videos Metrics (accuracy_abnormal, accuracy_normal, overall_accuracy, precision, recall, f1, cm):", all_metrics)
    print("Normal and Abnormal Videos Metrics (accuracy_abnormal, accuracy_normal, overall_accuracy, precision, recall, f1, cm):", normal_abnormal_metrics)
    print("Vague Normal and Vague Abnormal Videos Metrics (accuracy_abnormal/overall_accuracy):", vague_metrics)
    print("Categorical Metrics (accuracy_abnormal, accuracy_normal, overall_accuracy, precision, recall, f1, cm):", Categorical_metrics)
    
    # save results of VAD for major voting
    df_accuracy.to_csv(f'few_accuracy_{model_type}.csv', index=False)
