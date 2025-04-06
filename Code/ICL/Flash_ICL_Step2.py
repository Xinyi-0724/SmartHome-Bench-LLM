import json
import pandas as pd
import numpy as np
import re

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
def extract_json(filename):
    anomalies = {}

    # Read the JSON file line by line
    with open(filename, 'r') as file:
        for line in file:
            try:
                # Search for \"updated_anomaly\" and capture its value
                anomaly_match = re.search(r'\\\"anomaly\\\"\s*:\s*(\d+)', line)
                
                if anomaly_match:
                    # Extract the value after \"updated_anomaly\"
                    anomaly_value = anomaly_match.group(1)
                else:
                    anomaly_value = "NAN"  # No \"updated_anomaly\" found

                # Extract the video key from before the first colon
                video_key_match = re.match(r'^\{\\?"([^"]+)"\\?\s*:', line)
                
                if video_key_match:
                    video_key = video_key_match.group(1)
                    anomalies[video_key] = anomaly_value
                else:
                    # If the video key cannot be extracted, store the anomaly value under 'unknown'
                    anomalies['unknown'] = anomaly_value
                    print(f"Could not extract video key from line: {line.strip()}")

            except Exception as e:
                print(f"Error processing line: {line.strip()} - {e}")
                continue

    return anomalies

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
# Example usage
if __name__ == "__main__":
    model_type = 'flash'
    json_filename = f'response/responses_{model_type}_icl_1203.jsonl'
    csv_filename = 'Annotation_vad_1203.csv'

    anomalies = extract_json(json_filename)
    ground_truth_df = load_ground_truth(csv_filename)

    df_accuracy, all_metrics, normal_abnormal_metrics, vague_metrics = summarize_anomaly_accuracy(anomalies, ground_truth_df)
    
    categories = ["Baby Monitoring", "Wildlife", "Security", "Pet Monitoring", "Senior Care", "Kid Monitoring", "Other Category"]
    Categorical_metrics = summarize_category_accuracy(df_accuracy, categories)
    
    print("All Videos Metrics (accuracy_abnormal, accuracy_normal, overall_accuracy, precision, recall, f1, cm):", all_metrics)
    print("Normal and Abnormal Videos Metrics (accuracy_abnormal, accuracy_normal, overall_accuracy, precision, recall, f1, cm):", normal_abnormal_metrics)
    print("Vague Abnormal Videos Metrics (overall_accuracy):", vague_metrics)
    print("Categorical Metrics (accuracy_abnormal, accuracy_normal, overall_accuracy, precision, recall, f1, cm):", Categorical_metrics)
    
    # save results of VAD for major voting
    df_accuracy.to_csv(f'cot_accuracy_{model_type}.csv', index=False)
