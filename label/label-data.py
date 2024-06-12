import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

def identify_breaks(sensor_data, relevant_columns, window_size=25, variance_threshold=0.01):
    sensor_data['Variance'] = sensor_data[relevant_columns].var(axis=1)
    sensor_data['Rolling_Var'] = sensor_data['Variance'].rolling(window=window_size).mean()

    break_indices = sensor_data.index[sensor_data['Rolling_Var'] < variance_threshold].tolist()
    return break_indices

def label_sets(sensor_data, break_indices, rest_interval):
    break_indices = [-1] + break_indices + [len(sensor_data)]
    set_counter = 1
    
    for i in range(len(break_indices) - 1):
        if set_counter > 3:
            break
        start_idx = break_indices[i] + 1
        end_idx = break_indices[i + 1]
        
        if end_idx - start_idx > 0:
            sensor_data.loc[start_idx:end_idx, 'Label'] = set_counter
            set_counter += 1

    # Drop columns if they exist
    for col in ['Variance', 'Rolling_Var']:
        if col in sensor_data.columns:
            sensor_data = sensor_data.drop(columns=[col])
    
    return sensor_data

def label_and_save_data(aggregated_data_base_dir, labeled_data_base_dir):
    for session_folder in Path(aggregated_data_base_dir).iterdir():
        if session_folder.is_dir():
            participant_session = session_folder.name
            participant, rest_interval = participant_session.split('-')
            rest_interval = rest_interval.split('s')[0]
            
            acc_file = session_folder / 'Accelerometer-agg.csv'
            gyro_file = session_folder / 'Gyroscope-agg.csv'

            if acc_file.exists() and gyro_file.exists():
                acc_data = pd.read_csv(acc_file)
                gyro_data = pd.read_csv(gyro_file)

                relevant_columns = [col for col in gyro_data.columns if col.startswith('Gyro_')]
                
                # Detect and remove break times using gyroscope data (it is better for this purpose)
                if rest_interval != '0':
                    break_indices = identify_breaks(gyro_data, relevant_columns)
                    acc_data = label_sets(acc_data, break_indices, rest_interval)
                    gyro_data = label_sets(gyro_data, break_indices, rest_interval)
                else:
                    # For 0s rest intervals, split into 3 equal parts
                    total_length = len(acc_data)
                    part_length = total_length // 3
                    acc_data['Label'] = np.nan
                    gyro_data['Label'] = np.nan
                    acc_data.loc[:part_length, 'Label'] = 1
                    acc_data.loc[part_length:2*part_length, 'Label'] = 2
                    acc_data.loc[2*part_length:, 'Label'] = 3
                    gyro_data.loc[:part_length, 'Label'] = 1
                    gyro_data.loc[part_length:2*part_length, 'Label'] = 2
                    gyro_data.loc[2*part_length:, 'Label'] = 3
                
                # Drop rows that are not labeled
                acc_data = acc_data.dropna(subset=['Label'])
                gyro_data = gyro_data.dropna(subset=['Label'])
                
                # Convert Label column to integer type
                acc_data['Label'] = acc_data['Label'].astype(int)
                gyro_data['Label'] = gyro_data['Label'].astype(int)
                
                output_acc_file = labeled_data_base_dir / session_folder.name / 'Accelerometer-agg-labeled.csv'
                output_gyro_file = labeled_data_base_dir / session_folder.name / 'Gyroscope-agg-labeled.csv'
                output_acc_file.parent.mkdir(exist_ok=True, parents=True)
                acc_data.to_csv(output_acc_file, index=False)
                gyro_data.to_csv(output_gyro_file, index=False)
                print(f"Labeled data saved to {output_acc_file}")
                print(f"Labeled data saved to {output_gyro_file}")

def plot_labeled_data(sensor_data, output_file):
    # Remove NaN labels
    sensor_data = sensor_data.dropna(subset=['Label'])
    
    # Define colors for each set
    colors = {1: 'blue', 2: 'green', 3: 'red'}
    
    plt.figure(figsize=(12, 8))
    unique_labels = sensor_data['Label'].unique()
    
    for label in unique_labels:
        if label not in colors:
            print(f"Warning: Unknown label {label} found in the data.")
            continue
        
        subset = sensor_data[sensor_data['Label'] == label]
        color = colors[label]
        for col in ['Acc_X', 'Acc_Y', 'Acc_Z']:
            if col in subset.columns:
                plt.plot(subset['Timestamps'], subset[col], label=f'{col} {label}', color=color)
        for col in ['Gyro_X', 'Gyro_Y', 'Gyro_Z']:
            if col in subset.columns:
                plt.plot(subset['Timestamps'], subset[col], linestyle='--', label=f'{col} {label}', color=color)
    
    plt.title(f'Labeled Sensor Data')
    plt.xlabel('Timestamps')
    plt.ylabel('Sensor Readings')
    plt.legend()
    plt.grid(True)
    plot_file = output_file.parent / f'{output_file.stem}.png'
    plt.savefig(plot_file)
    plt.close()

# Base directories
aggregated_data_base_dir = Path('aggregated-data')
labeled_data_base_dir = Path('labeled-data')

# Ensure labeled data directory exists
labeled_data_base_dir.mkdir(exist_ok=True, parents=True)

label_and_save_data(aggregated_data_base_dir, labeled_data_base_dir)

# Plot labeled data for verification
# for session_folder in labeled_data_base_dir.iterdir():
#     if session_folder.is_dir():
#         for labeled_file in session_folder.iterdir():
#             if labeled_file.name.endswith('-labeled.csv'):
#                 sensor_data = pd.read_csv(labeled_file)
#                 plot_labeled_data(sensor_data, labeled_file)
