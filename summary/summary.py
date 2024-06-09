import pandas as pd
import numpy as np
import os
from pathlib import Path

def calculate_summary_statistics(df):
    summary = {}
    for axis in ['X', 'Y', 'Z']:
        cols = [c for c in df.columns if c.endswith(axis)]
        for col in cols:
            base_name = col.split('_')[0]  # Get the base name (e.g., Acc, Gyro)
            summary[f'Mean {col}'] = df[col].mean()
            summary[f'Std Dev {col}'] = df[col].std()
            summary[f'Min {col}'] = df[col].min()
            summary[f'Max {col}'] = df[col].max()
    return summary

def process_aggregated_data(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    for session_folder in input_path.iterdir():
        if session_folder.is_dir():
            # Create corresponding directory in the data-summary folder
            summary_session_folder = output_path / session_folder.name
            summary_session_folder.mkdir(parents=True, exist_ok=True)

            for sensor_file in session_folder.iterdir():
                if sensor_file.name.endswith('-agg.csv'):
                    # Read the aggregated data
                    df = pd.read_csv(sensor_file)

                    # Calculate summary statistics
                    summary_stats = calculate_summary_statistics(df)
                    
                    # Create a DataFrame for the summary statistics
                    summary_df = pd.DataFrame(summary_stats, index=[0])

                    # Determine output file name
                    if sensor_file.name.startswith('Accelerometer'):
                        output_file = summary_session_folder / 'summary-acc.csv'
                    elif sensor_file.name.startswith('Gyroscope'):
                        output_file = summary_session_folder / 'summary-gyro.csv'
                    else:
                        continue

                    # Save the summary statistics to a CSV file
                    summary_df.to_csv(output_file, index=False)
                    print(f"Summary saved to {output_file}")

output_dir = 'summary-data'
os.makedirs(output_dir, exist_ok=True)

process_aggregated_data('aggregated-data', output_dir)
