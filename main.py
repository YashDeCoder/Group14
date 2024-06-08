import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import re
import os

def read_and_combine_datasets(directory):
    # Initialize a dictionary to store dataframes for each xx value
    combined_dataframes = {'0': [], '10': [], '30': [], '60': []}
    
    # Regular expression to match directory names with pattern M$-xx or F$-xx
    pattern = re.compile(r'[MF](\d+)-(\d+)')
    
    # Iterate through directories in the given directory
    for folder_name in os.listdir(directory):
        match = pattern.match(folder_name)
        xx = match.group(2) #Extract break time
        folder_path = os.path.join(directory, folder_name)
        accel_file = os.path.join(folder_path, 'Accelerometer.csv')
        gyro_file = os.path.join(folder_path, 'Gyroscope.csv')
        df_accel = pd.read_csv(accel_file)
        df_gyro = pd.read_csv(gyro_file)
        # Combining everything
        combined_df = pd.merge(df_accel, df_gyro, on='Time (s)', suffixes=('_accel', '_gyro'))
        if xx in combined_dataframes:
            combined_dataframes[xx].append(combined_df)
    
    # Concatenate dataframes for each xx value into a single dataframe per xx value
    concatenated_dataframes = {}
    for xx, dfs in combined_dataframes.items():
        if dfs: 
            concatenated_dataframes[xx] = pd.concat(dfs, ignore_index=True)
    
    return concatenated_dataframes

# Example usage
directory = 'data'
dfs_by_xx = read_and_combine_datasets(directory)

# Access dataframes for each xx value
df_0 = dfs_by_xx['0']
df_10 = dfs_by_xx['10']
df_30 = dfs_by_xx['30']
df_60 = dfs_by_xx['60']
print(df_0.head(10))
