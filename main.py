import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import re
import os
from scipy.special import erfc

# TODO: Remove the breaks so that we get better results

def chauvenet_criterion(N, axis_acc_col, axis_gyro_col, acc_mean, acc_stdv, gyro_mean,gyro_stdv):
    criterion = 1.0/(2*N)         # Chauvenet's criterion
    # Removing Acc outliers
    d_acc = abs(axis_acc_col-acc_mean)/acc_stdv      # Distance of a value to mean in stdv's
    d_gyro = abs(axis_gyro_col-gyro_mean)/gyro_stdv      # Distance of a value to mean in stdv's
    prob_acc = erfc(d_acc)                 # Area normal dist.    
    prob_gyro = erfc(d_gyro)                 # Area normal dist.    
    
    # Identify data points that meet Chauvenet's criterion
    mask_acc = prob_acc >= criterion
    mask_gyro = prob_gyro >= criterion
    
    return axis_acc_col[mask_acc], axis_gyro_col[mask_gyro]

def filter_first_and_last_second(df, xx):
    sorted_df = df.sort_values(by='Timestamps')
    if xx == '0':
        end_time = 37.28
    elif xx == '10':
        end_time = 59.28
    elif xx == '30':
        end_time = 103.55
    else:
        end_time = 175.28
    # Filter rows for the first and last second
    first_second = sorted_df[sorted_df['Timestamps'] >= 1.0 ]
    last_second = sorted_df[sorted_df['Timestamps'] <= end_time]
    filtered_df = pd.concat([first_second, last_second])
    return filtered_df

def read_data(directory):
    # Initialize a dictionary to store dataframes for each xx value
    people = {}
    
    # Regular expression to match directory names with pattern M$-xx or F$-xx
    pattern = re.compile(r'[MF](\d+)-(\d+)')
    
    # Iterate through directories in the given directory
    for folder_name in os.listdir(directory):
        match = pattern.match(folder_name)
        person_xx = match.group(0)
        xx = match.group(2)
        if match:
            folder_path = os.path.join(directory, folder_name)
            # Combining everything into one dataset based off of directory
            if directory == 'aggregated-data':
                accel_file = os.path.join(folder_path, 'Accelerometer-agg.csv')
                gyro_file = os.path.join(folder_path, 'Gyroscope-agg.csv')
                # TODO: Figure out why merge returns empty values
                # df_accel = pd.read_csv(accel_file).dropna()
                # df_gyro = pd.read_csv(gyro_file).dropna()
                # combined_df = pd.merge(df_accel, df_gyro, on='Timestamps', suffixes=('_accel','_gyro'))
                # # Dropping values from the beginning and end then adding the dataset
                # combined_df = filter_first_and_last_second(combined_df, xx)     
            else:
                accel_file = os.path.join(folder_path, 'summary-acc.csv')
                gyro_file = os.path.join(folder_path, 'summary-gyro.csv')
            df_accel = pd.read_csv(accel_file).dropna()
            df_gyro = pd.read_csv(gyro_file).dropna()
            combined_df = pd.concat([df_accel, df_gyro], axis=1)
            # combined_df = filter_first_and_last_second(combined_df, xx)     
            people[str(person_xx)] = combined_df
    return people

def combine_aggregated_data(directory):
    # Initialize a dictionary to store dataframes for each xx value
    combined_dataframes = {'0': [], '10': [], '30': [], '60': []}
    
    # Regular expression to match directory names with pattern M$-xx or F$-xx
    pattern = re.compile(r'[MF](\d+)-(\d+)')
    
    # Iterate through directories in the given directory
    for folder_name in os.listdir(directory):
        match = pattern.match(folder_name)
        if match:
            folder_path = os.path.join(directory, folder_name)
            accel_file = os.path.join(folder_path, 'Accelerometer-agg.csv')
            gyro_file = os.path.join(folder_path, 'Gyroscope-agg.csv')
            df_accel = pd.read_csv(accel_file)
            df_gyro = pd.read_csv(gyro_file)
            # Combining everything
            combined_df = pd.merge(df_accel, df_gyro, on='Timestamps')
            xx = match.group(2) # Extract break time
            if xx in combined_dataframes:
                combined_dataframes[xx].append(combined_df)
    
    # Concatenate dataframes for each xx value into a single dataframe per xx value
    concatenated_dataframes = {}
    for xx, dfs in combined_dataframes.items():
        if dfs: 
            concatenated_dataframes[xx] = pd.concat(dfs, ignore_index=True)
    
    return concatenated_dataframes

def plot_categories(dfs_by_xx, xx):
    plt.figure(figsize=(30, 8))
    plt.title(f'Categories Plot after noise reduction - xx={xx}')
    time_column = 'Timestamps'
    
    # List of columns to plot
    categories = [col for col in dfs_by_xx.columns if col != time_column]

    for category in categories:
        plt.plot(dfs_by_xx[time_column], dfs_by_xx[category], label=category)
    
    plt.xlabel('Timestamps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

directory = 'summary-data'
dfs_stats_by_person_xx = read_data(directory)
directory = 'aggregated-data'
dfs_people = read_data(directory)

for people in dfs_people:
    person_df = dfs_people[people]
    stats_personxx_df = dfs_stats_by_person_xx[people]
    # Iterate over gyroscopic data columns (Gyro_X, Gyro_Y, Gyro_Z)
    for axis in ['X', 'Y', 'Z']:
        # Extract summary statistics for the current axis
        gyro_mean_col = stats_personxx_df[f'Mean Gyro_{axis}']
        acc_mean_col = stats_personxx_df[f'Mean Acc_{axis}']
        gyro_std_dev_col = stats_personxx_df[f'Std Dev Gyro_{axis}']
        acc_std_dev_col = stats_personxx_df[f'Std Dev Acc_{axis}']
        gyro_axis_col = person_df[f'Gyro_{axis}']
        acc_axis_col = person_df[f'Acc_{axis}']
        N = len(person_df)
        
        # Apply Chauvenet's criterion to the current axis
        acc_axis_col, gyro_axis_col = chauvenet_criterion(N, acc_axis_col,gyro_axis_col,acc_mean_col,acc_std_dev_col,gyro_mean_col,gyro_std_dev_col)
        person_df[f'Gyro_{axis}'] = gyro_axis_col
        person_df[f'Acc_{axis}'] = acc_axis_col
        if people == 'M1-0':
            plot_categories(person_df, '0')

# directory = 'aggregated-data'
# dfs_by_xx = read_and_combine_datasets(directory)
# # Access dataframes for each xx value
# df_0 = dfs_by_xx['0']
# df_10 = dfs_by_xx['10']
# df_30 = dfs_by_xx['30']
# df_60 = dfs_by_xx['60']
# # plot_categories(dfs_by_xx)