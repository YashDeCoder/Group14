import pandas as pd
import numpy as np
import os
from pathlib import Path
from scipy.special import erfc
import re
from ClassicalML import ClassicalML

bmi_values = {
    "F6": 24.7,
    "M6": 20.8,
    "F5": 17.9,
    "M5": 23.5,
    "F1": 20.8,
    "M1": 23.1,
    "M3": 25.3,
    "M4": 20.9,
    "F2": 22.3,
    "M2": 24.1,
    "F3": 20.7,
    "F4": 22.3,
    "M7": 22.1,
    "M8": 23.0,
    "M9": 23.7,
    "M10": 23.6,
    "M11": 21.6
}

def read_summary_data(directory):
    data = {}
    pattern = re.compile(r'[MF](\d+)-(\d+)')
    
    for folder_name in os.listdir(directory):
        match = pattern.match(folder_name)
        if match:
            folder_path = os.path.join(directory, folder_name)
            acc_file = os.path.join(folder_path, 'summary-acc.csv')
            gyro_file = os.path.join(folder_path, 'summary-gyro.csv')
            if os.path.exists(acc_file) and os.path.exists(gyro_file):
                df_acc = pd.read_csv(acc_file).dropna()
                df_gyro = pd.read_csv(gyro_file).dropna()
                combined_df = pd.concat([df_acc, df_gyro], axis=1)
                data[folder_name] = combined_df
            else:
                print(f"Summary files missing for {folder_name}")
    return data

def read_labeled_data(directory, sensor_type):
    data = {}
    pattern = re.compile(r'[MF](\d+)-(\d+)')
    
    for folder_name in os.listdir(directory):
        match = pattern.match(folder_name)
        if match:
            folder_path = os.path.join(directory, folder_name)
            sensor_file = os.path.join(folder_path, f'{sensor_type}-agg-labeled.csv')
            if os.path.exists(sensor_file):
                df = pd.read_csv(sensor_file).dropna()
                data[folder_name] = df
    return data

def combine_data_with_primary_gyro_label(acc_data, gyro_data):
    combined_data = {}
    pattern = re.compile(r'[MF](\d+)-(\d+)')
    for participant_session in acc_data:
        if participant_session in gyro_data:
            df_acc = acc_data[participant_session]
            df_gyro = gyro_data[participant_session]

            # Ensuring timestamps are aligned
            df_acc['Timestamps'] = pd.to_datetime(df_acc['Timestamps'], unit='ns')
            df_gyro['Timestamps'] = pd.to_datetime(df_gyro['Timestamps'], unit='ns')
            df_acc.set_index('Timestamps', inplace=True)
            df_gyro.set_index('Timestamps', inplace=True)

            # Reindex gyro data to the nearest acc timestamps
            df_combined = df_acc.join(df_gyro, how='outer', lsuffix='_acc', rsuffix='_gyro').interpolate(method='nearest').dropna().reset_index()
            
            # Use Label_gyro as primary label (as we used gyro data for break detection as well)
            df_combined['Label'] = df_combined['Label_gyro']

            # Select only required columns
            df_combined = df_combined[['Timestamps', 'Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Label']]

            # Int representation of the timestamp in Unix nanoseconds
            df_combined['Timestamps'] = df_combined['Timestamps'].astype(np.int64) // 10**6 * 10**6

            # Adding gender column
            match = pattern.match(participant_session)
            if match.group(0)[0] == 'M':
                female_male = 0
            else:
                female_male = 1
            df_combined['Gender'] = female_male

            # Adding BMI
            bmi = match.group(0)[0] + match.group(1)
            df_combined['BMI'] = bmi_values.get(bmi)

            combined_data[participant_session] = df_combined
        else:
            print(f"No gyroscope data for {participant_session}")
    return combined_data

def chauvenet_criterion(N, axis_col, mean, stdv):
    criterion = 1.0 / (2 * N)  # Chauvenet's criterion
    d = abs(axis_col - mean) / stdv  # Distance of a value to mean in stdv's
    prob = erfc(d)  # Area normal dist.    
    mask = prob >= criterion
    return axis_col[mask]

def clean_data(df, stats_df):
    N = len(df)
    for axis in ['X', 'Y', 'Z']:
        acc_mean_col = stats_df[f'Mean Acc_{axis}'].values[0]
        acc_std_dev_col = stats_df[f'Std Dev Acc_{axis}'].values[0]
        gyro_mean_col = stats_df[f'Mean Gyro_{axis}'].values[0]
        gyro_std_dev_col = stats_df[f'Std Dev Gyro_{axis}'].values[0]

        df[f'Acc_{axis}_cleaned'] = chauvenet_criterion(N, df[f'Acc_{axis}'], acc_mean_col, acc_std_dev_col)
        df[f'Gyro_{axis}_cleaned'] = chauvenet_criterion(N, df[f'Gyro_{axis}'], gyro_mean_col, gyro_std_dev_col)
    return df.dropna()

def save_combined_cleaned_data(cleaned_data, output_dir):
    arrayCleaned = []
    for participant_session, df in cleaned_data.items():
        output_session_dir = os.path.join(output_dir, participant_session)
        os.makedirs(output_session_dir, exist_ok=True)
        
        combined_file = os.path.join(output_session_dir, 'combined-agg-cleaned.csv')

        # Select only the cleaned columns and primary label
        df_cleaned = df[['Acc_X_cleaned', 'Acc_Y_cleaned', 'Acc_Z_cleaned', 'Gyro_X_cleaned', 'Gyro_Y_cleaned', 'Gyro_Z_cleaned', 'Label', 'Gender', 'BMI']]
        
        df_cleaned.to_csv(combined_file, index=False)
        
        arrayCleaned.append(df_cleaned)
    return list(arrayCleaned)


directory = 'labeled-data'
output_dir = 'cleaned-data'

acc_data = read_labeled_data(directory, 'Accelerometer')
gyro_data = read_labeled_data(directory, 'Gyroscope')

combined_data = combine_data_with_primary_gyro_label(acc_data, gyro_data)

summary_directory = 'summary-data'
summary_data = read_summary_data(summary_directory)

# Clean data
cleaned_data = {}
for participant_session, df in combined_data.items():
    stats_df = summary_data.get(participant_session)
    if stats_df is not None:
        cleaned_data[participant_session] = clean_data(df, stats_df)
    else:
        print(f"No summary data for {participant_session}")

# Save cleaned data
arrayCleanedData = save_combined_cleaned_data(cleaned_data, output_dir)

# Start of Classical Training
labels = ['Label']
c_ml = ClassicalML
df_train_X, df_test_X, df_train_Y, df_test_Y = c_ml.split_multiple_datasets_classification(c_ml,arrayCleanedData, labels, '', 0.7, unknown_users=True, temporal=True)
print(df_train_Y)