import pandas as pd
import numpy as np
import os
from pathlib import Path
from scipy.special import erfc
import re
from ClassicalML import ClassicalML
from ClassificationEvaluation import ClassificationEvaluation
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout,Flatten,Input
import matplotlib.pyplot as plt
from keras.layers import LSTM,Dense,Dropout,MaxPooling1D,TimeDistributed,Conv1D
from keras.models import load_model
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error

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
            df_combined['Set_Time'] = df_combined['Set_Time_acc']

            # Select only required columns
            df_combined = df_combined[['Timestamps', 'Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Label', 'Set_Time']]

            # Int representation of the timestamp in Unix nanoseconds
            df_combined['Timestamps'] = df_combined['Timestamps'].astype(np.int64) // 10**6 * 10**6
            df_combined['Set_Time'] = df_combined['Set_Time'].astype(np.int64) // 10**6 * 10**6

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
        df_cleaned = df[['Acc_X_cleaned', 'Acc_Y_cleaned', 'Acc_Z_cleaned', 'Gyro_X_cleaned', 'Gyro_Y_cleaned', 'Gyro_Z_cleaned', 'Label', 'Gender', 'BMI', 'Set_Time']]
        
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
df_train_X, df_test_X, df_train_Y, df_test_Y = c_ml.split_multiple_datasets_classification(c_ml,arrayCleanedData, labels, '', 0.7, unknown_users=True)

# pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y = c_ml.random_forest(c_ml, df_train_X, df_train_Y, df_test_X, print_model_details=True)

# # pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y = c_ml.naive_bayes(c_ml, df_train_X, df_train_Y, df_test_X)

# evaluator = ClassificationEvaluation

# # Compute metrics BUT THIS DOESN'T WORK
# test_accuracy = evaluator.accuracy(evaluator, df_test_Y, pred_test_y)
# test_precision = evaluator.precision(evaluator, df_test_Y, pred_test_y)
# test_recall = evaluator.recall(evaluator, df_test_Y, pred_test_y)
# test_f1 = evaluator.f1(evaluator, df_test_Y, pred_test_y)

# # Create a DataFrame to hold the results
# metrics_df = pd.DataFrame({
#     'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
#     'Test Set': [
#         test_accuracy, 
#         test_precision.mean(),  # Average precision across classes
#         test_recall.mean(),     # Average recall across classes
#         test_f1.mean(),         # Average F1 score across classes
#     ]
# })

# # Display the results
# # print(metrics_df)

# # Display detailed per-class metrics if needed
# detailed_metrics_df = pd.DataFrame({
#     'Class': [1, 2, 3],
#     'Test Precision': test_precision,
#     'Test Recall': test_recall,
#     'Test F1 Score': test_f1
# })

# print(detailed_metrics_df)

# Using LSTM or TCN

# Create the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(df_train_X.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(df_train_X, df_train_Y, epochs=50, batch_size=32, verbose=2)

# Make predictions
predictions = model.predict(df_test_X)

# Evaluate the model
mse = model.evaluate(df_test_X, df_test_Y)
print(f'MSE: {mse}')

# Plot the results
# plt.plot(y_test, label='Actual')
# plt.plot(predictions, label='Predicted')
# plt.legend()
# plt.show()

# Plot the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(df_test_Y, label='Actual')
plt.plot(predictions, label='Predicted')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Actual vs Predicted Stock Prices')
plt.legend()
plt.show()
