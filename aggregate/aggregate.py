import pandas as pd
import numpy as np
import re
import copy
import os
from datetime import datetime, timedelta
from pathlib import Path

class CreateDataset:

    base_dir = ''
    granularity = 0
    data_table = None

    def __init__(self, base_dir, granularity):
        self.base_dir = Path(base_dir)  
        self.granularity = granularity * 1e6  # granularity in nanoseconds

    # Create an initial data table with entries from start till end time, with steps of size granularity.
    def create_timestamps(self, start_time, end_time):
        print(f"Creating timestamps from {start_time} to {end_time} with granularity {self.granularity} ns")
        return np.arange(start_time, end_time + self.granularity, self.granularity)

    # Create a DataFrame with the specified columns and timestamps
    def create_dataset(self, start_time, end_time, cols, prefix):
        c = copy.deepcopy(cols)
        if not prefix == '':
            for i in range(0, len(c)):
                c[i] = str(prefix) + str(c[i])
        timestamps = self.create_timestamps(start_time, end_time)
        print(f"Number of timestamps created: {len(timestamps)}")
        self.data_table = pd.DataFrame(index=timestamps, columns=c, dtype=object)

    # Add numerical data to the dataset, aggregating it according to the specified granularity
    def add_numerical_dataset(self, file, timestamp_col, value_cols, aggregation='avg', prefix='', start_unix_time=0):
        print(f'Reading data from {file}')
        dataset = pd.read_csv(file, skipinitialspace=True)
        print(f"Columns in the dataset: {dataset.columns}")

        # Convert experiment time to Unix time in nanoseconds
        dataset['Unix_Time'] = dataset[timestamp_col] * 1e9 + start_unix_time

        print(f"Converted Unix Times: {dataset['Unix_Time'].head()}")

        # Create a table based on the times found in the dataset
        self.create_dataset(min(dataset['Unix_Time']), max(dataset['Unix_Time']), value_cols, prefix)

        # Iterate over all rows in the new table
        for i in range(0, len(self.data_table.index)):
            start_time = self.data_table.index[i]
            end_time = start_time + self.granularity  # Already in nanoseconds
            # Select the relevant measurements for the current interval
            relevant_rows = dataset[
                (dataset['Unix_Time'] >= start_time) &
                (dataset['Unix_Time'] < end_time)
            ]
            for col in value_cols:
                # Find the actual column name that starts with the specified prefix (X, Y, Z)
                actual_col = [c for c in dataset.columns if c.startswith(col)][0]
                # Take the average value
                if len(relevant_rows) > 0:
                    if aggregation == 'avg':
                        self.data_table.loc[start_time, str(prefix) + str(col)] = np.average(relevant_rows[actual_col])
                    else:
                        raise ValueError(f"Unknown aggregation {aggregation}")
                else:
                    self.data_table.loc[start_time, str(prefix) + str(col)] = np.nan
        print(f"Finished processing data from {file}")

    # Remove undesired characters from the names
    def clean_name(self, name):
        return re.sub('[^0-9a-zA-Z]+', '', name)

    # Get relevant columns that have one of the specified strings in their name
    def get_relevant_columns(self, ids):
        relevant_dataset_cols = []
        cols = list(self.data_table.columns)
        for id in ids:
            relevant_dataset_cols.extend([col for col in cols if id in col])
        return relevant_dataset_cols

# Function to read the start time from time.csv
def get_start_time(meta_dir):
    time_file = meta_dir / 'time.csv'
    time_data = pd.read_csv(time_file, skipinitialspace=True)
    start_time = time_data.loc[time_data['event'] == 'START', 'system time'].values[0]
    start_time_unix = float(start_time) * 1e9  # Convert to nanoseconds
    return start_time_unix

# Initialize the class with the base directory and granularity
dataset_creator = CreateDataset(base_dir='data', granularity=200)  # granularity in milliseconds

# Create the base directory for aggregated data
aggregated_data_base_dir = Path('aggregated-data')
aggregated_data_base_dir.mkdir(exist_ok=True)

# Iterate over all participant-session folders
base_dir = Path('data')
for session_folder in base_dir.iterdir():
    if session_folder.is_dir():
        participant_session = session_folder.name
        participant, rest_interval = participant_session.split('-')
        rest_interval += 's'
        
        # Create corresponding directory in the aggregated data folder
        aggregated_session_folder = aggregated_data_base_dir / session_folder.name
        aggregated_session_folder.mkdir(exist_ok=True)
        
        # Get the start time from the meta directory
        meta_dir = session_folder / 'meta'
        start_unix_time = get_start_time(meta_dir)
        print(f"Start Unix time for {session_folder.name}: {start_unix_time}")

        for sensor_file in session_folder.iterdir():
            if sensor_file.name.endswith('.csv') and sensor_file.name != 'time.csv':
                # Determine the sensor type and assign a prefix
                if sensor_file.name.startswith('Accelerometer'):
                    sensor_prefix = 'Acc_'
                    output_file = aggregated_session_folder / 'Accelerometer-agg.csv'
                elif sensor_file.name.startswith('Gyroscope'):
                    sensor_prefix = 'Gyro_'
                    output_file = aggregated_session_folder / 'Gyroscope-agg.csv'
                else:
                    continue
                
                dataset_creator.add_numerical_dataset(
                    file=sensor_file,
                    timestamp_col='Time (s)',
                    value_cols=['X', 'Y', 'Z'],
                    aggregation='avg',
                    prefix=sensor_prefix,
                    start_unix_time=start_unix_time
                )

                # Convert the aggregated timestamps to Unix time in nanoseconds
                dataset_creator.data_table['Timestamps'] = dataset_creator.data_table.index
                
                dataset_creator.data_table.reset_index(drop=True, inplace=True)
                cols = ['Timestamps'] + [col for col in dataset_creator.data_table.columns if col != 'Timestamps']
                dataset_creator.data_table = dataset_creator.data_table[cols]

                # Save aggregated data table to a CSV file
                dataset_creator.data_table.to_csv(output_file, index=False)
                print(f"Aggregated data saved to {output_file}")

# print(dataset_creator.data_table.head())
