import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_aggregated_data(base_dir):
    data = []
    for session_folder in Path(base_dir).iterdir():
        if session_folder.is_dir():
            participant_session = session_folder.name
            participant, rest_interval = participant_session.split('-')
            rest_interval = rest_interval.split('s')[0] 
            
            for sensor_file in session_folder.iterdir():
                if sensor_file.name.endswith('-agg.csv'):
                    sensor_data = pd.read_csv(sensor_file)
                    sensor_data['Participant'] = participant
                    sensor_data['Rest_Interval'] = rest_interval
                    data.append(sensor_data)
                    
    return pd.concat(data, ignore_index=True)


def plot_boxplots(data, session_type, output_dir):
    session_data = data[data['Rest_Interval'] == session_type]
    
    melted_data = session_data.melt(id_vars=['Timestamps', 'Participant', 'Rest_Interval'], 
                                    value_vars=['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z'],
                                    var_name='Sensor', value_name='Value')
    
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Sensor', y='Value', data=melted_data, palette="Set3")
    plt.title(f'Boxplot of Sensor Data for Rest Interval: {session_type}s')
    plt.xlabel('Sensor Type')
    plt.ylabel('Value')
    plt.grid(True)

    plot_file = output_dir / f'boxplot_rest_{session_type}s.png'
    plt.savefig(plot_file)
    plt.close()
    print(f"Boxplot saved to {plot_file}")

aggregated_data_base_dir = 'aggregated-data'
data = load_aggregated_data(aggregated_data_base_dir)

output_dir = Path('boxplots')
output_dir.mkdir(exist_ok=True)

# Plot boxplots for each session type
for session_type in ['0', '10', '30', '60']:
    plot_boxplots(data, session_type, output_dir)
