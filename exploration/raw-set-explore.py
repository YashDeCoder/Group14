import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Function to load data from a given file path
def load_data(file_path):
    return pd.read_csv(file_path)

# Load accelerometer and gyroscope data for different rest intervals
acc_data_10s = load_data('data/F3-0/Accelerometer.csv')
gyro_data_10s = load_data('data/F3-0/Gyroscope.csv')

# Colors from the Set2 palette
colors = sns.color_palette("Set2")
acc_color = colors[1]
gyro_color = colors[0]

# Plot combined accelerometer and gyroscope data for a single rest interval
def plot_combined_highlight(data_acc, data_gyro, title):
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Filter the data to only plot from 0 to 32 seconds
    data_acc = data_acc[data_acc["Time (s)"] <= 32]
    data_gyro = data_gyro[data_gyro["Time (s)"] <= 32]
    
    # Plot accelerometer data (Z-axis)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Acceleration (m/s^2)')
    ax1.plot(data_acc["Time (s)"], data_acc["Z (m/s^2)"], label='Acc Z', color=acc_color)
    ax1.tick_params(axis='y')
    
    # Highlight the first set (0-10 seconds)
    ax1.axvspan(1, 9.5, color='yellow', alpha=0.3, label='First Set')
    # Highlight the last set (21-30 seconds)
    ax1.axvspan(21, 30, color='pink', alpha=0.3, label='Last Set')
    
    # Plot gyroscope data (Y-axis) on the same plot with a secondary y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Angular Velocity (rad/s)')
    ax2.plot(data_gyro["Time (s)"], data_gyro["Y (rad/s)"], label='Gyro Y', color=gyro_color)
    ax2.tick_params(axis='y')

    plt.title(title)
    fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.grid(True)
    
    # Save the plot
    output_dir = Path('plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'tiredness.png', bbox_inches='tight')
    plt.show()

# Plot for 10s rest interval with highlights
plot_combined_highlight(acc_data_10s, gyro_data_10s, 'Accelerometer and Gyroscope Data for 0s Rest Interval')
