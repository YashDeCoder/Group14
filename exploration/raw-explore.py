import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load the data
acc_data = pd.read_csv('data/M2-10/Accelerometer.csv')
gyro_data = pd.read_csv('data/M2-10/Gyroscope.csv')


# Set the color palette
colors = sns.color_palette("Set2", 3)

# Directory for saving plots
output_dir = Path('plots')
output_dir.mkdir(exist_ok=True)

# Plot the accelerometer data
plt.figure(figsize=(12, 6))
plt.plot(acc_data['Time (s)'], acc_data['X (m/s^2)'], label='Acc X', color=colors[0])
plt.plot(acc_data['Time (s)'], acc_data['Y (m/s^2)'], label='Acc Y', color=colors[1])
plt.plot(acc_data['Time (s)'], acc_data['Z (m/s^2)'], label='Acc Z', color=colors[2])
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s^2)')
plt.title('Raw Accelerometer Data for Participant M1, Session 10')
plt.legend()
plt.grid(True)
plt.savefig(output_dir / 'acc_raw.png')

# Plot the gyroscope data
plt.figure(figsize=(12, 6))
plt.plot(gyro_data['Time (s)'], gyro_data['X (rad/s)'], label='Gyro X', color=colors[0])
plt.plot(gyro_data['Time (s)'], gyro_data['Y (rad/s)'], label='Gyro Y', color=colors[1])
plt.plot(gyro_data['Time (s)'], gyro_data['Z (rad/s)'], label='Gyro Z', color=colors[2])
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.title('Raw Gyroscope Data for Participant M1, Session 10')
plt.legend()
plt.grid(True)
plt.savefig(output_dir / 'gyro_raw.png')
