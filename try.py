import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the data from the JSON file
with open('g1_6n2p_np.json') as f:
    data = json.load(f)

av_backlog = data['av_backlog']

# Create a list to store each backlog's values
backlog_values = []
for backlog in av_backlog:
    backlog_values.append([item[1] for item in backlog])

# Calculate the sum of all backlogs at each time point
sum_backlog = [sum(values) for values in zip(*backlog_values)]

# Calculate the cumulative sum
cumulative_sum_backlog = np.cumsum(sum_backlog)

# Plot the cumulative sum of all backlogs
plt.figure(figsize=(10, 5))
plt.plot(cumulative_sum_backlog, label='Cumulative sum of all backlogs')

plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Cumulative Sum of all backlogs Over Time')
plt.legend()
plt.show()