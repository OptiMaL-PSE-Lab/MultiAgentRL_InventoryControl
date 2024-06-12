import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True

# Manually input the data into a DataFrame
data = {
    'Number of Agents': [6]*7 + [12]*7 + [18]*7 + [24]*7,
    'Algorithm': ['Static (s,S) Heuristic', 'Single Agent PPO', 'IPPO', 'MAPPO', 'GMAPPO', 'GP-MAPPO', 'Noise-GP-MAPPO']*4,
    'Cumulative Profit': [4320, 4176, 3280, 2548, 3899, 4160, 3893, 
                          11276, 13432, 13164, 12192, 11836, 11890, 14701, 
                          14300, 14765, 16148, 14313, 12928, 16788, 21718, 
                          11813, 9400, 12563, 8474, 10291, 12476, 16182],
    'Profit Std Dev': [1836, 342, 160, 185, 246, 214, 177, 
                       6097, 673, 366, 334, 350, 401, 402, 
                       9720, 1685, 555, 925, 642, 497, 918, 
                       8132, 1605, 1160, 834, 768, 1227, 1405],
    'Median Backlog': [17, 18, 10, 13, 7, 10, 7, 
                       71, 41, 74, 40, 41, 54, 32, 
                       935, 43, 66, 73, 43, 71, 132, 
                       845, 56, 117, 85, 104, 120, 52],
    'Average Inventory': [277, 78, 360, 357, 340, 346, 340, 
                          710, 68, 611, 630, 707, 607, 673, 
                          718, 66, 1013, 924, 1003, 988, 914, 
                          850, 52, 1331, 1286, 1366, 1276, 1317],
    'Inventory Std Dev': [90, 33, 80, 90, 94, 90, 95, 
                          110, 38, 180, 180, 152, 174, 163, 
                          123, 37, 300, 367, 308, 330, 302, 
                          120, 39, 438, 476, 434, 460, 470]
}

df = pd.DataFrame(data)

# Define the colors
colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']

# Define the function to create bar plots with error bars
def plot_with_error_bars(ax, df, y, yerr, title, ylabel, colors, save):
    algorithms = df['Algorithm'].unique()
    num_agents = df['Number of Agents'].unique()
    bar_width = 0.1  # width of the bars
    # Position of bars on x-axis
    r = [np.arange(len(num_agents)) + i*bar_width  for i in range(len(algorithms))]

    for i, alg in enumerate(algorithms):
        alg_data = df[df['Algorithm'] == alg]
        ax.bar(r[i], alg_data[y], color=colors[i % len(colors)], width=bar_width, edgecolor='grey', label=alg)
        ax.errorbar(r[i], alg_data[y], yerr=alg_data[yerr], fmt='none', c='black', capsize=5)

    # Add xticks on the middle of the group bars
    ax.set_xlabel('Number of Agents', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xticks([r + bar_width * 3 for r in range(len(num_agents))])
    ax.set_xticklabels(num_agents, fontsize=14)
    ax.tick_params(axis='y', labelsize=12)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.legend(fontsize=12, frameon = False)

    plt.savefig(save, dpi = 1100)
    #ax.grid(axis='y', linestyle='--', linewidth=0.7)

# Plot cumulative profit with error bars
#fig, ax = plt.subplots(figsize=(12, 8), layout = 'constrained')
#plot_with_error_bars(ax, df, 'Cumulative Profit', 'Profit Std Dev', 'Cumulative Profit by Algorithm', 'Cumulative Profit', colors, 'profit.png')
#plt.tight_layout()
#plt.show()

# Plot median backlog
fig, ax = plt.subplots(figsize=(12, 8), layout = 'constrained')
algorithms = df['Algorithm'].unique()
num_agents = df['Number of Agents'].unique()
bar_width = 0.1  # width of the bars
r = [np.arange(len(num_agents)) + i*bar_width for i in range(len(algorithms))]

for i, alg in enumerate(algorithms):
    alg_data = df[df['Algorithm'] == alg]
    ax.bar(r[i], alg_data['Median Backlog'], color=colors[i % len(colors)], width=bar_width, edgecolor='grey', label=alg)

ax.set_xlabel('Number of Agents', fontsize=14)
ax.set_ylabel('Median Backlog', fontsize=14)
ax.set_xticks([r + bar_width * 2 for r in range(len(num_agents))])
ax.set_xticklabels(num_agents, fontsize=14)
ax.tick_params(axis='y', labelsize=12)
ax.legend(fontsize=12, frameon = False)
ax.grid(axis='y', linestyle='--', linewidth=0.7)
plt.tight_layout()
plt.savefig('backlog.png', dpi = 1100)
plt.show()

# Plot average inventory with error bars
fig, ax = plt.subplots(figsize=(12, 8), layout = 'constrained')
plot_with_error_bars(ax, df, 'Average Inventory', 'Inventory Std Dev', 'Average Inventory by Algorithm', 'Average Inventory', colors, 'inventory.png')
plt.tight_layout()
plt.show()