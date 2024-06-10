from venv import create
import numpy as np 
import matplotlib.pyplot as plt
import json

#plt.rcParams['text.usetex'] = True

ippo18 = r"ng1_18.json"
mappo18 = r"ng2_18.json" 
gmappo18 = r"g1_18.json" 
g218 = r"g2_18.json" 
g218noise = r"g23_18.json"

ng2_disrupt =r"g2_1_18.json"
g2_disrupt =r"g23_18.json"


file_paths18 = [ippo18, mappo18, gmappo18, g218, g218noise]


ippo12 = r"ippo_12_1.json"
mappo12 = r"mappo_12_1.json"
gmappo12 = r"gmappo_12_1.json"
g212 =  r"g2_12_gnoise0.json"
g212noise =  r"g2_12_gnoise2.json"
file_paths12 = [ippo12, mappo12, gmappo12, g212, g212noise]

ippo24 = r"ippo_24.json"
mappo24 = r"mappo_24.json"
gmappo24 = r"gmappo_24.json"
g224 =  r"g2_24_gnoise0.json"
g224noise =  r"g2_24_gnoise2.json"
file_paths24 = [ippo24, mappo24, gmappo24, g224, g224noise]

ippo32 = r"ippo_32.json"
mappo32 = r"mappo_32.json"
gmappo32 = r"gmappo_32.json"
g232 =  r"g2_32_noise0.json"
g232noise =  r"g2_32_noise2.json"
file_paths32 = [ippo32, mappo32, gmappo32, g232, g232noise]

g2_noise00 = r"C:\Users\nk3118\Documents\sS\g2_18_nonoise.json"
g2_noise01 = r"C:\Users\nk3118\Documents\sS\g2_18_noise0.11.json"
g2_noise02 = r"C:\Users\nk3118\Documents\sS\g2_18_noise0.222.json"
g2_noise05 =r"C:\Users\nk3118\Documents\sS\g2_18_noise0.55.json"
g2_noise075 = r"C:\Users\nk3118\Documents\sS\g2_18_noise075.json"
g2_noise10 = r"C:\Users\nk3118\Documents\sS\g2_18_noise1.00.json"
g2_noise20 = r"C:\Users\nk3118\Documents\sS\g2_18_noise20.json"

files_noise = [g2_noise00, g2_noise01, g2_noise02, g2_noise05, g2_noise10, g2_noise20]

ippo6 = r"ippo_6.json"
mapppo6 = r"mappo_6.json"
gmappo6 = r"gmappo_6.json"
g2_6 = r"g2_6.json"
g2_6noise = r"g2_6_noise02.json"

file_paths6 = [ippo6, mapppo6, gmappo6, g2_6, g2_6noise]

def process_all_nodes_data(profits_dict, ax):
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    labels = ['IPPO', 'MAPPO', 'G-MAPPO', 'GP-MAPPO', 'Noise GP-MAPPO', 'N/A']
    
    for i, (key, av_profits) in enumerate(profits_dict.items()):
        average_profit_list = np.mean(av_profits, axis=0)
        std_profit_list = np.std(av_profits, axis=0)
        cumulative_profit_list = np.cumsum(average_profit_list, axis=0)

        period = range(1, len(average_profit_list) + 1)
        ax.plot(period, cumulative_profit_list, label=labels[i], color=colors[i], linestyle='solid')
        ax.fill_between(period, cumulative_profit_list + std_profit_list, cumulative_profit_list - std_profit_list, color=colors[i], alpha=0.2, linewidth=0)
    
    ax.set_xlabel('Period')
    ax.set_ylabel('Average Cumulative Profit')
    ax.set_xlim(0, 50)
    ax.legend(frameon=False, fontsize=14)
    


def process_all_nodes_data_from_file(file_paths):
    profits_dict = {}
    for file in file_paths:
        with open(file, 'r') as file:
            data = json.load(file)
        profits_dict[file] = data["av_profits"]
    return profits_dict

def create_subplots_for_different_paths(file_paths_list):
    n_sets = len(file_paths_list)
    fig, axes = plt.subplots(2, 2, figsize=(18, 6 * n_sets), constrained_layout=True)
    axes = axes.flatten()

    if n_sets == 1:
        axes = [axes]  # Ensure axes is always iterable
    labels = ['a)', 'b)', 'c)', 'd)']

    for ax, label in zip(axes.flatten(), labels):
        ax.text(0.5, -0.2, label, transform=ax.transAxes, 
            fontsize=14, va='center', ha='center')
        
    for file_paths, ax in zip(file_paths_list, axes):
        profits_dict = process_all_nodes_data_from_file(file_paths)
        process_all_nodes_data(profits_dict, ax)

    plt.show()
    #fig.savefig('execution_compile.png', dpi=1100)

file_paths_list = [file_paths6, file_paths12, file_paths18, file_paths24]

create_subplots_for_different_paths(file_paths_list)