import numpy as np 
import matplotlib.pyplot as plt 
import os 
import json 


#all nodes e.g. profit
def process_all_nodes_data(file_paths, config):
    num_files = len(file_paths)
    # Create a 2x2 grid for subplots if you have 4 files
    num_rows = 2
    num_cols = 2 if num_files >= 4 else 1  # Adjust as needed

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 10), constrained_layout=True)
    plt.tight_layout()

    # If you have less than 4 files, flatten axs to a 1D array
    axs = axs.flatten() if num_files < 4 else axs

    cols = ['tab:blue','tab:orange','tab:green','tab:red',\
    'tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']

    for i, file_path in enumerate(file_paths):
        with open(file_path, 'r') as file:
            data = json.load(file)

            av_profits1 = data['av_profits']
            av_profits2 = data['mav_profits']

            average_profit_list1  = np.mean(av_profits1, axis =0)
            std_profit_list1 = np.std(av_profits1, axis=0)
            average_profit_list2 = np.mean(av_profits2, axis=0)
            std_profit_list2 = np.std(av_profits2, axis =0)
   
            cumulative_profit_list1 = np.cumsum(average_profit_list1, axis =0)  # Calculate cumulative profit
            cumulative_profit_list2 = np.cumsum(average_profit_list2, axis = 0)


        period = range(1, len(average_profit_list1)+1)
        axs[i // num_cols, i % num_cols].plot(period, cumulative_profit_list1, color = cols[0], label="Single Agent",linestyle='solid')
        axs[i // num_cols, i % num_cols].fill_between(period, cumulative_profit_list1 + std_profit_list1, cumulative_profit_list1 - std_profit_list1,
                            color = cols[0], alpha=0.5, linewidth = 0 )
        axs[i // num_cols, i % num_cols].plot(period, cumulative_profit_list2, color = cols[1], label="Multi Agent", linestyle='dashed')
        axs[i // num_cols, i % num_cols].fill_between(period, cumulative_profit_list2 + std_profit_list2, cumulative_profit_list2 - std_profit_list2,
                            color = cols[1], alpha=0.5, linewidth = 0 )

        axs[i // num_cols, i % num_cols].set_ylabel('Average Cumulative Profit ($)')
        if file_path != "/Users/nikikotecha/Documents/PhD/sS/no_disruption_profit_data.json":
            if config["bullwhip"]: 
                axs[i // num_cols, i % num_cols].axvline(x=20, color=cols[2], linestyle='--', label='Disruption Start')
                axs[i // num_cols, i % num_cols].axvline(x=30, color=cols[2], linestyle='--', label='Disruption End')
        axs[i // num_cols, i % num_cols].legend(frameon = False)
        axs[i // num_cols, i % num_cols].set_xlabel('Period')
        # Add subtitles (a), (b), (c), (d)
        subtitle = f"({chr(97 + i)})"
        axs[i // num_cols, i % num_cols].text(0.5, 1.05, subtitle, transform=axs[i // num_cols, i % num_cols].transAxes,
                                          fontsize=10, fontweight='bold', va='bottom', ha='right')


    #axs[-1, -1].spines['right'].set_visible(False)
    #axs[-1 ,-1].spines['top'].set_visible(False)

    plt.tight_layout()
    plt.show()
    fig.savefig('your_plot.png', format='png', bbox_inches='tight', dpi=300)

file_paths = ["/Users/nikikotecha/Documents/PhD/sS/no_disruption_profit_data.json", 
              "/Users/nikikotecha/Documents/PhD/sS/bullwhip_profit_data.json",
              "/Users/nikikotecha/Documents/PhD/sS/node_price_profit_data.json",
              "/Users/nikikotecha/Documents/PhD/sS/node_cost_profit_data.json"]

config = {"bullwhip": True}

process_all_nodes_data(file_paths, config)