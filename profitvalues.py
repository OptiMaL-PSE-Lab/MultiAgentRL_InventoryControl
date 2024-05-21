
import numpy as np 
import matplotlib.pyplot as plt
import json

ippo18 = r"C:\Users\nk3118\Documents\sS\ng1_18.json"
mappo18 = r"C:\Users\nk3118\Documents\sS\ng2_18.json" 
gmappo18 = r"C:\Users\nk3118\Documents\sS\g1_18.json" 
g218 = r"C:\Users\nk3118\Documents\sS\g2_18.json" 
g218noise = r"C:\Users\nk3118\Documents\sS\g23_18.json"

ng2_disrupt =r"C:\Users\nk3118\Documents\sS\g2_1_18.json"
g2_disrupt =r"C:\Users\nk3118\Documents\sS\g23_18.json"


file_paths18 = [ippo18, mappo18, gmappo18, g218, g218noise]


ippo12 = r"C:\Users\nk3118\Documents\sS\ippo_12_1.json"
mappo12 = r"C:\Users\nk3118\Documents\sS\mappo_12_1.json"
gmappo12 = r"C:\Users\nk3118\Documents\sS\gmappo_12_1.json"
g212 =  r"C:\Users\nk3118\Documents\sS\g2_12_gnoise0.json"
g212noise =  r"C:\Users\nk3118\Documents\sS\g2_12_gnoise2.json"
file_paths12 = [ippo12, mappo12, gmappo12, g212, g212noise]

ippo24 = r"C:\Users\nk3118\Documents\sS\ippo_24.json"
mappo24 = r"C:\Users\nk3118\Documents\sS\mappo_24.json"
gmappo24 = r"C:\Users\nk3118\Documents\sS\gmappo_24.json"
g224 =  r"C:\Users\nk3118\Documents\sS\g2_24_gnoise0.json"
g224noise =  r"C:\Users\nk3118\Documents\sS\g2_24_gnoise2.json"
file_paths24 = [ippo24, mappo24, gmappo24, g224, g224noise]

ippo32 = r"C:\Users\nk3118\Documents\sS\ippo_32.json"
mappo32 = r"C:\Users\nk3118\Documents\sS\mappo_32.json"
gmappo32 = r"C:\Users\nk3118\Documents\sS\gmappo_32.json"
g232 =  r"C:\Users\nk3118\Documents\sS\g2_32_noise0.json"
g232noise =  r"C:\Users\nk3118\Documents\sS\g2_32_noise2.json"
file_paths32 = [ippo32, mappo32, gmappo32, g232, g232noise]

g2_noise00 = r"C:\Users\nk3118\Documents\sS\g2_18_nonoise.json"
g2_noise01 = r"C:\Users\nk3118\Documents\sS\g2_18_noise0.11.json"
g2_noise02 = r"C:\Users\nk3118\Documents\sS\g2_18_noise0.222.json"
g2_noise05 =r"C:\Users\nk3118\Documents\sS\g2_18_noise0.55.json"
g2_noise075 = r"C:\Users\nk3118\Documents\sS\g2_18_noise075.json"
g2_noise10 = r"C:\Users\nk3118\Documents\sS\g2_18_noise1.00.json"
g2_noise20 = r"C:\Users\nk3118\Documents\sS\g2_18_noise20.json"

files_noise = [g2_noise00, g2_noise01, g2_noise02, g2_noise05, g2_noise10, g2_noise20]

def process_all_nodes_data(profits_dict, config):
    for i, av_profits in enumerate(profits_dict.values(), start=1):
        average_profit_list = np.mean(av_profits, axis=0)
        std_profit_list = np.std(av_profits, axis=0)
        cumulative_profit_list = np.cumsum(average_profit_list, axis=0)

        print(f"Average Profit {i}: {cumulative_profit_list[-1]}")
        print(f"Standard Deviation Profit {i}: {std_profit_list[-1]}")
    

def process_all_nodes_data_from_file(file_paths, config):
    profits_dict = {}
    for file in file_paths:
        with open(file, 'r') as file:
            data = json.load(file)
        profits_dict[file] = data["av_profits"]
    config = {}
    process_all_nodes_data(profits_dict, config)

process_all_nodes_data_from_file(file_paths12, config = None)