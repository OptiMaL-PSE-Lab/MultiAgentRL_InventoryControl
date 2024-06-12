import shutil
import os

ippo6 = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-05-19_19-47-56f6z96d1b\result.json"]
mappo6 = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-05-19_19-47-15u94ac_tc\result.json"]
gmappo6 = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-05-19_19-49-37sbsnq6km\result.json"]
g2_6 = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-05-19_23-09-53h3ojljkh\result.json"]
g2_6noise = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-05-19_23-11-22vjup3rv8\result.json"]
file_paths6 = [ippo6, mappo6, gmappo6, g2_6, g2_6noise]

ippo12 = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-05-01_16-12-06owdt78yd\result.json"]
mappo12 = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-05-02_13-36-43l0j31ce0\result.json"]
gmappo12 = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-05-01_16-13-281w63y_j_\result.json"]
g2_12 = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-05-01_12-00-15zbm_m4ax\result.json"]
g2_12noise = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-05-01_16-11-11wakdzoi7\result.json"]
file_paths12 = [ippo12, mappo12, gmappo12, g2_12, g2_12noise]

ippo18 = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-04-11_09-18-09gpdriji_\result.json"]
mappo18 = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-04-10_20-41-03g7yt7xc3\result.json"]
gmappo18 = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-04-10_20-40-58vb6xruob\result.json"]
g2_18 = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-04-10_22-16-53zax54im2\result.json"]
g2_18noise = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-05-15_11-14-05l1k6u06k\result.json"]
file_paths18 = [ippo18, mappo18, gmappo18, g2_18, g2_18noise]

ippo24 = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-05-06_15-42-44nu1jjpuk\result.json"]
mappo24 = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-05-06_15-41-461marte8k\result.json"]
gmappo24 = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-05-06_15-45-588zbz5wvr\result.json"]
g2_24 = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-04-26_09-21-552nxyn_hr\result.json"]
g2_24noise = [r"C:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-05-07_08-35-34vpwck8mj\result.json"]

file_paths24 = [ippo24, mappo24, gmappo24, g2_24, g2_24noise]

def copy_files(file_paths, destination_folder):
    for file_path in file_paths:
        # Get the part of the file path after 'ray_results'
        sub_path = file_path[0].split('ray_results', 1)[1]

        # Combine the destination folder with the sub path
        destination_path = os.path.join(destination_folder, 'ray_results' + sub_path)

        # Create the destination directory if it doesn't exist
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)

        # Copy the file
        shutil.copy(file_path[0], destination_path)

destination_folder = r"C:\Users\nk3118\OneDrive - Imperial College London\Documents\sS"  # replace with your destination folder

all_file_paths = [file_paths6, file_paths12, file_paths18, file_paths24]
for file_paths in all_file_paths:
    copy_files(file_paths, destination_folder)
    print(f"Files copied to {destination_folder}")

