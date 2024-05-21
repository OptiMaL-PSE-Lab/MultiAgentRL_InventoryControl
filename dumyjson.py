import json
import os 

# Load the JSON file
ng1 =  r"c:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-03-05_10-58-52iozorpnu\result.json"
ng2 =  r"c:\Users\nk3118\ray_results\PPO_MultiAgentInvManagementDiv_2024-03-06_10-44-29fqk4q483\result.json"
objects = []
with open(ng1, 'r') as f:
    data1 = [json.loads(line) for line in f]

with open(ng2, 'r') as f:
    data2 = [json.loads(line) for line in f]

combineddata = data1 + data2

# Write the combined data to the third file
with open('result_g2_48.json', 'w') as f:
    for item in combineddata:
        f.write(json.dumps(item) + "\n")

with open('result_g2_48.json', 'r') as f:
    data3 = [json.loads(line) for line in f]

# Count the number of objects
num_objects = len(data3)

print(f'There are {num_objects} objects in the JSON file.')

file_path = os.path.abspath('file3.json')

# Get the directory of the file
file_dir = os.path.dirname(file_path)

print(f'The directory of the file is: {file_dir}')

