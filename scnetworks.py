import matplotlib.pyplot as plt
import networkx as nx
from pyproj import transform

plt.rcParams['text.usetex'] = True

"""
Code which makes the plots of the networks of the supply chain.
"""
# Define the supply chain configurations
supply_chain_6 = {
    0: [1, 2], 1: [3, 4], 2: [4, 5], 3: [], 4: [], 5: []
}

supply_chain_12 = {
    0: [1, 2], 1: [3, 4], 2: [5, 6], 3: [7, 8], 4: [9], 5: [10, 11], 6: [], 7: [], 8: [], 9: [11], 10: [11], 11: []
}

supply_chain_18 = {
    0: [1, 2], 1: [3, 4], 2: [5, 6], 3: [7, 8], 4: [9], 5: [10], 6: [],
    7: [11, 12, 13], 8: [12], 9: [14], 10: [15], 11: [16, 17], 12: [], 
    13: [17], 14: [17], 15: [], 16: [], 17: []
}

supply_chain_24 = {
    0: [1, 2], 1: [3, 4], 2: [5, 6], 3: [7, 8], 4: [9], 5: [10], 6: [],
    7: [11, 12, 13], 8: [12], 9: [14], 10: [15], 11: [16, 17], 12: [], 
    13: [17], 14: [17, 18], 15: [19], 16: [20], 17: [20, 21], 18: [22], 
    19: [22, 23], 20: [], 21: [], 22: [], 23: []
}

def draw_supply_chain_by_echelons(ax, chain, title):
    G = nx.DiGraph()
    for key, values in chain.items():
        for value in values:
            G.add_edge(key, value)
    
    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=True, node_size=200, node_color='skyblue', arrowsize=5, font_size=10, font_weight='bold', edge_color='black', ax=ax)
    ax.text(0.5, 0, title, ha='center', va = 'top', transform=ax.transAxes, fontsize=14)

# Create a 2x2 subplot for all supply chain configurations
fig, axes = plt.subplots(2, 2, figsize=(8, 6), layout = 'constrained')

draw_supply_chain_by_echelons(axes[0, 0], supply_chain_6, "a) 6 Nodes")
draw_supply_chain_by_echelons(axes[0, 1], supply_chain_12, "b) 12 Nodes")
draw_supply_chain_by_echelons(axes[1, 0], supply_chain_18, "c) 18 Nodes")
draw_supply_chain_by_echelons(axes[1, 1], supply_chain_24, "d) 24 Nodes")

#plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.tight_layout()
#plt.savefig('supply_chain_networks.png', dpi=600)
plt.show()

import matplotlib.pyplot as plt
import textwrap  # Import for text wrapping


# Define MARL methods and information levels
methods = ["IPPO", "MAPPO", "G-MAPPO", "GP-MAPPO", "Noise GP-MAPPO"]
information_levels = [1, 4, 3, 2, 2]  # Adjust based on your information level definition

# Create a colorful bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(methods, information_levels, color=['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan'])

# Add data labels above each bar
descriptions = [
    "Individual Agent's Observation (Local)",
    "Combined Agents' Observations (Local + Global)",
    "GNN Aggregated Agent Observations (Global)",
    "GNN Mean Pool (Global Summary)",
    "GNN Mean Pool (Global Summary)",
]
max_chars_per_line = 20 
for bar, desc in zip(bars, descriptions):
    yval = bar.get_height() + 0.1  # Adjust offset for label placement
    wrapped_text = '\n'.join(textwrap.fill(desc, max_chars_per_line).splitlines())  # Wrap and join lines

    plt.text(bar.get_x() + bar.get_width() / 2, yval, wrapped_text, ha='center', va='bottom', fontsize=12)

# Set labels and title
plt.xlabel("MARL Method")
plt.ylabel("Level of Information")  # Changed label to "Level of Information"
plt.xticks(rotation=45, ha="right")
plt.tick_params(left=False, labelleft=False)  # Hide ticks and labels on the left (y-axis)
ax = plt.gca()  # Get the current axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)  # Optional: Hide left spine for a more minimal look

plt.tight_layout()
plt.savefig('marl_information_levels.png', dpi=600)
# Display the chart
plt.show()

