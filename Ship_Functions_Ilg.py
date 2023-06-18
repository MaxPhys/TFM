import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Loading the Excel files we are manipulating
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

cargoes_file_path = 'Cargoes.xlsx'
xlsx = pd.ExcelFile(cargoes_file_path)

# Reading each sheet into a data frame
df_cargoes = pd.read_excel(xlsx, 'Cargoes')
df_chronology = pd.read_excel(xlsx, 'Chronology')
df_proven_boleans = pd.read_excel(xlsx, 'Proven_boleans')

# Remove possile whitespaces in strings
df_cargoes = df_cargoes.applymap(lambda x: x.strip() if isinstance(x, str) else x)
df_chronology = df_chronology.applymap(lambda x: x.strip() if isinstance(x, str) else x)
df_proven_boleans = df_proven_boleans.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Removing rows with empty cargo and 'Amphora type' as 'Unidentified'
df_cargoes = df_cargoes[(df_cargoes['Amphora type'] != 'Unidentified')]
# Drop the 3rd and 4th columns from the data frame
df_cargoes = df_cargoes.drop(columns=[df_cargoes.columns[2], df_cargoes.columns[3]])

# Merging 'Cargoes' and 'Chronology' data frames based on 'Amphora type'
merged_df = pd.merge(df_cargoes, df_chronology, on='Amphora type', how='left')
# Sorting for ships (Oxford_wreckID)
merged_df = merged_df.sort_values(merged_df.columns[0])

# Counting number of each value in the column
value_counts = merged_df['Oxford_wreckID'].value_counts()

# Removing single values (cargo containing only one amphora type)
single_occurrence_values = value_counts[value_counts == 1].index
merged_df_2amph = merged_df[~merged_df['Oxford_wreckID'].isin(single_occurrence_values)]
# Delete columns of 4, 3, 2 BC and 8 AD
merged_df_2amph = merged_df_2amph.drop(columns=[merged_df_2amph.columns[3], merged_df_2amph.columns[4], merged_df_2amph.columns[5], merged_df_2amph.columns[14]])
print(merged_df_2amph)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Get origins of amphora types
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Checking origins of each amphora type
df_proven_boleans = df_proven_boleans.drop(columns=[df_proven_boleans.columns[0]])

# Create an empty dictionary to store the origins for each amphora type
amphora_origins = {}

# Iterate over the rows of the dataframe
for index, row in df_proven_boleans.iterrows():
    amphora_type = row[0]  # Get the amphora type from the first column
    origins = []  # List to store the origins of the current amphora type

    # Iterate over the columns starting from the second column
    for column in df_proven_boleans.columns[1:]:
        if row[column] == 'yes':
            origins.append(column)  # Add the origin to the list

    amphora_origins[amphora_type] = origins  # Assign the origins to the amphora type in the dictionary

# Print the dictionary
for amphora_type, origins in amphora_origins.items():
    print(f"Amphora Type: {amphora_type}, Origins: {origins}")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Dating the shipwrecks
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

shipwreck_dates = {}
amphora_count = set()

for _, group in merged_df_2amph.groupby('Oxford_wreckID'):
    shipwreck_id = group['Oxford_wreckID'].iloc[0]
    amphora_types = group['Amphora type'].unique()
    amphora_count.update(amphora_types)
    centuries_overlap = []

    for century in group.columns[3:15]:
        if group[century].eq('YES').all():
            centuries_overlap.append(century)

    if len(centuries_overlap) > 0:
        for century in centuries_overlap:
            # Group 4-7 AD as one
            if century in ['4 AD', '5 AD', '6 AD', '7 AD']:
                century_label = '4-7 AD'
            else:
                century_label = century

            shipwreck_dates.setdefault(century_label, {}).setdefault(shipwreck_id, []).extend(amphora_types)

print("Shipwrecks Dated by Groups:")
for century, shipwrecks in shipwreck_dates.items():
    print(f"Century: {century}")
    for shipwreck_id, amphora_types in shipwrecks.items():
        print(f"Shipwreck ID: {shipwreck_id}, Amphora Types: {amphora_types}")
    print()

total_ships = len(merged_df_2amph['Oxford_wreckID'].unique())
total_amphoras = len(amphora_count)
print(f"Total Ships: {total_ships}")
print(f"Total Amphoras: {total_amphoras}")
print()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Keep amphora origins present in 'amphora_count'
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Create a new dictionary to store the shortened amphora origins
shortened_amphora_origins = {}

# Iterate over the keys in amphora_origins
for amphora_type in amphora_origins.keys():
    # Check if the amphora type is in the amphora_count set
    if amphora_type in amphora_count:
        shortened_amphora_origins[amphora_type] = amphora_origins[amphora_type]

# Update the amphora_origins dictionary with the shortened version
amphora_origins = shortened_amphora_origins

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Strength of connections
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Create an empty dictionary to store the strength of connection for each century
strength_of_connection = {}

# Iterate over the centuries
for century, shipwrecks in shipwreck_dates.items():
    century_label = century  # Keep the entire string for ranges like '4-7 AD'
    connections = {}

    # Iterate over the shipwrecks in the current century
    for shipwreck_id, amphora_types in shipwrecks.items():
        # Generate unique pairs of amphora types without considering the order
        pairs = {(min(amphora_types[i], amphora_types[j]), max(amphora_types[i], amphora_types[j])) for i in range(len(amphora_types)) for j in range(i + 1, len(amphora_types))}

        # Update the connections dictionary
        for pair in pairs:
            connections.setdefault(pair, 0)
            connections[pair] += 1

    # Store the connections for the current century
    strength_of_connection[century_label] = connections

# Print the strength of connection for each century
for century, connections in strength_of_connection.items():
    print(f"Century: {century}")
    for pair, strength in connections.items():
        print(f"Amphora Pair: {pair}, Strength of Connection: {strength}")
    print()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Empirical network
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Create a dictionary to store the networks for each century
amphora_networks = {}

# Iterate over the centuries
for century, shipwrecks in shipwreck_dates.items():
    century_label = century  # Keep the entire string for ranges like '4-7 AD'
    connections = {}

    # Create an empty network for the current century
    G = nx.Graph()

    # Iterate over the shipwrecks in the current century
    for shipwreck_id, amphora_types in shipwrecks.items():
        # Generate unique pairs of amphora types without considering the order
        pairs = {(min(amphora_types[i], amphora_types[j]), max(amphora_types[i], amphora_types[j])) for i in
                 range(len(amphora_types)) for j in range(i + 1, len(amphora_types))}

        # Add edges to the network, excluding self-loops
        for pair in pairs:
            if pair[0] != pair[1]:
                G.add_edge(pair[0], pair[1])

    # Get the unique origins from the amphora_origins dictionary
    all_origins = set()
    for origins in amphora_origins.values():
        all_origins.update(origins)

    # Generate a color palette with 25 different colors
    color_palette = plt.cm.Set1(np.linspace(0, 1, len(all_origins)))

    # Create a dictionary to store the color assignments for each origin
    origin_colors = {}
    for i, origin in enumerate(all_origins):
        origin_colors[origin] = color_palette[i]

    # Create a legend for the colors
    legend_labels = list(origin_colors.keys())
    legend_colors = list(origin_colors.values())

    # Assign colors to nodes based on their origins
    node_colors = [origin_colors.get(amphora_origins.get(node, [])[0]) for node in G.nodes]

    # Plot the network with colored nodes
    plt.figure()
    plt.figure(figsize=(15, 15))
    nx.draw(G, with_labels=True, node_color=node_colors)
    plt.title("Amphora Network - Century {century_label}")

    # Create proxy artists for the legend
    legend_handles = [Patch(facecolor=color) for color in legend_colors]

    # Create and display the legend using the proxy artists
    plt.legend(legend_handles, legend_labels)
    plt.show()
