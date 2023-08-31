import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import community
import numpy as np
import random
from itertools import combinations
import scienceplots

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Manipulating the Excel files for shipwrecks
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

cargoes_file_path = 'Cargoes.xlsx'
xlsx = pd.ExcelFile(cargoes_file_path)

# Reading each sheet into a data frame
df_cargoes = pd.read_excel(xlsx, 'Cargoes')
df_chronology = pd.read_excel(xlsx, 'Chronology')

# Getting rid of possible whitespaces in strings
df_cargoes = df_cargoes.applymap(lambda x: x.strip() if isinstance(x, str) else x)
df_chronology = df_chronology.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Removing rows with empty cargo and 'Amphora type' as 'Unidentified'
df_cargoes = df_cargoes[(df_cargoes['Amphora type'] != 'Unidentified')]
# Drop the 3rd and 4th columns from the data frame
df_cargoes = df_cargoes.drop(columns=[df_cargoes.columns[2], df_cargoes.columns[3]])

# Merging 'Cargoes' and 'Chronology' data frames based on 'Amphora type'
merged_df = pd.merge(df_cargoes, df_chronology, on='Amphora type', how='left')
# Sorting for ships (Oxford_wreckID)
merged_df = merged_df.sort_values(merged_df.columns[0])

# Number count of value in the column
value_counts = merged_df['Oxford_wreckID'].value_counts()

# We do not consider the columns of centuries 4 BC, 3 BC and 8 AD
merged_df_2amph = merged_df.drop(columns=[merged_df.columns[3], merged_df.columns[4],
                                          merged_df.columns[14]])

# print("All Shipwrecks")
# print(merged_df_2amph)
# print()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Get amphora production times of all shipwrecks
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

production_times = {}

for _, row in merged_df_2amph.iterrows():
    amphora_type = row['Amphora type']
    production_time = [col for col, value in row[3:].items() if value == 'YES']
    production_times[amphora_type] = production_time

print('Amphora production times:')
print(production_times)
print()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Just keep datable shipwrecks in merged_df_2amph
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

shipwreck_datable_all = {}
amphora_count_all = set()
shipwrecks_no_overlap_all = []

for _, group in merged_df_2amph.groupby('Oxford_wreckID'):
    shipwreck_id_all = group['Oxford_wreckID'].iloc[0]
    amphora_types_all = set(group['Amphora type'])
    amphora_count_all.update(amphora_types_all)
    centuries_overlap_all = []

    for century_all in group.columns[3:12]:
        if group[century_all].eq('YES').all():
            centuries_overlap_all.append(century_all)

    if len(amphora_types_all) > 1 and len(centuries_overlap_all) == 0:
        # Shipwreck has more than one amphora type but no overlap
        shipwrecks_no_overlap_all.append(shipwreck_id_all)

# print(shipwrecks_no_overlap_all)

# Filter merged_df_all based on 'Oxford_wreckID'
merged_df_all = merged_df_2amph[~merged_df_2amph['Oxford_wreckID'].isin(shipwrecks_no_overlap_all)]

print("Datable shipwrecks")
print(merged_df_all)
print()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Dating all shipwrecks
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

shipwreck_dates = {}
amphora_count = set()

for _, group in merged_df_all.groupby('Oxford_wreckID'):
    shipwreck_id = group['Oxford_wreckID'].iloc[0]
    amphora_types = set(group['Amphora type'])
    amphora_count.update(amphora_types)
    centuries_overlap = []

    for century in group.columns[3:12]:
        if group[century].eq('YES').all():
            centuries_overlap.append(century)

    if len(centuries_overlap) > 0:
        for century in centuries_overlap:
            # Group 2-1 BC and 4-7 AD as one
            if century in ['2 BC', '1 BC']:
                century_label = 'BC'
            elif century in ['4 AD', '5 AD', '6 AD', '7 AD']:
                century_label = '4-7 AD'
            else:
                century_label = century

            shipwreck_dates.setdefault(century_label, {}).setdefault(shipwreck_id, set()) \
                .update(amphora_types)

# print("All shipwrecks dated by groups:")
# for century, shipwrecks in shipwreck_dates.items():
#    print(f"Century: {century}")
#    for shipwreck_id, amphora_types in shipwrecks.items():
#        print(f"Shipwreck ID: {shipwreck_id}, Amphora Types: {amphora_types}")
#    print()

total_ships = len(merged_df_all['Oxford_wreckID'].unique())
total_amphoras = len(amphora_count)
print(f"Total number of ships: {total_ships}")
print(f"Total number of amphoras: {total_amphoras}")
print()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Get amphora origins
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Check for true origins (singles)
df_origins_updated = pd.read_excel(xlsx, 'Origins Updated')
df_origins_updated = df_origins_updated.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Filter out rows with empty origins
df_origins_updated = df_origins_updated[df_origins_updated['Origin'].notnull()]

amphora_origin_dict = {}

for row in df_origins_updated[['Amphora type', 'Origin']].values:
    amphora_type = row[0]
    origin = row[1]

    if amphora_type in amphora_origin_dict:
        amphora_origin_dict[amphora_type].append(origin)
    else:
        amphora_origin_dict[amphora_type] = [origin]

amphora_origins_all = amphora_origin_dict
# Print the updated amphora_origins_all dictionary
print(amphora_origins_all)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Strength of connections
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Dictionary to store the strength of connection for each century
strength_of_connection = {}

# Iterate over the centuries
for century, shipwrecks in shipwreck_dates.items():
    century_label = century  # Keep the entire string for ranges like '4-7 AD'
    connections = {}

    # Iterate over the shipwrecks in the current century
    for shipwreck_id, amphora_types in shipwrecks.items():
        # Generate unique pairs of amphora types without considering the order
        pairs = {(min(amphora_type_i, amphora_type_j), max(amphora_type_i, amphora_type_j)) for amphora_type_i in \
                 list(amphora_types) for amphora_type_j in list(amphora_types) if amphora_type_i != amphora_type_j}

        # Update the connections dictionary
        for pair in pairs:
            connections.setdefault(pair, 0)
            connections[pair] += 1

    # Storing connections for the current century
    strength_of_connection[century_label] = connections

# Print the strength of connection for each century
# for century, connections in strength_of_connection.items():
#    print(f"Century: {century}")
#    for pair, strength in connections.items():
#        print(f"Amphora Pair: {pair}, Strength of Connection: {strength}")
#    print()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Empirical network
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Dictionary to store the networks for each century (although not used in the following)
amphora_networks = {}

# Unique origins (provenances) from the amphora_origins_all dictionary
all_origins = list(set(origin for origins in amphora_origins_all.values() for origin in origins))

# Create a colormap using gist_rainbow
color_map = plt.cm.get_cmap('gist_rainbow', len(all_origins))

# Dictionary to map origins to unique colors
origin_colors = {origin: color_map(i) for i, origin in enumerate(all_origins)}

# Iterate over the centuries
for century, shipwrecks in shipwreck_dates.items():
    century_label = century  # Keep the entire string for ranges like '4-7 AD'
    connections = {}

    # Starting with an empty network for the current century
    G = nx.Graph()

    # Iterate over the shipwrecks in the current century
    for shipwreck_id, amphora_types in shipwrecks.items():
        # Generate unique pairs of amphora types without considering the order
        pairs = {(min(amphora_type_i, amphora_type_j), max(amphora_type_i, amphora_type_j)) for amphora_type_i in
                 amphora_types for amphora_type_j in amphora_types if amphora_type_i != amphora_type_j}

        # Update the connections dictionary and add edges to the network, excluding self-loops!
        for pair in pairs:
            if pair[0] != pair[1]:
                connections.setdefault(pair, 0)
                connections[pair] += 1
                G.add_edge(pair[0], pair[1], weight=connections[pair])

    # Storing connections for the current century
    strength_of_connection[century_label] = connections

    for node in G.nodes:
        origins = amphora_origins_all.get(node, [])
        G.nodes[node]['origins'] = origins

    # Legend for colors
    legend_labels = list(origin_colors.keys())
    legend_colors = list(origin_colors.values())

    # Assign colors to nodes based on their origins using the color palette defined above
    node_colors = [origin_colors.get(amphora_origins_all.get(node, [])[0]) for node in G.nodes]

    # Extract the weights from the network
    weights = [G[u][v]['weight'] for u, v in G.edges]

    # Scaleing weights
    min_weight = min(weights)
    max_weight = max(weights)
    scaled_weights = [5 * (weight - min_weight) / (max_weight - min_weight) + 1 for weight in weights]

    # Plot the current network
    plt.figure(figsize=(20, 20))

    # Draw node labels separately
    pos = nx.spring_layout(G)
    labels = {node: node for node in G.nodes}
    # Draw edges seperately (due to different transparency alpha)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, alpha=0.9, node_size=1000)
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=scaled_weights)
    nx.draw_networkx_labels(G, pos, labels, font_size=12, font_weight='bold', alpha=1)

    # Identify the unique origins present in the current network
    unique_origins = set(origin for node in G.nodes for origin in G.nodes[node]['origins'])

    # Filtered legend for the current network, so we can explain the nodes colors
    filtered_legend_labels = [origin for origin in legend_labels if origin in unique_origins]
    filtered_legend_handles = [Patch(facecolor=origin_colors[origin]) for origin in filtered_legend_labels]
    plt.legend(filtered_legend_handles, filtered_legend_labels, loc='upper right')
    plt.show()

    # Assign the network created for each time stamp to the dictionary
    amphora_networks[century_label] = G

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Randomization
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++


# Now we start with the randomization processes, where we exchange amphoras between the shipwrecks cargoes
all_shipwrecks = merged_df_all.groupby('Oxford_wreckID')['Amphora type'].apply(list).reset_index()


def interchange_cargoes(all_shipwrecks, production_times_all, num_randomizations):
    num_frames = 500
    data_frames = []

    for frame in range(num_frames):
        frame_data = all_shipwrecks.copy()  # Create a copy of all_shipwrecks for each new frame

        interchange_count = 0  # Count valid interchanges (We want 1000)

        while interchange_count < num_randomizations:

            # Step 1: Randomly select two shipwrecks
            index_1 = random.randint(0, len(frame_data) - 1)
            index_2 = random.randint(0, len(frame_data) - 1)
            shipwreck_1 = frame_data.iloc[index_1]
            shipwreck_2 = frame_data.iloc[index_2]

            # Step 2: Randomly select one amphora type from each cargo
            cargo_1 = shipwreck_1['Amphora type']
            cargo_2 = shipwreck_2['Amphora type']

            # Check if both cargoes are single values
            if len(cargo_1) == 1 and len(cargo_2) == 1:
                continue  # Skip the interchange and start again from step 1 (single amphora interchange has no impact)

            amphora_1 = random.choice(cargo_1)
            amphora_2 = random.choice(cargo_2)

            # Step 3: Check if the new cargoes have overlapping production times
            new_cargo_1 = cargo_1.copy()
            new_cargo_1.remove(amphora_1)
            new_cargo_1.append(amphora_2)

            new_cargo_2 = cargo_2.copy()
            new_cargo_2.remove(amphora_2)
            new_cargo_2.append(amphora_1)

            # Check if the new cargoes have overlapping production times in both
            if check_production_time_overlap(new_cargo_1, production_times_all) and \
               check_production_time_overlap(new_cargo_2, production_times_all):

                # Step 4: Check if the new cargoes are not already present in 'all_shipwrecks'
                if not check_duplicate_cargos(new_cargo_1, all_shipwrecks) and \
                   not check_duplicate_cargos(new_cargo_2, all_shipwrecks):
                    # Valid interchange found, accept it
                    frame_data.at[index_1, 'Amphora type'] = new_cargo_1
                    frame_data.at[index_2, 'Amphora type'] = new_cargo_2
                    interchange_count += 1

        data_frames.append(frame_data)  # Append the modified frame_data to data_frames

    return data_frames


def check_production_time_overlap(cargo, production_times_all):
    time_periods = [production_times_all.get(amphora_type_all, []) for amphora_type_all in cargo]
    all_time_periods = set.intersection(*map(set, time_periods))
    return bool(all_time_periods)


def check_duplicate_cargos(cargo, all_shipwrecks):
    for _, row in all_shipwrecks.iterrows():
        if set(cargo) == set(row['Amphora type']):
            return True
    return False


# Usage
rand_list_0 = [all_shipwrecks]
rand_list_100 = interchange_cargoes(all_shipwrecks, production_times, 100)
rand_list_300 = interchange_cargoes(all_shipwrecks, production_times, 300)
rand_list_500 = interchange_cargoes(all_shipwrecks, production_times, 500)
rand_list_800 = interchange_cargoes(all_shipwrecks, production_times, 800)
rand_list_1000 = interchange_cargoes(all_shipwrecks, production_times, 1000)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Manipulate evolution frames for metrics
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

# We start by deleting single values for all evolution data frames

def no_singles_in_rand(rand_list):
    for key, df in enumerate(rand_list):
        # Convert 'Amphora type' column to lists
        df['Amphora type'] = df['Amphora type'].apply(lambda x: [x] if isinstance(x, str) else x)
        # Count the number of entries in the 'Amphora type' column
        df['Amphora Count'] = df['Amphora type'].apply(len)
        # Make a copy of the sliced data frame
        df_filtered = df[df['Amphora Count'] > 1].copy()
        # Remove 'Amphora Count' column
        df_filtered.drop('Amphora Count', axis=1, inplace=True)
        # Update the data frame in the list
        rand_list[key] = df_filtered
    return rand_list


# Randomized networks (rand_list_0 is the empirical network)
rand_list_0 = no_singles_in_rand(rand_list_0)
rand_list_100 = no_singles_in_rand(rand_list_100)
rand_list_300 = no_singles_in_rand(rand_list_300)
rand_list_500 = no_singles_in_rand(rand_list_500)
rand_list_800 = no_singles_in_rand(rand_list_800)
rand_list_1000 = no_singles_in_rand(rand_list_1000)

# Copys of randomizations to analyze without slicing
no_slice_0 = rand_list_0
no_slice_100 = rand_list_100
no_slice_300 = rand_list_300
no_slice_500 = rand_list_500
no_slice_800 = rand_list_800
no_slice_1000 = rand_list_1000

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Date each rand_list_XXXX
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++


# Update get_ship_dates function to modify the dating periods based on time stamps
def get_ship_dates(df, production_times):
    ship_dates = set()
    for _, row in df.iterrows():
        cargo = row['Amphora type']
        cargo_production_times = [production_times.get(amphora, []) for amphora in cargo]
        # Check for overlaps among production times in the cargo
        overlaps = set.intersection(*map(set, cargo_production_times))
        if overlaps:
            ship_dates.update(overlaps)

    # Group the dating periods based on our five time stamps (BC, 1AD, 2AD, 3AD, 4-7AD)
    modified_ship_dates = set()
    for date in ship_dates:
        if date.endswith(" BC"):
            if date.startswith("2 "):
                modified_ship_dates.add("BC")
            if date.startswith("1 "):
                modified_ship_dates.add("BC")
        elif date.startswith("4 AD") or date.startswith("5 AD") or date.startswith("6 AD") or date.startswith("7 AD"):
            modified_ship_dates.add("4-7 AD")
        else:
            modified_ship_dates.add(date)
    return modified_ship_dates


# Update date_ships_in_list function to use the modified dating periods
def date_ships_in_list(df_list, production_times):
    for df in df_list:
        # Create a new column 'Dating Periods' in the data frame to store shipwrecks time stamp
        df['Dating Periods'] = ''
        # Iterate over each row in the data frame and get the time stamp for each shipwreck
        for _, group in df.groupby('Oxford_wreckID'):
            ship_dates = get_ship_dates(group, production_times)
            dating_periods = ', '.join(ship_dates) if ship_dates else 'Cannot be dated'
            df.loc[group.index, 'Dating Periods'] = dating_periods


date_ships_in_list(rand_list_0, production_times)
date_ships_in_list(rand_list_100, production_times)
date_ships_in_list(rand_list_300, production_times)
date_ships_in_list(rand_list_500, production_times)
date_ships_in_list(rand_list_800, production_times)
date_ships_in_list(rand_list_1000, production_times)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Print the updated data frames with the added 'Dating Periods' column
#for i, df in enumerate(rand_list_0):
#    print(f"Data Frame {i+1}:")
#    print(df)
#    print()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Filter for time stamps
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++


# Function to filter and group data frames based on the time stamps
def filter_and_group_dataframes(df_list):
    bc_ships_list = []
    ad1_ships_list = []
    ad2_ships_list = []
    ad3_ships_list = []
    ad4_7_ships_list = []

    # Iterate through each data frame in the list
    for df in df_list:
        # Filter shipwrecks dated in "BC" (2 BC and 1 BC)
        bc_ships = df[df['Dating Periods'].str.contains('BC')]
        bc_ships_list.append(bc_ships)

        # Filter for "1 AD"
        ad1_ships = df[df['Dating Periods'].str.contains('1 AD')]
        ad1_ships_list.append(ad1_ships)

        # Filter for "2 AD"
        ad2_ships = df[df['Dating Periods'].str.contains('2 AD')]
        ad2_ships_list.append(ad2_ships)

        # Filter for "3 AD"
        ad3_ships = df[df['Dating Periods'].str.contains('3 AD')]
        ad3_ships_list.append(ad3_ships)

        # Filter for "4-7 AD" (4 AD, 5 AD, 6 AD, and 7 AD)
        ad4_7_ships = df[df['Dating Periods'].str.contains('4-7 AD')]
        ad4_7_ships_list.append(ad4_7_ships)

    return bc_ships_list, ad1_ships_list, ad2_ships_list, ad3_ships_list, ad4_7_ships_list


# Apply the function to all six lists
bc_ships_0, ad1_ships_0, ad2_ships_0, ad3_ships_0, ad4_7_ships_0 = filter_and_group_dataframes(rand_list_0)
bc_ships_100, ad1_ships_100, ad2_ships_100, ad3_ships_100, ad4_7_ships_100 = filter_and_group_dataframes(rand_list_100)
bc_ships_300, ad1_ships_300, ad2_ships_300, ad3_ships_300, ad4_7_ships_300 = filter_and_group_dataframes(rand_list_300)
bc_ships_500, ad1_ships_500, ad2_ships_500, ad3_ships_500, ad4_7_ships_500 = filter_and_group_dataframes(rand_list_500)
bc_ships_800, ad1_ships_800, ad2_ships_800, ad3_ships_800, ad4_7_ships_800 = filter_and_group_dataframes(rand_list_800)
bc_ships_1000, ad1_ships_1000, ad2_ships_1000, ad3_ships_1000, ad4_7_ships_1000 = filter_and_group_dataframes(rand_list_1000)

# Print the lengths of each list to verify the number of shipwrecks in each time stamp
print("Number of shipwrecks in 'BC' time stamp:", len(bc_ships_0[0]))
print("Number of shipwrecks in '1 AD' time stamp:", len(ad1_ships_0[0]))
print("Number of shipwrecks in '2 AD' time stamp:", len(ad2_ships_0[0]))
print("Number of shipwrecks in '3 AD' time stamp:", len(ad3_ships_0[0]))
print("Number of shipwrecks in '4-7 AD' time stamp:", len(ad4_7_ships_0[0]))

# Now we have 5 list (for each time stamp) for 0, 100, 300, 500, 800, 1000 randomizations, respectively.
# From here we will analyze each data frame by creating a network and analyzing its properties, which will then be
# averaged, so that we obtain for each of the 5 investigated properties an average value for each time stamp at each
# number of realized randomizations.

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Analyze each networks properties for time stamps and randomization steps
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++


def create_weighted_graph(df):
    # Empty graph
    G = nx.Graph()
    # Drop "Dating Periods" column
    df_copy = df.drop(columns=['Dating Periods'])
    # Iterate over each row
    for _, row in df_copy.iterrows():
        cargo = row['Amphora type']
        # Add nodes to the graph for each amphora type in the cargo
        for amphora in cargo:
            G.add_node(amphora)
        # Add edges to the graph for all pairs of amphora types in the cargo
        for i in range(len(cargo)):
            for j in range(i + 1, len(cargo)):
                amphora_1, amphora_2 = cargo[i], cargo[j]
                if G.has_edge(amphora_1, amphora_2):
                    # If edge already exists, increment the weight by 1
                    G[amphora_1][amphora_2]['weight'] += 1
                else:
                    # If no edge yet, add it with weight 1
                    G.add_edge(amphora_1, amphora_2, weight=1)
    return G


###
# 0 Randomizations
###

# Create a list to store the graphs for the dating periods
bc_ships_graphs_0 = []
for df in bc_ships_0:
    graph = create_weighted_graph(df)
    bc_ships_graphs_0.append(graph)

# Create a list to store the graphs for the dating periods
ad1_ships_graphs_0 = []
for df in ad1_ships_0:
    graph = create_weighted_graph(df)
    ad1_ships_graphs_0.append(graph)

# Create a list to store the graphs for the dating periods
ad2_ships_graphs_0 = []
for df in ad2_ships_0:
    graph = create_weighted_graph(df)
    ad2_ships_graphs_0.append(graph)

# Create a list to store the graphs for the dating periods
ad3_ships_graphs_0 = []
for df in ad3_ships_0:
    graph = create_weighted_graph(df)
    ad3_ships_graphs_0.append(graph)

# Create a list to store the graphs for the dating periods
ad4_7_ships_graphs_0 = []
for df in ad4_7_ships_0:
    graph = create_weighted_graph(df)
    ad4_7_ships_graphs_0.append(graph)

###
# 100 Randomizations
###

# Create a list to store the graphs for the dating periods
bc_ships_graphs_100 = []
for df in bc_ships_100:
    graph = create_weighted_graph(df)
    bc_ships_graphs_100.append(graph)

# Create a list to store the graphs for the dating periods
ad1_ships_graphs_100 = []
for df in ad1_ships_100:
    graph = create_weighted_graph(df)
    ad1_ships_graphs_100.append(graph)
    
# Create a list to store the graphs for the dating periods
ad2_ships_graphs_100 = []
for df in ad2_ships_100:
    graph = create_weighted_graph(df)
    ad2_ships_graphs_100.append(graph)
    
# Create a list to store the graphs for the dating periods
ad3_ships_graphs_100 = []
for df in ad3_ships_100:
    graph = create_weighted_graph(df)
    ad3_ships_graphs_100.append(graph)
    
# Create a list to store the graphs for the dating periods
ad4_7_ships_graphs_100 = []
for df in ad4_7_ships_100:
    graph = create_weighted_graph(df)
    ad4_7_ships_graphs_100.append(graph)

###
# 300 Randomizations
###

# Create a list to store the graphs for the dating periods
bc_ships_graphs_300 = []
for df in bc_ships_300:
    graph = create_weighted_graph(df)
    bc_ships_graphs_300.append(graph)

# Create a list to store the graphs for the dating periods
ad1_ships_graphs_300 = []
for df in ad1_ships_300:
    graph = create_weighted_graph(df)
    ad1_ships_graphs_300.append(graph)

# Create a list to store the graphs for the dating periods
ad2_ships_graphs_300 = []
for df in ad2_ships_300:
    graph = create_weighted_graph(df)
    ad2_ships_graphs_300.append(graph)

# Create a list to store the graphs for the dating periods
ad3_ships_graphs_300 = []
for df in ad3_ships_300:
    graph = create_weighted_graph(df)
    ad3_ships_graphs_300.append(graph)

# Create a list to store the graphs for the dating periods
ad4_7_ships_graphs_300 = []
for df in ad4_7_ships_300:
    graph = create_weighted_graph(df)
    ad4_7_ships_graphs_300.append(graph)

###
# 500 Randomizations
###

# Create a list to store the graphs for the dating periods
bc_ships_graphs_500 = []
for df in bc_ships_500:
    graph = create_weighted_graph(df)
    bc_ships_graphs_500.append(graph)

# Create a list to store the graphs for the dating periods
ad1_ships_graphs_500 = []
for df in ad1_ships_500:
    graph = create_weighted_graph(df)
    ad1_ships_graphs_500.append(graph)

# Create a list to store the graphs for the dating periods
ad2_ships_graphs_500 = []
for df in ad2_ships_500:
    graph = create_weighted_graph(df)
    ad2_ships_graphs_500.append(graph)

# Create a list to store the graphs for the dating periods
ad3_ships_graphs_500 = []
for df in ad3_ships_500:
    graph = create_weighted_graph(df)
    ad3_ships_graphs_500.append(graph)

# Create a list to store the graphs for the dating periods
ad4_7_ships_graphs_500 = []
for df in ad4_7_ships_500:
    graph = create_weighted_graph(df)
    ad4_7_ships_graphs_500.append(graph)

###
# 800 Randomizations
###

# Create a list to store the graphs for the dating periods
bc_ships_graphs_800 = []
for df in bc_ships_800:
    graph = create_weighted_graph(df)
    bc_ships_graphs_800.append(graph)

# Create a list to store the graphs for the dating periods
ad1_ships_graphs_800 = []
for df in ad1_ships_800:
    graph = create_weighted_graph(df)
    ad1_ships_graphs_800.append(graph)

# Create a list to store the graphs for the dating periods
ad2_ships_graphs_800 = []
for df in ad2_ships_800:
    graph = create_weighted_graph(df)
    ad2_ships_graphs_800.append(graph)

# Create a list to store the graphs for the dating periods
ad3_ships_graphs_800 = []
for df in ad3_ships_800:
    graph = create_weighted_graph(df)
    ad3_ships_graphs_800.append(graph)

# Create a list to store the graphs for the dating periods
ad4_7_ships_graphs_800 = []
for df in ad4_7_ships_800:
    graph = create_weighted_graph(df)
    ad4_7_ships_graphs_800.append(graph)

###
# 1000 Randomizations
###

# Create a list to store the graphs for the dating periods
bc_ships_graphs_1000 = []
for df in bc_ships_1000:
    graph = create_weighted_graph(df)
    bc_ships_graphs_1000.append(graph)

# Create a list to store the graphs for the dating periods
ad1_ships_graphs_1000 = []
for df in ad1_ships_1000:
    graph = create_weighted_graph(df)
    ad1_ships_graphs_1000.append(graph)

# Create a list to store the graphs for the dating periods
ad2_ships_graphs_1000 = []
for df in ad2_ships_1000:
    graph = create_weighted_graph(df)
    ad2_ships_graphs_1000.append(graph)

# Create a list to store the graphs for the dating periods
ad3_ships_graphs_1000 = []
for df in ad3_ships_1000:
    graph = create_weighted_graph(df)
    ad3_ships_graphs_1000.append(graph)

# Create a list to store the graphs for the dating periods
ad4_7_ships_graphs_1000 = []
for df in ad4_7_ships_1000:
    graph = create_weighted_graph(df)
    ad4_7_ships_graphs_1000.append(graph)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Include same procedure for the non-sliced data frames after each randomization step
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

# 0
no_slice_ships_graphs_0 = []
for df in no_slice_0:
    graph = create_weighted_graph(df)
    no_slice_ships_graphs_0.append(graph)

# 100
no_slice_ships_graphs_100 = []
for df in no_slice_100:
    graph = create_weighted_graph(df)
    no_slice_ships_graphs_100.append(graph)

# 300
no_slice_ships_graphs_300 = []
for df in no_slice_300:
    graph = create_weighted_graph(df)
    no_slice_ships_graphs_300.append(graph)

# 500
no_slice_ships_graphs_500 = []
for df in no_slice_500:
    graph = create_weighted_graph(df)
    no_slice_ships_graphs_500.append(graph)

# 800
no_slice_ships_graphs_800 = []
for df in no_slice_800:
    graph = create_weighted_graph(df)
    no_slice_ships_graphs_800.append(graph)

# 1000
no_slice_ships_graphs_1000 = []
for df in no_slice_1000:
    graph = create_weighted_graph(df)
    no_slice_ships_graphs_1000.append(graph)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Average clustering coefficient's evolution
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

avg_cc_no_slice = []
avg_cc_bc = []
avg_cc_ad1 = []
avg_cc_ad2 = []
avg_cc_ad3 = []
avg_cc_ad4_7 = []

# Iterate over bc_ships_graphs_0, bc_ships_graphs_100, etc.
for graph_list in [bc_ships_graphs_0, bc_ships_graphs_100, bc_ships_graphs_300, bc_ships_graphs_500, bc_ships_graphs_800, bc_ships_graphs_1000]:
    avg_cc_per_set = []  # List to store average clustering coefficients for each graph in the set
    for graph in graph_list:
        avg_clustering_coefficient = nx.average_clustering(graph)
        avg_cc_per_set.append(avg_clustering_coefficient)
    overall_avg_cc = sum(avg_cc_per_set) / len(avg_cc_per_set)
    avg_cc_bc.append(overall_avg_cc)

# Repeat the same process for the other sets of graphs
for graph_list in [ad1_ships_graphs_0, ad1_ships_graphs_100, ad1_ships_graphs_300, ad1_ships_graphs_500, ad1_ships_graphs_800, ad1_ships_graphs_1000]:
    avg_cc_per_set = []  # List to store average clustering coefficients for each graph in the set
    for graph in graph_list:
        avg_clustering_coefficient = nx.average_clustering(graph)
        avg_cc_per_set.append(avg_clustering_coefficient)
    overall_avg_cc = sum(avg_cc_per_set) / len(avg_cc_per_set)
    avg_cc_ad1.append(overall_avg_cc)

# Repeat the same process for the other sets of graphs
for graph_list in [ad2_ships_graphs_0, ad2_ships_graphs_100, ad2_ships_graphs_300, ad2_ships_graphs_500, ad2_ships_graphs_800, ad2_ships_graphs_1000]:
    avg_cc_per_set = []  # List to store average clustering coefficients for each graph in the set
    for graph in graph_list:
        avg_clustering_coefficient = nx.average_clustering(graph)
        avg_cc_per_set.append(avg_clustering_coefficient)
    overall_avg_cc = sum(avg_cc_per_set) / len(avg_cc_per_set)
    avg_cc_ad2.append(overall_avg_cc)

# Repeat the same process for the other sets of graphs
for graph_list in [ad3_ships_graphs_0, ad3_ships_graphs_100, ad3_ships_graphs_300, ad3_ships_graphs_500, ad3_ships_graphs_800, ad3_ships_graphs_1000]:
    avg_cc_per_set = []  # List to store average clustering coefficients for each graph in the set
    for graph in graph_list:
        avg_clustering_coefficient = nx.average_clustering(graph)
        avg_cc_per_set.append(avg_clustering_coefficient)
    overall_avg_cc = sum(avg_cc_per_set) / len(avg_cc_per_set)
    avg_cc_ad3.append(overall_avg_cc)

# Repeat the same process for the other sets of graphs
for graph_list in [ad4_7_ships_graphs_0, ad4_7_ships_graphs_100, ad4_7_ships_graphs_300, ad4_7_ships_graphs_500, ad4_7_ships_graphs_800, ad4_7_ships_graphs_1000]:
    avg_cc_per_set = []  # List to store average clustering coefficients for each graph in the set
    for graph in graph_list:
        avg_clustering_coefficient = nx.average_clustering(graph)
        avg_cc_per_set.append(avg_clustering_coefficient)
    overall_avg_cc = sum(avg_cc_per_set) / len(avg_cc_per_set)
    avg_cc_ad4_7.append(overall_avg_cc)

# Iterate over non-sliced graphs
for graph_list in [no_slice_ships_graphs_0, no_slice_ships_graphs_100, no_slice_ships_graphs_300, no_slice_ships_graphs_500, no_slice_ships_graphs_800, no_slice_ships_graphs_1000]:
    avg_cc_per_set = []  # List to store average clustering coefficients for each graph in the set
    for graph in graph_list:
        avg_clustering_coefficient = nx.average_clustering(graph)
        avg_cc_per_set.append(avg_clustering_coefficient)
    overall_avg_cc = sum(avg_cc_per_set) / len(avg_cc_per_set)
    avg_cc_no_slice.append(overall_avg_cc)

# Plot the evolution
# X values for the time stamps
x_values = [0, 100, 300, 500, 800, 1000]

# Plotting the average clustering coefficients
plt.figure(figsize=(10, 6))
plt.style.use('science')
plt.errorbar(x_values, avg_cc_no_slice, yerr=np.std(avg_cc_no_slice), marker='x', color='black', linestyle='--', label='Total', capsize=5)
plt.errorbar(x_values, avg_cc_bc, yerr=np.std(avg_cc_bc), marker='o', linestyle='--', label='BC', capsize=5)
plt.errorbar(x_values, avg_cc_ad1, yerr=np.std(avg_cc_ad1), marker='v', linestyle='--', label='1 AD', capsize=5)
plt.errorbar(x_values, avg_cc_ad2, yerr=np.std(avg_cc_ad2), marker='s', linestyle='--', label='2 AD', capsize=5)
plt.errorbar(x_values, avg_cc_ad3, yerr=np.std(avg_cc_ad3), marker='^', linestyle='--', label='3 AD', capsize=5)
plt.errorbar(x_values, avg_cc_ad4_7, yerr=np.std(avg_cc_ad4_7), marker='D', linestyle='--', label='4-7 AD', capsize=5)
plt.xticks(x_values)
plt.xlabel('Randomization Counts')
plt.ylabel(r'Average Clustering Coefficient $\langle C \rangle$')
plt.legend()

plt.show()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Giant component's evolution
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

giant_size_no_slice = []
giant_size_bc = []
giant_size_ad1 = []
giant_size_ad2 = []
giant_size_ad3 = []
giant_size_ad4_7 = []

# Iterate over bc_ships_graphs_0, bc_ships_graphs_100, etc.
for graph_list in [bc_ships_graphs_0, bc_ships_graphs_100, bc_ships_graphs_300, bc_ships_graphs_500, bc_ships_graphs_800, bc_ships_graphs_1000]:
    giant_size_per_set = []  # List to store giant component sizes for each graph in the set
    for graph in graph_list:
        giant_component_size = len(max(nx.connected_components(graph), key=len))
        giant_size_per_set.append(giant_component_size)
    avg_giant_size = sum(giant_size_per_set) / len(giant_size_per_set)
    giant_size_bc.append(avg_giant_size)

# Repeat the same process for the other sets of graphs
for graph_list in [ad1_ships_graphs_0, ad1_ships_graphs_100, ad1_ships_graphs_300, ad1_ships_graphs_500, ad1_ships_graphs_800, ad1_ships_graphs_1000]:
    giant_size_per_set = []  # List to store giant component sizes for each graph in the set
    for graph in graph_list:
        giant_component_size = len(max(nx.connected_components(graph), key=len))
        giant_size_per_set.append(giant_component_size)
    avg_giant_size = sum(giant_size_per_set) / len(giant_size_per_set)
    giant_size_ad1.append(avg_giant_size)

# Repeat the same process for the other sets of graphs
for graph_list in [ad2_ships_graphs_0, ad2_ships_graphs_100, ad2_ships_graphs_300, ad2_ships_graphs_500, ad2_ships_graphs_800, ad2_ships_graphs_1000]:
    giant_size_per_set = []  # List to store giant component sizes for each graph in the set
    for graph in graph_list:
        giant_component_size = len(max(nx.connected_components(graph), key=len))
        giant_size_per_set.append(giant_component_size)
    avg_giant_size = sum(giant_size_per_set) / len(giant_size_per_set)
    giant_size_ad2.append(avg_giant_size)

# Repeat the same process for the other sets of graphs
for graph_list in [ad3_ships_graphs_0, ad3_ships_graphs_100, ad3_ships_graphs_300, ad3_ships_graphs_500, ad3_ships_graphs_800, ad3_ships_graphs_1000]:
    giant_size_per_set = []  # List to store giant component sizes for each graph in the set
    for graph in graph_list:
        giant_component_size = len(max(nx.connected_components(graph), key=len))
        giant_size_per_set.append(giant_component_size)
    avg_giant_size = sum(giant_size_per_set) / len(giant_size_per_set)
    giant_size_ad3.append(avg_giant_size)

# Repeat the same process for the other sets of graphs
for graph_list in [ad4_7_ships_graphs_0, ad4_7_ships_graphs_100, ad4_7_ships_graphs_300, ad4_7_ships_graphs_500, ad4_7_ships_graphs_800, ad4_7_ships_graphs_1000]:
    giant_size_per_set = []  # List to store giant component sizes for each graph in the set
    for graph in graph_list:
        giant_component_size = len(max(nx.connected_components(graph), key=len))
        giant_size_per_set.append(giant_component_size)
    avg_giant_size = sum(giant_size_per_set) / len(giant_size_per_set)
    giant_size_ad4_7.append(avg_giant_size)

# Iterate over non-sliced graphs
for graph_list in [no_slice_ships_graphs_0, no_slice_ships_graphs_100, no_slice_ships_graphs_300, no_slice_ships_graphs_500, no_slice_ships_graphs_800, no_slice_ships_graphs_1000]:
    giant_size_per_set = []  # List to store giant component sizes for each graph in the set
    for graph in graph_list:
        giant_component_size = len(max(nx.connected_components(graph), key=len))
        giant_size_per_set.append(giant_component_size)
    avg_giant_size = sum(giant_size_per_set) / len(giant_size_per_set)
    giant_size_no_slice.append(avg_giant_size)

# Plot the evolution
# X values for the time stamps
x_values = [0, 100, 300, 500, 800, 1000]

# Plotting the giant component sizes
plt.figure(figsize=(10, 6))
plt.style.use('science')
plt.errorbar(x_values, giant_size_no_slice, yerr=np.std(giant_size_no_slice), marker='x', color='black', linestyle='--', label='Total', capsize=5)
plt.errorbar(x_values, giant_size_bc, yerr=np.std(giant_size_bc), marker='o', linestyle='--', label='BC', capsize=5)
plt.errorbar(x_values, giant_size_ad1, yerr=np.std(giant_size_ad1), marker='v', linestyle='--', label='1 AD', capsize=5)
plt.errorbar(x_values, giant_size_ad2, yerr=np.std(giant_size_ad2), marker='s', linestyle='--', label='2 AD', capsize=5)
plt.errorbar(x_values, giant_size_ad3, yerr=np.std(giant_size_ad3), marker='^', linestyle='--', label='3 AD', capsize=5)
plt.errorbar(x_values, giant_size_ad4_7, yerr=np.std(giant_size_ad4_7), marker='D', linestyle='--', label='4-7 AD', capsize=5)
plt.xticks(x_values)
plt.xlabel('Randomization Counts')
plt.ylabel(r'Average Giant Component Size $S_{GC}$')
plt.legend()

plt.show()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Average degree's evolution
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

avg_degree_no_slice = []
avg_degree_bc = []
avg_degree_ad1 = []
avg_degree_ad2 = []
avg_degree_ad3 = []
avg_degree_ad4_7 = []

# Iterate over bc_ships_graphs_0, bc_ships_graphs_100, etc.
for graph_list in [bc_ships_graphs_0, bc_ships_graphs_100, bc_ships_graphs_300, bc_ships_graphs_500, bc_ships_graphs_800, bc_ships_graphs_1000]:
    avg_degree_per_set = []  # List to store average degrees for each graph in the set
    for graph in graph_list:
        avg_node_degree = sum(dict(graph.degree()).values()) / len(graph)
        avg_degree_per_set.append(avg_node_degree)
    overall_avg_degree = sum(avg_degree_per_set) / len(avg_degree_per_set)
    avg_degree_bc.append(overall_avg_degree)

# Repeat the same process for the other sets of graphs
for graph_list in [ad1_ships_graphs_0, ad1_ships_graphs_100, ad1_ships_graphs_300, ad1_ships_graphs_500, ad1_ships_graphs_800, ad1_ships_graphs_1000]:
    avg_degree_per_set = []  # List to store average degrees for each graph in the set
    for graph in graph_list:
        avg_node_degree = sum(dict(graph.degree()).values()) / len(graph)
        avg_degree_per_set.append(avg_node_degree)
    overall_avg_degree = sum(avg_degree_per_set) / len(avg_degree_per_set)
    avg_degree_ad1.append(overall_avg_degree)

# Repeat the same process for the other sets of graphs
for graph_list in [ad2_ships_graphs_0, ad2_ships_graphs_100, ad2_ships_graphs_300, ad2_ships_graphs_500, ad2_ships_graphs_800, ad2_ships_graphs_1000]:
    avg_degree_per_set = []  # List to store average degrees for each graph in the set
    for graph in graph_list:
        avg_node_degree = sum(dict(graph.degree()).values()) / len(graph)
        avg_degree_per_set.append(avg_node_degree)
    overall_avg_degree = sum(avg_degree_per_set) / len(avg_degree_per_set)
    avg_degree_ad2.append(overall_avg_degree)

# Repeat the same process for the other sets of graphs
for graph_list in [ad3_ships_graphs_0, ad3_ships_graphs_100, ad3_ships_graphs_300, ad3_ships_graphs_500, ad3_ships_graphs_800, ad3_ships_graphs_1000]:
    avg_degree_per_set = []  # List to store average degrees for each graph in the set
    for graph in graph_list:
        avg_node_degree = sum(dict(graph.degree()).values()) / len(graph)
        avg_degree_per_set.append(avg_node_degree)
    overall_avg_degree = sum(avg_degree_per_set) / len(avg_degree_per_set)
    avg_degree_ad3.append(overall_avg_degree)

# Repeat the same process for the other sets of graphs
for graph_list in [ad4_7_ships_graphs_0, ad4_7_ships_graphs_100, ad4_7_ships_graphs_300, ad4_7_ships_graphs_500, ad4_7_ships_graphs_800, ad4_7_ships_graphs_1000]:
    avg_degree_per_set = []  # List to store average degrees for each graph in the set
    for graph in graph_list:
        avg_node_degree = sum(dict(graph.degree()).values()) / len(graph)
        avg_degree_per_set.append(avg_node_degree)
    overall_avg_degree = sum(avg_degree_per_set) / len(avg_degree_per_set)
    avg_degree_ad4_7.append(overall_avg_degree)

# Iterate over non-sliced graphs
for graph_list in [no_slice_ships_graphs_0, no_slice_ships_graphs_100, no_slice_ships_graphs_300, no_slice_ships_graphs_500, no_slice_ships_graphs_800, no_slice_ships_graphs_1000]:
    avg_degree_per_set = []  # List to store average degrees for each graph in the set
    for graph in graph_list:
        avg_node_degree = sum(dict(graph.degree()).values()) / len(graph)
        avg_degree_per_set.append(avg_node_degree)
    overall_avg_degree = sum(avg_degree_per_set) / len(avg_degree_per_set)
    avg_degree_no_slice.append(overall_avg_degree)

# Plot the evolution
# X values for the time stamps
x_values = [0, 100, 300, 500, 800, 1000]

# Plotting the average degrees
plt.figure(figsize=(10, 6))
plt.style.use('science')
plt.errorbar(x_values, avg_degree_no_slice, yerr=np.std(avg_degree_no_slice), marker='x', color='black', linestyle='--', label='Total', capsize=5)
plt.errorbar(x_values, avg_degree_bc, yerr=np.std(avg_degree_bc), marker='o', linestyle='--', label='BC', capsize=5)
plt.errorbar(x_values, avg_degree_ad1, yerr=np.std(avg_degree_ad1), marker='v', linestyle='--', label='1 AD', capsize=5)
plt.errorbar(x_values, avg_degree_ad2, yerr=np.std(avg_degree_ad2), marker='s', linestyle='--', label='2 AD', capsize=5)
plt.errorbar(x_values, avg_degree_ad3, yerr=np.std(avg_degree_ad3), marker='^', linestyle='--', label='3 AD', capsize=5)
plt.errorbar(x_values, avg_degree_ad4_7, yerr=np.std(avg_degree_ad4_7), marker='D', linestyle='--', label='4-7 AD', capsize=5)
plt.xticks(x_values)
plt.xlabel('Randomization Counts')
plt.ylabel(r'Average Degree $\langle k \rangle$')
plt.legend()

plt.show()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Average weighted degree's evolution
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

avg_weighted_degree_no_slice = []
avg_weighted_degree_bc = []
avg_weighted_degree_ad1 = []
avg_weighted_degree_ad2 = []
avg_weighted_degree_ad3 = []
avg_weighted_degree_ad4_7 = []

# Iterate over bc_ships_graphs_0, bc_ships_graphs_100, etc.
for graph_list in [bc_ships_graphs_0, bc_ships_graphs_100, bc_ships_graphs_300, bc_ships_graphs_500, bc_ships_graphs_800, bc_ships_graphs_1000]:
    avg_weighted_degree_per_set = []  # List to store average weighted degrees for each graph in the set
    for graph in graph_list:
        avg_weighted_node_degree = sum(dict(graph.degree(weight='weight')).values()) / len(graph)
        avg_weighted_degree_per_set.append(avg_weighted_node_degree)
    overall_avg_weighted_degree = sum(avg_weighted_degree_per_set) / len(avg_weighted_degree_per_set)
    avg_weighted_degree_bc.append(overall_avg_weighted_degree)

# Repeat the same process for the other sets of graphs
for graph_list in [ad1_ships_graphs_0, ad1_ships_graphs_100, ad1_ships_graphs_300, ad1_ships_graphs_500, ad1_ships_graphs_800, ad1_ships_graphs_1000]:
    avg_weighted_degree_per_set = []  # List to store average weighted degrees for each graph in the set
    for graph in graph_list:
        avg_weighted_node_degree = sum(dict(graph.degree(weight='weight')).values()) / len(graph)
        avg_weighted_degree_per_set.append(avg_weighted_node_degree)
    overall_avg_weighted_degree = sum(avg_weighted_degree_per_set) / len(avg_weighted_degree_per_set)
    avg_weighted_degree_ad1.append(overall_avg_weighted_degree)

# Repeat the same process for the other sets of graphs
for graph_list in [ad2_ships_graphs_0, ad2_ships_graphs_100, ad2_ships_graphs_300, ad2_ships_graphs_500, ad2_ships_graphs_800, ad2_ships_graphs_1000]:
    avg_weighted_degree_per_set = []  # List to store average weighted degrees for each graph in the set
    for graph in graph_list:
        avg_weighted_node_degree = sum(dict(graph.degree(weight='weight')).values()) / len(graph)
        avg_weighted_degree_per_set.append(avg_weighted_node_degree)
    overall_avg_weighted_degree = sum(avg_weighted_degree_per_set) / len(avg_weighted_degree_per_set)
    avg_weighted_degree_ad2.append(overall_avg_weighted_degree)

# Repeat the same process for the other sets of graphs
for graph_list in [ad3_ships_graphs_0, ad3_ships_graphs_100, ad3_ships_graphs_300, ad3_ships_graphs_500, ad3_ships_graphs_800, ad3_ships_graphs_1000]:
    avg_weighted_degree_per_set = []  # List to store average weighted degrees for each graph in the set
    for graph in graph_list:
        avg_weighted_node_degree = sum(dict(graph.degree(weight='weight')).values()) / len(graph)
        avg_weighted_degree_per_set.append(avg_weighted_node_degree)
    overall_avg_weighted_degree = sum(avg_weighted_degree_per_set) / len(avg_weighted_degree_per_set)
    avg_weighted_degree_ad3.append(overall_avg_weighted_degree)

# Repeat the same process for the other sets of graphs
for graph_list in [ad4_7_ships_graphs_0, ad4_7_ships_graphs_100, ad4_7_ships_graphs_300, ad4_7_ships_graphs_500, ad4_7_ships_graphs_800, ad4_7_ships_graphs_1000]:
    avg_weighted_degree_per_set = []  # List to store average weighted degrees for each graph in the set
    for graph in graph_list:
        avg_weighted_node_degree = sum(dict(graph.degree(weight='weight')).values()) / len(graph)
        avg_weighted_degree_per_set.append(avg_weighted_node_degree)
    overall_avg_weighted_degree = sum(avg_weighted_degree_per_set) / len(avg_weighted_degree_per_set)
    avg_weighted_degree_ad4_7.append(overall_avg_weighted_degree)

# Iterate over non-sliced graphs
for graph_list in [no_slice_ships_graphs_0, no_slice_ships_graphs_100, no_slice_ships_graphs_300, no_slice_ships_graphs_500, no_slice_ships_graphs_800, no_slice_ships_graphs_1000]:
    avg_weighted_degree_per_set = []  # List to store average weighted degrees for each graph in the set
    for graph in graph_list:
        avg_weighted_node_degree = sum(dict(graph.degree(weight='weight')).values()) / len(graph)
        avg_weighted_degree_per_set.append(avg_weighted_node_degree)
    overall_avg_weighted_degree = sum(avg_weighted_degree_per_set) / len(avg_weighted_degree_per_set)
    avg_weighted_degree_no_slice.append(overall_avg_weighted_degree)

# Plot the evolution
# X values for the time stamps
x_values = [0, 100, 300, 500, 800, 1000]

# Plotting the average weighted degrees
plt.figure(figsize=(10, 6))
plt.style.use('science')
plt.errorbar(x_values, avg_weighted_degree_no_slice, yerr=np.std(avg_weighted_degree_no_slice), marker='x', color='black', linestyle='--', label='Total', capsize=5)
plt.errorbar(x_values, avg_weighted_degree_bc, yerr=np.std(avg_weighted_degree_bc), marker='o', linestyle='--', label='BC', capsize=5)
plt.errorbar(x_values, avg_weighted_degree_ad1, yerr=np.std(avg_weighted_degree_ad1), marker='v', linestyle='--', label='1 AD', capsize=5)
plt.errorbar(x_values, avg_weighted_degree_ad2, yerr=np.std(avg_weighted_degree_ad2), marker='s', linestyle='--', label='2 AD', capsize=5)
plt.errorbar(x_values, avg_weighted_degree_ad3, yerr=np.std(avg_weighted_degree_ad3), marker='^', linestyle='--', label='3 AD', capsize=5)
plt.errorbar(x_values, avg_weighted_degree_ad4_7, yerr=np.std(avg_weighted_degree_ad4_7), marker='D', linestyle='--', label='4-7 AD', capsize=5)
plt.xticks(x_values)
plt.xlabel('Randomization Counts')
plt.ylabel(r'Average Weighted Degree $\langle k_{W} \rangle$')
plt.legend()

plt.show()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Modularity's evolution
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

modularity_no_slice = []
modularity_bc = []
modularity_ad1 = []
modularity_ad2 = []
modularity_ad3 = []
modularity_ad4_7 = []


# Function to calculate the modularity
def calculate_modularity(graph):
    partition = community.best_partition(graph)
    modularity_value = community.modularity(partition, graph)
    return modularity_value


# Iterate over bc_ships_graphs_0, bc_ships_graphs_100, etc.
for graph_list in [bc_ships_graphs_0, bc_ships_graphs_100, bc_ships_graphs_300, bc_ships_graphs_500, bc_ships_graphs_800, bc_ships_graphs_1000]:
    modularity_per_set = []  # List to store modularity values for each graph in the set
    for graph in graph_list:
        modularity_value = calculate_modularity(graph)
        modularity_per_set.append(modularity_value)
    overall_modularity = sum(modularity_per_set) / len(modularity_per_set)
    modularity_bc.append(overall_modularity)

# Repeat the same process for the other sets of graphs
for graph_list in [ad1_ships_graphs_0, ad1_ships_graphs_100, ad1_ships_graphs_300, ad1_ships_graphs_500, ad1_ships_graphs_800, ad1_ships_graphs_1000]:
    modularity_per_set = []  # List to store modularity values for each graph in the set
    for graph in graph_list:
        modularity_value = calculate_modularity(graph)
        modularity_per_set.append(modularity_value)
    overall_modularity = sum(modularity_per_set) / len(modularity_per_set)
    modularity_ad1.append(overall_modularity)

# Repeat the same process for the other sets of graphs
for graph_list in [ad2_ships_graphs_0, ad2_ships_graphs_100, ad2_ships_graphs_300, ad2_ships_graphs_500, ad2_ships_graphs_800, ad2_ships_graphs_1000]:
    modularity_per_set = []  # List to store modularity values for each graph in the set
    for graph in graph_list:
        modularity_value = calculate_modularity(graph)
        modularity_per_set.append(modularity_value)
    overall_modularity = sum(modularity_per_set) / len(modularity_per_set)
    modularity_ad2.append(overall_modularity)

# Repeat the same process for the other sets of graphs
for graph_list in [ad3_ships_graphs_0, ad3_ships_graphs_100, ad3_ships_graphs_300, ad3_ships_graphs_500, ad3_ships_graphs_800, ad3_ships_graphs_1000]:
    modularity_per_set = []  # List to store modularity values for each graph in the set
    for graph in graph_list:
        modularity_value = calculate_modularity(graph)
        modularity_per_set.append(modularity_value)
    overall_modularity = sum(modularity_per_set) / len(modularity_per_set)
    modularity_ad3.append(overall_modularity)

# Repeat the same process for the other sets of graphs
for graph_list in [ad4_7_ships_graphs_0, ad4_7_ships_graphs_100, ad4_7_ships_graphs_300, ad4_7_ships_graphs_500, ad4_7_ships_graphs_800, ad4_7_ships_graphs_1000]:
    modularity_per_set = []  # List to store modularity values for each graph in the set
    for graph in graph_list:
        modularity_value = calculate_modularity(graph)
        modularity_per_set.append(modularity_value)
    overall_modularity = sum(modularity_per_set) / len(modularity_per_set)
    modularity_ad4_7.append(overall_modularity)

# Iterate over non-sliced graphs
for graph_list in [no_slice_ships_graphs_0, no_slice_ships_graphs_100, no_slice_ships_graphs_300, no_slice_ships_graphs_500, no_slice_ships_graphs_800, no_slice_ships_graphs_1000]:
    modularity_per_set = []  # List to store modularity values for each graph in the set
    for graph in graph_list:
        modularity_value = calculate_modularity(graph)
        modularity_per_set.append(modularity_value)
    overall_modularity = sum(modularity_per_set) / len(modularity_per_set)
    modularity_no_slice.append(overall_modularity)

# Plot the evolution
# X values for the time stamps
x_values = [0, 100, 300, 500, 800, 1000]

# Plotting the modularity values
plt.figure(figsize=(10, 6))
plt.style.use('science')
plt.errorbar(x_values, modularity_no_slice, yerr=np.std(modularity_no_slice), marker='x', color='black', linestyle='--', label='Total', capsize=5)
plt.errorbar(x_values, modularity_bc, yerr=np.std(modularity_bc), marker='o', linestyle='--', label='BC', capsize=5)
plt.errorbar(x_values, modularity_ad1, yerr=np.std(modularity_ad1), marker='v', linestyle='--', label='1 AD', capsize=5)
plt.errorbar(x_values, modularity_ad2, yerr=np.std(modularity_ad2), marker='s', linestyle='--', label='2 AD', capsize=5)
plt.errorbar(x_values, modularity_ad3, yerr=np.std(modularity_ad3), marker='^', linestyle='--', label='3 AD', capsize=5)
plt.errorbar(x_values, modularity_ad4_7, yerr=np.std(modularity_ad4_7), marker='D', linestyle='--', label='4-7 AD', capsize=5)
plt.xticks(x_values)
plt.xlabel('Randomization Counts')
plt.ylabel(r'Average Modularity $Q$')
plt.legend()

plt.show()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Create the empirical (observed) network
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

# For simplification purposes, switch keys and values in amphora_origins_all
origins_to_amphora_types = {}
for amphora_type, origins in amphora_origins_all.items():
    for origin in origins:
        if origin not in origins_to_amphora_types:
            origins_to_amphora_types[origin] = []
        origins_to_amphora_types[origin].append(amphora_type)

# Print the origins and associated amphora types
#for origin, amphora_types in origins_to_amphora_types.items():
#    print(f"Origin: {origin}, Amphora Types: {amphora_types}")

# Our five empirical graphs for each time stamp
emp_bc = bc_ships_0[0]
emp_ad1 = ad1_ships_0[0]
emp_ad2 = ad2_ships_0[0]
emp_ad3 = ad3_ships_0[0]
emp_ad4_7 = ad4_7_ships_0[0]


# Function fort extracting the unique origins from amphora types
def extract_unique_origins(amphora_type_lists):
    unique_origins = set()
    for amphora_type_list in amphora_type_lists:
        for amphora_type in amphora_type_list:
            if amphora_type in amphora_origins_all:
                origins = amphora_origins_all[amphora_type]
                unique_origins.update(origins)
    return list(unique_origins)


# Extract unique origins for each data frame
unique_origins_bc = extract_unique_origins(emp_bc['Amphora type'])
unique_origins_ad1 = extract_unique_origins(emp_ad1['Amphora type'])
unique_origins_ad2 = extract_unique_origins(emp_ad2['Amphora type'])
unique_origins_ad3 = extract_unique_origins(emp_ad3['Amphora type'])
unique_origins_ad4_7 = extract_unique_origins(emp_ad4_7['Amphora type'])

# Create the five empirical networks with origins as nodes
graph_bc = nx.Graph()
graph_bc.add_nodes_from(unique_origins_bc)

graph_ad1 = nx.Graph()
graph_ad1.add_nodes_from(unique_origins_ad1)

graph_ad2 = nx.Graph()
graph_ad2.add_nodes_from(unique_origins_ad2)

graph_ad3 = nx.Graph()
graph_ad3.add_nodes_from(unique_origins_ad3)

graph_ad4_7 = nx.Graph()
graph_ad4_7.add_nodes_from(unique_origins_ad4_7)

# Dictionary of origins with associated amphora types
origins_with_amphora_types = {}
for amphora_type, origins in amphora_origins_all.items():
    for origin in origins:
        if origin not in origins_with_amphora_types:
            origins_with_amphora_types[origin] = []
        origins_with_amphora_types[origin].append(amphora_type)

# Assigning Amphora Types to the Nodes in the graphs based on the amphora origins
for graph, data_frame in [(graph_bc, emp_bc), (graph_ad1, emp_ad1), (graph_ad2, emp_ad2), (graph_ad3, emp_ad3),
                          (graph_ad4_7, emp_ad4_7)]:
    for node in graph.nodes():
        amphora_types_from_origin = origins_with_amphora_types.get(node, [])
        amphora_types_in_data_frame = set()

        # Extract individual amphora types from the list in the data frame
        for amphora_type_list in data_frame['Amphora type']:
            amphora_types_in_data_frame.update(amphora_type_list)

        # Filter out those amphora types that are not present in the current data frame
        valid_amphora_types = [amphora_type for amphora_type in amphora_types_from_origin if
                               amphora_type in amphora_types_in_data_frame]

        graph.nodes[node]['Amphora types'] = valid_amphora_types

# Print node attributes for one of the graphs in order to verify
#for node, attributes in graph_ad3.nodes(data=True):
#    print(f"Node: {node}, Amphora Types: {attributes.get('Amphora types', [])}")

# Iterate through the graphs and their corresponding data frames
for graph, data_frame in [(graph_bc, emp_bc), (graph_ad1, emp_ad1), (graph_ad2, emp_ad2), (graph_ad3, emp_ad3),
                          (graph_ad4_7, emp_ad4_7)]:
    for _, row in data_frame.iterrows():
        cargo_amphora_types = row['Amphora type']
        cargo_origins = []

        # Get the origins corresponding to each amphora type in the cargo
        for amphora_type in cargo_amphora_types:
            origins = amphora_origins_all.get(amphora_type, [])
            cargo_origins.extend(origins)

        # Connect nodes with edges based on cargo origins
        for origin_1 in cargo_origins:
            for origin_2 in cargo_origins:
                if origin_1 != origin_2:
                    if graph.has_edge(origin_1, origin_2):
                        # If edge exists, increase weight by 1
                        graph[origin_1][origin_2]['weight'] += 1
                    else:
                        # If edge does not, add one with weight 1
                        graph.add_edge(origin_1, origin_2, weight=1)

    # Remove isolated nodes from the graph
    isolated_nodes = list(nx.isolates(graph))
    graph.remove_nodes_from(isolated_nodes)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Create the partitions
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++


# Extract provenance data from a graph
def extract_provenance_data(graph):
    provenances_list = list(graph.nodes())
    provenance_to_amphora_dict = {}
    for provenance in provenances_list:
        provenance_to_amphora_dict[provenance] = graph.nodes[provenance].get('Amphora types', [])
    return provenances_list, provenance_to_amphora_dict


# Function to create all possible partitions of provenances into two groups
def create_partitions(provenances_list):
    all_partitions = []
    for r in range(1, len(provenances_list)):
        partitions = combinations(provenances_list, r)
        all_partitions.extend(partitions)
    return all_partitions


# Check if a partition is valid based on side proportions with this function (20/80)
def is_partition_valid(side_A_percentage, side_B_percentage):
    return 20 <= side_A_percentage <= 80 and 20 <= side_B_percentage <= 80


# List of networks
networks = [graph_bc, graph_ad1, graph_ad2, graph_ad3, graph_ad4_7]

# Dictionary of valid partitions for each graph
valid_partitions_dict = {}

# List to store valid and invalid partitions counts for each network
valid_partitions_count_list = []
invalid_partitions_count_list = []

# Dictionary for the analysis results for each network
analysis_results_M_emp_dict = {}

# Iterate through networks and extract provenance data
for i, graph in enumerate(networks):
    prov_list, prov_to_amphora = extract_provenance_data(graph)
    partitions = create_partitions(prov_list)
    valid_partitions = []
    valid_count = 0
    invalid_count = 0
    graph_name = i

    for partition in partitions:
        side_A = partition
        side_B = [prov for prov in prov_list if prov not in partition]

        # Exclude 'Unknown/Uncertain' amphora types from both sides for the (20/80) check
        side_A_amphora = [amphora for provenance in side_A for amphora in prov_to_amphora[provenance] if
                          provenance != 'Uncertain/Unknown']
        side_B_amphora = [amphora for provenance in side_B for amphora in prov_to_amphora[provenance] if
                          provenance != 'Uncertain/Unknown']

        # Calculate the percentage of nodes in each side
        side_A_percentage = len(side_A) / len(prov_list) * 100
        side_B_percentage = len(side_B) / len(prov_list) * 100

        # Check if the partition is valid
        if is_partition_valid(side_A_percentage, side_B_percentage):
            valid_partitions.append((side_A, side_B))
            valid_count += 1
        else:
            invalid_count += 1

    # Store valid partitions for this graph in the dictionary
    valid_partitions_dict[graph_name] = valid_partitions
    valid_partitions_count_list.append(valid_count)
    invalid_partitions_count_list.append(invalid_count)

    # Now the analysis
    print(f"Analyzing valid partitions for Network {i + 1}:")

    # Initialize a list to store M_emp values for each partition
    M_emp_list = []

    for j, partition in enumerate(valid_partitions):
        side_A, side_B = partition

        # Initialize weights
        W_total_emp = 0
        W_s1_emp = 0
        W_s2_emp = 0

        # Calculate the total weight W_total_emp of edges in the graph
        for node1, node2 in graph.edges():
            W_total_emp += 1  # Increase the total weight for each edge

            # Calculate the total weight of edges between nodes that are on the same side of the partition
            if node1 in side_A and node2 in side_A:
                W_s1_emp += 1
            elif node1 in side_B and node2 in side_B:
                W_s2_emp += 1

        # Total weight of edges between nodes of different sides of the partition
        W_d_emp = W_total_emp - W_s1_emp - W_s2_emp

        # Mixing weight: M_emp = W_d_emp / W_total_emp
        M_emp = W_d_emp / W_total_emp if W_total_emp > 0 else 0

        # Append the M_emp value to the list
        M_emp_list.append(M_emp)

    analysis_results_M_emp_dict[graph_name] = M_emp_list

#print('Valid partitions:')
#print(valid_partitions_dict)
#print()

#print('Results M_emp dict:')
#print(analysis_results_M_emp_dict)
#print()

# Print the number of valid and invalid partitions for each network
for i, (valid_count, invalid_count) in enumerate(zip(valid_partitions_count_list, invalid_partitions_count_list)):
    print(f"Network {i + 1}:")
    print("Valid Partitions:", valid_count)
    print("Invalid Partitions:", invalid_count)
    print()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Create the randomized networks
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Extract unique origins from data frames, just as for the empirical networks
all_rand_unique_origins_bc = []
for df in bc_ships_1000:
    rand_origins_bc = extract_unique_origins(df['Amphora type'])
    all_rand_unique_origins_bc.append(rand_origins_bc)

all_rand_unique_origins_ad1 = []
for df in ad1_ships_1000:
    rand_origins_ad1 = extract_unique_origins(df['Amphora type'])
    all_rand_unique_origins_ad1.append(rand_origins_ad1)

all_rand_unique_origins_ad2 = []
for df in ad2_ships_1000:
    rand_origins_ad2 = extract_unique_origins(df['Amphora type'])
    all_rand_unique_origins_ad2.append(rand_origins_ad2)

all_rand_unique_origins_ad3 = []
for df in ad3_ships_1000:
    rand_origins_ad3 = extract_unique_origins(df['Amphora type'])
    all_rand_unique_origins_ad3.append(rand_origins_ad3)

all_rand_unique_origins_ad4_7 = []
for df in ad4_7_ships_1000:
    rand_origins_ad4_7 = extract_unique_origins(df['Amphora type'])
    all_rand_unique_origins_ad4_7.append(rand_origins_ad4_7)

# Create networks with origins as nodes
graphs_bc_random = []
for origins_bc_list in all_rand_unique_origins_bc:
    r_graph_bc = nx.Graph()
    r_graph_bc.add_nodes_from(origins_bc_list)
    graphs_bc_random.append(r_graph_bc)

graphs_ad1_random = []
for origins_ad1_list in all_rand_unique_origins_ad1:
    r_graph_ad1 = nx.Graph()
    r_graph_ad1.add_nodes_from(origins_ad1_list)
    graphs_ad1_random.append(r_graph_ad1)

graphs_ad2_random = []
for origins_ad2_list in all_rand_unique_origins_ad2:
    r_graph_ad2 = nx.Graph()
    r_graph_ad2.add_nodes_from(origins_ad2_list)
    graphs_ad2_random.append(r_graph_ad2)

graphs_ad3_random = []
for origins_ad3_list in all_rand_unique_origins_ad3:
    r_graph_ad3 = nx.Graph()
    r_graph_ad3.add_nodes_from(origins_ad3_list)
    graphs_ad3_random.append(r_graph_ad3)

graphs_ad4_7_random = []
for origins_ad4_7_list in all_rand_unique_origins_ad4_7:
    r_graph_ad4_7 = nx.Graph()
    r_graph_ad4_7.add_nodes_from(origins_ad4_7_list)
    graphs_ad4_7_random.append(r_graph_ad4_7)

# Fill the dictionary with amphora types for each origin
for graph_list, rand_data_frame_list in zip([graphs_bc_random, graphs_ad1_random, graphs_ad2_random,
                                             graphs_ad3_random, graphs_ad4_7_random],
                                            [bc_ships_1000, ad1_ships_1000, ad2_ships_1000,
                                             ad3_ships_1000, ad4_7_ships_1000]):
    for graph, rand_data_frame in zip(graph_list, rand_data_frame_list):
        for node in graph.nodes():
            rand_amphora_types_from_origin = origins_with_amphora_types.get(node, [])
            rand_amphora_types_in_data_frame = set()

            for cargo_amphora_types in rand_data_frame['Amphora type']:
                rand_amphora_types_in_data_frame.update(cargo_amphora_types)

            valid_amphora_types = [amphora_type for amphora_type in rand_amphora_types_from_origin if
                                   amphora_type in rand_amphora_types_in_data_frame]

            graph.nodes[node]['Amphora types'] = valid_amphora_types

# Iterate through the nodes and assign attributes
for graph_list, rand_data_frame_list in zip([graphs_bc_random, graphs_ad1_random, graphs_ad2_random,
                                             graphs_ad3_random, graphs_ad4_7_random],
                                            [bc_ships_1000, ad1_ships_1000, ad2_ships_1000,
                                             ad3_ships_1000, ad4_7_ships_1000]):
    for graph, rand_data_frame in zip(graph_list, rand_data_frame_list):
        for node in graph.nodes():
            rand_amphora_types_from_origin = origins_with_amphora_types.get(node, [])
            rand_amphora_types_in_data_frame = set()

            for cargo_amphora_types in rand_data_frame['Amphora type']:
                rand_amphora_types_in_data_frame.update(cargo_amphora_types)

            valid_amphora_types = [amphora_type for amphora_type in rand_amphora_types_from_origin if
                                   amphora_type in rand_amphora_types_in_data_frame]

            graph.nodes[node]['Amphora types'] = valid_amphora_types

        for _, row in rand_data_frame.iterrows():
            cargo_amphora_types = row['Amphora type']
            rand_cargo_origins = []

            for amphora_type in cargo_amphora_types:
                rand_origins = amphora_origins_all.get(amphora_type, [])
                rand_cargo_origins.extend(rand_origins)

            for origin_1 in rand_cargo_origins:
                for origin_2 in rand_cargo_origins:
                    if origin_1 != origin_2:
                        if graph.has_edge(origin_1, origin_2):
                            graph[origin_1][origin_2]['weight'] += 1
                        else:
                            graph.add_edge(origin_1, origin_2, weight=1)

        # Remove isolated nodes from the graph
        isolated_nodes = list(nx.isolates(graph))
        graph.remove_nodes_from(isolated_nodes)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Analyzing the partitions of the empirical networks applied on the randomized networks
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

# List of lists of randomized networks
random_networks = [graphs_bc_random, graphs_ad1_random, graphs_ad2_random, graphs_ad3_random, graphs_ad4_7_random]

random_networks_analysis_results = []

for idx, random_graphs in enumerate(random_networks):
    random_network_results = {}

    # Get partitions for the current time-stamped set of networks
    partitions_list = valid_partitions_dict[idx]

    for partition_idx, (side_A, side_B) in enumerate(partitions_list):
        M_partition_dict = {}  # Dictionary to store M values for each randomized network

        for network_idx, random_graph in enumerate(random_graphs):
            W_total = 0
            W_s1 = 0
            W_s2 = 0

            for node1, node2 in random_graph.edges():
                W_total += 1

                if node1 in side_A and node2 in side_A:
                    W_s1 += 1
                elif node1 in side_B and node2 in side_B:
                    W_s2 += 1

            W_d = W_total - W_s1 - W_s2
            M = W_d / W_total if W_total > 0 else 0

            M_partition_dict[network_idx] = M

        random_network_results[partition_idx] = M_partition_dict

    random_networks_analysis_results.append(random_network_results)

#print('Mixing weights of randomized networks:')
#for idx, random_network_results in enumerate(random_networks_analysis_results):
#    print(f"Random Network {idx + 1}:")
#    for partition_idx, network_results in random_network_results.items():
#        print(f"Partition {partition_idx + 1}: {network_results}")
#    print()

# The dictionary random_networks_analysis_results contains 5 sub-dictionaries (0, 1, 2, 3, 4) for each time stamp of
# our randomized networks. Each sub-dicitonary has keys that stand for the partition and values, representing the
# corresponding mixing weight of each randomized network of respective time stamp

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# P-value calculation
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

# List of observed graphs
observed_graphs = [graph_bc, graph_ad1, graph_ad2, graph_ad3, graph_ad4_7]

# List to store highly significant partitions for each observed graph
highly_significant_partitions_list = []

# Iterate through observed graphs
for i, observed_graph in enumerate(observed_graphs):
    print(f"Processing Observed Graph {i + 1}:")

    valid_partitions = valid_partitions_dict[i]  # Get valid partitions for the current observed graph
    M_emp_list = analysis_results_M_emp_dict[i]  # Get M_emp values for the current observed graph

    # List to store highly significant partitions for the current observed graph
    highly_significant_partitions = []

    # Iterate through valid partitions for the current observed graph
    for j, partition in enumerate(valid_partitions):
        M_emp = M_emp_list[j]  # Get M_emp value for the current partition from the list above

        p_value_count = 0  # Counter for p-values less than 0.005

        # Iterate through randomized networks and calculate p-value
        for random_network_results in random_networks_analysis_results:
            M_partition_dict = random_network_results[i]  # Get M values for the current observed graph

            if j in M_partition_dict:
                # Count M values lower than M_emp
                lower_M_count = sum(1 for M in M_partition_dict.values() if M < M_emp)

                # Calculate p-value
                p_value = lower_M_count / len(M_partition_dict)

                # Check if p-value is highly significant
                if p_value < 0.005:
                    p_value_count += 1

        # If there are highly significant partitions, add the partition index to the list
        if p_value_count > 0:
            highly_significant_partitions.append(j)

    highly_significant_partitions_list.append(highly_significant_partitions)

# Print number of highly significant partitions for each observed graph
for i, observed_graph in enumerate(observed_graphs):
    graph_name = f"Graph {i + 1}"
    num_highly_significant = len(highly_significant_partitions_list[i])
    print(f"{graph_name}: Number of Highly Significant Partitions = {num_highly_significant}")

    valid_partitions = valid_partitions_dict[i]  # Valid partitions for the current observed graph
    M_emp_list = analysis_results_M_emp_dict[i]  # M_emp values for the current observed graph

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Connection weight between two provenances
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

# List of observed graphs
observed_graphs = [graph_bc, graph_ad1, graph_ad2, graph_ad3, graph_ad4_7]

# List of lists of randomized graphs
random_networks = [graphs_bc_random, graphs_ad1_random, graphs_ad2_random, graphs_ad3_random, graphs_ad4_7_random]

# List to store the fractions I=(A,B) and fraction of total weight for each pair of provenances
fraction_list = []

# Iterate through observed graphs
for i, observed_graph in enumerate(observed_graphs):
    print(f"Processing Observed Graph {i + 1}:")

    valid_partitions = valid_partitions_dict[i]  # Valid partitions for the current observed graph
    significant_partitions = highly_significant_partitions_list[i]  # Get highly significant partitions

    graph_fraction_list = []  # List to store fractions for the current graph

    total_weight = sum(weight for _, _, weight in observed_graph.edges(data='weight'))  # Total weight of the graph

    # Iterate through pairs of provenances (A, B)
    for A in observed_graph.nodes():
        for B in observed_graph.nodes():
            if A != B:
                S_A = S_B = 0  # Counters for significant partitions where A and B are on the same side
                weight_AB = 0  # Weight of links connecting A and B

                # Iterate through significant partitions
                for partition_idx in significant_partitions:
                    side_A, side_B = valid_partitions[partition_idx]  # Get sides of the partition

                    if A in side_A and B in side_A:
                        S_A += 1
                    elif A in side_B and B in side_B:
                        S_B += 1

                # Calculate the fraction I=(A,B)
                if S_A + S_B > 0:
                    I = (S_A + S_B) / len(significant_partitions)
                else:
                    I = 0

                # Then the weight of links connecting A and B
                if observed_graph.has_edge(A, B):
                    weight_AB = observed_graph[A][B]['weight']

                graph_fraction_list.append((A, B, I, weight_AB / total_weight))

    fraction_list.append(graph_fraction_list)

# Iterate through observed graphs and randomized networks to again calculate p-values
for i, (observed_graph, random_graphs) in enumerate(zip(observed_graphs, random_networks)):
    graph_name = f"Graph {i + 1}"
    print(f"{graph_name}")
    for A, B, I, fraction_weight in fraction_list[i]:
        weight_rand_list = []  # List to store weight_rand(A, B) values for randomized networks

        # Iterate through the randomized networks
        for random_graph in random_graphs:
            weight_rand_AB = 0  # Weight of connections between A and B in the randomized graph
            if random_graph.has_edge(A, B):
                weight_rand_AB = random_graph[A][B]['weight']
            weight_rand_list.append(weight_rand_AB)

        # Calculate p-value p(weight(A, B))
        p_value = sum(1 for weight_rand in weight_rand_list if weight_rand >= fraction_weight) / len(weight_rand_list)

        print(f"({A}, {B}): I = {I}, Fraction of Total Weight = {fraction_weight}, p(weight(A, B)) = {p_value}")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Just print the significant connections
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Iterate through observed graphs and randomized networks to calculate p-values
for i, (observed_graph, random_graphs) in enumerate(zip(observed_graphs, random_networks)):
    graph_name = f"Graph {i + 1}"
    print(f"{graph_name}")
    for A, B, I, fraction_weight in fraction_list[i]:
        weight_rand_list = []  # List to store weight_rand(A, B) values for randomized networks

        # Iterate through randomized networks
        for random_graph in random_graphs:
            weight_rand_AB = 0  # Weight of links connecting A and B in the randomized graph
            if random_graph.has_edge(A, B):
                weight_rand_AB = random_graph[A][B]['weight']
            weight_rand_list.append(weight_rand_AB)

        # Calculate p-value p(weight(A, B))
        p_value = sum(1 for weight_rand in weight_rand_list if weight_rand >= fraction_weight) / len(weight_rand_list)

        # Check if p-value is less than 0.05 before printing
        if p_value < 0.05:
            print(f"({A}, {B}): I = {I}, Fraction of Total Weight = {fraction_weight}, p(weight(A, B)) = {p_value}")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Final provenance plots
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Iterate through observed graphs and randomized networks
for i, (observed_graph, random_graphs) in enumerate(zip(observed_graphs, random_networks)):
    # New graph for the current observed graph
    provenance_graph = nx.Graph()

    # Iterate through the fraction_list for the current graph
    for A, B, I, fraction_weight in fraction_list[i]:
        # Get the observed weight for the current edge (A, B)
        weight_observed = observed_graph[A][B]['weight'] if observed_graph.has_edge(A, B) else 0

        # List to store randomized weights for the current edge (A, B)
        weight_rand_list = []
        for random_graph in random_graphs:
            # Get the randomized weight for the current edge (A, B) in the current random graph
            weight_rand_AB = random_graph[A][B]['weight'] if random_graph.has_edge(A, B) else 0
            weight_rand_list.append(weight_rand_AB)

        # Calculate the p-value p(weight(A, B))
        p_value = sum(1 for weight_rand in weight_rand_list if weight_rand >= weight_observed) / len(weight_rand_list)

        # Add edge to the provenance graph with weight and p-value
        provenance_graph.add_edge(A, B, p_value=p_value, weight=weight_observed)

    # Position nodes
    pos = nx.spring_layout(provenance_graph)

    # Retrieve edges and the data (p_values and weights) from the provenance graph
    edges = provenance_graph.edges(data=True)

    # Generate a list of node colors based on the origins (same colors as for the empirical graphs)
    node_colors = [origin_colors.get(node, 'gray') for node in provenance_graph.nodes]

    # Retrieve the weights from the edges
    weights = [data['weight'] for _, _, data in edges]
    min_weight = min(weights)
    max_weight = max(weights)

    # Draw the graph
    plt.figure(figsize=(20, 20))
    nx.draw_networkx_nodes(provenance_graph, pos, node_color=node_colors, alpha=0.9, node_size=1000)
    for (u, v, data), weight in zip(edges, weights):
        p_value = data['p_value']
        edge_color = 'blue' if p_value < 0.05 else 'gray'
        edge_width = 5 * (weight - min_weight) / (max_weight - min_weight) + 1
        nx.draw_networkx_edges(provenance_graph, pos, edgelist=[(u, v)], alpha=0.5, edge_color=edge_color, width=edge_width)
    nx.draw_networkx_labels(provenance_graph, pos, font_size=12, font_weight='bold')
    plt.show()

# +++++++++++++++++++++++++++++++++++++++++++++ TEST +++++++++++++++++++++++++++++++++++++++++++++

print()
print('Graph BC')
print(graph_bc)
for node in graph_bc.nodes():
    attributes = graph_bc.nodes[node]
    print(f"Node {node} attributes:", attributes)
print('Graph BC Random')
print(graphs_bc_random[0])
# Iterate through nodes in the graph
for node in graphs_bc_random[0].nodes():
    attributes = graphs_bc_random[0].nodes[node]
    print(f"Node {node} attributes:", attributes)

print()
print('Graph AD 1')
print(graph_ad1)
for node in graph_ad1.nodes():
    attributes = graph_ad1.nodes[node]
    print(f"Node {node} attributes:", attributes)
print('Graph AD 1 Random')
print(graphs_ad1_random[0])
# Iterate through nodes in the graph
for node in graphs_ad1_random[0].nodes():
    attributes = graphs_ad1_random[0].nodes[node]
    print(f"Node {node} attributes:", attributes)

print()
print('Graph AD 2')
print(graph_ad2)
for node in graph_ad2.nodes():
    attributes = graph_ad2.nodes[node]
    print(f"Node {node} attributes:", attributes)
print('Graph AD 2 Random')
print(graphs_ad2_random[0])
# Iterate through nodes in the graph
for node in graphs_ad2_random[0].nodes():
    attributes = graphs_ad2_random[0].nodes[node]
    print(f"Node {node} attributes:", attributes)

print()
print('Graph AD 3')
print(graph_ad3)
for node in graph_ad3.nodes():
    attributes = graph_ad3.nodes[node]
    print(f"Node {node} attributes:", attributes)
print('Graph AD 3 Random')
print(graphs_ad3_random[0])
# Iterate through nodes in the graph
for node in graphs_ad3_random[0].nodes():
    attributes = graphs_ad3_random[0].nodes[node]
    print(f"Node {node} attributes:", attributes)

print()
print('Graph AD 4-7')
print(graph_ad4_7)
for node in graph_ad4_7.nodes():
    attributes = graph_ad4_7.nodes[node]
    print(f"Node {node} attributes:", attributes)
print('Graph AD 4-7 Random')
print(graphs_ad4_7_random[0])
# Iterate through nodes in the graph
for node in graphs_ad4_7_random[0].nodes():
    attributes = graphs_ad4_7_random[0].nodes[node]
    print(f"Node {node} attributes:", attributes)
