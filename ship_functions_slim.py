import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import scienceplots
import itertools
import community
import numpy as np
import random
import matplotlib.colors as mcolors
import statistics

plt.style.use('science')

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Manipulating the Excel files for shipwrecks
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

cargoes_file_path = 'Cargoes.xlsx'
xlsx = pd.ExcelFile(cargoes_file_path)

# Reading each sheet into a data frame
df_cargoes = pd.read_excel(xlsx, 'Cargoes')
df_chronology = pd.read_excel(xlsx, 'Chronology')
df_proven_boleans = pd.read_excel(xlsx, 'Proven_boleans')

# Remove possible whitespaces in strings
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

# Delete columns of 4 and 3 BC and 8 AD
merged_df_2amph = merged_df.drop(columns=[merged_df.columns[3], merged_df.columns[4],
                                                merged_df.columns[14]])

# print("All Shipwrecks")
# print(merged_df_2amph)
# print()

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
# for amphora_type, origins in amphora_origins.items():
#    print(f"Amphora Type: {amphora_type}, Origins: {origins}")
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

# Filter merged_df_all based on 'Oxford_wreckID' using shipwreck_dates_all keys
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

print("All shipwrecks dated by groups:")
for century, shipwrecks in shipwreck_dates.items():
    print(f"Century: {century}")
    for shipwreck_id, amphora_types in shipwrecks.items():
        print(f"Shipwreck ID: {shipwreck_id}, Amphora Types: {amphora_types}")
    print()

total_ships = len(merged_df_all['Oxford_wreckID'].unique())
total_amphoras = len(amphora_count)
print(f"Total number of ships: {total_ships}")
print(f"Total number of amphoras: {total_amphoras}")
print()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Just keep amphora origins of amphora present in 'amphora_count'
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Keep the whole list
amphora_origins_all = amphora_origins

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
        pairs = {(min(amphora_type_i, amphora_type_j), max(amphora_type_i, amphora_type_j)) for amphora_type_i in \
                 list(amphora_types) for amphora_type_j in list(amphora_types) if amphora_type_i != amphora_type_j}

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
        pairs = {(min(amphora_type_i, amphora_type_j), max(amphora_type_i, amphora_type_j)) for amphora_type_i in \
                 amphora_types for amphora_type_j in amphora_types if amphora_type_i != amphora_type_j}

        # Update the connections dictionary and add edges to the network, excluding self-loops
        for pair in pairs:
            if pair[0] != pair[1]:
                connections.setdefault(pair, 0)
                connections[pair] += 1
                G.add_edge(pair[0], pair[1], weight=connections[pair])

    # Store the connections for the current century
    strength_of_connection[century_label] = connections

    # Get the unique origins from the amphora_origins dictionary
    all_origins = set()
    for origins in amphora_origins.values():
        all_origins.update(origins)

    # Generate a color palette with 25 different colors
    color_palette = plt.cm.Set2(np.linspace(0, 1, len(all_origins)))

    # Create a dictionary to store the color assignments for each origin
    origin_colors = {}
    for i, origin in enumerate(all_origins):
        origin_colors[origin] = color_palette[i]

    # Create a legend for the colors
    legend_labels = list(origin_colors.keys())
    legend_colors = list(origin_colors.values())

    # Assign colors to nodes based on their origins
    node_colors = [origin_colors.get(amphora_origins.get(node, [])[0]) for node in G.nodes]

    # Extract the weights from the network
    weights = [G[u][v]['weight'] for u, v in G.edges]

    # Scale the weights to the desired range for line width
    min_weight = min(weights)
    max_weight = max(weights)
    scaled_weights = [weight for weight in weights] # NOT SCALED: (weight-min_weight)/(max_weight-min_weight)

    # Plot the network with colored nodes and scaled line widths
    plt.figure()
    plt.figure(figsize=(15, 15))
    nx.draw(G, with_labels=True, node_color=node_colors, width=scaled_weights)
    plt.title(f"Amphora Network - Century {century_label}")

    # Create proxy artists for the legend
    legend_handles = [Patch(facecolor=color) for color in legend_colors]

    # Create and display the legend using the proxy artists
    plt.legend(legend_handles, legend_labels)
    plt.show()

    # Assign network created for each century to the amphora_networks dictionary
    amphora_networks[century_label] = G

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Randomization
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

all_shipwrecks = merged_df_all.groupby('Oxford_wreckID')['Amphora type'].apply(list).reset_index()


def interchange_cargoes(all_shipwrecks, production_times_all, num_randomizations):
    num_frames = 10
    data_frames = []

    for frame in range(num_frames):
        frame_data = all_shipwrecks.copy()  # Create a copy of all_shipwrecks for each frame

        interchange_count = 0  # Counter for valid interchanges

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
                continue  # Skip the interchange and start again from step 1

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
rand_list_0 = interchange_cargoes(all_shipwrecks, production_times, 0)
rand_list_100 = interchange_cargoes(all_shipwrecks, production_times, 100)
rand_list_300 = interchange_cargoes(all_shipwrecks, production_times, 300)
rand_list_500 = interchange_cargoes(all_shipwrecks, production_times, 500)
rand_list_800 = interchange_cargoes(all_shipwrecks, production_times, 800)
rand_list_1000 = interchange_cargoes(all_shipwrecks, production_times, 1000)

# Print the original data frame
# print("Original Data Frame:")
# print(all_shipwrecks.to_string(index=False))
# print()

# Print all data frames in new_rand_df
# for i, df in enumerate(rand_list_1000):
#     print("Data Frame", i+1)
#     print(df.to_string(index=False))
#     print()

print(rand_list_0)

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
        # Create a copy of the sliced DataFrame
        df_filtered = df[df['Amphora Count'] > 1].copy()
        # Remove the 'Amphora Count' column
        df_filtered.drop('Amphora Count', axis=1, inplace=True)
        # Update the data frame in the list
        rand_list[key] = df_filtered
    return rand_list


rand_list_0 = no_singles_in_rand(rand_list_0)
rand_list_100 = no_singles_in_rand(rand_list_100)
rand_list_300 = no_singles_in_rand(rand_list_300)
rand_list_500 = no_singles_in_rand(rand_list_500)
rand_list_800 = no_singles_in_rand(rand_list_800)
rand_list_1000 = no_singles_in_rand(rand_list_1000)

print(rand_list_100)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Date each rand_list_XXXX
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++


# Update get_ship_dates function to modify the dating periods based on grouping
def get_ship_dates(df, production_times):
    ship_dates = set()
    for _, row in df.iterrows():
        cargo = row['Amphora type']
        cargo_production_times = [production_times.get(amphora, []) for amphora in cargo]
        # Check for overlaps among production times that are common to all amphoras in the cargo
        overlaps = set.intersection(*map(set, cargo_production_times))
        if overlaps:
            ship_dates.update(overlaps)

    # Group the dating periods based on your desired grouping
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
        # Create a new column 'Dating Periods' in the DataFrame to store ship dating information
        df['Dating Periods'] = ''
        # Iterate over each row in the DataFrame and get the dating periods for each ship
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
for i, df in enumerate(rand_list_100):
    print(f"Data Frame {i+1}:")
    print(df)
    print()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Filter for time stamps
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++


# Function to filter and group data frames based on dating periods
def filter_and_group_dataframes(df_list):
    bc_ships_list = []
    ad1_ships_list = []
    ad2_ships_list = []
    ad3_ships_list = []
    ad4_7_ships_list = []

    # Iterate through each data frame in the list
    for df in df_list:
        # Filter ships dated in "BC" (2 BC and 1 BC)
        bc_ships = df[df['Dating Periods'].str.contains('BC')]
        bc_ships_list.append(bc_ships)

        # Filter ships dated in "1 AD"
        ad1_ships = df[df['Dating Periods'].str.contains('1 AD')]
        ad1_ships_list.append(ad1_ships)

        # Filter ships dated in "2 AD"
        ad2_ships = df[df['Dating Periods'].str.contains('2 AD')]
        ad2_ships_list.append(ad2_ships)

        # Filter ships dated in "3 AD"
        ad3_ships = df[df['Dating Periods'].str.contains('3 AD')]
        ad3_ships_list.append(ad3_ships)

        # Filter ships dated in "4-7 AD" (4 AD, 5 AD, 6 AD, and 7 AD)
        ad4_7_ships = df[df['Dating Periods'].str.contains('4-7 AD')]
        ad4_7_ships_list.append(ad4_7_ships)

    return bc_ships_list, ad1_ships_list, ad2_ships_list, ad3_ships_list, ad4_7_ships_list


# Apply the function to all five lists
bc_ships_0, ad1_ships_0, ad2_ships_0, ad3_ships_0, ad4_7_ships_0 = filter_and_group_dataframes(rand_list_0)
bc_ships_100, ad1_ships_100, ad2_ships_100, ad3_ships_100, ad4_7_ships_100 = filter_and_group_dataframes(rand_list_100)
bc_ships_300, ad1_ships_300, ad2_ships_300, ad3_ships_300, ad4_7_ships_300 = filter_and_group_dataframes(rand_list_300)
bc_ships_500, ad1_ships_500, ad2_ships_500, ad3_ships_500, ad4_7_ships_500 = filter_and_group_dataframes(rand_list_500)
bc_ships_800, ad1_ships_800, ad2_ships_800, ad3_ships_800, ad4_7_ships_800 = filter_and_group_dataframes(rand_list_800)
bc_ships_1000, ad1_ships_1000, ad2_ships_1000, ad3_ships_1000, ad4_7_ships_1000 = filter_and_group_dataframes(rand_list_1000)

# Print the lengths of each list to verify the number of ships in each time stamp
print("Number of ships in 'BC' time stamp in rand_list_100:", len(bc_ships_100[0]))
print("Number of ships in '1 AD' time stamp in rand_list_100:", len(ad1_ships_100[0]))
print("Number of ships in '2 AD' time stamp in rand_list_100:", len(ad2_ships_100[0]))
print("Number of ships in '3 AD' time stamp in rand_list_100:", len(ad3_ships_100[0]))
print("Number of ships in '4-7 AD' time stamp in rand_list_100:", len(ad4_7_ships_100[0]))

# Print the list ad4_7_ships_100
for i, df in enumerate(ad4_7_ships_100):
    print(f"Data Frame List {i + 1}:")
    print(df)
    print()

# Now we have 5 list (for each time stamp) for 100, 300, 500, 800, 1000 randomizations, respectively.
# From here we will analyze each data frame by creating a network and analyzing its properties, which will then be
# averaged, so that we obtain for each of the 5 investigated properties an average value for each time stamp at each
# number of realized randomizations.

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Analyze each networks properties for time stamps and randomization steps
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++


def create_weighted_graph(df):
    # Create an empty graph
    G = nx.Graph()
    # Drop the "Dating Periods" column from the DataFrame
    df_copy = df.drop(columns=['Dating Periods'])
    # Iterate over each row in the DataFrame
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
                    # If the edge already exists, increment the weight by 1
                    G[amphora_1][amphora_2]['weight'] += 1
                else:
                    # If the edge doesn't exist, add it with weight 1
                    G.add_edge(amphora_1, amphora_2, weight=1)
    return G


###
### 0 Randomizations
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
# Average clustering coefficient's evolution
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

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

# Plot the evolution
# X values for the time stamps
x_values = [0, 100, 300, 500, 800, 1000]

# Plotting the average clustering coefficients
plt.figure(figsize=(10, 6))

plt.errorbar(x_values, avg_cc_bc, yerr=np.std(avg_cc_bc), color='blue', marker='o', linestyle='--', label='BC')
plt.errorbar(x_values, avg_cc_ad1, yerr=np.std(avg_cc_ad1), color='red', marker='v', linestyle='--', label='1 AD')
plt.errorbar(x_values, avg_cc_ad2, yerr=np.std(avg_cc_ad2), color='seagreen', marker='s', linestyle='--', label='2 AD')
plt.errorbar(x_values, avg_cc_ad3, yerr=np.std(avg_cc_ad3), color='purple', marker='^', linestyle='--', label='3 AD')
plt.errorbar(x_values, avg_cc_ad4_7, yerr=np.std(avg_cc_ad4_7), color='yellow', marker='D', linestyle='--', label='4-7 AD')
plt.xticks(x_values)
plt.xlabel('Randomization Count')
plt.ylabel(r'$\langle C \rangle$')
plt.legend()

plt.show()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Giant component's evolution
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

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

# Plot the evolution
# X values for the time stamps
x_values = [0, 100, 300, 500, 800, 1000]

# Plotting the giant component sizes
plt.figure(figsize=(10, 6))

plt.errorbar(x_values, giant_size_bc, yerr=np.std(giant_size_bc), color='blue', marker='o', linestyle='--', label='BC')
plt.errorbar(x_values, giant_size_ad1, yerr=np.std(giant_size_ad1), color='red', marker='v', linestyle='--', label='1 AD')
plt.errorbar(x_values, giant_size_ad2, yerr=np.std(giant_size_ad2), color='seagreen', marker='s', linestyle='--', label='2 AD')
plt.errorbar(x_values, giant_size_ad3, yerr=np.std(giant_size_ad3), color='purple', marker='^', linestyle='--', label='3 AD')
plt.errorbar(x_values, giant_size_ad4_7, yerr=np.std(giant_size_ad4_7), color='yellow', marker='D', linestyle='--', label='4-7 AD')
plt.xticks(x_values)
plt.xlabel('Randomization Count')
plt.ylabel(r'$\langle S_{GC} \rangle$')
plt.legend()

plt.show()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Average degree's evolution
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

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

# Plot the evolution
# X values for the time stamps
x_values = [0, 100, 300, 500, 800, 1000]

# Plotting the average degrees
plt.figure(figsize=(10, 6))

plt.errorbar(x_values, avg_degree_bc, yerr=np.std(avg_degree_bc), color='blue', marker='o', linestyle='--', label='BC')
plt.errorbar(x_values, avg_degree_ad1, yerr=np.std(avg_degree_ad1), color='red', marker='v', linestyle='--', label='1 AD')
plt.errorbar(x_values, avg_degree_ad2, yerr=np.std(avg_degree_ad2), color='seagreen', marker='s', linestyle='--', label='2 AD')
plt.errorbar(x_values, avg_degree_ad3, yerr=np.std(avg_degree_ad3), color='purple', marker='^', linestyle='--', label='3 AD')
plt.errorbar(x_values, avg_degree_ad4_7, yerr=np.std(avg_degree_ad4_7), color='yellow', marker='D', linestyle='--', label='4-7 AD')
plt.xticks(x_values)
plt.xlabel('Randomization Count')
plt.ylabel(r'$\langle k \rangle$')
plt.legend()

plt.show()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Average weighted degree's evolution
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

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

# Plot the evolution
# X values for the time stamps
x_values = [0, 100, 300, 500, 800, 1000]

# Plotting the average weighted degrees
plt.figure(figsize=(10, 6))

plt.errorbar(x_values, avg_weighted_degree_bc, yerr=np.std(avg_weighted_degree_bc), color='blue', marker='o', linestyle='--', label='BC')
plt.errorbar(x_values, avg_weighted_degree_ad1, yerr=np.std(avg_weighted_degree_ad1), color='red', marker='v', linestyle='--', label='1 AD')
plt.errorbar(x_values, avg_weighted_degree_ad2, yerr=np.std(avg_weighted_degree_ad2), color='seagreen', marker='s', linestyle='--', label='2 AD')
plt.errorbar(x_values, avg_weighted_degree_ad3, yerr=np.std(avg_weighted_degree_ad3), color='purple', marker='^', linestyle='--', label='3 AD')
plt.errorbar(x_values, avg_weighted_degree_ad4_7, yerr=np.std(avg_weighted_degree_ad4_7), color='yellow', marker='D', linestyle='--', label='4-7 AD')
plt.xticks(x_values)
plt.xlabel('Randomization Count')
plt.ylabel(r'$\langle k_{W} \rangle$')
plt.legend()

plt.show()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Modularity's evolution
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

modularity_bc = []
modularity_ad1 = []
modularity_ad2 = []
modularity_ad3 = []
modularity_ad4_7 = []


# Define a function to calculate modularity
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

# Plot the evolution
# X values for the time stamps
x_values = [0, 100, 300, 500, 800, 1000]

# Plotting the modularity values
plt.figure(figsize=(10, 6))

plt.errorbar(x_values, modularity_bc, yerr=np.std(modularity_bc), color='blue', marker='o', linestyle='--', label='BC')
plt.errorbar(x_values, modularity_ad1, yerr=np.std(modularity_ad1), color='red', marker='v', linestyle='--', label='1 AD')
plt.errorbar(x_values, modularity_ad2, yerr=np.std(modularity_ad2), color='seagreen', marker='s', linestyle='--', label='2 AD')
plt.errorbar(x_values, modularity_ad3, yerr=np.std(modularity_ad3), color='purple', marker='^', linestyle='--', label='3 AD')
plt.errorbar(x_values, modularity_ad4_7, yerr=np.std(modularity_ad4_7), color='yellow', marker='D', linestyle='--', label='4-7 AD')
plt.xticks(x_values)
plt.xlabel('Randomization Count')
plt.ylabel(r'$Q$')
plt.legend()

plt.show()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Analyze each networks properties for time stamps and randomization steps
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++


'''
# We start by deleting single values for all evolution data frames
for key, df_dict in evolution_rand_df.items():
    for frame_num, df in df_dict.items():
        # Convert 'Amphora type' column to lists
        df['Amphora type'] = df['Amphora type'].apply(lambda x: [x] if isinstance(x, int) else x)

        # Count the number of entries in the 'Amphora type' column
        df['Amphora Count'] = df['Amphora type'].apply(len)

        # Create a copy of the sliced DataFrame
        df_filtered = df[df['Amphora Count'] > 1].copy()

        # Remove the 'Amphora Count' column
        df_filtered.drop('Amphora Count', axis=1, inplace=True)

        # Update the data frame in the dictionary
        df_dict[frame_num] = df_filtered

    # Update the modified data frames in the dictionary
    evolution_rand_df[key] = df_dict

# Next re-group all randomized data frames with same keys (all the 100s, 300s, ...)
grouped_data_frames = {}

for key, df_dict in evolution_rand_df.items():
    for frame_num, df in df_dict.items():
        if frame_num not in grouped_data_frames:
            grouped_data_frames[frame_num] = {}
        grouped_data_frames[frame_num][key] = df

# Divide grouped_data_frames, by creating five lists, each containing all data frames
# after 100, 300, 500, 800 and 1000 randomizations, respectively
df_after_100 = []
df_after_300 = []
df_after_500 = []
df_after_800 = []
df_after_1000 = []

for key, df_dict in grouped_data_frames.items():
    if key == 100:
        df_after_100.extend(df_dict.values())
    elif key == 300:
        df_after_300.extend(df_dict.values())
    elif key == 500:
        df_after_500.extend(df_dict.values())
    elif key == 800:
        df_after_800.extend(df_dict.values())
    elif key == 1000:
        df_after_1000.extend(df_dict.values())

print(df_after_100)


# Now we are going to create 5 time stamps for all of them
def date_shipwrecks(mixed_list):
    new_dict = {}  # Dictionary to store the periods for each shipwreck
    for i in range(len(mixed_list)):
        df = mixed_list[i]
        shipwreck_id = df['Oxford_wreckID'].iloc[0]  # Assuming each dataframe corresponds to a single shipwreck

        # Initialize a set to store the possible centuries for the current shipwreck
        possible_centuries = set()

        for index, row in df.iterrows():
            amphora_types = row['Amphora type']

            # Check the production century of the first amphora type
            first_amphora_type = amphora_types[0]
            if first_amphora_type in production_times:
                production_century = production_times[first_amphora_type][0]
                possible_centuries.add(production_century)

        # Add the possible centuries for the shipwreck to the new_dict dictionary
        for century in possible_centuries:
            if century in new_dict:
                new_dict[century][shipwreck_id] = amphora_type
            else:
                new_dict[century] = {shipwreck_id: possible_centuries}
    return new_dict



# Call the function for each dictionary
dated_shipwrecks_100 = date_shipwrecks(df_after_100)
dated_shipwrecks_300 = date_shipwrecks(df_after_300)
dated_shipwrecks_500 = date_shipwrecks(df_after_500)
dated_shipwrecks_800 = date_shipwrecks(df_after_800)
dated_shipwrecks_1000 = date_shipwrecks(df_after_1000)


# Print the updated groups
print("Group 100:")
print(dated_shipwrecks_100)
print()



# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Country trade from the randomized network
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Manipulate amphora_origins
provenances = {}

for amphora, origins in amphora_origins.items():
    for origin in origins:
        if origin not in provenances:
            provenances[origin] = []
        provenances[origin].append(amphora)

print(amphora_origins)
print()
print(provenances)
print()
print(shipwreck_dates)
print()

# Create the graph for each time stamp
graphs = {}

# Iterate over each time stamp
for time_stamp in shipwreck_dates:
    # Convert the values in the provenances dictionary to sets
    for provenance, amphora_types in provenances.items():
        provenances[provenance] = set(amphora_types)

    # Create the graph
    graph = nx.Graph()

    # Add nodes for origins
    for provenance in provenances:
        graph.add_node(provenance)

    # Add links between origins based on amphora types in shipwrecks
    for shipwreck, amphora_types in shipwreck_dates[time_stamp].items():
        origins = set()
        for amphora_type in amphora_types:
            for provenance, amphora_list in provenances.items():
                if amphora_type in amphora_list:
                    origins.add(provenance)

        for origin_a in origins:
            for origin_b in origins:
                if origin_a != origin_b:
                    if graph.has_edge(origin_a, origin_b):
                        # Increase the edge weight if the edge already exists
                        graph[origin_a][origin_b]['weight'] += 1
                    else:
                        # Add a new edge with weight 1
                        graph.add_edge(origin_a, origin_b, weight=1)

    graphs[time_stamp] = graph

# Plot the graphs
for time_stamp, graph in graphs.items():
    plt.figure()
    plt.title(f"Graph {time_stamp}")

    pos = nx.kamada_kawai_layout(graph, scale=10000)

    # Extract edge weights
    edge_weights = [graph[u][v]['weight'] for u, v in graph.edges()]

    # Generate random colors for nodes
    random_colors = [random.choice(list(mcolors.CSS4_COLORS.keys())) for _ in graph.nodes()]

    # Draw nodes with random colors
    nx.draw_networkx_nodes(graph, pos, node_color=random_colors)

    # Draw edges with thinner widths based on the edge weights
    edge_widths = [5 / w for w in edge_weights]
    nx.draw_networkx_edges(graph, pos, width=edge_widths)

    # Create legend
    legend_labels = [node for node in graph.nodes()]
    legend_handles = [plt.Line2D([], [], marker='o', markersize=10, color=color, label=label) for color, label in zip(random_colors, legend_labels)]
    plt.legend(handles=legend_handles)

    plt.axis('off')
    plt.show()

# Store each graph
graph_bc = graphs['BC']
graph_1ad = graphs['1 AD']
graph_2ad = graphs['2 AD']
graph_3ad = graphs['3 AD']
graph_4_7ad = graphs['4-7 AD']

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Partitions of the empirical network
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++


# Function to perform Girvan-Newman algorithm
def girvan_newman(graph):
    def find_best_edge(G0):
        eb = nx.edge_betweenness_centrality(G0)
        return sorted(eb.items(), key=lambda x: x[1], reverse=True)[0][0]

    components = [c for c in nx.connected_components(graph)]
    while len(components) == 1:
        edge_to_remove = find_best_edge(graph)
        graph.remove_edge(*edge_to_remove)
        components = [c for c in nx.connected_components(graph)]

    return components


# Perform Girvan-Newman algorithm for each century's graph
partitions = {}
for century, graph in amphora_networks.items():
    components = girvan_newman(graph)
    partitions[century] = components

# Filter partitions based on size criteria
filtered_partitions = {}
for century, components in partitions.items():
    filtered_components = []
    for component in components:
        num_nodes = len(component)
        if num_nodes > 0.2 * graph.number_of_nodes() and num_nodes < 0.8 * graph.number_of_nodes():
            filtered_components.append(component)
    filtered_partitions[century] = filtered_components

# Print the filtered partitions
for century, components in filtered_partitions.items():
    print(f"Partitions for Century {century}:")
    for i, component in enumerate(components):
        print(f"Partition {i+1}: {component}")
    print()

# a) Calculate W_tot
W_tot = 0
# Iterate over the edges of the graph
for u, v, data in graph.edges(data=True):
    provenance_u = amphora_origins.get(u, [None])[0]  # Get the provenance of node u
    provenance_v = amphora_origins.get(v, [None])[0]  # Get the provenance of node v

    if provenance_u != provenance_v:
        weight = data.get('weight', 0)  # Get the weight of the edge
        W_tot += weight

print(f"Total weight of links between different provenances: {W_tot}")

# b) Calculate W_S
# Calculate W_S for each partition
W_S = {}
for century, components in filtered_partitions.items():
    W_S[century] = 0

    for component in components:
        # Define the set of provenances belonging to side S of the partition
        partition_S = set()

        # Iterate over the nodes in side S and add their provenances to the set
        for node in component:
            provenance = amphora_origins.get(node, [None])[0]  # Get the provenance of the node
            if provenance is not None:
                partition_S.add(provenance)

        # Iterate over the edges of the graph
        for u, v, data in amphora_networks[century].edges(data=True):
            provenance_u = amphora_origins.get(u, [None])[0]  # Get the provenance of node u
            provenance_v = amphora_origins.get(v, [None])[0]  # Get the provenance of node v

            # Check if nodes u and v belong to the same side S of the partition
            if provenance_u in partition_S and provenance_v in partition_S:
                # Check if nodes u and v have different provenances
                if provenance_u != provenance_v:
                    weight = data.get('weight', 0)  # Get the weight of the edge
                    W_S[century] += weight

    print(f"Total weight of links between nodes belonging to the same side of the partition (Century {century}): {W_S[century]}")

# c) Calculate W_sameprov and M
# Calculate W_sameprov and M for each partition
W_sameprov = {}
mixing_weights = {}  # Define the mixing_weights dictionary
for century, components in filtered_partitions.items():
    W_tot = 0
    W_S = 0

    for component in components:
        # Define the set of provenances belonging to side S of the partition
        partition_S = set()

        # Iterate over the nodes in side S and add their provenances to the set
        for node in component:
            provenance = amphora_origins.get(node, [None])[0]  # Get the provenance of the node
            if provenance is not None:
                partition_S.add(provenance)

        # Iterate over the edges of the graph
        for u, v, data in amphora_networks[century].edges(data=True):
            provenance_u = amphora_origins.get(u, [None])[0]  # Get the provenance of node u
            provenance_v = amphora_origins.get(v, [None])[0]  # Get the provenance of node v

            weight = data.get('weight', 0)  # Get the weight of the edge
            W_tot += weight

            # Check if nodes u and v belong to the same side S of the partition
            if provenance_u in partition_S and provenance_v in partition_S:
                # Check if nodes u and v have different provenances
                if provenance_u != provenance_v:
                    W_S += weight

    # Calculate W_sameprov
    W_sameprov[century] = W_tot - W_S

    print(f"Total weight of links between nodes belonging to different sides of the partition (Century {century}): {W_sameprov[century]}")

    # Calculate mixing weight (M)
    if W_tot != 0:
        M = W_sameprov[century] / W_tot
    else:
        M = 0

    mixing_weights[century] = M

    print(f"Mixing weight (M) for Century {century}: {M}")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Manipulate randomized network for partitions
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Now we want to do the same for randomized graphs
# For that we start by manipulating our data frames
# Iterate over each data frame in rand_list_1000
for i in range(len(rand_list_1000)):
    # Filter rows with multiple values in 'Amphora type'
    rand_list_1000[i] = rand_list_1000[i][rand_list_1000[i]['Amphora type'].apply(len) > 1]

print(rand_list_1000)


# Define the check_overlap function to check if all cargo production times overlap
def check_overlap(cargo_times):
    specified_centuries = ['2 BC', '1 BC', '1 AD', '2 AD', '3 AD', '4 AD', '5 AD', '6 AD', '7 AD']
    overlapping_centuries = []
    for century in specified_centuries:
        if all(any(century in times for times in cargo) for cargo in cargo_times):
            overlapping_centuries.append(century)
    return overlapping_centuries


# Create a list to store the shipwreck dictionaries
shipwreck_dict_list = []

# Iterate through the data frames in the 'rand_list_1000' list
for df in rand_list_1000:
    for index, row in df.iterrows():
        shipwreck_dict = {}  # Dictionary to store the shipwreck details
        shipwreck_dict['Shipwreck ID'] = row['Oxford_wreckID']
        cargo = row['Amphora type']

        # Get the production times for each amphora type in the cargo
        cargo_times = [production_times.get(amphora, []) for amphora in cargo]

        # Check if any cargo production times overlap with the specified centuries
        overlapping_centuries = check_overlap(cargo_times)
        if overlapping_centuries:
            for century in overlapping_centuries:
                shipwreck_dict = {}
                shipwreck_dict['Shipwreck ID'] = row['Oxford_wreckID']
                shipwreck_dict['Century'] = century
                shipwreck_dict['Cargo'] = cargo
                shipwreck_dict_list.append(shipwreck_dict)
        else:
            century_dict = {
                '2 BC': any('2 BC' in times for times in cargo_times),
                '1 BC': any('1 BC' in times for times in cargo_times),
                '1 AD': any('1 AD' in times for times in cargo_times),
                '2 AD': any('2 AD' in times for times in cargo_times),
                '3 AD': any('3 AD' in times for times in cargo_times),
                '4 AD': any('4 AD' in times for times in cargo_times),
                '5 AD': any('5 AD' in times for times in cargo_times),
                '6 AD': any('6 AD' in times for times in cargo_times),
                '7 AD': any('7 AD' in times for times in cargo_times)
            }

            if century_dict['2 BC']:
                shipwreck_dict['Century'] = '2 BC'
            elif century_dict['1 BC']:
                shipwreck_dict['Century'] = '1 BC'
            elif century_dict['1 AD']:
                shipwreck_dict['Century'] = '1 AD'
            elif century_dict['2 AD']:
                shipwreck_dict['Century'] = '2 AD'
            elif century_dict['3 AD']:
                shipwreck_dict['Century'] = '3 AD'
            elif century_dict['4 AD']:
                shipwreck_dict['Century'] = '4 AD'
            elif century_dict['5 AD']:
                shipwreck_dict['Century'] = '5 AD'
            elif century_dict['6 AD']:
                shipwreck_dict['Century'] = '6 AD'
            elif century_dict['7 AD']:
                shipwreck_dict['Century'] = '7 AD'
            else:
                for century, has_century in century_dict.items():
                    if has_century:
                        shipwreck_dict['Century'] = century
                        break

            shipwreck_dict['Cargo'] = cargo
            shipwreck_dict_list.append(shipwreck_dict)

# Group shipwrecks by century
shipwrecks_by_century = {'2 BC': [], '1 BC': [], '1 AD': [], '2 AD': [], '3 AD': [], '4 AD': [], '5 AD': [], '6 AD': [], '7 AD': []}

# Iterate through the shipwreck dictionaries
for shipwreck_dict in shipwreck_dict_list:
    century = shipwreck_dict['Century']
    shipwrecks_by_century[century].append(shipwreck_dict)

# Print the resulting dictionary with centuries as keys and shipwrecks and cargo as values
for century, shipwrecks in shipwrecks_by_century.items():
    print(century)
    for shipwreck in shipwrecks:
        print(shipwreck['Shipwreck ID'])
        print(shipwreck['Cargo'])


# Create an empty dictionary to store the shipwrecks
shipwrecks_by_century = {}

# Iterate through the shipwreck dictionaries
for shipwreck_dict in shipwreck_dict_list:
    century = shipwreck_dict['Century']
    shipwreck_id = shipwreck_dict['Shipwreck ID']
    cargo = shipwreck_dict['Cargo']

    # Create the century key if it doesn't exist
    if century not in shipwrecks_by_century:
        shipwrecks_by_century[century] = {}

    # Create the shipwreck ID key if it doesn't exist
    if shipwreck_id not in shipwrecks_by_century[century]:
        shipwrecks_by_century[century][shipwreck_id] = set()

    # Add the cargo to the shipwreck's cargo set
    shipwrecks_by_century[century][shipwreck_id].update(cargo)

# Print the resulting dictionary
# print(shipwrecks_by_century)

# Create a list to store the dictionaries for each data frame
shipwrecks_by_century_list = []

# Iterate through the data frames in 'rand_list_1000'
for df in rand_list_1000:
    # Create an empty dictionary to store the shipwrecks
    shipwrecks_by_century = {}

    # Iterate through the shipwreck dictionaries
    for shipwreck_dict in shipwreck_dict_list:
        century = shipwreck_dict['Century']
        shipwreck_id = shipwreck_dict['Shipwreck ID']
        cargo = shipwreck_dict['Cargo']

        # Create the century key if it doesn't exist
        if century not in shipwrecks_by_century:
            shipwrecks_by_century[century] = {}

        # Create the shipwreck ID key if it doesn't exist
        if shipwreck_id not in shipwrecks_by_century[century]:
            shipwrecks_by_century[century][shipwreck_id] = set()

        # Add the cargo to the shipwreck's cargo set
        shipwrecks_by_century[century][shipwreck_id].update(cargo)

    # Append the shipwrecks by century dictionary to the list
    shipwrecks_by_century_list.append(shipwrecks_by_century)

# Print the resulting list of dictionaries
for idx, shipwrecks_by_century in enumerate(shipwrecks_by_century_list):
    print(f"Dictionary for Data Frame {idx + 1}:")
    for century, shipwrecks in shipwrecks_by_century.items():
        print(century)
        for shipwreck_id, cargo in shipwrecks.items():
            print(shipwreck_id)
            print(cargo)
        print()

# Create the time stamps
for dictionary in shipwrecks_by_century_list:
    # Merge '1 BC' and '2 BC' into 'BC'
    dictionary['BC'] = {**dictionary.pop('1 BC', {}), **dictionary.pop('2 BC', {})}

    # Merge '4 AD', '5 AD', '6 AD', and '7 AD' into '4-7 AD'
    dictionary['4-7 AD'] = {
        **dictionary.pop('4 AD', {}),
        **dictionary.pop('5 AD', {}),
        **dictionary.pop('6 AD', {}),
        **dictionary.pop('7 AD', {})
    }

    # Rename '1 AD' as is
    # Rename '2 AD' as is
    # Rename '3 AD' as is

    # Remove any other keys not needed
    keys_to_remove = ['2 BC', '1 BC', '4 AD', '5 AD', '6 AD', '7 AD']
    for key in keys_to_remove:
        dictionary.pop(key, None)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Create the graphs for the randomized network
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Create the graphs for the randomized network
amphora_networks_by_century = []  # Create a list to store the networks for each time stamp
amphora_networks_dict = {}  # Create a dictionary to store the networks with keys as numbers of the current dictionary

for idx, dictionary in enumerate(shipwrecks_by_century_list):
    connections = {}
    G_rand = nx.Graph()

    for century_label, shipwrecks in dictionary.items():
        for shipwreck_id, amphora_types in shipwrecks.items():
            pairs = {(min(amphora_type_i, amphora_type_j), max(amphora_type_i, amphora_type_j)) for amphora_type_i in amphora_types for amphora_type_j in amphora_types if amphora_type_i != amphora_type_j}

            for pair in pairs:
                if pair[0] != pair[1]:
                    connections.setdefault(pair, 0)
                    connections[pair] += 1
                    G_rand.add_edge(pair[0], pair[1], weight=connections[pair])

    strength_of_connection = connections
    amphora_networks_by_century.append(G_rand)
    amphora_networks_dict[idx] = G_rand

print(amphora_networks_dict)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Partitions for the randomized network
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Perform Girvan-Newman algorithm and calculate statistics for each network
avg_stats_by_century = []

for graph in amphora_networks_by_century:
    components = girvan_newman(graph)

    # Filter partitions based on size criteria
    filtered_components = []
    for component in components:
        num_nodes = len(component)
        if num_nodes > 0.2 * graph.number_of_nodes() and num_nodes < 0.8 * graph.number_of_nodes():
            filtered_components.append(component)

    # Calculate statistics for the current network
    W_tot = 0
    W_S = 0
    W_sameprov = 0

    for u, v, data in graph.edges(data=True):
        provenance_u = amphora_origins.get(u, [None])[0]  # Get the provenance of node u
        provenance_v = amphora_origins.get(v, [None])[0]  # Get the provenance of node v

        if provenance_u != provenance_v:
            weight = data.get('weight', 0)  # Get the weight of the edge
            W_tot += weight

        if provenance_u == provenance_v:
            W_S += weight
        else:
            W_sameprov += weight

    if W_tot != 0:
        M = W_sameprov / W_tot
    else:
        M = 0

    # Append statistics to the list for the current century
    avg_stats_by_century.append({
        'W_tot': W_tot,
        'W_S': W_S,
        'W_sameprov': W_sameprov,
        'M': M
    })

# Calculate the average statistics for each century
avg_stats_over_time = {
    'W_tot': 0,
    'W_S': 0,
    'W_sameprov': 0,
    'M': 0
}

for i, stats in enumerate(avg_stats_by_century):
    print(f"Statistics for Century {i+1}:")
    print(f"Average total weight of links between different provenances: {stats['W_tot']}")
    print(f"Average total weight of links between nodes belonging to the same side of the partition: {stats['W_S']}")
    print(f"Average total weight of links between nodes belonging to different sides of the partition: {stats['W_sameprov']}")
    print(f"Average mixing weight (M): {stats['M']}")
    print()

    # Accumulate statistics for averaging over all time stamps
    avg_stats_over_time['W_tot'] += stats['W_tot']
    avg_stats_over_time['W_S'] += stats['W_S']
    avg_stats_over_time['W_sameprov'] += stats['W_sameprov']
    avg_stats_over_time['M'] += stats['M']

# Average the statistics over all time stamps
num_centuries = len(avg_stats_by_century)
avg_stats_over_time['W_tot'] /= num_centuries
avg_stats_over_time['W_S'] /= num_centuries
avg_stats_over_time['W_sameprov'] /= num_centuries
avg_stats_over_time['M'] /= num_centuries

# Print average statistics over all time stamps
print("Average Statistics over all Time Stamps:")
print(f"Average total weight of links between different provenances: {avg_stats_over_time['W_tot']}")
print(f"Average total weight of links between nodes belonging to the same side of the partition: {avg_stats_over_time['W_S']}")
print(f"Average total weight of links between nodes belonging to different sides of the partition: {avg_stats_over_time['W_sameprov']}")
print(f"Average mixing weight (M): {avg_stats_over_time['M']}")'''
