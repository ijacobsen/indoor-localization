"""
Created on Fri Jul  8 10:22:37 2016

Author: Ian Jacobsen
"""
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
########################### Parse Fingerprint File ############################
###############################################################################
#
# //////////////////////////////// DESCRIPTION \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#   this function takes in a raw data file containing position, RSSI, and MAC, 
# then returns the parsed data in the form of a dictionary
# 
# ////////////////////////////////// INPUTS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# file_name: 
#   - type: string
#   -description: (windows) path to file containing data (.txt, comma delimted)
# --------------------
# network_name:
#   - type: string
#   - description: name of the WLAN network that is being used
# 
# ////////////////////////////////// OUTPUTS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# MAC_addresses: 
#   - type: dictionary (key: string, values: list of [integer tuple, integer])
#   - description: stores the position and RSSI measurement as a list of entries
# associated with each MAC address
#

def parse_fingerprinting_data(file_name, network_name):
    # create empty dictionary for MAC addresses
    MAC_addresses = {}
    
    # open file, read only
    with open(file_name, 'r') as input_file:
    
        # iterate over each line in the file    
        for line in input_file:
            
            # seperate values based on comma delimitation, store in list
            current_line = line.split(",")
     
            # form a position tuple
            pos_tuple = (int(current_line[1]), int(current_line[2]))
    
            # loop over the length of the current line, searching for the MAC addresses        
            for i in range(len(current_line)):
                
                # if a MAC address is found
                if ((len(current_line[i]) == 17) and (current_line[i].count(':') == 5)):
    
                    # extract current MAC address and current RSSI value                
                    current_MAC = current_line[i]
                    current_RSSI = int(current_line[i+1])
                    
                    # if the MAC address exists in the dictionary, append a new entry                
                    if (current_MAC in MAC_addresses):
                        MAC_addresses[current_MAC].append([pos_tuple, current_RSSI])
                    
                    # otherwise, add MAC address to dictionary and assign first element                 
                    else:
                        MAC_addresses[current_MAC] = [[pos_tuple, current_RSSI]]
    return MAC_addresses

###############################################################################
############################### Fill Data Grid ################################
###############################################################################
#
# //////////////////////////////// DESCRIPTION \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#   this function takes in the MAC dictionary and an empty grid, then
# populates the empty grid with RSSI values that were stored in the dictionary
# 
# ////////////////////////////////// INPUTS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# MAC_dict: 
#   - type: dictionary (key: string, values: list of [integer tuple, integer])
#   - description: stores the position and RSSI measurement as a list of entries
# associated with each MAC address
# --------------------
# empty_grid:
#   - type: np.array[Y x X x N]
#   - description: empty numpy array of correct dimensionality (Ymax, Xmax, 
# Number of APs)
# 
# ////////////////////////////////// OUTPUTS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# empty_grid: 
#   - type: np.array[Y x X x N]
#   - description: (not actually empty!) each row corresponds to a Y coordinate,
# each column to an X coordinate, each slice to an AP
#

def populate_grid(MAC_dict, empty_grid):
    AP_num = 0
    for key in MAC_dict:
        for item in MAC_dict[key]:
            x = item[0][0]
            y = item[0][1]
            empty_grid[x, y, AP_num] = item[1]
        AP_num = AP_num+1
    
    return empty_grid
    
###############################################################################
############################## Create Empty Grid ##############################
###############################################################################
#
# //////////////////////////////// DESCRIPTION \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#   this function takes in the MAC dictionary the step size that the fingerprints
# were taken at, and returns an empty grid of the appropriate size. the empty
# grid is a stack of 2D position grids. each slice in the stack corresponds to
# a specific MAC address
# 
# ////////////////////////////////// INPUTS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# MAC_dict: 
#   - type: dictionary (key: string, values: list of [integer tuple, integer])
#   - description: stores the position and RSSI measurement as a list of entries
# associated with each MAC address
# --------------------
# step_size:
#   - type: integer
#   - description: distance between each measurement in the fingerprint. it is
# important to note that the distance between each X measurement must be consistent,
# as well as the distance between each Y measurement, AND the distance between
# X and Y must be the same! confusing to word, so here's an example:
#   x coordinates: 0, 4, 8, 12, 16, 20, 24, 28, 32, ...
#   y coordinates: 0, 4, 8, 12, 16, 20, 24, 28, 32, ... 
#   *** notice that the step size in X is the same as the step size in Y!
# 
# ////////////////////////////////// OUTPUTS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# grid: 
#   - type: np.array[Y x X x N]
#   - description: each row corresponds to a Y coordinate, each column to an 
# X coordinate, each slice to an AP
#

def form_grid(MAC_dict, step_size):
    # initial values for maximums
    x_max = 0
    y_max = 0
    # loop over each key in the dictionary
    for key in MAC_dict:
        
        # initialize empty lists
        x = []
        y = []
        
        # loop over each nested list in the value of the current key
        for item in MAC_dict[key]:
            x.append(item[0][0])
            y.append(item[0][1])
        
        # check if maximums are exceded, and if so update
        if (max(x) > x_max):
            x_max = max(x)
        if (max(y) > y_max):
            y_max = max(y)

    # find the number of APs in the dictionary
    num_APs = len(MAC_dict)    
    
    # form an empty grid
    grid = np.zeros([x_max+1, y_max+1, num_APs])
    
    # populate grid
    grid = populate_grid(MAC_dict, grid)
    return grid

###############################################################################
################################# Clean Grid ##################################
###############################################################################
#
# //////////////////////////////// DESCRIPTION \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#   this function takes in the RSSI grid (where the 0th and 1st axis represents 
# the position coordinates and the 2nd axis represents the APs), and the percent
# of measurements that each AP should contain in order to be used in the analysis.
# this function will filter out the slices where the AP does not have an adequite
# amount of RSSI values.
# 
# ////////////////////////////////// INPUTS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# grid: 
#   - type: np.array[Y x X x N]
#   - description: each row corresponds to a Y coordinate, each column to an 
# X coordinate, each slice to an AP
# --------------------
# percent_keep:
#   - type: float
#   - description: percentage (range [0, 1]) of measurements required to use 
# 
# ////////////////////////////////// OUTPUTS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# sig_grid: 
#   - type: np.array[Y x X x N]
#   - description: each row corresponds to a Y coordinate, each column to an 
# X coordinate, each slice to an AP
#

def clean_grid(grid, percent_keep):
    signf = np.floor(percent_keep*grid.shape[0]*grid.shape[1])
    
    # find APs which have enough associated measurements to be useful
    count = 0
    for i in range(grid.shape[2]):
        if (np.count_nonzero(grid[:, :, i]) >= signf):
            grid[:, :, count] = grid[:, :, i]
            count = count + 1
    
    # significant grid
    sig_grid = grid[:, :, :count]
    return sig_grid
    
###############################################################################
############################# Find Zero Positions #############################
###############################################################################
#
# //////////////////////////////// DESCRIPTION \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#   this function takes in the RSSI grid (where the 0th and 1st axis represents 
# the position coordinates and the 2nd axis represents the APs), and finds the
# position of 0 entries. the function returns a list of lists of tuples. each
# nested list is associated with a specific grid, and each tuple correpsonds
# to a coordinate position where a 0 was entered.
# 
# ////////////////////////////////// INPUTS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# grid: 
#   - type: np.array[Y x X x N]
#   - description: each row corresponds to a Y coordinate, each column to an 
# X coordinate, each slice to an AP 
# 
# ////////////////////////////////// OUTPUTS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# zeros_pos_tupl_lst: 
#   - type: list of lists of tuples
#   - description: each nested list is associated with a specific grid, and 
# each tuple correpsonds to a coordinate position where a 0 was entered.
#

def find_zeros(grid):
    zero_pos_tupl_lst = [[] for i in range(grid.shape[2])]
    for i in range(grid.shape[2]):
        curr_grid = grid[:, :, i]
        
        # if 0 exists in the current grid
        if (0 in curr_grid):       
            
            # find all of the coordinate pairs where the 0's exist
            zero_pos = (np.where(curr_grid == 0)[0], np.where(curr_grid == 0)[1])
            
            # loop over the number of 0's in the grid        
            for j in range(zero_pos[0].shape[0]):
                
                # extract coordinate tuples and append to list
                zero_pos_tupl_lst[i].append((zero_pos[0][j], zero_pos[1][j]))
    return zero_pos_tupl_lst

###############################################################################
############################ Interpolate and Insert ###########################
###############################################################################
#
# //////////////////////////////// DESCRIPTION \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#   this function takes in the RSSI grid (where the 0th and 1st axis represents 
# the position coordinates and the 2nd axis represents the APs), and finds the
# position of 0 entries. the function returns a list of lists of tuples. each
# nested list is associated with a specific grid, and each tuple correpsonds
# to a coordinate position where a 0 was entered.
# 
# ////////////////////////////////// INPUTS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# grid: 
#   - type: np.array[Y x X x N]
#   - description: each row corresponds to a Y coordinate, each column to an 
# X coordinate, each slice to an AP 
# 
# ////////////////////////////////// OUTPUTS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# zeros_pos_tupl_lst: 
#   - type: list of lists of tuples
#   - description: each nested list is associated with a specific grid, and 
# each tuple correpsonds to a coordinate position where a 0 was entered.
#

def intrplte_zeros(grid, zero_pos):
    xmax = grid.shape[0]
    ymax = grid.shape[1]
    for i in range(len(zero_pos)):
        pos_list = zero_pos[i]
        # form coordinate pairs, store as tuples
        for item in pos_list:
            nx = item[0]
            ny = item[1]
            average_list = []
            tuple_list = []
            x = np.arange(nx-1, nx+2, 1)
            y = np.arange(ny-1, ny+2, 1)
            for x_i in x:
                for y_i in y:
                    tuple_list.append((x_i, y_i))
            
            # if nonzero entry, add to list to average
            for cord in tuple_list:
                x = cord[0]
                y = cord[1]    
                
                # if indices out of bounds, skip to next interation
                if (x < 0 or y < 0 or x >= xmax or y >= ymax):
                    continue
                
                if (grid[x, y, i] != 0):
                    average_list.append(grid[x, y, i])
            grid[nx, ny, i] = int(np.mean(average_list))
    return grid
    
###############################################################################
############################## Call to Functions ##############################
###############################################################################

# parse and extract data file
file_name = 'C:\\Users\\GMRD43\\Documents\\Indoor Tracking\\Data Files\\wifi_trace_manual_scan_7_18_cafeteria.txt'
network_name = 'M-Wireless'
MAC_dict = parse_fingerprinting_data(file_name, network_name)

# form fingerprint grid
step_size = 1
grid = form_grid(MAC_dict, step_size)

# remove grids that contain less than 70% of the size of the grid
percent_keep = 0.9
grid = clean_grid(grid, percent_keep)

# find all 0's and store their positions in a list of tuples
zero_pos = find_zeros(grid)
    
# average all non-zero surrounding values and insert to 0 position
grid = intrplte_zeros(grid, zero_pos)

# plot raw (with filled in 0's) and convolved
for i in range(grid.shape[2]):
    plt.subplot(3, 2, i+1)
    plt.contourf(grid[:, :, i])

