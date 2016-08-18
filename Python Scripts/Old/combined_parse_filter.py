"""
Created on Fri Jul  8 10:22:37 2016

Author: Ian Jacobsen
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.signal as signal

#*****************************************************************************#
#*****************************************************************************#
###############################################################################
############################# Function Definitions ############################
###############################################################################
#*****************************************************************************#
#*****************************************************************************#


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
            pos_tuple = (int(current_line[0]), int(current_line[1]))
    
            # loop over the length of the current line, searching for the MAC addresses        
            for i in range(len(current_line)):
                
                # if the network name is found, then the next entry is a MAC address
                if (current_line[i] == network_name):
    
                    # extract current MAC address and current RSSI value                
                    current_MAC = current_line[i+1]
                    current_RSSI = int(current_line[i+2])
                    
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
############################ Resampling Technique #############################
###############################################################################
#
# //////////////////////////////// DESCRIPTION \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#   this function takes in the vector of current particles, as well as the 
# weights associated with each particle. the function outputs a new set of
# particles which correspond to the current time instant. this algorithm was
# translated from the psuedo-code that was provided in Algorithm 2 in the paper:
# "A Tutorial on Particle Filters for Online Nonlinear/ Non-Gaussian Bayesian
# Tracking", written by M. Sanjeev Arulampalam et al.
# 
# ////////////////////////////////// INPUTS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# sp: 
#   - type: np.array
#   - description: this array contains particles corresponding to the current
# time period. each entry in the array is a float. the particles have the following
# components: x position, y position, theta
# --------------------
# wp:
#   - type: np.array
#   - description: these are the weights corresponding to each of the current 
# particles. each weight is a float.
# 
# ////////////////////////////////// OUTPUTS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# spt: 
#   - type: np.array
#   - description: new, updated particles
#
 
def resample(sp, wp):
    
    # find the number of particles
    M = sp.shape[0]
    
    # form cumulative distribution
    cdf = np.cumsum(wp)
    
    # initialize at a random starting point
    strtpt = np.random.uniform(0, 1/M)
    
    # start at the bottom of the cdf    
    i = 0

    # initialize 0's
    spt = np.zeros(sp.shape)
    W = np.empty([M])

    # navigate through the cdf. a random seed is used to eliminate small probabilities.
    # we will reuse particles with significant weights.
    for j in range(M):
        uj = strtpt + (j-1)/M
        while (uj > cdf[i]):  
            i = i + 1
        
        # assign new particle
        spt[j, :] = sp[i, :]
        
        # uniform weighting
        W[j] = 1/M
        
    spt[0, :] = spt[1, :]               # remove erronous particle
    return spt
  
###############################################################################
######################## Fingerprint Based Observation ########################
###############################################################################
# 
# *******************************************IAN THIS NEEDS WORK
# have to figure out how to deal with no measurement, filling in holes, etc
#  
# //////////////////////////////// DESCRIPTION \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# 
# ////////////////////////////////// INPUTS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#  
#   - type: 
#   - description: 
# --------------------
#   - type: 
#   - description: 
# --------------------
# 
# ////////////////////////////////// OUTPUTS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#   - type: 
#   - description: 
# 
  
def fp_estimate(xx, yy, sample_RSSI, fp_RSSI_grid):
    num_samps = sample_RSSI.shape[0]
    x_max = xx.shape[1]
    y_max = yy.shape[0]
    RSSI_sqrt_SSE = np.empty((y_max, x_max))
    most_likely_pos = np.empty((num_samps, 2))
    min_index = np.empty((num_samps, 2))    
    for i in range(num_samps):
        current_samp = sample_RSSI[i, :]
        dist = fp_RSSI_grid - current_samp
        SSE = np.sum(dist**2, axis=2)
        RSSI_sqrt_SSE = np.sqrt(SSE)
        min_index[i, :] = np.unravel_index(RSSI_sqrt_SSE.argmin(), RSSI_sqrt_SSE.shape)
        most_likely_pos[i, :] = [xx.T[min_index[i, 1]], yy[min_index[i, 0]]]       
    return most_likely_pos

###############################################################################
############################ Convolution Smoothing ###########################
###############################################################################
# 
# *******************************************IAN THIS NEEDS WORK
# have to figure out how to deal with no measurement, filling in holes, etc
#  
# //////////////////////////////// DESCRIPTION \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# 
# ////////////////////////////////// INPUTS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#  
#   - type: 
#   - description: 
# --------------------
#   - type: 
#   - description: 
# --------------------
# 
# ////////////////////////////////// OUTPUTS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#   - type: 
#   - description: 
# 

def smoothed_rssi_grid(fp_RSSI_grid):
    original = np.copy(fp_RSSI_grid)
    num_APs = original.shape[2]
    conv_filter = np.array([[1/9, 1/9, 1/9],[1/9, 1/9, 1/9],[1/9, 1/9, 1/9]])
    smoothed = np.empty(fp_RSSI_grid.shape)
    for i in range(num_APs):
        smoothed[:, :, i] = signal.convolve2d(original[:, :, i], conv_filter, boundary='symm', mode='same')
    return smoothed
    
#*****************************************************************************#
#*****************************************************************************#
###############################################################################
########################### Beginning of Test Script ##########################
###############################################################################
#*****************************************************************************#
#*****************************************************************************#

plt.close("all")

# parameters that we can vary
plot_flag = False

# parse and extract data file
file_name = 'C:\\Users\\GMRD43\\Documents\\Indoor Tracking\\Data Files\\wifi_trace_manual_scan_with_delay.txt'
network_name = 'ZWireless'
MAC_dict = parse_fingerprinting_data(file_name, network_name)

# form fingerprint grid
step_size = 1
grid = form_grid(MAC_dict, step_size)

# form grids for fingerprinting, each of area specified by fp_res^2


# smooth RSSI grid
smoothed_RSSI_grid = smoothed_rssi_grid(averaged_RSSI_grid)

# simulate then calculate distance and estimated RSSI at each sampled point in the trajectory to each of the APs
num_aver = 10
[sample_distances, sample_RSSI] = distance_rssi_sample(sample_points, AP_location, num_aver)

# find most likely position of the sample by minimizing mean squared error of RSSI
most_likely_pos_og = fp_estimate(xx, yy, sample_RSSI, fp_RSSI_grid)
most_likely_pos_av = fp_estimate(xx, yy, sample_RSSI, averaged_RSSI_grid)
most_likely_pos_sm = fp_estimate(xx, yy, sample_RSSI, smoothed_RSSI_grid)
most_likely_pos = most_likely_pos_sm

# useful plots before particle filter
if (plot_flag == True):
    # display contour plot of RSSI based off of distance from AP
    plt.figure(1) 
    for i in range(num_APs):
        plt.subplot(2, 2, i+1)        
        plt.contourf(xx[0],yy.T[0],fp_RSSI_grid[:, :, i])
        plt.plot(traj_x, traj_y)
        plt.plot(sample_points[:, 0], sample_points[:, 1], 'ro')
        plt.plot(AP_location[:, 0], AP_location[:, 1], '^g', markersize=32)
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')    
    plt.show()
    plt.suptitle('Raw RSSI Contour Plot')    
    
    # display contour plot of averaged RSSI
    plt.figure(2)    
    for i in range(num_APs):
        plt.subplot(2, 2, i+1)        
        plt.contourf(xx[0],yy.T[0],averaged_RSSI_grid[:, :, i])
        plt.plot(traj_x, traj_y)
        plt.plot(sample_points[:, 0], sample_points[:, 1], 'ro')
        plt.plot(AP_location[:, 0], AP_location[:, 1], '^g', markersize=32)
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')    
    plt.show()  
    plt.suptitle('Averaged RSSI Contour Plot')
    
    # display contour plot of smoothed RSSI
    plt.figure(3)  
    for i in range(num_APs):
        plt.subplot(2, 2, i+1)        
        plt.contourf(xx[0],yy.T[0],smoothed_RSSI_grid[:, :, i])
        plt.plot(traj_x, traj_y)
        plt.plot(sample_points[:, 0], sample_points[:, 1], 'ro')
        plt.plot(AP_location[:, 0], AP_location[:, 1], '^g', markersize=32)    
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')    
    plt.show()
    plt.suptitle('Smoothed RSSI Contour Plot')

    # display actual position vs MMSE estimate
    plt.figure(4)
    plt.subplot(3, 1, 1)
    plt.plot(sample_points[:, 0], sample_points[:, 1])
    plt.subplot(3, 1, 2)
    plt.plot(most_likely_pos[:, 0], most_likely_pos[:, 1])
    plt.subplot(3, 1, 3)
    plt.plot(sample_points[:, 0], sample_points[:, 1])
    plt.plot(most_likely_pos[:, 0], most_likely_pos[:, 1], 'ro')
    plt.show()
    plt.suptitle('Actual Position vs MMSE Estimate')


###############################################################################
############################### Particle Filter ###############################
###############################################################################

# given most_likely_pos, want to ESTIMATE sample_points

# parameters
n_samp = num_samps      # number of samples
M = 800                 # number of particles
std_v = 0.0001               # stddev of observation process o(t)
sigma_mult = 1

# initialization
s = np.empty([n_samp, 2])                       # nx2, [sx, sy]
v = np.random.normal(0, std_v, [n_samp, 2])     # nx2, [vx, vy]
w = np.empty([M, 1])                            # Mx1, [w]
s_est = np.zeros([n_samp, 3])
weighted_sum = np.zeros([n_samp, 3])                       # weighted mean
MAP_estimate = np.zeros([n_samp, 3])                                   # most likely weight
windowed_MAP = np.zeros([n_samp, 3])   

# observations are the most likely positions
o = most_likely_pos + v                        # minimum squared error + zero mean noise

# generate initial particles from a uniform distribution
sp = np.zeros([M, 3])
sp[:, 0] = np.random.uniform(xx[0, 0], xx[0, -1], M)
sp[:, 1] = np.random.uniform(yy.T[0, 0], yy.T[0, -1], M)
sp[:, 2] = np.random.uniform(0, 2*np.pi, M)


# calculate weights using Gaussian kernel
log_w = -(((o[0, 0] - sp[:, 0])**2)/(2*(sigma_mult*1)**2) + ((o[0, 1] - sp[:, 1])**2)/(2*(sigma_mult*1))**2)
w_max = np.max(log_w)
log_w_shifted = log_w - w_max
w = np.exp(log_w_shifted)                   # back to exponential
w_nrmlzd = w/np.sum(w)                      # sum of weights = 1
w_shaped = w_nrmlzd.reshape(M, 1)           # reshape format of data 
s_est[0, :] = np.sum(w_shaped*sp, axis=0)   # form estimates

# resample
spt = resample(sp, w)

# display trajectory, particles, and estimate
plt.figure(5)
plt.plot(sp[:, 0], sp[:, 1], 'ro', ms = 8)
plt.plot(AP_location[:, 0], AP_location[:, 1], '^g', markersize=32)
plt.pause(10)

for t in range(1, n_samp):
    
    # generate random depdendent parameters for motion model using Box-Muller transformation
    x0 = np.random.uniform(0, 1, M)
    x1 = np.random.uniform(0, 1, M)
    delD = 0.55*np.sqrt(-2*np.log(x0))*np.cos(2*np.pi*x1)            #0.55 is good
    theta_noise = np.sqrt(-2*np.log(x0))*np.sin(2*np.pi*x1)
    
    # theta = arctan(dely/delx) + noise    
    theta = theta_noise + np.arctan((o[t, 1] - o[t-1, 1])/(o[t, 0] - o[t-1, 0]))
    
    # propagation of particles through motion model
    sp[:, 2] = (spt[:, 2] + theta)%(2*np.pi)
    sp[:, 0] = spt[:, 0] + delD*np.cos(sp[:, 2])
    sp[:, 1] = spt[:, 1] + delD*np.sin(sp[:, 2])
    
    # calculate weights
    rho = np.corrcoef(sp.T)[1, 0]
    k_term_one = ((-sp[:, 0] + o[t, 0])**2)/(2*(sigma_mult*np.std(sp[:, 0]))**2)
    k_term_two = (2*rho*(-sp[:, 0] + o[t, 0])*(-sp[:, 1] + o[t, 1]))/(sigma_mult*np.std(sp[:, 0])*(sigma_mult*np.std(sp[:, 1])))
    k_term_three =  ((-sp[:, 1] + o[t, 1])**2)/(2*(sigma_mult*np.std(sp[:, 1]))**2) 
    log_w = -(k_term_one - k_term_two + k_term_three)
    w_max = np.max(log_w)
    log_w_shifted = log_w - w_max
    w = np.exp(log_w_shifted)
    w_nrmlzd = w/np.sum(w)
    w_shaped = w_nrmlzd.reshape(M, 1)

    # current estimate
    robust_mean_num = 3
    max_index = np.argsort(w)[-robust_mean_num:] 
    weighted_sum[t, :] = np.sum(w_shaped*sp, axis=0)                         # weighted mean
    MAP_estimate[t, :] = sp[max_index[-1]]                                   # most likely weight
    windowed_MAP[t, :] = np.mean(sp[max_index], axis=0)                      # robust mean
    current_estimate = windowed_MAP[t, :]    
    
    # estimate of position
    if t > 2*smplng_frq:      # first 2 seconds go with raw estimate
        temp = np.ma.average(np.vstack((s_est[t-3:t-1, :], current_estimate)), axis=0, weights=[.5, .6, .4])
        #temp = np.ma.average(np.vstack((s_est[t-5:t-1, :], current_estimate)), axis=0, weights=[.4, .4, .5, .9, .7])
    else:
        temp = current_estimate

    # final estimate    
    s_est[t, :] = [temp[0], temp[1], temp[2]]

    # resample
    spt = resample(sp, w_shaped)
    
    if (plot_flag == True):
        # display trajectory, particles, and estimate
        plt.clf()
        plt.plot(AP_location[:, 0], AP_location[:, 1], '^g', markersize=32, label='WiFi AP')
        plt.plot(sample_points.T[0, 0:t], sample_points.T[1, 0:t], label='Actual Trajectory')
        plt.plot(s_est[0:t, 0], s_est[0:t, 1], 'go', linewidth = 2, label='Estimated Trajectory')
        plt.plot(s_est[t, 0], s_est[t, 1], 'yo', ms = 14, label='Point Estimate')
        plt.plot(sp[:, 0], sp[:, 1], 'ro', ms = 4, label='Particles')
        #plt.plot(weighted_sum[0:t, 0], weighted_sum[0:t, 1], 'y', label='Weighted Sum')  
        #plt.plot(MAP_estimate[0:t, 0], MAP_estimate[0:t, 1], 'c', label='MAP Estimate')        
        #plt.plot(windowed_MAP[0:t, 0], windowed_MAP[0:t, 1], 'k', label='Windowed MAP')        
        plt.legend(loc=0) 
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Trajectory Across Room')
        plt.pause(.001)

#*****************************************************************************#
#*****************************************************************************#
###############################################################################
################################## Evaluation #################################
###############################################################################
#*****************************************************************************#
#*****************************************************************************#

# evaluate performance, print, and plot
pf_me = np.mean(np.linalg.norm((s_est[:, 0:2] - sample_points[:, :]), axis=1))
pf_std = np.std(np.linalg.norm((s_est[:, 0:2] - sample_points[:, :]), axis=1))
fp_me = np.mean(np.linalg.norm((o[:, :] - sample_points[:, :]), axis=1))
fp_std = np.std(np.linalg.norm((o[:, :] - sample_points[:, :]), axis=1))

ws_me = np.mean(np.linalg.norm((weighted_sum[:, 0:2] - sample_points[:, :]), axis=1))
map_me = np.mean(np.linalg.norm((MAP_estimate[:, 0:2] - sample_points[:, :]), axis=1))
wmap_me = np.mean(np.linalg.norm((windowed_MAP[:, 0:2] - sample_points[:, :]), axis=1))

print('\n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n')
print('========================================================================')
print('            Mean Euclidian Distance of Different Estimates (m)          ')
print('========================================================================')
print('ARMA Estimate | Weighted Sum | MAP Estimate | Windowed MAP | Fingerprint')
print('==============|==============|==============|==============|============')
print(' %.2f         | %.2f         | %.2f         | %.2f         | %.2f       ' %(pf_me, ws_me, map_me, wmap_me, fp_me))
print('\n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n')

plt.figure(4)
plt.clf()
plt.subplot(221)
plt.title('ARMA Estimate')
plt.plot(sample_points[:, 0], sample_points[:, 1], 'b')
plt.plot(s_est[0:t, 0], s_est[0:t, 1], 'g')
plt.subplot(222)
plt.title('Weighted Sum Estimate')
plt.plot(sample_points[:, 0], sample_points[:, 1], 'b')
plt.plot(weighted_sum[0:t, 0], weighted_sum[0:t, 1], 'g')
plt.subplot(223)
plt.title('MAP Estimate')
plt.plot(sample_points[:, 0], sample_points[:, 1], 'b')
plt.plot(MAP_estimate[0:t, 0], MAP_estimate[0:t, 1], 'g')
plt.subplot(224)
plt.title('Windowed MAP Estimate')
plt.plot(sample_points[:, 0], sample_points[:, 1], 'b')
plt.plot(windowed_MAP[0:t, 0], windowed_MAP[0:t, 1], 'g')
plt.tight_layout()
plt.show()


plt.figure(5)
plt.clf()
plt.plot(AP_location[:, 0], AP_location[:, 1], '^g', markersize=32, label='WiFi AP')
plt.plot(sample_points.T[0, 0:t], sample_points.T[1, 0:t], linewidth = 4, label='Actual Trajectory')
plt.plot(s_est[0:t, 0], s_est[0:t, 1], 'go', linewidth = 2, label='Estimated Trajectory using Proposed Method')
plt.plot(most_likely_pos[0:t, 0], most_likely_pos[0:t, 1], 'r', linewidth = 1, label='Estimated Trajectory using Finger Printing')
#plt.plot(s_est[t, 0], s_est[t, 1], 'yo', ms = 14, label='Point Estimate')
#plt.plot(sp[:, 0], sp[:, 1], 'ro', ms = 4, label='Particles')
#plt.plot(weighted_sum[0:t, 0], weighted_sum[0:t, 1], 'y', label='Weighted Sum')  
#plt.plot(MAP_estimate[0:t, 0], MAP_estimate[0:t, 1], 'c', label='MAP Estimate')        
#plt.plot(windowed_MAP[0:t, 0], windowed_MAP[0:t, 1], 'k', label='Windowed MAP')   
plt.legend(loc=0) 
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Trajectory Across Room')

        
        


