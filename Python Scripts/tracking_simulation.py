# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 10:50:24 2016

@author: gmrd43
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.signal as signal


#***************************************************************#
#***************************************************************#
#################################################################
###################### Function Definitions #####################
#################################################################
#***************************************************************#
#***************************************************************#
 
def resample(sp, wp):
    M = sp.shape[0]
    cdf = np.cumsum(wp)
    strtpt = np.random.uniform(0, 1/M)
    i = 0
    ind = np.zeros(M)
    spt = np.zeros(sp.shape)
    W = np.empty([M])
    for j in range(M):
        uj = strtpt + (j-1)/M
        while (uj > cdf[i]):
            i = i + 1
        spt[j, :] = sp[i, :]
        W[j] = 1/M
        ind[j] = i
    spt[0, :] = spt[1, :]               # remove erronous particle
    return spt

def pathloss_model(dist):
    alpha_pl = 4.02
    r_sens = -90.
    sigma_pl = 4#7.36
    RSSI = r_sens - 10 * alpha_pl * np.log10(dist) + np.random.normal(0, sigma_pl, dist.shape)
    return RSSI

def trajectory_model(num_time, time_length, traj):
    traj_amp = 30                         # amplitude of trajectory
    traj_var = .2                         # variance of noise present in trajectory
    traj_freq = np.pi/16                 # frequency of cosine trajectory
    traj_x = np.linspace(0, time_length, num_time)
    
    if (traj == "cos"):    
        alpha_decay = 0.01                   # damping factor
        traj_y = np.exp(-alpha_decay*traj_x) * (traj_amp * np.cos(traj_freq*traj_x) + np.random.normal(0, traj_var, num_time))
    elif (traj == "cir"):           
        traj_y = traj_amp*np.cos(traj_x*traj_freq)
        traj_x = 40 + traj_amp*np.sin(traj_x*traj_freq)    
    return [traj_x, traj_y]
   
def random_traj_sample(num_time, num_samps, traj_x, traj_y):
    some_perm = np.random.permutation(num_time)
    selected_points = some_perm[:num_samps]
    selected_points.sort()
    traj_x_samp = np.array(traj_x[selected_points])
    traj_y_samp = np.array(traj_y[selected_points])
    sample_points = np.vstack((traj_x_samp, traj_y_samp))
    return sample_points.T

def form_grid(AP_location, fp_res):
    minimum_x_co = math.ceil((1.2+10)*np.amin(AP_location[:, 0]))
    minimum_y_co = math.ceil(1.2*np.amin(AP_location[:, 1]))
    maximum_x_co = math.ceil(1.2*np.amax(AP_location[:, 0]))
    maximum_y_co = math.ceil(1.2*np.amax(AP_location[:, 1]))
    xgrid = np.arange(minimum_x_co, maximum_x_co, fp_res)
    ygrid = np.arange(minimum_y_co, maximum_y_co, fp_res)
    xx, yy = np.meshgrid(xgrid, ygrid, sparse=True)
    return [xx, yy]

def distance_rssi_grid(x_max, y_max, AP_location, xx, yy):
    num_APs = len(AP_location)
    fp_distance_grid = np.empty([y_max, x_max, num_APs])
    fp_RSSI_grid = np.empty([y_max, x_max, num_APs])
    for n in range(num_APs):
        fp_distance_grid[:, :, n] = np.linalg.norm([xx, yy] - AP_location[n, :])
        fp_RSSI_grid[:, :,  n] = pathloss_model(fp_distance_grid[:, :, n])
    return [fp_distance_grid, fp_RSSI_grid]

def distance_rssi_sample(sample_points, AP_location, num_aver):
    num_APs = len(AP_location)
    num_samps = len(sample_points)
    sample_distances = np.empty((num_samps, num_APs))
    sample_RSSI_temp = np.empty((num_samps, num_APs, num_aver))
    sample_RSSI = np.empty((num_samps, num_APs))
    for n in range(num_APs):
        sample_distances[:, n] = np.linalg.norm(sample_points[:, :] - AP_location[n, :], axis=1)
        for i in range(num_aver):
            sample_RSSI_temp[:, n, i] = pathloss_model(sample_distances[:, n])
        sample_RSSI[:, n] = np.mean(sample_RSSI_temp[:, n, :], axis=1)
    return [sample_distances, sample_RSSI]
    
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
        most_likely_pos[i, :] = [xx.T[int(min_index[i, 1])], yy[int(min_index[i, 0])]]       
    return most_likely_pos

def averaged_rssi_grid(fp_distance_grid, num_aver):
    num_APs = fp_distance_grid.shape[2]
    y_max = fp_distance_grid.shape[0]
    x_max = fp_distance_grid.shape[1]
    fp_RSSI_grid_temp = np.empty([y_max, x_max, num_APs, num_aver])
    for i in range(num_aver):
        for n in range(num_APs):
            fp_RSSI_grid_temp[:, :,  n, i] = pathloss_model(fp_distance_grid[:, :, n])
    averaged = np.mean(fp_RSSI_grid_temp, axis=3) 
    return averaged

def smoothed_rssi_grid(fp_RSSI_grid):
    original = np.copy(fp_RSSI_grid)
    num_APs = original.shape[2]
    #conv_filter = np.array([[0, .2, 0],[.2, .2, .2],[0,.2, 0]])
    conv_filter = np.array([[1/9, 1/9, 1/9],[1/9, 1/9, 1/9],[1/9, 1/9, 1/9]])
    smoothed = np.empty(fp_RSSI_grid.shape)
    for i in range(num_APs):
        smoothed[:, :, i] = signal.convolve2d(original[:, :, i], conv_filter, boundary='symm', mode='same')
    return smoothed
    
#***************************************************************#
#***************************************************************#
#################################################################
#################### Beginning of Test Script ###################
#################################################################
#***************************************************************#
#***************************************************************#

def main(traj):
    
    # parameters that we can vary
    time_length = 90                # length of trajectory
    percent_samps = 0.7             # percentage of samples to take (btwn 0 and 1)
    AP_0_cord = [-5, -50]
    AP_1_cord = [-5, 50]
    AP_2_cord = [95, -50]
    AP_3_cord = [95, 50]
    fp_res = 4
    smplng_frq = 32
    plot_flag = False
    
    # constants
    num_time = time_length*smplng_frq
    num_samps = int(num_time*percent_samps)
    
    # generate true trajectory
    if (traj == "cos"):
        [traj_x, traj_y] = trajectory_model(num_time, time_length, "cos")
    elif (traj == "cir"):
        [traj_x, traj_y] = trajectory_model(num_time, time_length, "cir")
    
    
    # random sample from true trajectory
    sample_points = random_traj_sample(num_time, num_samps, traj_x, traj_y)
    
    # location of APs
    AP_location = np.array([AP_0_cord, AP_1_cord, AP_2_cord, AP_3_cord])
    
    # form grids for fingerprinting, each of area specified by fp_res^2
    [xx, yy] = form_grid(AP_location, fp_res)
    x_max = np.size(xx)
    y_max = np.size(yy)
    
    # calculate distance at each point in the grid to each of the APs, as well as estimated RSSI
    [fp_distance_grid, fp_RSSI_grid] = distance_rssi_grid(x_max, y_max, AP_location, xx, yy)
    
    # average RSSI grid over num_aver measurements
    num_aver = 20               # can vary this
    averaged_RSSI_grid = averaged_rssi_grid(fp_distance_grid, num_aver)
    
    # smooth RSSI grid
    smoothed_RSSI_grid = smoothed_rssi_grid(averaged_RSSI_grid)
    
    # simulate then calculate distance and estimated RSSI at each sampled point in the trajectory to each of the APs
    num_aver = 8
    [sample_distances, sample_RSSI] = distance_rssi_sample(sample_points, AP_location, num_aver)
    
    # find most likely position of the sample by minimizing mean squared error of RSSI
    #most_likely_pos_og = fp_estimate(xx, yy, sample_RSSI, fp_RSSI_grid)
    #most_likely_pos_av = fp_estimate(xx, yy, sample_RSSI, averaged_RSSI_grid)
    most_likely_pos_sm = fp_estimate(xx, yy, sample_RSSI, smoothed_RSSI_grid)
    most_likely_pos = most_likely_pos_sm
    
    # useful plots before particle filter
    if (plot_flag == True):
        '''
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
    '''
    
    #################################################################
    ######################## Particle Filter ########################
    #################################################################
    
    # given most_likely_pos, want to ESTIMATE sample_points
    
    # parameters
    n_samp = num_samps      # number of samples
    M = 800                 # number of particles
    std_v = 0.0001               # stddev of observation process o(t)
    sigma_mult = 1
    
    # initialization
    v = np.random.normal(0, std_v, [n_samp, 2])     # nx2, [vx, vy]
    w = np.empty([M, 1])                            # Mx1, [w]
    s_est = np.zeros([n_samp, 3])
    weighted_sum = np.zeros([n_samp, 3])                       # weighted mean
    MAP_estimate = np.zeros([n_samp, 3])                       # most likely weight
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
    plt.clf()
    plt.plot(sp[:, 0], sp[:, 1], 'ro', ms = 8)
    plt.plot(AP_location[:, 0], AP_location[:, 1], '^g', markersize=32)
    plt.pause(4)
    
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
    
    #***************************************************************#
    #***************************************************************#
    #################################################################
    ########################### Evaluation ##########################
    #################################################################
    #***************************************************************#
    #***************************************************************#
    
    # evaluate performance, print, and plot
    pf_me = np.mean(np.linalg.norm((s_est[:, 0:2] - sample_points[:, :]), axis=1))
    #pf_std = np.std(np.linalg.norm((s_est[:, 0:2] - sample_points[:, :]), axis=1))
    fp_me = np.mean(np.linalg.norm((o[:, :] - sample_points[:, :]), axis=1))
    #fp_std = np.std(np.linalg.norm((o[:, :] - sample_points[:, :]), axis=1))
    
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
    
    '''
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
    '''    
    
    plt.figure(5)
    plt.clf()
    plt.plot(AP_location[:, 0], AP_location[:, 1], '^g', markersize=32, label='WiFi AP')
    plt.plot(sample_points.T[0, 0:t], sample_points.T[1, 0:t], linewidth = 4, label='Actual Trajectory')
    plt.plot(s_est[0:t, 0], s_est[0:t, 1], 'go', linewidth = 2, label='Estimated Trajectory using Proposed Method')
    #plt.plot(most_likely_pos[0:t, 0], most_likely_pos[0:t, 1], 'r', linewidth = 1, label='Estimated Trajectory using Finger Printing')
    #plt.plot(s_est[t, 0], s_est[t, 1], 'yo', ms = 14, label='Point Estimate')
    #plt.plot(sp[:, 0], sp[:, 1], 'ro', ms = 4, label='Particles')
    #plt.plot(weighted_sum[0:t, 0], weighted_sum[0:t, 1], 'y', label='Weighted Sum')  
    #plt.plot(MAP_estimate[0:t, 0], MAP_estimate[0:t, 1], 'c', label='MAP Estimate')        
    #plt.plot(windowed_MAP[0:t, 0], windowed_MAP[0:t, 1], 'k', label='Windowed MAP')   
    plt.legend(loc=0) 
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Trajectory Across Room')
    plt.pause(15)

#***************************************************************#
#***************************************************************#
#################################################################
########################## Call to Main #########################
#################################################################
#***************************************************************#
#***************************************************************#

main("cos")
'''
while(1):
    main("cos")    
    main("cir")
'''        

