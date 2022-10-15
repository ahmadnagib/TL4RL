import numpy as np
import pandas as pd
import scipy.io as sio

np.random.seed(2021)

# reward function weights of the basic expert agents
weights = [[0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.1, 0.45, 0.45], [0.1, 0.7, 0.2], [0.1, 0.2, 0.7],
    [0.1, 0.6, 0.3], [0.1, 0.3, 0.6], [0.1, 0.5, 0.4], [0.1, 0.4, 0.5], [0.3333, 0.3333, 0.3333],
    [0.8, 0.1, 0.1], [0.4, 0.5, 0.1], [0.4, 0.4, 0.2], [0.4, 0.1, 0.5], [0.4, 0.2, 0.4], [0.4, 0.3, 0.3]]

# reward function weights of the learner agents
weights_for_acc = [[0.1, 0.1, 0.8], [0.1, 0.8, 0.1], [0.8, 0.1, 0.1], [0.6, 0.2, 0.2], [0.2, 0.6, 0.2],[0.2, 0.2, 0.6], [0.4, 0.3, 0.3], [0.3, 0.4, 0.3], [0.3, 0.3, 0.4], [0.2, 0.4, 0.4], 
[0.4, 0.2, 0.4], [0.4, 0.4, 0.2], [0.1, 0.2, 0.7], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7], [0.2, 0.7, 0.1], [0.7, 0.2, 0.1], [0.7, 0.1, 0.2], [0.1, 0.3, 0.6], [0.1, 0.6, 0.3], 
[0.3, 0.1, 0.6], [0.3, 0.6, 0.1], [0.6, 0.3, 0.1], [0.6, 0.1, 0.3], [0.2, 0.3, 0.5], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5], [0.3, 0.5, 0.2], [0.5, 0.3, 0.2], [0.5, 0.2, 0.3], 
[0.1, 0.4, 0.5], [0.1, 0.5, 0.4], [0.4, 0.1, 0.5], [0.4, 0.5, 0.1], [0.5, 0.4, 0.1], [0.5, 0.1, 0.4], [0.1, 0.15, 0.75], [0.1, 0.85, 0.05], [0.8, 0.15, 0.05], [0.6, 0.25, 0.15],
[0.2, 0.55, 0.25], [0.2, 0.25, 0.55], [0.4, 0.35, 0.25], [0.3, 0.45, 0.25], [0.3, 0.35, 0.35], [0.2, 0.45, 0.35], [0.4, 0.25, 0.35], [0.4, 0.45, 0.15], [0.1, 0.25, 0.65], [0.1, 0.75, 0.15], 
[0.2, 0.15, 0.65], [0.2, 0.75, 0.05], [0.7, 0.25, 0.05], [0.7, 0.15, 0.15], [0.1, 0.35, 0.55], [0.1, 0.65, 0.25], [0.3, 0.15, 0.55], [0.3, 0.65, 0.05], [0.6, 0.35, 0.05], [0.6, 0.15, 0.25], 
[0.2, 0.35, 0.45], [0.2, 0.55, 0.25], [0.3, 0.25, 0.45], [0.3, 0.55, 0.15], [0.5, 0.25, 0.15], [0.5, 0.25, 0.25], [0.1, 0.45, 0.45], [0.1, 0.55, 0.35], [0.4, 0.15, 0.45], [0.4, 0.55, 0.05], 
[0.5, 0.45, 0.05], [0.5, 0.15, 0.35], [0.15, 0.1, 0.75], [0.15, 0.8, 0.05], [0.85, 0.1, 0.05], [0.65, 0.2, 0.15], [0.25, 0.5, 0.25], [0.25, 0.2, 0.55], [0.45, 0.3, 0.25], [0.35, 0.4, 0.25], 
[0.35, 0.3, 0.35], [0.25, 0.4, 0.35], [0.45, 0.2, 0.35], [0.45, 0.4, 0.15], [0.15, 0.2, 0.65], [0.15, 0.7, 0.15], [0.25, 0.1, 0.65], [0.25, 0.7, 0.05], [0.75, 0.2, 0.05], [0.75, 0.1, 0.15], 
[0.15, 0.3, 0.55], [0.15, 0.6, 0.25], [0.35, 0.1, 0.55], [0.35, 0.6, 0.05], [0.65, 0.3, 0.05], [0.65, 0.1, 0.25], [0.25, 0.3, 0.45], [0.25, 0.5, 0.25], [0.35, 0.2, 0.45], [0.35, 0.5, 0.15], 
[0.55, 0.2, 0.15], [0.55, 0.2, 0.25], [0.15, 0.4, 0.45], [0.15, 0.5, 0.35], [0.45, 0.1, 0.45], [0.45, 0.5, 0.05], [0.55, 0.4, 0.05], [0.55, 0.1, 0.35]]

def get_reward_weights(seed):
    return weights[seed]
    
def get_reward_weights_acc(seed, order):
    weights_set = seed+(order*10)
    return weights_for_acc[weights_set]
    
def generate_data(num_users_volte, num_users_video, num_users_urllc, num_tti):
    
    # define the seed so that results can be reproduced
    seed = 20
    rand_state = 20
    
    rand = np.random.RandomState(seed)
    
    traffic_types = ['volte', 'video', 'urllc']
    dist_list = ['uniform']
    param_list = ['0,160']

    traffic = {}
    traffic[traffic_types[0]] = {}
    traffic[traffic_types[1]] = {}
    traffic[traffic_types[2]] = {}


    traffic[traffic_types[0]]['num_users'] = 2
    traffic[traffic_types[1]]['num_users'] = 2
    traffic[traffic_types[2]]['num_users'] = 2
    
    # sample traffic data - can be exchanged with real data or data generated from mathematical models 
    arrival_times_initial1 = np.array([0, 7,8,9, 10, 10, 10, 10, 11, 12, 13, 14, 15, 15, 16, 16, 17, 18, 18, 19, \
                                       20, 27,28,29, 30, 30, 30, 30, 31, 32, 33, 34, 35, 35, 36, 36, 37, 38, 38, 39
                                      , 45, 46, 46, 46, 47, 48, 48, 49, 50, 51, 51, 52, 52, 52, 54, 55, 56, 57, \
                                       58, 59\
                                      , 65, 66, 66, 66, 67, 68, 68, 69, 70, 71, 71, 72, 72, 72, 74, 75, 76, 77, \
                                       78, 79])
    
    packet_sizes_initial1 = np.array([3, 5, 3, 5, 4, 3, 2, 6, 5, 5, 5, 6, 4, 4, 4, 4, 3, 6, 7, 4, \
                                      3, 5, 3, 5, 4, 3, 2, 6, 5, 5, 5, 6, 4, 4, 4, 4, 3, 6, 7, 4, \
                                      5, 6, 6, 6, 7, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 5, 6, 3, 3, 3, \
                                      5, 6, 6, 6, 7, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 5, 6, 3, 3, 3])*2

    arrival_times_initial2 = np.array([0, 1, 1, 2, 2, 3, 4, 4, 5, 6, 7, 8, 8, 9, 14, 14, 15, 16, \
                                       20, 21, 21, 22, 22, 23, 24, 24, 25, 26, 27, 28, 28, 29, 34, 34, 35, 36, \
                                       40, 40, 40, 42, 45, 47, 49, 50, 50 ,52, 55, 58, 58, 58, \
                                       60, 60, 60, 62, 65, 67, 69, 70, 70 ,72, 75, 78, 78, 78])
    
    packet_sizes_initial2 = np.array([3, 3, 3, 2, 2, 3, 4, 4, 5, 6, 7, 4, 4, 4, 4, 4, 5, 6, \
                                      3, 3, 3, 2, 2, 3, 4, 4, 5, 6, 7, 4, 4, 4, 4, 4, 5, 6, \
                                      5, 6, 5, 5, 5, 7, 2, 3, 3 ,3, 5, 8, 8, 8,\
                                      5, 6, 5, 5, 5, 7, 2, 3, 3 ,3, 5, 8, 8, 8])*4
    


    arrival_times_initial3 = np.array([0, 4, 6, 7, 11, 12, 14, 17, 18, \
                                       20, 21, 24, \
                                       40, 42, 45, 47, 49, 50, 55, 58, \
                                       60, 65, 67, 69, 70, 72, 75, 78])

    packet_sizes_initial3 = np.array([3, 3, 3, 2, 2, 3, 4, 4, 5,  \
                                      3, 3, 3,  \
                                      5, 6, 5, 5, 5, 7, 2, 3,\
                                      5, 6, 5, 5, 5, 7, 2, 3])*15
    
    
    arrival_times1 = arrival_times_initial1.copy()
    packet_sizes1 = packet_sizes_initial1.copy()
    for i in range(199):
        l1 = arrival_times_initial1 + (80*(i+1))
        arrival_times1 = np.append(arrival_times1, l1)

        l2 = packet_sizes_initial1
        packet_sizes1 = np.append(packet_sizes1, l2)
        
        
    arrival_times2 = arrival_times_initial2.copy()
    packet_sizes2 = packet_sizes_initial2.copy()
    for i in range(199):
        l1 = arrival_times_initial2 + (80*(i+1))
        arrival_times2 = np.append(arrival_times2, l1)

        l2 = packet_sizes_initial2
        packet_sizes2 = np.append(packet_sizes2, l2)
        
    arrival_times3 = arrival_times_initial3.copy()
    packet_sizes3 = packet_sizes_initial3.copy()
    for i in range(199):
        l1 = arrival_times_initial3 + (80*(i+1))
        arrival_times3 = np.append(arrival_times3, l1)

        l2 = packet_sizes_initial3
        packet_sizes3= np.append(packet_sizes3, l2)

    # compile the generated traffic in one dictionary for easy access
    for traffic_type in traffic_types:
        traffic[traffic_type]['users'] = {}
        # randomly generated with a certain seed if possible
        for user in range (1, traffic[traffic_type]['num_users']+1):
            traffic[traffic_type]['users'][user] = {}
            if traffic_type == 'volte':
                traffic[traffic_type]['users'][user]['packet_size'] = packet_sizes1
                traffic[traffic_type]['users'][user]['actual_arrival_time'] = arrival_times1
            
            elif traffic_type == 'video':
                traffic[traffic_type]['users'][user]['packet_size'] = packet_sizes2
                traffic[traffic_type]['users'][user]['actual_arrival_time'] = arrival_times2
                
            elif traffic_type == 'urllc':
                traffic[traffic_type]['users'][user]['packet_size'] = packet_sizes3
                traffic[traffic_type]['users'][user]['actual_arrival_time'] = arrival_times3    
    
    
    count = 0
    for traffic_type in traffic.keys():

        for user in traffic[traffic_type]['users'].keys():

            l1 = [traffic_type] * len(traffic[traffic_type]['users'][user]['actual_arrival_time'].tolist()) 
            l2 = [user] * len(traffic[traffic_type]['users'][user]['actual_arrival_time'].tolist()) 
            l3 = traffic[traffic_type]['users'][user]['actual_arrival_time'].tolist()
            l4 = traffic[traffic_type]['users'][user]['packet_size'].tolist() 

            data_tuples = list(zip(l1,l2,l3,l4))
            if count == 0:
                traffic_df  = pd.DataFrame(data_tuples, columns=['type', 'user', 'arrival_times', 'packet_sizes'])
            else:
                temp_df = pd.DataFrame(data_tuples, columns=['type', 'user', 'arrival_times', 'packet_sizes'])
                traffic_df = pd.concat([traffic_df,temp_df])
            count+=1
    traffic_df_final = traffic_df[(traffic_df['arrival_times']<=400000)].sort_values(['arrival_times', 'packet_sizes'])

    return traffic_df_final
