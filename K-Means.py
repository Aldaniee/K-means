import math
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv
import random as rand






#
# --- SETTINGS ---
#
DATA_NAME = 'clusterFun_2_keep.csv'     # [2D] cluster01 [3D] clusterFun_2 [LABLED cluster]
UPPER_K_LIM = 8                 # must be < n
DISPLAY_DOT_PLOTS = True        # Set False to not show plots 
LABELED = True                  # Is the training set labeled?
VALIDATION_PERCENT = 20/100     # percent of data used for validation set
ERROR_TESTING = False           # set true for accurate non-determinstic error testing                     
#
# --- SETTINGS ---
#





# Debug
PRINT_INPUT = False             # Display input numbers
REPS = 100 
# Static Constants
COLORS = 'bgrcmy' # All colors for 2D plots
# All colors for 3D plots
cmaps =    ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
ACCURACY_CONSTANT = 20          # number of decimal places to round to (slower if higher but more accurate)

# in:  x - number of zeros
# out: arr - filled array with x zeros
def zero_init(x):
    arr = []
    for i in range(x):
        arr.append(0.0)
    return arr
# in: data - an array of data
# out bool - True if the array is a nested list
def is_nested(data):
    return not any(isinstance(sub, list) for sub in data)

# in:  name - name of data file
# out: data - collection of data
def aquire_data(name):
    data = []
    global DIM
    if(LABELED):
        global labels
        labels = []
    with open(name, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for entry in reader:
            if(LABELED):
                DIM = len(entry) - 1
                labels.append(int(entry[DIM]))
            else:
                DIM = len(entry)
            temp = []
            for d in range(DIM):
                temp.append(float(entry[d]))
            data.append(temp)
    if PRINT_INPUT:
        print(input_data)
        plot_iteration(input_data)
    return data

# in: data - data to plot
def plot_iteration(data):
    if DIM == 2:
        plt.figure(num = 'K-Means Clustering')
        plt.title("K = " + str(len(data)))
    elif DIM == 3:
        ax = plt.axes(projection='3d')
    for i, cluster in enumerate(data): # for each cluster
        dimens = []
        for d in range(DIM):
            dimens.append(np.zeros((len(cluster[1:]),), dtype=float))
        for j, point in enumerate(cluster[1:]): # iterate number of points minus 1 for centroid
            for d in range(DIM):
                dimens[d][j] = point[d]
        if DIM == 2:
            plt.plot(dimens[0], dimens[1], str(COLORS[i % 6] + 'o'))  # plot data in next available color
            plt.plot(cluster[0][0], cluster[0][1], 'k^', markersize = 8)     # plot the centroid in black
        elif DIM == 3:
            ax.scatter3D(dimens[0], dimens[1], dimens[2], c=dimens[2], marker='o', cmap=cmaps[i % 18])  # plot data in next available color
        else:
            print("More than three dimensions cannot be plotted")
    plt.ylabel('y')
    plt.xlabel('x')
    plt.show()

# in: x, y - data to plot
def plot_elbow(x, y):
    plt.figure(num = 'K-Means Clustering')
    plt.plot(x, y, 'o-')  # plot data in next available color
    plt.ylabel('Distance from clusters')
    plt.xlabel('K number of clusters')
    plt.title("Elbow Graph")
    plt.show()
# in:    p - list of points
# out:   total - integer euclidean distance between the points
def euclidean_distance(p1, p2):
    total = 0.0
    for d in range(DIM):
        total += ((p1[d] - p2[d]) ** 2)
    return math.sqrt(total)

# in:   p - a list of points or a single point
# out: sum - a point that is an average of all those points
def point_avg(p):
    if(is_nested(p)):
        return p
    sum = zero_init(DIM)
    for i in p:
        for j in range(DIM):
            sum[j] += i[j]
    for j in range(DIM):
        sum[j] /= len(p)
        sum[j] = round(sum[j], ACCURACY_CONSTANT)
    return sum
# in: data – a list of points to assign
#     d    - a list of euclidian distances
#     c    - a list of centroids to assign data to
def assign(data, d, c, k):
    while(len(data) > 0):
        if(is_nested(data)): #if data is not a nested list (has one entry)
            x = data
        else:
            x = data.pop(0);
        for i in range(k):
            d[i] = euclidean_distance(point_avg(c[i]), x)
        i = d.index(min(d))
        if(is_nested(c[i])): # if c1 is not a nested list (has one entry)
            temp = c[i]
            c[i] = []
            c[i].append(temp)
        c[i].append(x)

# in: data – a list of points to assign
#     d    - a list of euclidian distances
#     c    - a list of finished centroids to assign data to
def validate(data, d, c):
    while(len(data) > 0):
        if(is_nested(data)): #if data is not a nested list (has one entry)
            x = data
        else:
            x = data.pop(0);
        for i in range(k):
            if(is_nested(c[i])): #if c[i] is not a nested list (has one entry)
                d[i] = euclidean_distance(c[i], x)
            else:
                d[i] = euclidean_distance(c[i][0], x)
        i = d.index(min(d))
        if(is_nested(c[i])): # if c[i] is not a nested list (has one entry)
            temp = c[i]
            c[i] = []
            c[i].append(temp)
        c[i].append(x)
    print("Validation Data:")

def cluster_labeled_data(input_data):
    clusters = max(labels)
    c = []
    for i in range(clusters):
        c.append([])
    for j in range(len(input_data)):
        for i in range(clusters):
            if(labels[j] == i + 1):
                if(is_nested(c[i])): # if c1 is not a nested list (has one entry)
                    temp = c[i]
                    c[i] = []
                    c[i].append(temp)
                c[i].append(input_data[j])
    for i in range(clusters):
        del c[i][0]
    reps = 1
    r = 0
    if VALIDATION_PERCENT > 0 and ERROR_TESTING:
        reps = REPS
    correct_points = zero_init(reps + 1)
    while r < reps:
        c_g = cluster(input_data, clusters) # generate clusters with algorithm
        if DISPLAY_DOT_PLOTS and not ERROR_TESTING:
            print("Labeled Clustering for K =", str(clusters))
            plot_iteration(c)
            print("K-Means Clustering for K =", str(clusters))
            plot_iteration(c_g)
        total = zero_init(clusters)
        for k in range(clusters):
            temp_total = zero_init(clusters)
            for c1 in range(clusters):
                cn = (c1 + k) % clusters
                for point in c_g[cn]:
                    if point in c[c1]:
                        temp_total[cn] += 1
            for l in range(clusters):
                if total[l] < temp_total[l]:
                    total[l] = temp_total[l]
        r += 1
        correct_points[r] = (np.sum(total))
    if ERROR_TESTING:
        print(str(int(round(np.average(correct_points)))), "/", str(len(input_data)), "points clustered correctly")
        print("Error", str((1 - np.average(correct_points)/len(input_data)) * 100), "%")
    

def cluster(input_data, k):
    if DIM == 3:
        print("(computing K =", str(k), "...)")
    data = copy.copy(input_data)
    c = []
    for i in range(k):
        c.append(data.pop(0))
    run = True
    while run:
        assign(data, zero_init(len(c)), c, k)
        data = copy.copy(input_data)
        equal = True
        for i in c:
            if(i[0] != point_avg(i)):
                equal = False
        if equal:
            run = False
        else:
            for i in range(k):
                c[i] = point_avg(c[i])
    return c
    
# in: input_data - a list of points gathered from input
#              k - the number of clusters
#         k_vals - list of k_vals used for output
#      distances - list of distances for output
def iteration(input_data, k, k_vals, distances):
    c = cluster(input_data, k)
    distance_sum = 0.0
    for i in c:
        for j in i[1:]:
            distance_sum += euclidean_distance(i[0], j)
    distances.append(distance_sum)
    if DISPLAY_DOT_PLOTS:
        plot_iteration(c)
    return c

# main script

input_data = aquire_data(DATA_NAME)
data = copy.copy(input_data)
validation_data = []
for i in range(int(len(input_data) * VALIDATION_PERCENT)):
    index = rand.randrange(len(data))
    labels.pop(index)
    validation_data.append(data.pop(index)) 
k_vals = []
distances = []

for k in range(1, UPPER_K_LIM + 1):
    iteration(data, k, k_vals, distances)
    k_vals.append(k)
plot_elbow(k_vals, distances)
if LABELED:
    cluster_labeled_data(data)
