import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.mlab as mlab

MAX_X_SIZE = 1e6

# function for counting particles in bins:
def sample_bins(x_min, x_max, values, n_boxes):
    bin_size = float(float(x_max - x_min) / float(n_boxes))
    last_bin = n_boxes - 1
    bins = np.zeros(n_boxes)
    bin_x = np.zeros(n_boxes)
    for value in values:
        index = np.floor((value - x_min) / bin_size).astype(int)
        if 0 <= index and index <= last_bin:
            bins[index] += 1
            bin_x[index] = x_min + bin_size
    return bins, bin_x
    # We take the decimal number of buckets from the start, and we round it down
    # to get the index.

    # if we have the range 0-12, and a bucket size of 4(three buckets:0,1,2), then 12 will fall
    # at bucket 3 index of bucket 2. To fix that, round just the maximal value back to the previous bucket.

def count(matrix_buckets):
    sum_of_square = 0
    for item in matrix_buckets:
         sum_of_square += (item ** 2)
    psi_x = matrix_buckets / (np.sqrt(sum_of_square))# gives the parameters of the wave function.
    # for receiving the probability we should take the square of the values in the matrix.
    return psi_x

def m(W_x):
    return np.minimum(np.trunc(W_x + np.random.uniform(0.0, 1.0, W_x.size)), 3).astype(int)

def V(x):
    return 0.5 * (x ** 2)

def W(V_x, e_r, dt):
    return np.exp(-(V_x - e_r) * dt)

def particle_locations(x, dt):
    sigma = np.sqrt(dt)
    row = np.random.randn(len(x))
    location = x + sigma*row
    return location

def energy(v_x, n_pre, n_0, dt):
    return np.average(v_x) + (1.0 - float(n_pre) / n_0) / dt
    # sum_e_r += e_r
    # avg_e_r = sum_e_r / time_interval
    # if (previous * 100 / avg_e_r) <= 5:
    # time_interval.append(time)

def run_dmc(dt, n_times, n_0, x_min, x_max, n_boxes, sample_from_iteration):
    x = np.zeros(n_0, dtype=np.float)
    V_x = V(x)
    e_r = np.average(V_x)
    e_rs = [e_r]
    bins = np.zeros(n_boxes)

    for i in range(n_times):
        print i
        # creates a vector of number with step dt. i gives the items in the list
        x = particle_locations(x, dt)
        V_x = V(x)
        W_x = W(V_x, e_r, dt)
        m_x = m(W_x)
        x = np.repeat(x, m_x)
        n_previous = len(x)
        # print('Round %d m_x: %s' % (i, np.mean(m_x)))
        e_r = energy(V_x, n_previous, n_0, dt)
        # previous_avg = np.average(e_rs)
        e_rs.append(e_r)
        if len(x) > MAX_X_SIZE:
            raise Exception('x is too big, aborting!')
        if i > sample_from_iteration:
            bins += sample_bins(x_min, x_max, x, n_boxes)

    avg_e_r = np.average(e_rs[sample_from_iteration:])
    standard_dev = np.std(e_rs[sample_from_iteration:])

    plt.plot(bins)
    # the histogram of the data
    # n, bins, patches = plt.hist(x_0, n_boxes,y, normed=1, facecolor='green', alpha=0.75)
    # plt.axis([x_min, x_max, 0, 1.0])
    plt.grid(True)
    plt.show()

    return standard_dev, avg_e_r



            # if (psy_s[i-51] * 100 / psy) <= 1 & (previous_avg * 100 /avg_e_r) <= 1  :


#animation of psy


    # # initial values: N_1=500, E_r=0.5, N_max=2000
# time_interval = []
# # initializing the matrix(==delta function) in zero
# for time in np.arange(0.0, t_0, dt,
#                       dtype=float):

#     if (previous * 100 / avg_e_r) <= 5:
#         time_interval.append(time)
#     else:
#         previous = avg_e_r



if __name__ == "__main__":
    # execute only if run as a script
    n_0 = 500
    x_min = -5.0
    x_max = 5.0
    n_boxes = 200
    dt = 0.1
    n_times = 2000
    sample_from_iteration = 50
    print run_dmc(dt = dt, n_times = n_times, n_0 = n_0, x_min = x_min, x_max = x_max, n_boxes = n_boxes, sample_from_iteration=sample_from_iteration)
