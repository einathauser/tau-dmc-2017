import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.mlab as mlab
# # function for counting particles in buckets:

# we get a vector where the valueof x is the index of box it belongs to
# These are the values of the particles in the matrix. to use the function i should write:
# bucketing(item) for item in x_0.(values=x_0)
def bucketing(x_min, x_max, values, n_boxes):
    bucket_size = float(float(x_max - x_min) / float(n_boxes))
    last_bucket = n_boxes - 1
    matrix_buckets = np.zeros(n_boxes)  # [0] * n_boxes
    for value in values:
        index = int((value - x_min) / bucket_size)
        if index > last_bucket:
           index = last_bucket
        matrix_buckets[index] += 1
    return matrix_buckets
    # We take the decimal number of buckets from the start, and we round it down
    # to get the bucket index (Starting from 0)

    # if we have the range 0-12, and a bucket size of 4(three buckets:0,1,2), then 12 will fall
    # at bucket 3 index of bucket 2. To fix that, round just the maximal value back to the previous bucket.
def matrix_location(x_min, x_max, n_boxes, values):
    bucket_size = float(float(x_max - x_min) / float(n_boxes))
    last_bucket = n_boxes - 1
    indices = []
    for value in values:
        index = int((value - x_min) / bucket_size)
        if index > last_bucket:
           index = last_bucket
        else:
            index = index
        indices.append(index)
    return indices


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

def run_dmc(dt, n_times, n_0, x_min, x_max, n_boxes):
    x = np.zeros(n_0, dtype=np.float)
    V_x = V(x)
    e_r = np.average(V_x)
    e_rs = [e_r]
    new_psy = 0
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
        previous_avg = np.average(e_rs)
        e_rs.append(e_r)
        psy_s = []
        buckets = []
        if len(x) > 30 * 1000 * 1000:
            raise Exception('Basa')
        if i >= 50 & i < n_times:
            avg_e_r = np.average(e_rs[49:])
            standard_dev = np.std(e_rs[49:])
            num_buckets = bucketing(x_min, x_max, x, n_boxes)
            buckets += matrix_location(x_min, x_max, x, n_boxes)
            psy = new_psy
            new_psy = count(num_buckets)
    mu, sigma = ((x_max + x_min)/2), 5)
    x_0 = mu + sigma * np.random.randn(n_boxes)

    # the histogram of the data
    n, bins, patches = plt.hist(x_0, n_boxes, normed=1, facecolor='green', alpha=0.75)
    plt.axis([x_min, x_max, 0, 1.0])
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



# if __name__ == "__main__":
    # execute only if run as a script
    # count(np.random.uniform(0, 1999, 10000), 2000)
    n_0 = 500
    x_min = -2.0
    x_max = 2.0
    n_boxes = 200
    dt = 0.1
    n_times = 2000
print run_dmc(dt = 0.1, n_times = 2000, n_0 = 500, x_min = -2.0, x_max = 2.0, n_boxes = 200)
