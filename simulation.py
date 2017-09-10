import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.mlab as mlab

MAX_X_SIZE = 1e6

# function for counting particles in bins:
# We take the decimal number of buckets from the start, and we round it down to get the index.

# if we have the range 0-12, and a bucket size of 4(three buckets:0,1,2), then 12 will fall
# at bucket 3 index of bucket 2. To fix that, round just the maximal value back to the previous bucket.
def sample_bins(x_min, x_max, values, n_boxes):
    bins, _ = np.histogram(values, bins=np.linspace(x_min, x_max, n_boxes + 1))
    return bins

def axis_x(x_min, x_max, n_boxes):
    a = np.arange(n_boxes)
    bin_size = float(x_max - x_min) / float(n_boxes)
    return (x_min + (bin_size * (a + 0.5)))


def wave_function(matrix_bins):
    sum_of_square = 0
    bin_size = float(x_max - x_min) / float(n_boxes)
    for item in matrix_bins:
        sum_of_square += item ** 2
    psi_x = (matrix_bins).astype(float) / float(np.sqrt(bin_size*(sum_of_square)))# gives the parameters of the wave function.
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
    # if (previous * 100 / avg_e_r) <= 5:
    # time_interval.append(time)

def run_dmc(dt, n_times, n_0, x_min, x_max, n_boxes, sample_from_iteration):
    x = np.zeros(n_0, dtype=np.float)
    V_x = V(x)
    e_r = np.average(V_x)
    e_rs = [e_r]
    bins = np.zeros(n_boxes)
    psi = 0
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
            psi = wave_function(bins)
    bin_size = axis_x(x_min, x_max, n_boxes)
    avg_e_r = np.average(e_rs[sample_from_iteration:])
    standard_dev = np.std(e_rs[sample_from_iteration:])



    x = np.array(bin_size)
    y = np.array(psi)
    # yerr
    # xerr
    # First illustrate basic pyplot interface, using defaults where possible.
    # plt.figure()
    # plt.errorbar(x, y, xerr, yerr)
    plt.title("DMC Harmonic oscillator")
    z = (np.pi**(-0.25))*(np.exp(-(x**2)/2))
    plt.plot(x, y, color = 'green')
    plt.plot(x, z, color = 'blue')
    plt.show()

    return standard_dev, avg_e_r



            # if (psy_s[i-51] * 100 / psy) <= 1 & (previous_avg * 100 /avg_e_r) <= 1  :


#animation of psy


    # # initial values: N_1=500, E_r=0.5, N_max=2000
# time_interval = []
#     if (previous * 100 / avg_e_r) <= 5:
#         time_interval.append(time)



if __name__ == "__main__":
    # execute only if run as a script
    n_0 = 500
    x_min = -20.0
    x_max = 20.0
    n_boxes = 200
    dt = 0.1
    n_times = 2000
    sample_from_iteration = 50
    print run_dmc(dt, n_times, n_0, x_min, x_max, n_boxes, sample_from_iteration)
