import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.mlab as mlab

MAX_X_SIZE = 1e6

def sample_bins(x_min, x_max, x, n_boxes):
    distance = np.array(np.sqrt(x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2)).astype(float)
    bins, _ = np.histogram(distance, bins=np.linspace(x_min, x_max, n_boxes + 1))
    return bins

def axis_x(x_min, x_max, n_boxes):
    a = np.arange(n_boxes)
    bin_size = float(x_max - x_min) / float(n_boxes)
    return (x_min + (bin_size * (a + 0.5)))


def wave_function(matrix_bins):
    sum_of_square = 0
    for item in matrix_bins:
        sum_of_square += item ** 2
    psi_x = (matrix_bins).astype(float) / float(np.sqrt(sum_of_square))# gives the parameters of the wave function.
    # for receiving the probability we should take the square of the values in the matrix.
    return psi_x

def m(W_x):
    return np.minimum(np.trunc(W_x + np.random.uniform(0.0, 1.0, W_x.size)), 3).astype(int)

def V(e, x):
    distance = np.array(np.sqrt(x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2)).astype(float)
    return ((-e**2) / distance) # returns the third column, with is r.

def W(V_x, e_r, dt):
    return np.exp(-(V_x - e_r) * dt)

def particle_locations(x, dt):
    sigma = np.sqrt(dt)
    row = np.random.randn(len(x), 3)
    location = x + sigma*row
    return location

def energy(v_x, n_pre, n_0, dt):
    return np.average(v_x) + (1.0 - float(n_pre) / n_0) / dt
    # if (previous * 100 / avg_e_r) <= 5:
    # time_interval.append(time)

def run_dmc(dt, n_times, n_0, x_min, x_max, n_boxes, sample_from_iteration, e):
    x = np.array([[0, 0, 1]]*n_0).astype(float)
    V_x = V(e, x)
    e_r = np.average(V_x)
    e_rs = [e_r]
    bins = np.zeros(n_boxes)
    psi = 0
    for i in range(n_times):
        # creates a vector of number with step dt. i gives the items in the list
        x = particle_locations(x, dt)
        V_x = V(e,x)
        W_x = W(V_x, e_r, dt)
        m_x = m(W_x)
        x = np.repeat(x, m_x, axis=0)
        n_previous = len(x)
        # print('Round %d m_x: %s' % (i, np.mean(m_x)))
        e_r = energy(V_x, n_previous, n_0, dt)
        print e_r
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
    plt.title("DMC Hydrogen atom")
   # z = 2*np.exp(-x)
    r = x * np.array(psi)
    plt.plot(x, y, color = 'green')
    # plt.plot(e_rs)
    # plt.plot(x, z, color = 'blue')
    plt.plot(x, r, color = 'red')
    plt.show()

    return standard_dev, avg_e_r

if __name__ == "__main__":
    # execute only if run as a script
    n_0 = 5000
    x_min = 0.0
    x_max = 5.0
    n_boxes = 2000
    dt = 0.1
    n_times = 2000
    sample_from_iteration = 100
    e = 1
    print run_dmc(dt, n_times, n_0, x_min, x_max, n_boxes, sample_from_iteration, e)