import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.mlab as mlab

MAX_X_SIZE = 1e6

def sample_bins(x_max, x, n_boxes):
    r_max = np.sqrt(float(3*x_max**2))
    radius = np.array(np.sqrt(x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2)).astype(float)
    bins, _ = np.histogram(radius, bins=np.linspace(0.0, r_max, n_boxes + 1))
    return bins

def wave_function(density, x_max, n_boxes, bin_size):
    sum_of_square = 0
    delta_r = np.sqrt(3*x_max**2).astype(float)/n_boxes
    new_matrix = (density**2)
    for item in new_matrix:
        sum_of_square += item
    psi_x = density.astype(float) / ((bin_size)*float(np.sqrt(delta_r*(sum_of_square))))
    return psi_x

def m(W_x):
    return np.minimum(np.trunc(W_x + np.random.uniform(0.0, 1.0, W_x.size)), 3).astype(int)

def V(e, x):
    distance = np.array(np.sqrt(x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2)).astype(float)
    return ((-e**2) / distance)

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
    a = np.arange(n_boxes)
    bin_size = (np.sqrt(3*x_max**2).astype(float)/n_boxes)*(a + 0.5)
    matrix_of_volumes = (0+(4*np.pi*(float(np.sqrt(3*(x_max**2)))**3) / (3*n_boxes**3)))*(a + 0.5)
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
            bins += sample_bins(x_max, x, n_boxes)
            density = bins/matrix_of_volumes
            psi = wave_function(density, x_max, n_boxes, bin_size)
    avg_e_r = np.average(e_rs[sample_from_iteration:])
    standard_dev = np.std(e_rs[sample_from_iteration:])

    r_psi = psi*bin_size
    plt.xlim(0.0, 7.5)
    plt.ylim(0.0, 2.2)
    plt.title("DMC Hydrogen atom")
    psi_analytic = 2*np.exp(-bin_size)
    r_psi_analytic = bin_size*psi_analytic
    plt.plot(bin_size, r_psi, color = 'green')
    # plt.plot(e_rs)
    plt.plot(bin_size, psi_analytic, color = 'blue')
    plt.plot(bin_size, psi, color = 'red')
    plt.plot(bin_size, r_psi_analytic, color='yellow')
    plt.show()

    return standard_dev, avg_e_r

if __name__ == "__main__":
    # execute only if run as a script
    n_0 = 5000
    x_min = 0.0
    x_max = 10.0
    # x_max=y_max=z_max. these are the values which indicate location that you can enter the matrix
    # the values can be different but their range can't because it's a sphere.
    n_boxes = 2000
    dt = 0.1
    n_times = 2000
    sample_from_iteration = 100
    e = 1
    print run_dmc(dt, n_times, n_0, x_min, x_max, n_boxes, sample_from_iteration, e)