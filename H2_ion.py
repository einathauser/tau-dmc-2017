import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.mlab as mlab

MAX_X_SIZE = 1e6

def sample_bins(x_min, x_max, x, n_boxes):
    bins_x, _ = np.histogram(x[:, 0], bins=np.linspace(x_min, x_max, n_boxes + 1))
    bins_y, _ = np.histogram(x[:, 1], bins=np.linspace(x_min, x_max, n_boxes + 1))
    bins_z, _ = np.histogram(x[:, 2], bins=np.linspace(x_min, x_max, n_boxes + 1))
    bins = [bins_x, bins_y, bins_z]
    return bins

def center_of_bins(x_min, x_max, n_boxes):
    a = np.arange(n_boxes)
    return (x_min +(float(x_max - x_min)/n_boxes)) * (a + 0.5)
# we can use this function for all axises: x,y,z.

def wave_function(matrix_bins, x_min, x_max, n_boxes):
    # matrix_bins- the matrix we get from the function "sample_bins"
    sum_of_square = np.zeros(3)
    bin_size = float(x_max - x_min)/n_boxes
    for item in matrix_bins:
        sum_of_square += item**2
        # we receive separate number for each axis.
    psi_x = (matrix_bins).astype(float) / float(np.sqrt(bin_size*(sum_of_square)))
    return psi_x

def m(W_x):
    return np.minimum(np.trunc(W_x + np.random.uniform(0.0, 1.0, W_x.size)), 3).astype(int)

def distance(vector):
    return np.sqrt(vector[:, 0]**2 + vector[:, 1]**2 + vector[:, 2]**2)

def V(e, x, r):
    r_vector = np.zeros(3, dtype=np.float)
    r_vector[2] = r
    distance_from_atom1 = distance(np.absolute(x+0.5*r_vector))
    distance_from_atom2 = distance(np.absolute(x-0.5*r_vector))
   # tot_distance = (distance_from_atom1+distance_from_atom2)/(distance_from_atom1*distance_from_atom2)
    return (-e**2)*((1 / distance_from_atom1) + (1 / distance_from_atom2))
# r_vector = (0, 0, R) = (0, 0, 2)
def W(V_x, e_r, dt):
    return np.exp(-(V_x - e_r) * dt)

def particle_locations(x, dt):
    sigma = np.sqrt(dt)
    row = np.random.randn(len(x), 3)
    location = x + sigma*row
    return location

def energy(v_x, n_pre, n_0, dt):
    return np.average(v_x) + (1.0 - float(n_pre) / n_0) / dt

def run_dmc(dt, n_times, n_0, x_min, x_max, n_boxes, sample_from_iteration, e, r):
    x = np.zeros((n_0, 3), dtype=np.float)
    electron_points = np.zeros((n_0, 3), dtype=np.float)
   # r_vector = np.array([[0.0, 0.0, r]]).astype(float)
    V_x = V(e, x, r)
    e_r = np.average(V_x)
    e_rs = [e_r]
    # bins = np.zeros(n_boxes)
    # psi = 0
    for i in range(n_times):
        x = particle_locations(x, dt)
        V_x = V(e, x, r)
        W_x = W(V_x, e_r, dt)
        m_x = m(W_x)
        x = np.repeat(x, m_x, axis=0)
        n_previous = len(x)
        # print('Round %d m_x: %s' % (i, np.mean(m_x)))
        e_r = energy(V_x, n_previous, n_0, dt)
        electron_points = np.concatenate((electron_points, x), axis=0)
        print e_r
        # previous_avg = np.average(e_rs)
        e_rs.append(e_r)
        if len(x) > MAX_X_SIZE:
            raise Exception('x is too big, aborting!')
     #   if i > sample_from_iteration:
           # bins += sample_bins(x_min, x_max, x, n_boxes)
           # psi = wave_function(bins)
   # bin_size = center_of_bins(x_min, x_max, n_boxes)
    avg_e_r = np.average(e_rs[sample_from_iteration:])
    standard_dev = np.std(e_rs[sample_from_iteration:])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = electron_points[:, 0]
    y = electron_points[:, 1]
    z = electron_points[:, 2]
    ax.scatter(x, y, z, c='r', marker='o')
   # plt.title("DMC H2_ion")
    #plt.plot(x, y, z)
    # plt.plot(e_rs)
    plt.show()

    return standard_dev, avg_e_r

if __name__ == "__main__":
    # execute only if run as a script
    n_0 = 500
    x_min = 0.0
    x_max = 5.0
    n_boxes = 200
    dt = 0.1
    n_times = 2000
    sample_from_iteration = 100
    e = 1.0
    r = 2.0
    print run_dmc(dt, n_times, n_0, x_min, x_max, n_boxes, sample_from_iteration, e, r)