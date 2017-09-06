import numpy as np
import matplotlib.pyplot as plt
import sys


# # function for counting particles in buckets:
def bucketing(x_min, x_max, x, n_boxes):
# we get a vector where the valueof x is the index of box it belongs to
#  These are the values of the particles in the matrix. to use the function i should write:
# bucketing(item) for item in x_0.(values=x_0)
    bucket_size = (x_max - x_min) / n_boxes
    last_bucket = n_boxes - 1
    matrix_buckets = np.zeros(last_bucket)
    for replica in x:
         index = int((replica - x_min) / bucket_size)
         if index > last_bucket:
             matrix_buckets[last_bucket] += 1
         else:
             matrix_buckets[index] += 1
    return matrix_buckets
    # We take the decimal number of buckets from the start, and we round it down
    # to get the bucket index (Starting from 0)

    # if we have the range 0-12, and a bucket size of 4(three buckets:0,1,2), then 12 will fall
    # at bucket 3 index of bucket 2. To fix that, round just the maximal value back to the previous bucket.


     def count(number_of_particles):
     # This function gives the "wave function"
     sum_of_square = 0
     for item in number_of_particles:
         sum_of_square += (item ** 2)
     psi_x = (number_of_particles % (np.sqrt(sum_of_square)) / 10
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
    return np.average(v_x) + (1 - (n_0 / n_pre)) / dt
    # sum_e_r += e_r
    # avg_e_r = sum_e_r / time_interval
    # if (previous * 100 / avg_e_r) <= 5:
    # time_interval.append(time)

def run_dmc(dt, n_times, n_0):
    x = np.zeros(n_0, dtype=np.float)
    V_x = V(x)
    e_r = np.average(V_x)
    e_rs = [e_r]
    for i in range(n_times):
        # creates a vector of number with step dt. i gives the items in the list
        x = particle_locations(x, dt)
        n_previous = len(x)
        V_x = V(x)
        W_x = W(V_x, e_r, dt)
        m_x = m(W_x)
        x = np.repeat(x, m_x)
        e_r = energy(V_x, n_previous, n_0, dt)
        previous_avg = np.average(e_rs)
        e_rs.append(e_r)
        psy_s = []
        if i >= 50:
            avg_e_r = np.average(e_rs)
            num_buckets = bucketing(x_min, x_max, x_0, n_boxes)
            psy = count(num_buckets)
            psy_s
            if (previous_avg * 100 / avg_e_r) <= 1 & :

         else:
    #         previous = avg_e_r
    # while time > time_interval[0]:
    #     buckets = bucketing(x_min, x_max, x_0, n_boxes)
    #     psy = count(buckets)
    #     standard_dev = np.std(e_r)



    plt.plot(e_rs)
    plt.show()



    # # initial values: N_1=500, E_r=0.5, N_max=2000
# time_interval = []
# previous = 0
# sum_e_r = 0
# # initializing the matrix(==delta function) in zero
# for time in np.arange(0.0, t_0, dt,
#                       dtype=float):
#     row = np.random.randn(n_1)
#     x_0 += (dt ** 0.5) * row  # gives the new value=position of every particle after dt.
#     V_x = 0.5 * (x_0 ** 2)
#     W = np.exp ** (-(V_x - e_r) * dt)
#     N_0 = n_1  # we alway save the number of particles from the former stage for the energy.
#     # it means that Nj-1=Nj and on the next stage we will examine Nj+1 so it's important to maintain Nj.
#     u = np.random.random((n_1,))
#     for j in n_1:
#         m_n = np.min([int(W[j] + u[j]), 3])  # finds the new number for each column.
#         # if m_n == 1 we don't need to do anything and therefore there is no iterable for it.
#  n_0 = len(x)
#         if m_n[j] == 0:
#             n_1 -= 1
#             x_0[1][j] = x_0[0][j]  # deletes particle j.
#             # command for deleting items in a matrix:np.delete(x_0, 0, j)
#         elif m_n == 2:
#             n_1 += 1
#             x_0.append(x_0[1][j])
#         # the new particle receives the value of the particle it was created from.
#         # - it starts in the same position
#         elif m_n == 3:
#             n_1 += 2
#             x_0.append(x_0[1][j])
#             x_0.append(x_0[1][j])
#         if n_1 > n_max:
#             n_1 = n_max
#             x_0 = x_0[:n_max + 1]
#   e_r = V_x + (1 - (n_1 / N_0)) / dt
#     sum_e_r += e_r
#     avg_e_r = sum_e_r / time_interval
#     if (previous * 100 / avg_e_r) <= 5:
#         time_interval.append(time)
#     else:
#         previous = avg_e_r
# while time > time_interval[0]:
#     buckets = bucketing(x_min, x_max, x_0, n_boxes)
#     psy = count(buckets)
#     standard_dev = np.std(e_r)
#
#
#     # finding wave functions and energies


if __name__ == "__main__":
    # execute only if run as a script
    # count(np.random.uniform(0, 1999, 10000), 2000)
    n_0 = 1000
    dt = 0.1
    n_times = 200
    run_dmc(dt, n_times, n_0)