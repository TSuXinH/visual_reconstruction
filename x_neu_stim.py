import h5py
import numpy as np
from scipy import io as scio
import matplotlib.pyplot as plt


def plot10(mat):
    idx_list = np.random.randint(0, len(mat), size=(10, ))
    for item in idx_list:
        print(item)
        plt.plot(mat[item])
    plt.show(block=True)


neu_path = r'./x_data/neurons.mat'
data = scio.loadmat(neu_path)  # this can not be open using `h5py.File`
print(data.keys())

# convert size to [neuron number, time]
c_mat = data['C_mat'].T
f_dff_mat = data['F_dff_mat'].T
r_mat = data['R_mat'].T
s_mat = data['S_mat'].T
yra_mat = data['YrA_mat'].T

stim_start_point = [134, 1515, 2896, 4278, 5659, 7040]
stim_end_point = [1138, 2519, 3900, 5282, 6663, 8044]

for idx in range(6):
    print(stim_end_point[idx] - stim_start_point[idx])

plot10(c_mat)
plot10(f_dff_mat)
plot10(r_mat)
plot10(yra_mat)

stim_point_arr = np.array([stim_start_point, stim_end_point]).T
f_list = []
for idx in range(len(stim_point_arr)):
    f_list.append(f_dff_mat[:, stim_point_arr[idx, 0]: stim_point_arr[idx, 1]])
f_tensor = np.concatenate(f_list, axis=-1)

