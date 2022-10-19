from sklearn.cross_decomposition import PLSRegression
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pickle

neural_file = "./data/valid_C_Z.mat"
ae_file = './data/ae_data.npy'
pls_dim=6

# neural_data = h5py.File(neural_file,'r')
# neural_data = np.asarray(neural_data['valid_C_Z']).astype(np.float32) #valid_C_200

# ae_data = np.load(ae_file)

# pls2 = PLSRegression(n_components=pls_dim)
# pls2.fit(neural_data, ae_data)
# neural_pls = pls2.transform(neural_data)


f = h5py.File('./data/neural_pls100.mat','r') 
neural_pls = np.asarray(f[list(f.keys())[0]][()]).astype(np.float32)

# filename = 'test_pls.json'
# # with open(filename, 'wb') as f:
# #     pickle.dump(pls2,f)

# with open(filename,'rb') as f:
#     pls2 = pickle.load(f)
#     computed = pls2.transform(neural_data)

# with h5py.File(neural_file+'.mat', 'w', libver='latest', swmr=True) as f:
#     f.create_dataset('test_pls', data=neural_pls, dtype='float32')

print(neural_pls[1])
# plt.plot(computed[1])
plt.plot(neural_pls[1])
plt.show()

