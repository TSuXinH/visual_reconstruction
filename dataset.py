from torchvision import transforms
import h5py, os ,torch
import numpy as np
import tifffile as tif
import cv2
import pickle
from tqdm import tqdm
from scipy import io as scio


class CalDataset(torch.utils.data.Dataset):
    
    def __init__(self, path, train=True, offsets=[-5,6], padding=False,target='single_frame'):
        """
        for offset it means [lower, upper)
        """
        self.path = path   # path should be root dir of train datasets
        if train == True:   
            file_name='train'
        else: 
            file_name = 'test'
        self.offsets=offsets
        self.tif_imgs = tif.imread(os.path.join(self.path,"%s_target.tif"%file_name,))
        self.tif_imgs = self.tif_imgs[:,45:45+240,20:20+320]
        self.data = h5py.File(os.path.join(self.path,"%s_C.mat"%file_name,),'r')
        self.data = np.asarray(self.data['valid_C_%s'%file_name]).astype(np.float32)
        self.data_size,self.data_dim=self.data.shape
        _,self.tif_hight,self.tif_width = self.tif_imgs.shape
        self.padding=padding
        self.target=target
        
    def __getitem__(self, index):
        if self.padding == True:    
            lower_bound = index+self.offsets[0]
            upper_bound = index+self.offsets[1]
            if lower_bound<0:
                comp_null = np.zeros((-lower_bound, self.data_dim))
                data = self.data[:upper_bound]
                data = np.concatenate((comp_null,data),axis=0)

                comp_null_tif = np.zeros((-lower_bound,self.tif_hight,self.tif_width))
                tif_imgs = self.tif_imgs[:upper_bound]
                tif_imgs = np.concatenate((comp_null_tif,tif_imgs),axis=0)

            elif upper_bound>self.data_size:
                comp_null = np.zeros((upper_bound-self.data_size,self.data_dim))
                data = self.data[lower_bound:]
                data = np.concatenate((data,comp_null),axis=0)

                comp_null_tif = np.zeros((upper_bound-self.data_size,self.tif_hight,self.tif_width))
                tif_imgs = self.tif_imgs[lower_bound:]
                tif_imgs = np.concatenate((tif_imgs,comp_null_tif),axis=0)
            else:
                data = self.data[lower_bound:upper_bound]
                tif_imgs = self.tif_imgs[lower_bound:upper_bound]
        else:
            if self.offsets[0]<0:
                index = index-self.offsets[0]
            lower_bound = index+self.offsets[0]
            upper_bound = index+self.offsets[1]
            data = self.data[lower_bound:upper_bound]
            tif_imgs = self.tif_imgs[lower_bound:upper_bound]
        
        if self.target == 'single_frame':
            tif_imgs=self.tif_imgs[index:index+1,:,:]
        elif self.target == 'frames_with_bof':
            comp_null = np.zeros((1,self.data_dim))
            data = np.concatenate((comp_null,data),axis=0)
            comp_null_tif = np.zeros((1,self.tif_hight,self.tif_width))
            tif_imgs = np.concatenate((comp_null_tif,tif_imgs),axis=0)
        return data.astype(np.float32), tif_imgs.astype(np.float32)

    def __len__(self):
        if self.padding==True:
            return self.data_size
        else:
            return self.data_size - max(0,self.offsets[1]) + min(0,self.offsets[0])


def get_frames(cap):
    """Helper function to load video segments.

    Parameters
    ----------
    cap : :obj:`cv2.VideoCapture` object
    idxs : :obj:`array-like`
        frame indices into video

    Returns
    -------
    obj:`array-like`
        returned frames of shape shape (n_frames, y_pix, x_pix)

    """
    n_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = np.zeros((n_total_frames, 1, 256, 256), dtype='uint8')
    for i in tqdm(range(n_total_frames)):
        ret, frame = cap.read()
        frames[i, 0, :, :] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frames


class neural_ae_Dataset(torch.utils.data.Dataset):
    def __init__(self,ae_file,neural_file,avi_file, train=True,
                  offsets=[-5,6],reduce='single_frame',return_frames=False, transform_file=None,top_neurals=100):
        """
        for offset it means [lower, upper)
        """
        
        self.offsets=offsets
        self.return_frames=return_frames
        # load data
        self.target = np.load(ae_file) # (T,latent_num)
        # try:
        #     f = h5py.File(neural_file,'r')
        #     print('open with h5py')
        # except:
        #     f = scio.loadmat(neural_file)
        #     print('open with scio')
        # self.data = np.asarray(f[list(f.keys())[0]][()]).astype(np.float32)
        self.data = neural_file.T
        if transform_file:
            with open(transform_file, 'rb') as f:
                pls2 = pickle.load(f)
                self.data = pls2.transform(self.data)[:,:top_neurals]
        n_times,n_neurals = self.data.shape
        # self.data = self.data[:,:20000]
        video_cap = cv2.VideoCapture(avi_file)
        self.frames = get_frames(video_cap).astype(np.float32)/255
        # check size
        target_size,_ = self.target.shape
        data_size,_=self.data.shape
        # print(target_size, data_size)
        # assert data_size==target_size, 'data time sequence not matched'
        self.frames = self.frames[: data_size]
        self.target = self.target[: data_size]
 
        if train == True:   
            valid_idx=range(int(data_size*0.7))
        else: 
            valid_idx=range(int(data_size*0.7),data_size)
        self.data = self.data[valid_idx]
        self.target = self.target[valid_idx]
        self.frames = self.frames[valid_idx]
        self.data_size,self.data_dim = self.data.shape
        self.reduce=reduce

    def __getitem__(self, index):
        if self.offsets[0]<0:
            index = index-self.offsets[0]
        lower_bound = index+self.offsets[0]
        upper_bound = index+self.offsets[1]
        data = self.data[lower_bound:upper_bound]
        target = self.target[lower_bound:upper_bound]
        frames = self.frames[lower_bound:upper_bound] 
        if self.reduce=='single_frame':
            target = self.target[upper_bound-1:upper_bound]
            frames = self.frames[upper_bound-1:upper_bound]
        if self.return_frames:
            return data.astype(np.float32), target.astype(np.float32), frames.astype(np.float32)
        else:
            return data.astype(np.float32), target.astype(np.float32),

    def __len__(self):
        return self.data_size - max(0,self.offsets[1]) + min(0,self.offsets[0])
