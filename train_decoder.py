# from numpy.core.defchararray import mod
import torch
import torch.nn as nn
import numpy as np
import os 
from dataset import neural_ae_Dataset
from models.model import FCN_ae
from metrics import SSIM,PSNR,multi_roi_loss
import cv2
from torch.utils.tensorboard import SummaryWriter
import lpips
from scipy import io as scio
from tqdm import tqdm

def normalize_img(img):
    return img.repeat(1,3,1,1)*2-1

def train(model,train_loader,test_loader,num_epochs,Result_DIR,device='cuda'):
    writer = SummaryWriter(Result_DIR)
    # Loss and optimizer
    criterion_MSE = nn.MSELoss()
    criterion_perc = lpips.LPIPS(net='alex').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  
    model.train()
    # Train the model
    total_step = len(train_loader)
    ae_model = get_ae_from_me().to(device)
    print('training')
    print('ae model training: {}'.format(ae_model.training))
    for epoch in range(num_epochs):
        for i, (src,tgt,img) in enumerate(train_loader):
            # Move tensors to the configured device
            src = src.to(device)
            tgt = tgt.to(device)
            img = img.to(device)
            
            output = model(src)
            img_recon = ae_model.decoding(output.squeeze(), None, None, dataset=None)
            loss_mse = criterion_MSE(output,tgt)
            loss_perc = criterion_perc(normalize_img(img_recon),normalize_img(img[:,0])).mean()*100
            loss = loss_mse + loss_perc
            
            # Backprpagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 10 == 0:
                writer.add_scalar('train_loss',loss.item(),epoch*total_step+i)
                writer.add_scalar('train_loss_mse',loss_mse.item(),epoch*total_step+i)
                writer.add_scalar('train_loss_perc',loss_perc.item(),epoch*total_step+i)
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, mse: {:.4f},perc:{:.4f}'
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item(),loss_mse.item(),loss_perc.item()))

            
        if (epoch+1) % 10== 0:
            # Save the model checkpoint
            torch.save(model.state_dict(), os.path.join(Result_DIR,'model_fcn%d.ckpt'%(epoch+1)))

        if (epoch+1) % 2 == 0:
            test_mse,test_perc,_ = test(model,test_loader,criterion_MSE,criterion_perc,device,ae_model=ae_model)
            writer.add_scalar('test_loss_mse',test_mse,epoch*total_step+i)
            writer.add_scalar('test_loss_perc',test_perc,epoch*total_step+i)
            print('test mse: {:.4f}, perc:{:.4f}'.format(test_mse,test_perc))   

class Loss_Acc():
    def __init__(self):
        self.accmulator = 0
        self.num = 0
    
    def update(self,value,n):
        self.accmulator+=value*n
        self.num +=n
    
    def read_avg(self):
        if self.num !=0:
            avg = self.accmulator/self.num
        else:
            avg = 0
        self.accmulator = 0
        self.num = 0
        return avg


def test(model,test_loader,criterion_mse,criterion_perc,device='cuda',load=False,checkpoint_file=None,ae_model=None):
    if load:
        # load model
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint)
    model.eval()
    mse_acc = Loss_Acc()
    perc_acc = Loss_Acc()
    with torch.no_grad():
        output_list = []
        print('testing')
        for i, (src,tgt,img) in tqdm(enumerate(test_loader)):
            n=src.shape[0]
            src = src.to(device)
            tgt = tgt.to(device)
            img = img.to(device)

            # Forward pass
            output = model(src)
            output_list.append(output.detach().cpu().squeeze().numpy())

            mse_acc.update(criterion_mse(output,tgt).item(),n) 

            # reconstruct image
            if ae_model:
                img_recon = ae_model.decoding(output.squeeze(), None, None, dataset=None)
                perc_acc.update(criterion_perc(normalize_img(img_recon),normalize_img(img[:,0])).mean()*100,n)
            
        mse_avg = mse_acc.read_avg()
        perc_acc = perc_acc.read_avg()
        outputs = np.concatenate(output_list,axis=0)
    model.train()
    return mse_avg, perc_acc, outputs


def get_ae_from_me():
    from models.aes import AE as Model
    import json
    with open('./models/ae_model_arch.json') as f:
        hparams = json.loads(f.read())
    model_ae = Model(hparams)
    model_ae.version = int(0)
    model_ae.load_state_dict(torch.load('./models/ae_model_param.pt', map_location=lambda storage, loc: storage))
    model_ae.to(hparams['device'])
    model_ae.eval()
    return model_ae


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
    for i in range(n_total_frames):
        ret, frame = cap.read()
        frames[i, 0, :, :] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frames



def train_model(neural_file,transform_file,Result_DIR,top_neurals):
    ae_file = './data/ae_data.npy'
    avi_file ="./data/Reaglined MRI video.avi"
    
    # Check Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print('device is:',device)

    # Define Hyper-parameters 
    output_size=12
    offsets=[0,1]

    # rois = [[1,'whole',whole_roi]]
    num_epochs = 100
    batch_size = 64

    train_dataset=neural_ae_Dataset(ae_file=ae_file,neural_file=neural_file,avi_file=avi_file, transform_file=transform_file,
                                     train=True, offsets=offsets,reduce='single_frame',return_frames=True,top_neurals=top_neurals)
    test_dataset=neural_ae_Dataset(ae_file=ae_file,neural_file=neural_file,avi_file=avi_file, transform_file=transform_file,
                                     train=False, offsets=offsets,reduce='single_frame',return_frames=True,top_neurals=top_neurals)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size, 
                                            shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                            batch_size=batch_size,shuffle=False)    
    print('data loader is constructed')
    # model = GRU_ae(train_dataset.data_dim,output_size,device=device)
    print('population vector dim is {}'.format(train_dataset.data_dim))
    model = FCN_ae(train_dataset.data_dim,output_size,offsets[1]-offsets[0]).to(device)

    train(model,train_loader,test_loader,num_epochs,Result_DIR,device)


def generate_videos(checkpoint_file,neural_file,out_avi_file):
    ae_file = './data/ae_data.npy'
    avi_file ="./data/Reaglined MRI video.avi"
    
    neural_file = neural_file
    test_dataset =neural_ae_Dataset(ae_file=ae_file,neural_file=neural_file,avi_file=avi_file,
                            train=False, offsets=[0,1],reduce='single_frame',return_frames=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                             batch_size=128, 
                                             shuffle=False)
    model = FCN_ae(test_dataset.data_dim,12,1).to('cuda')
    ae_model = get_ae_from_me()
    criterion_MSE = nn.MSELoss()
    checkpoint_file = checkpoint_file # 'G:\\BBNC\\xiaoguihua\\result\\self-exp\\model_rnn100.ckpt'
    loss,psnr, latents = test(model,test_loader,criterion_MSE,None,device='cuda',load=True,checkpoint_file=checkpoint_file)
    latents = torch.from_numpy(latents)
    ims_recon_neural = ae_model.decoding(latents.squeeze(), None, None, dataset=None)
    ims_recon_neural = (ims_recon_neural*255).detach().numpy().astype(np.uint8)
    

    video_cap = cv2.VideoCapture(avi_file)
    ims_orig = get_frames(video_cap)[-(test_dataset.__len__()):]
    T,C,H,W  =ims_orig.shape
    print(T, C, H, W)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(out_avi_file,fourcc,10,(W*2,H),0)
    combine_frame = np.concatenate([ims_orig,ims_recon_neural],axis=3)
    print(combine_frame.shape)
    print('generating video')
    for idx in tqdm(range(T)):
        video.write(combine_frame[idx,0])
    cv2.destroyAllWindows()
    video.release()
    

if __name__ == '__main__':
    from x_dataset import proc_neu_data, stim_point_arr
    neu_path = r'./x_data/neurons.mat'
    video_path = r'./x_data/frames_cropped'
    data = scio.loadmat(neu_path)  # this can not be opened using `h5py.File
    f_dff_mat = data['F_dff_mat'].T
    neu_data = f_dff_mat
    #
    # # neural_file = './x_data/neurons.mat'
    # transform_file = None
    # Result_DIR = r'E:\xuxh\menet_svr\alter'
    # for topk_neurals in [10,20,50,100,200]:  # 10,20,50,100,200,500,1000
    #     result_dir = Result_DIR+str(topk_neurals)
    #     train_model(neural_file,transform_file,result_dir,topk_neurals)
    checkpoint_model = r'./alter10/model_fcn10.ckpt'

    out_avi_file = r'./output_video/output.avi'
    generate_videos(checkpoint_model,neu_data,out_avi_file)
