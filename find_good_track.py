import torch
from torch.utils.data import DataLoader, random_split
from TrDataset import TrDataset
from DNN import DNN
import matplotlib.pyplot as plt
import numpy as np
import datetime
np.random.seed(29)
torch.manual_seed(29) 

trainset = TrDataset("train",using_sample=False)
testset = TrDataset("test",using_sample=False)
valset_len = int(len(trainset) / 8)
trainset, valset = random_split(trainset, (len(trainset)-valset_len, valset_len))

train_loader = DataLoader(dataset=trainset,batch_size=1010,shuffle=False)
test_loader = DataLoader(dataset=testset,batch_size=len(testset),shuffle=False)
val_loader = DataLoader(dataset=valset, batch_size=len(valset), shuffle=False)

num_hidden = 1
hidden_size = 10
num_epoch = 50
print("Num Hidden: "+str(num_hidden))
print("Hidden Size: "+str(hidden_size))
print("Num Epoch: "+str(num_epoch))


def find_good_track(data_loader):
    net.load_state_dict(torch.load("logs/dicts/epoch50_numh1_hsize10_2020-05-02-20-54-25_ct_tr2020-05-02-20-35-31"))
    for k in range(0,len(testset),101):
        with torch.no_grad():
            for i,(init_pos,t_pos) in enumerate(data_loader):
                init_pos = init_pos[k:k+101]
                t_pos = t_pos[k:k+101]
            pre_tpos = net(init_pos)
        fig = plt.figure()
        plt.subplot(3,1,1)
        plt.plot(pre_tpos[:, 0], pre_tpos[:, 1], color='red',linestyle='--')
        plt.plot(t_pos[:,0],t_pos[:,1], color='red',linestyle='-')
        
        plt.plot(pre_tpos[:, 2], pre_tpos[:, 3], color='blue',linestyle='--')
        plt.plot(t_pos[:,2],t_pos[:,3], color='blue',linestyle='-')
        
        plt.plot(pre_tpos[:, 4], pre_tpos[:, 5], color='green',linestyle='--')
        plt.plot(t_pos[:,4],t_pos[:,5], color='green',linestyle='-')

        plt.subplot(3,1,2)
        plt.plot(pre_tpos[:, 0], pre_tpos[:, 1], color='red',linestyle='--')
        plt.plot(pre_tpos[:, 2], pre_tpos[:, 3], color='blue',linestyle='--')
        plt.plot(pre_tpos[:, 4], pre_tpos[:, 5], color='green',linestyle='--')
        
        plt.subplot(3,1,3)
        plt.plot(t_pos[:,0],t_pos[:,1], color='red',linestyle='-')
        plt.plot(t_pos[:,2],t_pos[:,3], color='blue',linestyle='-')
        plt.plot(t_pos[:,4],t_pos[:,5], color='green',linestyle='-')

        fig_path = 'logs/figs/find_good_track/epoch170/' + str(int(k/101)) + '.png'
        fig.savefig(fig_path)
        plt.close()
    
net = DNN(num_hidden=num_hidden,hidden_size=hidden_size)
find_good_track(test_loader)





