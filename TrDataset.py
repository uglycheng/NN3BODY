import numpy as np 
from torch.utils.data import Dataset

class TrDataset(Dataset):
    def __init__(self,data,using_sample=True):
        if data == "train":
            if using_sample:
                self.data = np.load('./datasets/trainset_sample.npy')
            else:
                self.data = np.load('./datasets/trainset.npy')
        elif data == "test":
            if using_sample:
                self.data = np.load('./datasets/testset_sample.npy')
            else:
                self.data = np.load('./datasets/testset.npy')
        else:
            print("required argument for data is train or test")
    
    def __getitem__(self,index):
        return self.data[index][:7], self.data[index][7:]
    
    def __len__(self):
        return self.data.shape[0]