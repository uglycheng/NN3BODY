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

time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
net = DNN(num_hidden=num_hidden,hidden_size=hidden_size)
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(net.parameters(),lr=0.00001)


def evaluate_mse(criterion, data_loader):
    net.eval()
    with torch.no_grad():
        for i,(init_pos,t_pos) in enumerate(data_loader):
            # print(init_pos[0])
            pre_tpos = net(init_pos)
            mse = criterion(pre_tpos,t_pos)
    return mse

def train(criterion, optimizer, train_loader, val_loader,ct_tr = False):
    if ct_tr:
        net.load_state_dict(torch.load("logs/dicts/epoch10_numh1_hsize10_2020-05-02-20-35-31_ct_tr2020-05-02-20-16-16"))
    for epoch in range(num_epoch):
        net.train()
        for i,(init_pos,t_pos) in enumerate(train_loader):
            optimizer.zero_grad()
            pre_tpos = net(init_pos)
            loss = criterion(pre_tpos,t_pos)
            loss.backward()
            optimizer.step()
            #print(str(i)+" LOSS: "+str(loss.item()))
        mse = evaluate_mse(criterion,val_loader)
        print("Val MSE: " + str(mse.item()))
        tr_mse_log.write("Val MSE: " + str(mse.item()) + '\n')
        # print(" ")
    path = 'logs/dicts/epoch{0}_numh{1}_hsize{2}_{3}'.format(num_epoch, num_hidden, hidden_size, time)
    if ct_tr:
        path += "_ct_tr2020-05-02-20-35-31"
    torch.save(net.state_dict(), path)
    net.load_state_dict(torch.load(path))

def get_track(data_loader,ct_tr = False):
    with torch.no_grad():
        for i,(init_pos,t_pos) in enumerate(data_loader):
            init_pos = init_pos[:101]
            t_pos = t_pos[:101]
        pre_tpos = net(init_pos)
    print(pre_tpos)
    tr_mse_log.write("Prediction Data for the Figure Plot: \n")
    tr_mse_log.write(str(pre_tpos))
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

    fig_path = 'logs/figs/epoch{0}_numh{1}_hsize{2}_{3}.png'.format(num_epoch, num_hidden, hidden_size, time)
    if ct_tr:
        fig_path = fig_path[:-4] + "_ct_tr2020-05-02-20-35-31.png" 
    fig.savefig(fig_path)
    plt.show()

def find_good_track(data_loader):
    net.load_state_dict(torch.load("logs/dicts/epoch10_numh1_hsize10_2020-05-02-20-35-31_ct_tr2020-05-02-20-16-16"))
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

        fig_path = 'logs/figs/find_good_track/epoch120/' + str(int(k/101)) + '.png'
        fig.savefig(fig_path)
        plt.close()
    
tr_mse_log = open('logs/tr_mse_logs/epoch{0}_numh{1}_hsize{2}_{3}.txt'.format(num_epoch, num_hidden, hidden_size, time),'w')
train(criterion,optimizer,train_loader,val_loader,ct_tr=True)
test_mse = evaluate_mse(criterion,test_loader)
print("Test MSE: " + str(test_mse.item()))
tr_mse_log.write("Test MSE: " + str(test_mse.item())+"\n")
#print(testset[0])
get_track(test_loader,ct_tr=True)
tr_mse_log.close()

# find_good_track(test_loader)





