import torch.nn as nn

class DNN(nn.Module):
    def __init__(self,num_hidden=100,hidden_size=128):
        super(DNN,self).__init__()
        self.input_layer = nn.Linear(7,hidden_size)
        self.hiddens = []
        for i in range(num_hidden):
            self.hiddens.append(nn.Linear(hidden_size,hidden_size))
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size,6)
        
    def forward(self, x):
        temp = self.input_layer(x)
        for i in range(len(self.hiddens)):
            temp = self.relu(temp)
            temp = self.hiddens[i](temp)
        temp = self.relu(temp)
        result = self.output(temp)
        return result

if __name__ == '__main__':
    import torch
    input_ = torch.randn(3,7)
    print(input_)
    model = DNN(hidden_size=10,num_hidden=2)
    result = model(input_)
    print(result)