import torch
import torch.nn as nn
import torch.nn.functional as F

class Graph(nn.Module):
    def __init__(self):
        super(Graph, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        #self.conv2 = nn.Conv2d(40, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        #x = torch.cat([x, x], 1)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def create_model():
    return Graph()

def create_dummy_input():
    return torch.rand((64, 1, 28, 28))

# could also register data loader and training approach here
# ... ...