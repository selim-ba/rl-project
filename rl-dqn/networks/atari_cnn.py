# networks/atari_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# Q-Network (based on the Nature paper)
class QNetwork(nn.Module):
    """
        CNN du Deep Q-Network décrit dans https://www.nature.com/articles/nature14236
        Input : preprocessed image of size 84x84
        Output : Q(s,a)
    """
    def __init__(self,input_channels: int, num_actions: int):
        super(QNetwork,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels,out_channels=32,kernel_size=8,stride=4)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=4,stride=2)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1)
        
        self.fcn1 = nn.Linear(in_features=7*7*64,out_features=512) # 7*7*64 = 3136
        self.fcn2 = nn.Linear(in_features=512,out_features=num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept uint8 or float in [0,255]; scale to [0,1] once here
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)
        elif x.dtype.is_floating_point and x.max() > 1:
            x = x.div(255.0)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fcn1(x))
        x = self.fcn2(x) # <-- q-values Q(s,a)
        return x