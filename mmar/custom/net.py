import torch
import torch.nn as nn
from pylon.pylon import PylonConfig
from pylon.utils.pretrain import *
from model_constants import classes

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        net_conf = PylonConfig(
            n_in=1,
            n_out=classes,
            up_type='2layer',
            # pretrain_conf=PretrainConfig(
            #     pretrain_name='nih',
            #     path='/content/pylon/pylon,nih,256.pkl',
            # ),
            # freeze='enc',
        )
        self.image_model = net_conf.make_model()
    
    def forward(self,image):
        image_output = self.image_model(image)
        return image_output