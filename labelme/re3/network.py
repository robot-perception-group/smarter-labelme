# @article{gordon2018re3,
#   title={Re3: Real-Time Recurrent Regression Networks for Visual Tracking of Generic Objects},
#   author={Gordon, Daniel and Farhadi, Ali and Fox, Dieter},
#   journal={IEEE Robotics and Automation Letters},
#   volume={3},
#   number={2},
#   pages={788--795},
#   year={2018},
#   publisher={IEEE}
# }
import torch
import torch.nn as nn

class Re3Net(nn.Module):
    def __init__(self):
        super(Re3Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.LocalResponseNorm(2, alpha=2e-5, beta=0.75)
        )
        self.conv1_skip = nn.Sequential(
            nn.Conv2d(96, 16, 1),
            nn.PReLU(num_parameters=16)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, stride=1, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2),
            nn.LocalResponseNorm(2, alpha=2e-5, beta=0.75)
        )
        self.conv2_skip = nn.Sequential(
            nn.Conv2d(256, 32, 1),
            nn.PReLU(num_parameters=32)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, stride=1, padding=1, groups=2),
            nn.ReLU(inplace=True)
        )
        self.conv5_1 = nn.Sequential(
            nn.Conv2d(384, 256, 3, stride=1, padding=1, groups=2),
            nn.ReLU(inplace=True)
        )
        self.conv5_2 = nn.MaxPool2d(3,2)
        self.conv5_skip = nn.Sequential(
            nn.Conv2d(256, 64, 1),
            nn.PReLU(num_parameters=64)
        )
        self.fc6 = nn.Sequential(
            nn.Linear(74208, 1024),
            nn.ReLU(inplace=True)
        ) 
        self.lstm1 = nn.LSTM(input_size=1024, hidden_size=512, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=1536, hidden_size=512, batch_first=True)
        self.fc7 = nn.Linear(512,4)

    def forward(self, x, num_unrolls = 1, batch_size = None, prevLstmState = None):
        if batch_size is None:
            batch_size = int(x.shape[0]/num_unrolls/2)
        
        # Conv layers
        conv_reshaped = self._conv_layers(x, num_unrolls, batch_size)

        # FC 6
        fc6_out = self.fc6(conv_reshaped)

        # LSTM 1
        if(prevLstmState is None):
            lstm1_out, state1 = self.lstm1(fc6_out)
        else:
            lstm1_out, state1 = self.lstm1(fc6_out, prevLstmState[0])

        # LSTM 2
        lstm2_inputs = torch.cat((fc6_out, lstm1_out), 2)
        if(prevLstmState is None):
            lstm2_out, state2 = self.lstm2(lstm2_inputs)
        else:
            lstm2_out, state2 = self.lstm2(lstm2_inputs, prevLstmState[1])
        lstm_reshaped = lstm2_out.contiguous().view(num_unrolls*batch_size, 512)

        # FC 7
        out = self.fc7(lstm_reshaped)

        return out, (state1, state2)

    def _conv_layers(self, x, num_unrolls, batch_size):
        # Conv 1
        x1 = self.conv1(x)
        
        # Conv 1 skipped layer
        x1_skip = self.conv1_skip(x1)
        x1_skip_flat = x1_skip.view(x1_skip.shape[0], 16*27*27)
        
        # Conv 2
        x2 = self.conv2(x1)
        
        # Conv 2 skipped layer
        x2_skip = self.conv2_skip(x2)
        x2_skip_flat = x2_skip.view(x2_skip.shape[0], 32*13*13)
        
        # Conv 3
        x3 = self.conv3(x2)
        
        # Conv 4
        x4 = self.conv4(x3)
        
        # Conv 5
        x5_1 = self.conv5_1(x4)
        x5_2 = self.conv5_2(x5_1)        
        x5_flat = x5_2.view(x5_2.shape[0], 256*6*6)

        # Conv 5 skipped layer
        x5_skip = self.conv5_skip(x5_1)
        x5_skip_flat = x5_skip.view(x5_skip.shape[0], 64*13*13)
        
        # Concat all layers
        x_cat = torch.cat((x1_skip_flat, x2_skip_flat, x5_skip_flat, x5_flat), 1)
        x_reshape = x_cat.view(batch_size, num_unrolls, 2, x_cat.shape[-1])
        x_reshape = x_reshape.view(batch_size, num_unrolls, 2 * x_cat.shape[-1])
        
        return x_reshape

if __name__ == "__main__":
    net = Re3Net()
    pretrained_model = torch.load('caffenet_param.pt')
    model_dict = net.state_dict()
    model_dict.update(pretrained_model)
    net.load_state_dict(model_dict)
    for param_tensor in net.state_dict():
        print(param_tensor, "\t", net.state_dict()[param_tensor].size())
    torch.save(net.state_dict(),'pretrained_model.pth')
        
