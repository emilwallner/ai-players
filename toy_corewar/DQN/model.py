import torch
import torch.nn as nn
import torch.optim as optim
import config

CFG = config.get_cfg().settings.toy_corewar

torch.set_default_tensor_type('torch.FloatTensor')

class Dueling_DQN(nn.Module):
    def __init__(self, middle_size, middle_layers):
        super(Dueling_DQN, self).__init__()

        input_size = CFG.N_TARGETS * 2

        self.fc_input = nn.Linear(in_features=input_size, out_features=middle_size)

        self.fc_list = nn.ModuleList([nn.Linear(middle_size, middle_size) for _ in range(middle_layers)])

        self.fc_adv = nn.Linear(in_features=middle_size, out_features=CFG.NUM_ACTIONS)
        self.fc_val = nn.Linear(in_features=middle_size, out_features=1)
        
        self.relu = nn.ReLU()
    
    def forward(self, state):

        # Process state vector in 2 FC layers
        x = self.relu(self.fc_input(state.float()))

        for fc in self.fc_list:
            x = self.relu(fc(x))
        
        # Split processing in 2 streams: value and advantage
        adv = self.fc_adv(x)
        val = self.fc_val(x).expand(-1, CFG.NUM_ACTIONS)
        
        x = val + adv - adv.mean().expand(CFG.NUM_ACTIONS)
        
        return x
