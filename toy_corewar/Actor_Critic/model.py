import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import config

CFG = config.get_cfg()
CWCFG = CFG.settings.toy_corewar

torch.set_default_tensor_type('torch.FloatTensor')

class AC_Model(nn.Module):
    def __init__(self, middle_size, middle_layers):
        super(AC_Model, self).__init__()

        input_size = CWCFG.N_TARGETS * 2

        self.fc_s1 = nn.Linear(in_features=input_size, out_features=middle_size)
        self.fc_s2 = nn.Linear(in_features=middle_size, out_features=middle_size)

        self.fc_actor = nn.Linear(in_features=middle_size, out_features=CWCFG.NUM_ACTIONS)
        self.fc_critic = nn.Linear(in_features=middle_size, out_features=1)

        self.relu = nn.ReLU()

    def forward(self, state):

        # Process state vector in 2 FC layers
        s = self.relu(self.fc_s1(state.float()))
        s = self.relu(self.fc_s2(s.float()))

        action_scores = F.softmax(self.fc_actor(s), dim=-1)
        state_value = self.fc_critic(s)

        return action_scores, state_value
