import sigopt
from reward import *
import torch
import math


MAX_EPISODES = 50000


def search(reward_func, M):
    conn = sigopt.Connection(client_token='QWILQOQCAHZIVDIMLOLIQZGYCBTIKTVTEFQMJXSAVWQVBLTA')
    
    DQN_experiment = conn.experiments().create(
        name= reward_func.__name__ + 'Dueling_DQN',
        observation_budget=128,
        parameters=[
            dict(name='h_size', type='int', bounds=dict(min=20,max=100)),
            dict(name='middle_size', type='int', bounds=dict(min=30, max=250)),
            dict(name='lstm_layers', type='int', bounds=dict(min=1, max=2)),
            dict(name='epsilon_decay_steps', type='int', bounds=dict(min=M/10, max=M)),
            dict(name='learning_starts', type='int', bounds=dict(min=0, max=M/2)),
            dict(name='learning_freq', type='int', bounds=dict(min=1, max=15)),
            dict(name='log_lr', type='double', bounds=dict(min=math.log(0.00001), max=math.log(1.0))),
            dict(name='gamma', type='double', bounds=dict(min=0.5, max=0.9999)),
            dict(name='batch_size', type='int', bounds=dict(min=10, max=500)),
            dict(name='replay_buffer_size', type='int', bounds=dict(min=1000, max=1000000)),  
        ]
    )
    
    experiment = DQN_experiment
    for _ in range(experiment.observation_budget):
        suggestion = conn.experiments(experiment.id).suggestions().create()
        
        objective_metric = run_environment(
            h_size=suggestion.assignments['h_size'],
            middle_size=suggestion.assignments['middle_size'],
            lstm_layers=suggestion.assignments['lstm_layers'],
            epsilon_decay_steps=suggestion.assignments['epsilon_decay_steps'],
            learning_starts=suggestion.assignments['learning_starts'],
            lr=math.exp(suggestion.assignments['log_lr']),
            gamma=suggestion.assignments['gamma'],
            batch_size=suggestion.assignments['batch_size'],
            replay_buffer_size=suggestion.assignments['replay_buffer_size']
        )
        
        conn.experiments(experiment.id).observations().create(
            suggestion=suggestion.id,
            value=objective_metric
        )
    
if __name__=="__main__":
    search(specific_register_values, 50000)