import os
import time, datetime
import config
import torch
from game.environment import Env
from tensorboardX import SummaryWriter
from collections import namedtuple
import numpy as np

CFG = config.get_cfg()
CWCFG = CFG.settings.toy_corewar

Task = namedtuple('Task', ('reward_function', 'reg_init', 'total_episodes', 'best_score'))

class Agent:
    def __init__(self, verbose, log_dir):
        self.best_score = -float('Inf')
        self.best_episode = 0
        self.verbose = verbose
        self.log_dir = log_dir
        self.writer = SummaryWriter() if log_dir else None
        self.model = None
        self.best_model = None
        self.total_episodes = 0
    
    ## Methods that need to be implemented in the child classes
    
    def train(self, reward_func, reg_init, episodes):
        raise NotImplementedError("You need to implement a train method in your class!")
    
    def act(self, state):
        raise NotImplementedError("You need to implement an act method in your class!")
    
    def load(self, path):
        raise NotImplementedError("You need to implement an act method in your class!")

        
    ## Methods that are implemented in the Agent class

    def save(self, name, best=False):
        if self.log_dir is not None:
            path = os.path.join(self.log_dir, "models")
            os.makedirs(path, exist_ok=True)
            path = os.path.join(path, name)
        else:
            path = name
        if best:
            torch.save(self.best_model.state_dict(), path)
        else:
            torch.save(self.model.state_dict(), path)


    def assess(self, reward_func, reg_init=None, episode=None, print=False, file=None):
        env = Env(reward_func)
        s = env.reset(reg_init)
        for t in range(CWCFG.MAX_LENGTH):
            a = self.act(s)
            s_prime, reward, done, _ = env.step(a)
            s = s_prime
            if done:
                break
        performance = env.performance
        total_reward = env.total_reward
        
        if print:
            env.print_details(file=file)
        return performance, total_reward


    def evaluate(self, log=False):
        performances = []
        total_rewards = []
        if log:
            filename = os.path.join(self.log_dir, "{:07}_Evaluation".format(self.total_episodes))
        else:
            filename = os.devnull
        with open(filename, 'w') as f:
            print("Evaluation over {} tasks".format(len(self.tasks)), file=f)
            print("Algorithm: {}".format(self.__class__.__name__), file=f)
            current_time = datetime.timedelta(seconds=(time.time() - self.start_time))
            print("Time: {}".format(str(current_time)), file=f)

            for reward_func, reg_init in self.tasks:
                print("Task:  {}".format(reward_func), file=f)
                print("Initialization: {}".format(reg_init), file=f)
                performance, total_reward = self.assess(reward_func, reg_init, print=True, file=f)
                performances.append(performance)
                total_rewards.append(total_reward)
                print("\n\n", file=f)

            mean_perf = np.mean(performances)
            mean_reward = np.mean(total_rewards)
            print("Mean performance: {}".format(mean_perf), file=f)
            print("Mean reward: {}".format(mean_reward), file=f)

        if self.verbose:
            print("Currently at episode {}".format(self.total_episodes))
        if log:
            self.writer.add_scalars(self.log_dir, {'Mean performance': mean_perf}, self.total_episodes)
            self.writer.add_scalars(self.log_dir, {'Mean total reward': mean_reward}, self.total_episodes)

        if mean_perf > self.best_score:
            self.best_score = mean_perf
            self.best_episode = self.total_episodes
            self.best_model.load_state_dict(self.model.state_dict())

        return mean_perf, mean_reward


    def generalize(self, Reward_func, num_tasks, reward_settings=None, reg_zero_init=True, log=False):
        np.random.seed(0)
        performances = []
        total_rewards = []
        if log:
            filename = os.path.join(self.log_dir, "{:07}_Generalization".format(self.total_episodes))
        else:
            filename = os.devnull
        with open(filename, 'w') as f:
            print("Generalization over {} test tasks".format(num_tasks), file=f)
            print("Algorithm: {}".format(self.__class__.__name__), file=f)
            current_time = datetime.timedelta(seconds=(time.time() - self.start_time))
            print("Time: {}".format(str(current_time)), file=f)
            for n in range(num_tasks):
                reward_function = Reward_func(None, reward_settings)
                if reg_zero_init:
                    reg_init = np.zeros(CWCFG.NUM_REGISTERS, dtype=int)
                else:
                    reg_init = np.random.randint(0, 256, CWCFG.NUM_REGISTERS)
                print("Task {}:  {}".format(n + 1, reward_function), file=f)
                print("Initialization: {}".format(reg_init), file=f)
                performance, total_reward = self.assess(reward_function, reg_init, print=True, file=f)
                performances.append(performance)
                total_rewards.append(total_reward)
                print("\n\n", file=f)
            mean_perf = np.mean(performances)
            mean_reward = np.mean(total_rewards)
            print("Mean performance: {}".format(mean_perf), file=f)
            print("Mean reward: {}".format(mean_reward), file=f)
        np.random.seed()

        if mean_perf > self.best_score:
            self.best_score = mean_perf
            self.best_episode = self.total_episodes
            self.best_model.load_state_dict(self.model.state_dict())

        return mean_perf, mean_reward

    def best_performance(self):
        return self.best_score, self.best_episode

    def __del__(self):
        if self.writer is not None:
            self.writer.close()

    # def update_reward_function(self, Reward_func, targets, reward_settings, episode, episodes):
    #     if targets is None:
    #         target = None
    #         change_frequency = 1
    #     elif isinstance(targets, int):
    #         change_frequency = episodes // targets
    #         target = None
    #     elif isinstance(targets, list):
    #         change_frequency = episodes // len(targets)
    #         index =
    #         target = np.array(targets[index], dtype=int)
    #     else:
    #         raise ValueError("Unrecognized data type for 'targets': {}".format(targets))
    #     if episode % (change_frequency) == 0:
    #         self.reward_func = Reward_func(target, reward_settings)

    # def log_init(self, episodes, reward_func):
    #     self.log_num += 1
    #     # Console output
    #     if self.verbose:
    #         print("Starting training [algo = {}, reward = {}, version {}] for {} episodes...".format(
    #             self.__class__.__name__, reward_func.__class__.__name__, self.log_num, episodes))
    #     # Logging file output
    #     if self.log_dir is not None:
    #         with open(os.path.join(self.log_dir, "logs{}".format(self.log_num)), "w") as f:
    #             print("Starting training for {} episodes...".format(episodes), file=f)
    #             print("Algorithm: {}".format(self.__class__.__name__), file=f)
    #             print("Reward function:  {}\n\n\n".format(reward_func), file=f)
    #

    # def log(self, episode, reward_func, start_time):
    #     # to console
    #     if self.verbose:
    #         print("Episode {} completed for {}, {}_{}".format(
    #             episode + 1, self.__class__.__name__, reward_func.__class__.__name__, self.log_num))
    #
    #     # to log file and Tensorboard
    #     if self.log_dir is not None:
    #         with open(os.path.join(self.log_dir, "logs{}".format(self.log_num)), "a") as f:
    #             current_time = datetime.timedelta(seconds=(time.time()-start_time))
    #             print("Episode {}: [time:  {}]\n".format(episode+1, str(current_time)), file=f)
    #             score = self.assess(reward_func, episode=episode, print=True, file=f)
    #             print("\n\n\n", file=f)
    #             # log to Tensorboard
    #             self.writer.add_scalars(self.log_dir, {'rewards': score}, episode)

# def multi_train(self, Reward_func, targets, reg_init_freq, episodes):
    #     ''' Performs multiple trainings on the same task, but with different target values and register initializations.
    #     Arg types:
    #     - reward_func: [string] name of a Reward_function class
    #     - targets: [integer] number of random target values to generate
    #     - reg_init_freq: [integer] indicates after how many episodes register initializations are randomly reset.
    #                     A value of O means that all registers are initialized at 0.
    #     - episodes: [integer] the total number of episodes done, divided among all training subtasks'''
    #
    #     if isinstance(targets, list) and isinstance(targets[0], np.ndarray):
    #         num_targets = len(targets)
    #     elif episodes % num_targets == 0:
    #         num_targets = targets
    #         episodes_per_target = episodes // num_targets
    #         targets = [None for _ in range(num_targets)] # 'None' means the Reward_function object is constructed with random target values
    #     else:
    #         raise ValueError("Need episodes({}) % num_targets({}) == 0".format(episodes, num_targets))
    #
    #     if reg_init_freq < 0 :
    #         raise ValueError("Negative reg_init_freq: {}".format(reg_init_freq))
    #     elif reg_init_freq == 0:
    #         zero_init = True
    #         reg_init = np.zeros(CWCFG.NUM_REGISTERS, dtype=int)
    #         reg_init_freq = episodes_per_target
    #     elif episodes_per_target % reg_init_freq == 0:
    #         zero_init = False
    #     else:
    #         raise ValueError("Need (episodes({}) // num_targets({})) % reg_init_freq({}) == 0".format(episodes, num_targets, reg_init_freq))
    #
    #     # Create training tasks
    #     for target in targets:
    #         reward = reward_func(target)
    #         for _ in range(episodes_per_target // reg_init_freq):
    #             if not zero_init:
    #                 reg_init = np.random.randint(0, 256, CWCFG.NUM_REGISTERS)
    #             self.tasks.append(Task(reward, reg_init, 0, -float('Inf')))
    #
    #     task = self.tasks[0]
    #     self.train(reward, reg_init, reg_init_freq)
    #     self.save("End_multi_training", best=False)
