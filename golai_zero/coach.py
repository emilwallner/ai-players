import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../../game/GOLAI')))
from arena import Arena
from collections import deque
from MCTS import MCTS
import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle


class Coach():
    
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []
        self.allOpponents = createRandomOpp(self.args.numEps)
    
    def createRandomOpp(self):
        randomPrograms = []
        
        for i in range(self.args.numEps):
            value = [1, 0]  
            dist = random.random()
            randomPrograms.append(np.random.choice(value, (self.args.programSize, self.args.programSize), p=[dist, (1.0 - dist)]))

        return np.array(randomPrograms, dtype=np.int8)
        
    
    def executeEpisode(self, eps):
        
        trainExamples = []
        self.curProgram = self.game.getInitProgram()
        self.curOpponent = self.trainOpponents[eps]
        episodeStep = 0
        
        while True:
            episodeStep += 1
            temp = int(episodeStep < self.args.tempThreshold)
            
            pi = self.mcts.getActionProb(self.curProgram, self.curOpponent, selftemp=temp)
            trainExamples.append([self.curProgram, pi, None])
            action = np.random.choice(len(pi), p=pi)
            self.game.getNextState(self.curProgram, self.curOpponent, action)
            
            if episodeStep == self.args.vocabLen:
                self.allOpponents.append(self.curProgram)
                r = self.game.getGameEnded(self.curProgram, self.curOpponent, episodeStep)
                return[(x[0], x[1], r) for x in trainExamples]
            
        def learn(self):
            
            for i in range(1, self.args.numIters+1):
                
                print('---------Iter ' + str(i) + '---------')

                    
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                eps_time = AverageMeter()
                bar = Bar('Self Play', max=self.args.numEps)
                end = time.time()
                self.trainOpponents = self.allOpponents[:self.args.numEps]
                shuffle(self.trainOpponents)
                
                for eps in range(self.args.numEps):
                    self.mcts = MCTS(self.game, self.nnet, self.args)
                    iterationTrainExamples += self.executeEpisode(eps)
                    eps_time.update(time.time() - end)
                    end = time.time()
                    bar.suffix = '({eps}/{maxeps} Eps Time: {et:.3f} | Total: {total:} | ETA: \
                    {eta:}'.format(eps=eps+1, maxeps=self.args.numEps, et=eps_time.avg,\
                    total=bar.elapsed_td, eta=bar.eta_td)

                    bar.next()
                bar.finished()

                self.trainExamplesHistory.append(interationTrainExamples)
                
                if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                    print("len(trainExamplesHistory) =", len(self.trainExamplesHistory), \
                          " => remove the oldest trainExamples")
                    self.trainExamplesHistory.pop(0)
                
                self.saveTrainExamples(i-1)
                trainExamples = []
                
                for e in self.trainExamplesHistory:
                    trainExamples.extend(e)
                shuffle(trainExamples)
                
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
                
                mcts = MCTS(self.game, self.nnet, self.args) ## why?
                self.nnet.train(trainExamples)
            
            
            def getCheckpointFile(self, iteration):
                return 'checkpoint_' + str(iteration) + '.pth.tar'
            
            def saveTrainExamples(self, iteration):
                folder = self.args.checkpoint
                
                if not os.path.exists(folder):
                    os.makedirs(folder)
                filename = os.path.join(folder, self.getCheckpointFile(iteration)+".examples")
                
                with open(filename, "wb+") as f:
                    Pickler(f).dump(self.trainExamplesHistory)
                f.closed
            
            def loadTrainExamples(self):
                modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
                examplesFile = modelFile+".examples"
                          
                if not os.path.isfile(examplesFile):
                          print(examplesFile)
                          r = input("File with trainExamples not found. Continue? [y|n]")
                          if r != "y":
                              sys.exit()
                else: 
                          
                    print("File with trainExamples found. Read it.")
                    with open(examplesFile, "rb") as f:
                          self.trainExamplesHistory = Unpickler(f).load()
                    f.closed
                    self.skipFirstSelfPlay = True
                          
# Structure from: https://github.com/suragnair/alpha-zero-general/blob/master/Coach.py      
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
            
                
                
                
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    