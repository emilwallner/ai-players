from collections import deque
from Arena import Arena
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
        self.skipFirstSelfPlay = False
        
    def executeEpisode(self, oponent):
        
        trainExamples = []
        program = self.game.getInitProgram()
        self.curPlayer = 1
        episodeStep = 0
        
        while True:
            episodeStep += 1
            temp = int(episodeStep < self.args.temThreshold)
            
            pi = self.mcts.getActionProb(program, temp=temp)
            action = np.random.choice(len(pi), p=pi)
            program, self.curPlayer = self.game.getNextState(program, action)
            
            r = self.game.getGameEnded(program, oponent, episodeStep)
            
            if r!=0:
                return[(x[0], x[2], r*((-1)**(x[1]!=self.curPlayer))) for x in trainExamples]
            
        def learn(self):
            
            for i in range(1, self.args.numIters+1):
                
                print('---------Iter ' str(i) + '---------')
                
                if not self.skipFirstSelfPlay or i > 1:
                    
                    iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
                    
                    eps_time = AverageMeter()
                    bar = Bar('Self Play', max=self.args.numEps)
                    end = time.time()
                    
                    for eps in range(self.args.numEps):
                        self.mcts = MCTS(self.game, self.nnet, self.args)
                        iterationTrainExamples += self.executeEpisode()

                        eps_time.update(time.time() - end)
                        end = time.time()
                        bar.suffix = '({eps}/{maxeps} Eps Time: (et:.3f | Total: {total:} | ETA: \
                        {eta:}'.format(eps=eps+1, maxeps=self.args.numEps, et=eps_time.avg,\
                        bar.elapsed_td, eta=bar.eta_td)

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
                
                mcts = MCTS(self.game, self.nnet, self.args)
                self.nnet.train(trainExamples)
            
            
            def getCheckpointFile(self, iteration):
                return 'checkpoint_' + str(iteration) + '.pth.tar'
            
            def saveTrainExamples(self, iteration):
                folder = self.args.checkpoint
                
                if not os.path.exists(folder):
                    os.makedirs(folder)
                filename = os.apth.join(folder, self.getCheckpointFile(iteration)+".examples)
                
                with open(filename, "wb+" as f:
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
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
            
                
                
                
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    