from coach import Coach
from model import GolaiZero
from game import Game

args = dotdict({
    'numIters': 1000,
    'numEps': 100,
    'vocabWidth': 3, 
    'vocabHeight': 3,
    'programSize': 9,
    'vocabLen': 9,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,
    'arenaCompare': 40,
    'cpuct': 1,
    
    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})

if __name__=="__main__":
    g = Game(6)
    nnet = GolaiZero()
    
    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
        
    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
    
    
    
    
# Structure from: https://github.com/suragnair/alpha-zero-general/blob/master/main.py