from GOLAI import arena
import random
from settings import *
from torch import Tensor

class Game():
    
    def __init__(self):
        self.game_round = 0
        self.game_steps = 100
        self.program_width = 9
        self.program_height = 9
    
    def getInitProgram(self):
        
        return np.zeros((9,9), dtype=np.int8)
    
    def getBoardSize(self):
        return arena.size()
    
    def getActionSize(self):
        return(VOCAB_SIZE)
    
    def getNextState(self, program, action):
        
        """
        
        Turns the action/vocab into a board piece.
        Adds it to the board.
        Creates -1 for next action square.
        
        """
    
    def getGameEnded(self, playerOne, playerTwo):
        
        if self.game_round < self.game_round_limit:
            return 0
        else:
            player1, player2 = convertToArenaPlayers(playerOne, playerTwo)
            arena.add_players(playerOne, playerTwo)
            arena.run_steps(self.game_steps)
            return selectWinner(arena.grid())
    
    def selectWinner(self, board):
        ones = 0, twos = 0

        for i in game_result:
            if i == 1:
                ones += 1
            elif i == 2:
                twos += 1

        if ones > twos:
            winner = Tensor(1.0)
        elif ones > twos:
            winner = Tensor(-1.0)
        else:
            winner = Tensor(random.uniform(0.001, 0.1))
            
        return winner
        
    def convertToArenaPlayers(player1, player2):
        
        """
        
        Turns the working format into a 2D numpy array int8
        
        """
        
        return player1, player2
    
    def stringRepresentation(self, program):
        
        """
        
        Turn board into a string used for the MCST hashing.
        
        """
        
        program_string = ""
        for integer in board:
            program_string += str(integer)
            
        return program_string
    
    def integerImageRepresentation(self, board):
        
        """
        
        Creates the input for the neural network:
        
        """
        
        return board.reshape(self.program_width, self.program_height)
    

# Structure from: https://github.com/suragnair/alpha-zero-general/blob/master/Game.py