import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../..')))
from game.GOLAI.arena import Arena
import random
from torch import Tensor

class Game():
    
    def __init__(self):
        self.game_round = 0
        self.game_steps = 50
        self.program_width = 6
        self.program_height = 6
        self.start = True
        self.vocab = 16
        self.program_size = 36
        self.vocab_w = 2
        self.vocab_h = 2
        self.prediction_len = self.program_size // self.vocab_w
    
    def getNextState(self, program, action):
        
        """
        
        Turns the action/vocab into a board piece.
        Adds it to the board.
        Creates -1 for next action square.
        
        """
        
        for i in program:
            if i == -1:
                program[i] = action
                return program
        print("getNextState was called with a full board")
        return program
        
    def integerImageRepresentation(self, sequence):

        """ Creates the input for the neural network: """
        
        self.x = ((self.program_size - self.vocab_w) // 2) - 1
        self.y = ((self.program_size - self.vocab_h) // 2) - 1
        self.program = np.full((self.program_size, self.program_size), -1, dtype=np.int8)
        self.create_player_from_sequence(sequence)
        return self.program
    
    def create_player_from_sequence(self, sequence):  
        for digit in sequence:
            if digit == -1:
                break
            grid = digit_to_grid(self, digit)
            add_grid_to_program(self, grid)
            next_cord(self)
                
    def digit_to_grid(self, digit):
        # Turn digit into binary representation to create a word
        binary = "{0:b}".format(digit)
        binary = binary.zfill(4)
        return np.array(binary).reshape((self.vocab_w, self.vocab_h))
    
    def add_grid_to_program(self, grid):
        for y in self.vocab_h:
            for x in self.vocab_w:
                self.program[self.x + x][self.y + y] = grid[x][y]
    
    def next_cord(self):
        #The program is initialized with -1, if it's something else we know its been filled already.
        #The program is added in a spiral shape starting by moving to the right. Move down if left block 
        #is filled and bottom is emtpy, or move left if top is filled, or move up if right is filled, else 
        #move right.
        
        if self.start:
            self.x += self.vocab_w
            self.start = False

        if self.x != 0 and self.program[self.x - 1, self.y] != -1 \
        and self.program[self.x, self.y + self.vocab_h] == -1:
            self.y += self.vocab_h
        elif self.y != 0 and self.program[self.x, self.y - 1] != -1:
            self.x -= self.vocab_w
        elif self.x + self.vocab_w != self.program_size and self.program[self.x + self.vocab_w, self.y] != -1:
            self.y -= self.vocab_h
        else:
            self.x += self.vocab_w
    
    def getInitProgram(self):
        
        return np.full((self.prediction_len), -1, dtype=np.int8)
    
    def getBoardSize(self):
        return arena.size()
    
    def getActionSize(self):
        return(self.vocab_size)
    
    def getGameEnded(self, playerOne, playerTwo):
        
        player1, player2 = convertToArenaPlayers(playerOne, playerTwo)
        arena.add_players(playerOne, playerTwo)
        arena.run_steps(self.game_steps)
        return selectWinner(arena.grid())
    
    def selectWinner(self, board):
        ones = 0
        twos = 0

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
    
  

# Structure from: https://github.com/suragnair/alpha-zero-general/blob/master/Game.py
