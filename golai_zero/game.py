import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../../game/GOLAI')))
from arena import Arena
import random
from torch import Tensor

import random
from torch import Tensor

class Game():
    
    def __init__(self, args):
        self.args = args
        
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
        
        self.start = True
        self.x = (self.args.programWidth // 2) - (self.args.vocabWidth // 2) 
        self.y = (self.args.programHeight // 2) - (self.args.vocabHeight // 2) 
        self.program = np.full((self.args.programWidth, self.args.programHeight), -1, dtype=np.int8)
        self.create_player_from_sequence(sequence)
        return self.program
    
    def create_player_from_sequence(self, sequence):  
        for digit in sequence:
            print(digit)
            if digit == -1:
                break
            grid = self.digit_to_grid(digit)
            self.add_grid_to_program(grid)
            self.next_cord()
                
    def digit_to_grid(self, digit):
        # Turn digit into binary representation to create a word
        binary = "{0:b}".format(digit)
        binary = binary.zfill(4)
        binary = list(str(binary))
        return np.array(binary).reshape((self.args.vocabWidth, self.args.vocabHeight))
    
    def add_grid_to_program(self, grid):
        for x in range(self.args.vocabWidth):
            for y in range(self.args.vocabHeight):
                print(self.x + x, self.y + y, x, y)
                self.program[self.x + x][self.y + y] = grid[x][y]
    
    def next_cord(self):
        
         #The program is initialized with -1, if it's something else we know its been filled already.\
         #The program is added in a spiral shape starting by moving to the right. Move down if left block \
         #is filled and bottom is emtpy, or move left if top is filled, or move up if right is filled, else \
         #move right.
        
        if self.start:
            self.x += self.args.vocabWidth
            print(self.x)
            self.start = False
        elif self.x != 0 and self.program[self.x - 1, self.y] != -1 \
        and self.program[self.x, self.y + self.args.vocabHeight] == -1:
            self.y += self.args.vocabHeight
        elif self.y != 0 and self.program[self.x, self.y - 1] != -1:
            self.x -= self.args.vocabWidth
        elif self.x + self.args.vocabWidth != self.args.programWidth and self.program[self.x + self.args.vocabWidth, self.y] != -1:
            self.y -= self.args.vocabHeight
        else:
            self.x += self.args.vocabWidth
    
    def getInitProgram(self):
        
        return np.full((self.args.predictionLen), -1, dtype=np.int8)
    
    def getBoardSize(self):
        return arena.size()
    
    def getActionSize(self):
        return(self.args.vocabLen)
    
    def getGameEnded(self, playerOne, playerTwo):
        player1, player2 = convertToArenaPlayers(playerOne, playerTwo)
        arena.add_players(playerOne, playerTwo)
        arena.run_steps(self.args.gameSteps)
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
            winner = Tensor(random.uniform(0.000001, 0.000000001))
            
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