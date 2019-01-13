import keras
from keras.models import Model
import numpy as np

import random
from collections import namedtuple
from game2048.game import Game
from game2048.expectimax import board_to_move

OUT_SHAPE = (4,4)
CAND = 16
map_table = {2**i : i for i in range(1,CAND)}
map_table[0] = 0
vmap = np.vectorize(lambda x: map_table[x])

def grid_ohe(arr):
    ret = np.zeros(shape=OUT_SHAPE+(CAND,),dtype=bool)  
    for r in range(OUT_SHAPE[0]):
        for c in range(OUT_SHAPE[1]):
            ret[r,c,arr[r,c]] = 1
    return ret

Guide = namedtuple('Guides', ('state', 'action'))

class Guides:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Guide(*args)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
   
    def ready(self,batch_size):
        return len(self.memory) >= batch_size

    def __len__(self):
        return len(self.memory)

       
class ModelWrapper:
   
    def __init__(self, model, capacity):
        self.model = model
        self.memory = Guides(capacity)
        self.trainning_step = 0
        
    def predict(self, board):
        return model.predict(np.expand_dims(board,axis=0))
    
    def move(self, game):
        ohe_board = grid_ohe(vmap(game.board))
        suggest = board_to_move(game.board)
        direction = self.predict(ohe_board).argmax()
        game.move(direction)
        self.memory.push(ohe_board, suggest)
        
    def train(self, batch):
        if self.memory.ready(batch):
            guides = self.memory.sample(batch)
            X = []
            Y = []
            for guide in guides:
                X.append(guide.state)
                ohe_action = [0]*4
                ohe_action[guide.action] = 1
                Y.append(ohe_action)
            loss, acc = self.model.train_on_batch(np.array(X), np.array(Y))
            self.trainning_step += 1

MEMORY = 16384
BATCH = 1024

model = keras.models.load_model('best/model.h5')
mw = ModelWrapper(model,MEMORY)

while True:
    game = Game(4, random=False)
    while not game.end:
        mw.move(game)
    print('score:',game.score, end='\t')

    mw.train(BATCH)

    if(mw.trainning_step%10==0):
        model.save('best/model.h5')
