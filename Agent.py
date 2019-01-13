from game2048.game import Game
from game2048.agents import Agent
import numpy as np
import keras
import time

model = keras.models.load_model('best/model.h5')

OUT_SHAPE = (4,4)
CAND = 16
map_table = {2**i : i for i in range(1,CAND)}
map_table[0] = 0

def grid_ohe(arr):
    ret = np.zeros(shape=OUT_SHAPE+(CAND,),dtype=bool)
    for r in range(OUT_SHAPE[0]):
        for c in range(OUT_SHAPE[1]):
            ret[r,c,arr[r,c]] = 1
    return ret

class MyOwnAgent(Agent):
    def __init__(self, game, display=None):
        super().__init__(game, display)
        self.testgame = Game(4, random=False)
        self.testgame.enable_rewrite_board = True

    def step(self):
        a = time.time()
        piece = [map_table[k] for k in self.game.board.astype(int).flatten().tolist()]
        x0 = np.array([ grid_ohe(np.array(piece).reshape(4,4)) ])
        preds = list(model.predict(x0))
        direction = np.argmax(preds[0])
        b = time.time()
        step_time = b-a
        return direction, step_time
