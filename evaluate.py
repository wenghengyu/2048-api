from game2048.game import Game
from game2048.displays import Display


def single_run(size, score_to_win, AgentClass, **kwargs):
    game = Game(size, score_to_win)
    agent = AgentClass(game, display=Display(), **kwargs)
    time = agent.play(verbose=True)
    return game.score, time


if __name__ == '__main__':
    GAME_SIZE = 4
    SCORE_TO_WIN = 2048
    N_TESTS = 50

    '''====================
    Use your own agent here.'''
    from Agent import MyOwnAgent as TestAgent
    '''===================='''

    scores = []
    times = []
    Agent_score = [0, 0, 0, 0, 0, 0]
    for _ in range(N_TESTS):
        score, time = single_run(GAME_SIZE, SCORE_TO_WIN,
                           AgentClass=TestAgent)
        if(score==32):
            Agent_score[0] += 1
        elif(score==64):
            Agent_score[1] += 1
        elif(score==128):
            Agent_score[2] += 1
        elif(score==256):
            Agent_score[3] += 1
        elif(score==512):
            Agent_score[4] += 1
        elif(score==1024):
            Agent_score[5] += 1

        scores.append(score)
        times.append(time)

    print("Average scores: @%s times" % N_TESTS, sum(scores) / len(scores))
    print("Score distribution(times): 32:%d ,64:%d, 128:%d, 256:%d, 512:%d, 1024:%d" % (Agent_score[0], Agent_score[1], Agent_score[2], Agent_score[3], Agent_score[4], Agent_score[5]))
    average_step_time = sum(times) / len(times)
    print("Average time for each predicted step is %f s" % average_step_time)
