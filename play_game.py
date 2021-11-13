import cv2
import os
import sys
import pygame
import random

import numpy as np
import flappy_bird as game
import tensorflow as tf

sys.path.append("game/")

from collections import deque
from tqdm import tqdm
from predictor import Predictor

GAME = 'Flappy Bird' # the name of the game being played for log files
N_ACTIONS = 2 # number of valid actions
N_STATES = 80*80
BATCH = 32 # size of minibatch
LR = 0.01
EPSILON = 0.1
GAMMA = 0.9 # decay rate of past observations
TARGET_REPLACE_ITER = 100 # target network 更新間隔
MEMORY_CAPACITY = 100 # number of previous transitions to remember
EPOCH = 4000

FPS = 30
SCREENWIDTH  = 288
SCREENHEIGHT = 512

os.environ["cuda_visible_devices"] = "-1"

def trainNetwork():
    # define predictor for training network
    predictor = Predictor(N_STATES, N_ACTIONS, BATCH, LR, EPSILON, GAMMA, TARGET_REPLACE_ITER, MEMORY_CAPACITY)

    # open up a game state to communicate with emulator
    FPSCLOCK = pygame.time.Clock()
    game_state = game.GameState()

    # initailize state
    do_nothing = 0
    state, reward, done = game_state.frame_step(do_nothing)
    # RGB 2 Gray, resize 2 (80, 80)
    state = cv2.cvtColor(cv2.resize(state, (80, 80)), cv2.COLOR_BGR2GRAY)
    # x_t > ret = 255, x_t < ret = 0
    ret, state = cv2.threshold(state,1,255,cv2.THRESH_BINARY)
    state = np.stack((state, state, state, state), axis=2).astype(float)

    # learning
    for ep in range(EPOCH):

        t, rewards = 0, 0
        while True:
            FPSCLOCK.tick(FPS)
            # 選擇 action
            action = predictor.choose_action(tf.constant(state.reshape((1, 80, 80, 4))))
            next_state, reward, done = game_state.frame_step(action)
            # RGB 2 Gray, resize 2 (80, 80)
            next_state = cv2.cvtColor(cv2.resize(next_state, (80, 80)), cv2.COLOR_BGR2GRAY)
            # x_t > ret = 255, x_t < ret = 0
            ret, next_state = cv2.threshold(next_state,1,255,cv2.THRESH_BINARY)

            # 儲存 experience
            predictor.store_transition(state.reshape(-1), action, reward, next_state.reshape(-1))

            # 累積 reward
            rewards += reward

            # 有足夠 experience 後進行訓練
            if predictor.memory_counter > MEMORY_CAPACITY: predictor.run_epoch()

            # 進入下一 state
            state = np.append(next_state.reshape((80,80,1)), state[..., :3], axis=2)

            if done:  
                print('Episode finished after {} timesteps, total rewards {}'.format(t+1, rewards))
                break

            t += 1



def playGame():
    
    pygame.init()
    SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
    pygame.display.set_caption(GAME)

    trainNetwork()

def main():
    playGame()

if __name__ == "__main__":
    main()