# Used code from:
# The Pacman AI projects were developed at UC Berkeley found at
# http://ai.berkeley.edu/project_overview.html

import numpy as np
import random
import agents.util as util
import time
import sys

# Pacman game
from pacman import Directions
from game import Agent
import game

# Replay memory
from collections import deque
from dqn.ReplayBuffer import *

# Tiles tracker
from dqn.tiles_tracker import *

import tensorflow as tf

# Defunct (No need to adjust)
NUM_CHANNELS = 6
VISIBLE_LIMIT = 5
AUDIBLE_LIMIT = 2
CHANNEL_RADIUS = max(VISIBLE_LIMIT, AUDIBLE_LIMIT)
channel_height = 7
channel_width = 7
is_partially_observable = False
AUDIBLE_LIMIT = 0

#########################################
########## Adjustables (start) ##########

# Neural nets (Choose the neural network)

# from dqn.CNN1 import *
# from dqn.TwoPlusOneD_CNN_with_time import *
from dqn.FullyConnected_NN import *
# from dqn.MHA_NN_with_time import *
# from dqn.MHA_NN import *
# from dqn.CNN2 import *
# from dqn.CNN3 import *
# from dqn.CNN4 import *
# from dqn.CNN1_with_LSTM import *

# Field of vision (if true, restrict the field of vision to 7x7)
is_sliced = True

# Training or testing (if false, model will always exploit and will not be updated)
TRAINING_MODE = True

# Logfile
NAMENAME = "-CNN2_trained_originalClassic"

# save the neural network weights to this location
save_path_targetqnet = 'saves/FC_NN/target_qnet' #filepath
save_path_qnet = 'saves/FC_NN/qnet' #filepath

# load the neural network weights from this location
load_file_targetqnet = 'saves/originalClassic/CNN2/target_qnet/ckpt-81' #None or str: foldername/filename
load_file_qnet = 'saves/originalClassic/CNN2/qnet/ckpt-81' #None or str: foldername/filename

########## Adjustables (end) ##########
#######################################

if is_sliced:
    from dqn.matrices2 import prepare_observation_matrix
else:
    from dqn.matrices import prepare_observation_matrix

params = {
    # Model backups
    'ToLoad' : False,       # True or False
    'ToSave' : True,        # True or False
    'save_interval' : 100,  # Save after save_interval number of episodes

    # Training parameters
    'train_start': 256,     # Episodes before training starts
    'batch_size': 32,       # Replay memory batch size (batch_size < train_start)
    'mem_size': 15000,      # Replay memory size

    'discount': 0.9,        # Discount rate (gamma value)
    'lr': 0.00025,          # Learning rate

    # Experience replay
    'priority_scale': 0.6,  # 0 reverts to a uniform selection

    # Prioritised memory replay
    'annealment_growthrate': 1.01,
    'annealment_max': 1,
    'annealment_base': 0.4,

    # Epsilon value (epsilon-greedy)
    'eps_start': 1,         # Epsilon start value
    'eps_final': 0.05,      # Epsilon end value (0.05 as per paper: HUMAN-LEVEL CONTROL THROUGH DRL)
    'eps_step': 0.9,        # 0 < Decay factor < 1 (higher means eps decrease slower)

    # Training delay (update params after how many steps)
    'update_target_freq': 10000,

    # Video time step
    'video_time_stamp': 0,

    # Algo
    'DQN_or_DDQN':'DQN',    # DQN or DDQN

    # Defunct (No need to adjust)
    # Stalling threshold: saves its last 20 positions. If the number of uniques are less than 20 / 2, then it gets penalised.
    # no_new_position_threshold: if it fails to find a new unique position in 10 steps, penalise.
    'stalling_threshold': 10,
    'no_new_position_threshold': 3,
}



class PacmanTDQN(game.Agent):
    def __init__(self, args):

        print("Initialise DQN Agent")

        # Load parameters from user-given arguments
        self.params = params

        if is_partially_observable == True or is_sliced == True:
            self.params['width'] = channel_width
            self.params['height'] = channel_height
        else:
            self.params['width'] = args['width']
            self.params['height'] = args['height']
        self.params['num_training'] = args['numTraining']
        self.params['NUM_CHANNELS'] = NUM_CHANNELS

        self.params['load_file'] = load_file_targetqnet
        self.params['save_file'] = save_path_targetqnet
        self.target_qnet = DQN(self.params)  # Instantiate the DQN model: For predicting

        self.params['load_file'] = load_file_qnet
        self.params['save_file'] = save_path_qnet
        self.qnet = DQN(self.params) # Instantiate the DQN model: Updated every step

        self.params.pop('load_file', None)
        self.params.pop('save_file', None)

        # Load the checkpoint if the 'load_file' parameter is specified
        if self.params['ToLoad']:
            print(" ")
            self.qnet.load_checkpoint()
            self.target_qnet.load_checkpoint()
            input("CHECKPOINT: Press any key to proceed to training...")
        else:
            self.target_qnet.set_weights(self.qnet.get_weights()) # If it is not loading, then it must be a new instance.
            print(" ")
            input("CHECKPOINT: This is a new instance. Press any key to proceed to training...")

        self.params['eps'] = self.params['eps_start']
        self.annealment_power = self.params['annealment_base']

        # time started
        self.general_record_time = time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime())

        # Q and cost
        self.Q_global = deque(maxlen=50)
        self.cost_disp = 0

        # Stats
        self.local_cnt = 0 # Number of steps taken from the first training episode
        self.stepinep = 0 # Number of steps in an episode
        self.numeps = 0 # Number of episodes aka games
        self.last_score = 0
        self.s = time.time()
        self.last_reward = 0.

        self.num_uniq_state = 0

        self.replay_mem = ReplayBuffer(maxlen=self.params['mem_size'])


    def getMove(self, state):
        # Exploit / Explore
        if np.random.rand() > self.params['eps']:

            if self.params['video_time_stamp'] > 0:
                exp_hist = []
                k_history = self.params['video_time_stamp']-1
                if len(self.replay_mem.buffer) >= k_history:
                    '''
                    Extract self.params['video_time_stamp']-1 experience ago only when there are enough memories
                    Access experiences from the end of the buffer, collecting the latest k_history experiences
                    '''
                    start_index = len(self.replay_mem.buffer) - k_history
                    for i in range(start_index, len(self.replay_mem.buffer)):
                        exp, _, _, _, _ =  self.replay_mem.buffer[i]
                        exp_hist.append(exp)
                    exp_hist.append(self.current_state)

                    current_state = np.stack(exp_hist, axis=0)

                else:
                    '''
                    If there is not enough experience, make the 5D tensor out of 0
                    Fill in the last part of exp_hist with current_state and sets the current_state as the last slice
                    '''
                    final_shape = (k_history+1, self.params['width'], self.params['height'], NUM_CHANNELS)
                    exp_hist = np.zeros(final_shape)
                    current_state_reshaped = self.current_state
                    exp_hist[-1] = current_state_reshaped

                current_state = np.expand_dims(exp_hist, axis=0)
            elif self.params['video_time_stamp'] == 0:
                current_state = self.current_state[None,:,:,:]

            # Exploit action - direct model call
            self.Q_pred = self.qnet(current_state, training=False).numpy()[0]

            self.Q_global.append(max(self.Q_pred))  # Q_global starts out empty.
            a_winner = np.argwhere(self.Q_pred == np.amax(self.Q_pred))

            if len(a_winner) > 1:
                move = self.get_direction(a_winner[np.random.randint(0, len(a_winner))][0])
            else:
                move = self.get_direction(a_winner[0][0])

        else:
            # Random action:
            move = self.get_direction(np.random.randint(0, 4))

        # Save last_action
        self.last_action = self.get_value(move)

        return move


    def get_value(self, direction):
        if direction == Directions.NORTH:
            return 0.
        elif direction == Directions.EAST:
            return 1.
        elif direction == Directions.SOUTH:
            return 2.
        else:
            return 3.

    def get_direction(self, value):
        if value == 0.:
            return Directions.NORTH
        elif value == 1.:
            return Directions.EAST
        elif value == 2.:
            return Directions.SOUTH
        else:
            return Directions.WEST

    def observation_step(self, state):
        if self.last_action is not None:
            # Process current experience state
            self.last_state = np.copy(self.current_state)
            self.current_state = np.transpose(self.getStateMatrices(state), (2, 1, 0))

            # Process current experience reward
            self.current_score = state.getScore()
            reward = self.current_score - self.last_score
            self.last_score = self.current_score

            self.last_reward = reward

            pacmanPosition = state.getPacmanPosition()

            _, _, self.num_uniq_state = self.visitedTracker.visit(pacmanPosition)
            self.won = state.isWin()

            self.ep_rew += self.last_reward

            experience = (self.last_state,
                          float(self.last_reward),
                          self.last_action,
                          self.current_state,
                          self.terminal)
            self.replay_mem.add(experience)

            # Process current terminal state
            self.last_terminal = self.terminal

            self.train()

        # Next
        self.local_cnt += 1
        self.frame += 1

        if TRAINING_MODE == True:
            self.params['eps'] = max(self.params['eps_final'], self.params['eps_start']*(self.params['eps_step']**float(self.qnet.train_step_counter.numpy())))
        else:
            self.params['eps'] = 0
        self.stepinep += 1


    def observationFunction(self, state):
        # Do observation
        self.terminal = False
        self.observation_step(state)

        return state


    def final(self, state):
        # Next
        self.ep_rew += self.last_reward

        # Do observation
        self.terminal = True
        self.observation_step(state)

        # Print stats
        # numeps (#): the number of games played
        # local_cnt (steps): number of steps taken (e.g., 1 unit up is 1 step). Cumulative for the training sessions
        # cnt (steps_t): cumulative number of training iter (1 forward + 1 backward passes = 1 cnt)
        # time (t) is time
        # ep_rew (r): reward of that episode
        # Q: maximum Q-value of that episode
        # step_ep: number of steps taken in that episode
        log_file = open('./logs/'+str(self.general_record_time)+'-l-'+str(self.params['width'])+'-m-'+str(self.params['height'])+'-x-'+str(self.params['num_training'])+str(NAMENAME)+'.log','a')
        log_file.write("# %4d | steps: %5d | w_upd: %5d | t: %4f | r: %12f | epsilon: %10f " %
                         (self.numeps, self.local_cnt, self.qnet.train_step_counter.numpy(), time.time()-self.s, self.ep_rew, self.params['eps']))
        log_file.write("| Q: %10f | won: %r " % ((max(self.Q_global, default=float('nan')), self.won)))
        log_file.write("| stepinep: %4d| nn_loss: %8.1f| score: %10.2f| num_uniq_state: %3d \n" % (self.stepinep, self.cost_disp, self.current_score, self.num_uniq_state))
        sys.stdout.write("# %4d | steps: %5d | w_upd: %8d | t: %8.1f | r: %6d | epsilon: %0.5f " %
                         (self.numeps, self.local_cnt, self.qnet.train_step_counter.numpy(), time.time()-self.s, self.ep_rew, self.params['eps']))
        sys.stdout.write("| Q: %6.1f | won: %r" % ((max(self.Q_global, default=float('nan')), self.won)))
        sys.stdout.write("| stepinep: %4d| nn_loss: %8.1f| score: %8.1f| num_uniq_state: %3d \n" % (self.stepinep, self.cost_disp, self.current_score, self.num_uniq_state))
        sys.stdout.flush()

        # Save model
        if self.params['ToSave']:
            if self.local_cnt > self.params['train_start'] and self.numeps % self.params['save_interval'] == 0:
                save_path_qn = self.qnet.checkpoint_manager.save()
                save_path_tqn = self.target_qnet.checkpoint_manager.save()
                print(f'Model saved at {save_path_qn} & {save_path_tqn}')


    def train(self):
        # Train the network
        if self.local_cnt > self.params['train_start']:
            # Sample a minibatch from the replay memory
            samples, importance, indices = self.replay_mem.sample(self.params['batch_size'], priority_scale=self.params['priority_scale'])
            batch_s, batch_r, batch_a, batch_n, batch_nt = zip(*samples)

            if self.params['video_time_stamp'] > 0:
                # Initialize an empty array for storing sequences
                batch_s_sequences = np.zeros((self.params['batch_size'], self.params['video_time_stamp'], self.params['width'], self.params['height'], NUM_CHANNELS))
                batch_n_sequences = np.zeros((self.params['batch_size'], self.params['video_time_stamp'], self.params['width'], self.params['height'], NUM_CHANNELS))

                for i, idx in enumerate(indices):
                # Ensure you do not go out of bounds when fetching earlier frames
                    start_idx = max(idx - self.params['video_time_stamp'] + 1, 0)
                    sequence = [self.replay_mem.buffer[j][0] for j in range(start_idx, idx + 1)]  # Assuming [0] is the state in the tuple
                    sequencen = [self.replay_mem.buffer[j][3] for j in range(start_idx, idx + 1)]  # Assuming [3] is the next_state in the tuple

                    # If there are not enough past states, prepend with zeros
                    if len(sequence) < self.params['video_time_stamp']:
                        needed = self.params['video_time_stamp'] - len(sequence)
                        sequence = [np.zeros((self.params['width'], self.params['height'], NUM_CHANNELS))] * needed + sequence
                        sequencen = [np.zeros((self.params['width'], self.params['height'], NUM_CHANNELS))] * needed + sequencen

                    batch_s_sequences[i] = np.array(sequence)
                    batch_n_sequences[i] = np.array(sequencen)

                batch_s = batch_s_sequences # so that the variables below do not need adjustments
                batch_n = batch_n_sequences # so that the variables below do not need adjustments

            elif self.params['video_time_stamp'] == 0:
                batch_s = np.array(batch_s)
                batch_n = np.array(batch_n)

            batch_r = np.array(batch_r)
            batch_a = np.array(batch_a)
            batch_nt = np.array(batch_nt, dtype=np.float32)

            # One-hot encode the actions
            batch_a = self.get_onehot(batch_a)

            if self.params['DQN_or_DDQN']=='DDQN':
                # Double DQN
                # Step 1: Action selection using the primary network (qnet) for the next states
                actions_from_qnet = np.argmax(self.qnet(batch_n, training=False).numpy(), axis=1) # Shape will be (batch_size,)

                # Step 2: Evaluate the Q-value of the selected actions using the target network (target_qnet)
                q_values_from_target_qnet = self.target_qnet(batch_n, training=False).numpy()
                q_values_chosen = q_values_from_target_qnet[np.arange(self.params['batch_size']), actions_from_qnet]

                # Step 3: Calculate the target Q-values considering terminal states
                self.yj = batch_r + (1 - batch_nt) * self.params['discount'] * q_values_chosen
                # Double DQN

            elif self.params['DQN_or_DDQN']=='DQN':
                #### DQN ###
                self.yj = batch_r + (1 - batch_nt)*self.params['discount']*np.max(self.target_qnet(batch_n, training=False).numpy(),axis=1)
                #### DQN ###

            # Train with the minibatch
            if self.annealment_power < self.params['annealment_max']:
                self.annealment_power = min(self.params['annealment_max'], self.params['annealment_base'] * self.params['annealment_growthrate']**float(self.qnet.train_step_counter.numpy()))

            beta_importance = importance ** self.annealment_power

            if TRAINING_MODE==True:
                self.cost_disp, td_errors = self.qnet.train_step(batch_s, batch_a, self.yj, beta_importance)
            else:
                self.cost_disp, td_errors = (0,0)

            # Update Priorities in Replaymemory
            self.replay_mem.set_priorities(indices, td_errors)

            # Update target network weights
            if self.local_cnt % self.params['update_target_freq'] == 0:
                self.target_qnet.set_weights(self.qnet.get_weights()) # hard update

    def get_onehot(self, actions):
        """ Create list of vectors with 1 values at index of action in list """
        return np.eye(4)[actions.astype(int)]


    def mergeStateMatrices(self, stateMatrices):
        """ Merge state matrices to one state tensor """
        stateMatrices = np.swapaxes(stateMatrices, 0, 2)
        total = np.zeros((7, 7))
        for i in range(len(stateMatrices)):
            total += (i + 1) * stateMatrices[i] / 6
        return total


    def getStateMatrices(self, state):
        observation = prepare_observation_matrix(
            state, NUM_CHANNELS, channel_height, channel_width,
            VISIBLE_LIMIT, AUDIBLE_LIMIT, CHANNEL_RADIUS,
            is_partially_observable=is_partially_observable, is_sliced=is_sliced)

        return observation


    def registerInitialState(self, state): # inspects the starting state

        # Reset reward
        self.last_score = 0
        self.current_score = 0
        self.last_reward = 0.
        self.ep_rew = 0

        # Reset state
        self.last_state = None
        self.current_state = np.transpose(self.getStateMatrices(state), (2, 1, 0))

        # Reset actions
        self.last_action = None

        # Reset vars
        self.last_terminal = None
        self.terminal = None
        self.won = True
        self.Q_global = []
        self.delay = 0

        # Reset trackers
        self.stepinep = 0 # Newly added to initialise it for debugging

        self.visitedTracker = VisitedTilesTracker(stalling_threshold=self.params['stalling_threshold'],
                                                  no_new_position_threshold=self.params['no_new_position_threshold'])

        # Next
        self.frame = 0
        self.numeps += 1

    def getAction(self, state):
        move = self.getMove(state)

        # Stop moving when not legal
        legal = state.getLegalActions(0)
        if move not in legal:
            legal.remove(Directions.STOP)
            move = random.choice(legal)

        return move
