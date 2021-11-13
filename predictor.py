import numpy as np
import pandas as pd
import tensorflow as tf

import pickle
import glob

from module import QDN
class Predictor():
    def __init__(self, n_states, n_actions, batch_size, lr, epsilon, gamma, target_replace_iter, memory_capacity):

        self.eval_net, self.target_net = QDN(), QDN()

        # 每個 memory 中的 experience 大小為 (state + reward + action + next state )
        self.memory = np.zeros((memory_capacity, n_states * 5 + 2)) 

        self.memory_counter = 0
        self.learn_step_counter = 0 # 讓 target network 知道什麼時候要更新

        self.n_states = n_states
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.target_replace_iter = target_replace_iter
        self.memory_capacity = memory_capacity

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

        #self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)


    def choose_action(self, state):
        #x = torch.unsqueeze(torch.FloatTensor(state), 0)

        # epsilon-greedy
        if np.random.uniform() < self.epsilon: # 隨機
            action = np.random.randint(0, self.n_actions)
        else: # 根據現有 policy 做最好的選擇
            actions_value = self.eval_net(state) # 以現有 eval net 得出各個 action 的分數
            action = tf.math.argmax(tf.reshape(actions_value, -1)) # 挑選最高分的 action

        return int(action)


    def store_transition(self, state, action, reward, next_state):
        # 打包 experience
        transition = np.hstack((state, [action, reward], next_state))

        # 存進 memory；舊 memory 可能會被覆蓋
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1


    @tf.function
    def train_step(self, b_state, b_action, b_reward, b_next_state):

        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            q_eval = tf.gather(self.eval_net(b_state), b_action, axis=1) # 重新計算這些 experience 當下 eval net 所得出的 Q value
            q_next = tf.stop_gradient(self.target_net(b_next_state)) # detach 才不會訓練到 target net
            q_target = b_reward + self.gamma * tf.math.reduce_max(q_next,1) # 計算這些 experience 當下 target net 所得出的 Q value

            loss = self.loss_object(q_eval, q_target)
        gradients = tape.gradient(loss, self.eval_net.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.eval_net.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(q_eval, q_target)
        print('gggggggggggggggggg')
        # 每隔一段時間 (target_replace_iter), 更新 target net，即複製 eval net 到 target net
        self.learn_step_counter += 1
        checkpoint = tf.train.Checkpoint(self.eval_net)
        save_path = checkpoint.write(f'/ckpt/training_checkpoints_{self.learn_step_counter}')
        if self.learn_step_counter % self.target_replace_iter == 0:
            checkpoint = tf.train.Checkpoint(self.target_net)
            checkpoint.read(save_path)


    def run_epoch(self):
        ''' traning model from train and evaluation by valid
        Args:
            train, valid (torch dataset)
        '''
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_state = tf.reshape(tf.constant(b_memory[:, :self.n_states*4], dtype=float), [self.batch_size, 80, 80, 4])
        b_action = tf.constant(b_memory[:, self.n_states*4:self.n_states*4+1].astype(int))
        b_reward = tf.constant(b_memory[:, self.n_states*4+1:self.n_states*4+2], dtype=float)
        b_next_state = tf.reshape(tf.constant(b_memory[:, -self.n_states:], dtype=float), [self.batch_size, 80, 80, 1])

        # Reset the metrics at the start of the next epoch
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()

        self.train_step(b_state, b_action, b_reward, b_next_state)
        '''
        print(
            f'Loss: {self.train_loss.result()}, '
            f'Accuracy: {self.train_accuracy.result() * 100}, '
            f'Train Loss: {self.train_loss.result()}, '
            f'Train Accuracy: {self.train_accuracy.result() * 100}'
        )
        '''

    
    def load_model(self, path):
        ''' load model_state_dict from path
        '''
        folder = '/'.join(path.split('/')[:-1])
        epoch = path.split('/')[-1]
        model_path = glob.glob('{0}/*'.format(folder))
        
        for p in model_path:
          
          if epoch in p:
            print(p)
            break
        self.model.load_state_dict( torch.load(p) )


    def save_hist(self, hist, path):
        ''' save history data to path
        '''
        with open(path, 'wb') as f:
            pickle.dump(hist, f)

    def save_result(self, predict, result_path):
        ''' save prediction to path
        '''
        pass

