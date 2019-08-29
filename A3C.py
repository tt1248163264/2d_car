'''
This is the A3C algo for my pygame_car_env
'''
import multiprocessing 
import threading
import os
import shutil
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pygame_car_env import CarEnv
from tensorflow.contrib import slim

OUTPUT_GRAPH = True
LOG_DIR = './log'

N_WORKS = multiprocessing.cpu_count()
MAX_GLOBAL_EP = 1000
GLOBAL_NET_SCOPE = 'global'
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
ENTROPY_BETA = 0.001

LR_A = 0.001
LR_C = 0.001
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0

env = CarEnv()
n_liner = env.n_liner
n_angular = env.n_angular
n_agent = env.n_agent
image_shape = env.image_shape
# liner, angular
n_state = 2

config = tf.ConfigProto(gpu_options = tf.GPUOptions(allow_growth = True))

class ACNet(object):
    def __init__(self, scope, globalAC=None):

        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.state_input = tf.placeholder(tf.float32,[None, n_state], 'State')
                self.image_input = tf.placeholder(tf.float32,[None,image_shape[0],image_shape[1],image_shape[2]],'Image')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.state_input = tf.placeholder(tf.float32,[None, n_state], 'State')
                self.iamge_input = tf.placeholder(tf.float32,[None,image_shape[0],image_shape[1],image_shape[2]],'Image')
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A_his')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                self.liner_prob, self.angular_prob, self.liner_v, self.angular_v, self.a_params, self.c_params = self._build_net(scope)
                # 可能需要影响因子来控制线速度、角速度的优先性
                self.v = tf.add(self.liner_v, self.angular_v, name='V')
                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('a_loss'):
                    liner_log_prob = tf.reduce_sum(tf.log(self.liner_prob + 1e-5) * tf.one_hot(self.a_his[0], n_liner, dtype=tf.float32), axis=1, keep_dims=True)
                    angular_log_prob = tf.reduce_sum(tf.log(self.angular_prob + 1e-5) * tf.one_hot(self.a_his[1], n_liner, dtype=tf.float32), axis=1, keep_dims=True)
                    exp_liner_v = liner_log_prob * tf.stop_gradient(td)
                    exp_angular_v = angular_log_prob * tf.stop_gradient(td)
                    entropy_liner = -tf.reduce_sum(self.liner_prob * tf.log(self.liner_prob + 1e-5),
                                            axis=1, keep_dims=True)  # encourage exploration
                    entropy_angular = -tf.reduce_sum(self.angular_prob * tf.log(self.angular_prob + 1e-5),
                                            axis=1, keep_dims=True)  # encourage exploration
                    self.exp_v = ENTROPY_BETA * (entropy_angular + entropy_liner) + exp_liner_v + exp_angular_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):
        with tf.variable_scope('actor'):
            with tf.variable_scope('image'):
                # 160 160 3
                image_a = slim.conv2d(self.image_input, 32, [3, 3], stride=2, padding='VALID', scope='conv_1')
                # 79 79 32
                image_a = slim.conv2d(image_a, 64, [3, 3], stride=1, padding='SAME', scope='conv_2')
                # 79 79 64
                image_a = slim.max_pool2d(image_a, [3, 3], stride=2, padding='VALID', scope='max_pool_1')
                # 39 39 64
                image_a = slim.conv2d(image_a, 80, [3, 3], stride=2, padding='VALID', scope='conv_3')
                # 19 19 80
                image_a = slim.conv2d(image_a, 192,[3, 3], stride=1, padding='VALID', scope='conv_4')
                # 19 19 192
                image_a = slim.max_pool2d(image_a, [3, 3], stride=2, padding='VALID', scope='max_pool_2' )
                # 9 9 192
                image_a = slim.conv2d(image_a, 192,[3, 3], stride=1, padding='SAME',scope='conv_5')
                # 9 9 192
            with tf.variable_scope('state'):
                state_a = slim.fully_connected(self.state_input, 192, activation_fn=tf.nn.relu, scope='state_fc') 
                state_a = tf.reshape(state_a, [-1, 1, 1, 192])
                state_a = tf.tile(state_a, [1, 9, 9, 1])
            with tf.variable_scope('connect'):
                # 9 9 192
                net = tf.add(image_a, state_a)
            with tf.variable_scope('feature'):
                net = slim.conv2d(net, 192, [1, 1], stride=1, padding='VALID', scope='conv_6')
                # 9 9 192
                net = slim.conv2d(net, 192, [3, 3], stride=1, padding='VALID', scope='conv_7')
                # 7 7 256
                net = slim.conv2d(net, 256, [1, 1], stride=1, padding='VALID', scope='conv_7')
                # 7 7 256
                net = slim.conv2d(net, 256, [3, 3], stride=1, padding='VALID',scope='conv_8')
                # 5 5 256
                net = slim.flatten(net, scope='flatten')
                net = slim.dropout(net, 0.8, scope='dropout')
            with tf.variable_scope('liner_out'):
                liner_a = slim.fully_connected(net, num_outputs=512, activation_fn=tf.nn.relu, scope='liner_fc_1')
                liner_a = slim.fully_connected(liner_a, num_outputs=n_liner, activation_fn=None, scope='liner_fc_2')
                liner_prob = tf.nn.softmax(liner_a, name='liner')
            with tf.variable_scope('angular_out'):
                angular_a = slim.fully_connected(net, num_outputs=512, activation_fn=tf.nn.relu, scope='angular_fc_1')
                angular_a = slim.fully_connected(angular_a, num_outputs=n_angular, activation_fn=None, scope='angular_fc_2')
                angular_prob = tf.nn.softmax(angular_a,name='angular')
        with tf.variable_scope('critic'):
            with tf.variable_scope('image'):
                # 160 160 3
                image_c = slim.conv2d(self.image_input, 32, [3, 3], stride=2, padding='VALID', scope='conv_1')
                # 79 79 32
                image_c = slim.conv2d(image_c, 64, [3, 3], stride=1, padding='SAME', scope='conv_2')
                # 79 79 64
                image_c = slim.max_pool2d(image_c, [3, 3], stride=2, padding='VALID', scope='max_pool_1')
                # 39 39 64
                image_c = slim.conv2d(image_c, 80, [3, 3], stride=2, padding='VALID', scope='conv_3')
                # 19 19 80
                image_c = slim.conv2d(image_c, 192,[3, 3], stride=1, padding='VALID', scope='conv_4')
                # 19 19 192
                image_c = slim.max_pool2d(image_c, [3, 3], stride=2, padding='VALID', scope='max_pool_2' )
                # 9 9 192
                image_c = slim.conv2d(image_c, 192,[3, 3], stride=1, padding='SAME',scope='conv_5')
                # 9 9 192
            with tf.variable_scope('state'):
                state_c = slim.fully_connected(self.state_input, 192, activation_fn=tf.nn.relu, scope='state_fc') 
                state_c = tf.reshape(state_c, [-1, 1, 1, 192])
                state_c = tf.tile(state_c, [1, 9, 9, 1])
            with tf.variable_scope('connect'):
                # 9 9 192
                net = tf.add(image_c, state_c)
            with tf.variable_scope('feature'):
                net = slim.conv2d(net, 192, [1, 1], stride=1, padding='VALID', scope='conv_6')
                # 9 9 192
                net = slim.conv2d(net, 192, [3, 3], stride=1, padding='VALID', scope='conv_7')
                # 7 7 256
                net = slim.conv2d(net, 192, [1, 1], stride=1, padding='VALID', scope='conv_7')
                # 7 7 256
                net = slim.conv2d(net, 192, [3, 3], stride=1, padding='VALID',scope='conv_8')
                # 5 5 256
                net = slim.flatten(net, scope='flatten')
            with tf.variable_scope('liner_v'):
                liner_v = slim.fully_connected(net, num_outputs=1, activation_fn=None, scope='liner_c')
            with tf.variable_scope('angular_v'):
                angular_v = slim.fully_connected(net, num_outputs=1, activation_fn=None, scope='angular_c')
        
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return liner_prob, angular_prob, liner_v, angular_v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, car_state, last_action, image):  # run by a local
        '''
        选择动作：通过局部路径规划或者网络模型计算两种方式
        ---
        由于在dwa算法中，使用的信息是数据，需要使用到:
        obs,vel,goal
        state[x,y,r,liner,angular]四方面的信息，这与网络模型带入的参数不一样
        
        param dwa:[state,vel,goal,obs]
        param net:[state(liner,angular), image]
        '''
        prob_weights = SESS.run(self.a_prob, feed_dict={self.s: s[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action


class Worker(object):
    def __init__(self, agent_id, globalAC):
        self.agent = env.agents[agent_id]
        self.id = agent_id
        self.name = 'agent_{0}'.format(agent_id)
        self.AC = ACNet(name, globalAC)

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        
        #memory = []
        buffer_state, buffer_action, buffer_reward, buffer_image = [], [], [], []

        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            state, image = self.agent.reset()
            action = [0, 0]
            ep_r = 0
            while True:
                action = self.AC.choose_action(self.agent.car_state, action, image)
                image, reward, is_collision, done = self.agent.step(action)
                
                ep_r += reward
                #memory.append([state,action,reward,image])
                buffer_state.append(state)
                buffer_action.append(action)
                buffer_reward.append(reward)
                buffer_image.append(image)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = sess.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_reward[::-1]:    # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)

                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * ep_r)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                          )
                    GLOBAL_EP += 1
                    break

def train():
    pass

if __name__ == "__main__":
    sess = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.device('/cpu:0'):
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
        workers = []
        # Create worker
        for i in range(n_agent):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(i, GLOBAL_AC))

    COORD = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, sess.graph)

    worker_threads = []
    for work in workers:
        job = lambda:work.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)