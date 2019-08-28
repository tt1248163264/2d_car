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

                self.a_prob, self.v, self.a_params, self.c_params = self._build_net(scope)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(tf.log(self.a_prob + 1e-5) * tf.one_hot(self.a_his, N_A, dtype=tf.float32), axis=1, keep_dims=True)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5),
                                             axis=1, keep_dims=True)  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
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
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            a_prob = tf.layers.dense(l_a, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return a_prob, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, car_state, last_action, image):  # run by a local
        '''
        选择动作：通过局部路径规划或者网络模型计算两种方式
        
        由于在dwa算法中，使用的信息是数据，需要使用到:
        obs,vel,goal
        state[x,y,r,liner,angular]四方面的信息，这与网络模型带入的参数不一样
        ---
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
        buffer_state, buffer_action, buffer_reward, buffer_image = [], [], [], []

        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            state, image = self.agent.reset()
            action = [0, 0]
            ep_r = 0
            while True:
                action = self.AC.choose_action(self.agent.car_state, action, image)
                image, reward, done = self.agent.step(action)
                if done: r = -5
                ep_r += r
                buffer_state.append(s)
                buffer_action.append(a)
                buffer_reward.append(r)
                buffer_image.append()

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