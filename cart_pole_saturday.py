import numpy as np
try:
    import cPickle as pickle
except:
    import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import math

import tensorflow.contrib.slim as slim

try:
    xrange = xrange
except:
    xrange = range

import gym
env = gym.make('CartPole-v0')


#hyper paramaters

# Hyper Ps
H = 10
batch_size = 5
learning_rate = 1e-2
gamma = 0.99

D = 4 #input dimensions

tf.reset_default_graph()


observations = tf.placeholder(tf.float32, [None,D], name='observation')
W1 = tf.get_variable("W1", shape=[D,H], initializer=tf.contrib.layers.xavier_initializer())

layer_1 = tf.nn.relu(tf.matmul(observations,W1))

W2 = tf.get_variable("W2", shape=[H,1], initializer=tf.contrib.layers.xavier_initializer())

score = tf.matmul(layer_1, W2)
probability = tf.nn.sigmoid(score)

tvars = tf.trainable_variables()
input_y = tf.placeholder(tf.float32,[None,1], name="actions")
advantages = tf.placeholder(tf.float32,name='reward_signal')

loglik = tf.log(input_y*(input_y - probability) + (1-input_y)*(input_y + probability))
loss = -tf.reduce_mean(loglik*advantages)
newGrads = tf.gradients(loss,tvars)


adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
w1Grad = tf.placeholder(tf.float32,name='batch_grad1')
w2Grad = tf.placeholder(tf.float32,name='batch_grad2')
batch_grad = [w1Grad, w2Grad]
updateGrads = adam.apply_gradients(zip(batch_grad, tvars))



def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

xs,hs,dlogs,drs,ys,tfps= [],[],[],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 1
total_episodes = 10000
init = tf.global_variables_initializer()

with tf.Session() as sess:
    rendering = False
    sess.run(init)
    observation = env.reset()

    gradBuffer = sess.run(tvars)
    for indx,grad in enumerate(gradBuffer):
        gradBuffer[indx] = grad * 0



    while episode_number <= total_episodes:

        if reward_sum > 975 or rendering == True:
            env.render()
            rendering = True


        x = np.reshape(observation, [1,D])

        tfprob = sess.run(probability,feed_dict={observations:x})
        action = 1 if np.random.uniform() < tfprob else 0

        xs.append(x)

        y = 1 if action == 0 else 0
        ys.append(y)

        observation, reward, done, info = env.step(action)
        reward_sum += reward

        drs.append(reward)

        if done:
            episode_number +=1

            epx = np.vstack(xs) # states / observation of states
            epy = np.vstack(ys) # inputs
            epr = np.vstack(drs) # #rewards
            tfp = tfps
            xs,hs,dlogps,drs,ys,tfps = [],[],[],[],[],[]

            discounted_epr = discount_rewards(epr)

            discounted_epr -= np.mean(discounted_epr)
            discounted_epr //= np.std(discounted_epr)

            tGrad = sess.run(newGrads, feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})

            for ix,grad in enumerate(tGrad):
                gradBuffer[ix] += grad

            if episode_number % batch_size == 0:
                sess.run(updateGrads, feed_dict={w1Grad: gradBuffer[0], w2Grad:gradBuffer[1]})

                for indx,grad in enumerate(gradBuffer):
                    gradBuffer[indx] = grad * 0

                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                print('Average reward for episode %f.  Total average reward %f.' % (reward_sum//batch_size, running_reward//batch_size))

                if reward_sum//batch_size > 200:
                    print("Task solved in", episode_number, "episodes!")
                    break

                reward_sum = 0

            observation = env.reset()

print(episode_number, 'episodes completed.')
