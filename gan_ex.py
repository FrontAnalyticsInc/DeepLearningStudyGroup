import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os



X_dim = 784
z_dim = 100
y_dim = 10
h_dim = 128


def xavier_init(size):
    '''stddev is standard deviation.
    This function ensures the weights are initialized in the goldylocks zone'''
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# Descriminator Net
with tf.name_scope("Descriminator_Variables"):
    X = tf.placeholder(tf.float32, shape=[None, 784], name='x')


    D_W1= tf.Variable(xavier_init([X_dim+y_dim, 128]), name='D_W1')



    D_b1= tf.Variable(tf.zeros(shape=[128]), name='D_b1')

    D_W2= tf.Variable(xavier_init([128, 1]), name='D_W2')
    D_b2= tf.Variable(tf.zeros(shape=[1]), name='D_b2')

theta_D = [D_W1, D_W2, D_b1, D_b2]


# Generator Net


y = tf.placeholder(tf.float32, shape=[None, y_dim], name='Generator_Conditions')

with tf.name_scope("Generator_Variables"):
    Z = tf.placeholder(tf.float32, shape=[None, 100], name='z')

    G_W1 = tf.Variable(xavier_init([z_dim+y_dim, 128]), name='G_W1')

    G_b1 = tf.Variable(tf.zeros(shape=[128]), name='G_b1')

    G_W2 = tf.Variable(xavier_init([128,784]), name="G_W2")
    G_b2 = tf.Variable(tf.zeros(shape=[784]), name="G_b2")

theta_G = [G_W1, G_W2, G_b1, G_b2]



def generator(z, y):

    '''takes 100-dimensional vector and
        returns 784-dimensional vector,
        which is MNIST image (28x28)'''
    with tf.name_scope("generator_net"):
        inputs = tf.concat(values=[z, y], axis=1)
        G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)

        G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
        G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob

def descriminator(x, y):
    '''takes MNIST image(s) and return a scalar
     which represents a probability of real MNIST image.'''
    inputs = tf.concat(values=[x, y], axis=1)
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)

    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig



G_sample = generator(Z, y)
with tf.name_scope("descriminator_net"):
    D_real, D_logit_real = descriminator(X, y)
    D_fake, D_logit_fake = descriminator(G_sample, y)

# G_sample = generator(Z)
# with tf.name_scope("descriminator_net"):
#     D_real, D_logit_real = descriminator(X)
#     D_fake, D_logit_fake = descriminator(G_sample)



'''we use negative sign for the loss
 functions because they need to be maximized,
 whereas TensorFlow’s optimizer can only do minimization.'''
'''we want to made the descriminator make a mistake, so we maximize it's error'''
with tf.name_scope("D_loss"):
    D_loss = -tf.reduce_mean( tf.log(D_real) + tf.log(1. - D_fake) )
with tf.name_scope("G_loss"):
    G_loss = -tf.reduce_mean( tf.log(D_fake) )
'''G_loss is the error from passing in the generated image to the descriminator'''


# Only update D(X)'s parameters, so var_list = theta_D
with tf.name_scope("D_Train"):
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
with tf.name_scope("G_Train"):
    # Only update G(X)'s parameters, so var_list = theta_G
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)



mb_size = 128
Z_dim = 100

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./logs/1/train')
    writer.add_graph(sess.graph)

    if not os.path.exists('out/'):
        os.makedirs('out/')

    i = 0


    for it in range(1000000):
        if it % 1000 == 0:

            y_sample = np.zeros(shape=[16, y_dim])
            y_sample[:, 5] = 1
            samples = sess.run(G_sample, feed_dict={Z:sample_Z(16, Z_dim), y:y_sample})
            # samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})

            fig = plot(samples)
            plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)
            # Z_sample = sample_Z(n_sample, Z_dim)

# Create conditional one-hot vector, with index 5 = 1

        X_mb, y_mb = mnist.train.next_batch(mb_size)

        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim), y:y_mb})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim), y:y_mb})



        if it % 1000 == 0:
            print('Iter: {}'.format(it))
            print('D loss: {:.4}'. format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))
            print()
