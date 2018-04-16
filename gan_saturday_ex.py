import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os



y_dim = 10

def xavier_init(size):
    '''stddev is standard deviation.
    This function ensures the weights are initialized in the goldylocks zone'''
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# Tom Hanks == Descriminator Net
with tf.name_scope("Discriminator_Variables"):
    X = tf.placeholder(tf.float32, shape=[None, 784], name='x')

    D_W1 = tf.Variable(xavier_init([784+y_dim, 128]))
    D_b1 = tf.Variable(tf.zeros(shape=[128]))

    D_W2 = tf.Variable(xavier_init([128, 1]))
    D_b2 = tf.Variable(tf.zeros(shape=[1]))


theta_D = [D_W1, D_W2, D_b1, D_b2]

# tf.summary.histogram('d_1_weights', D_W1)
# tf.summary.histogram('d_2_weights', D_W2)

# Leonardo == Generaotor
Y = tf.placeholder(tf.float32, shape=[None, y_dim])
with tf.name_scope('Generator_Variables'):
    Z = tf.placeholder(tf.float32, shape=[None, 100], name='z')

    G_W1 = tf.Variable(xavier_init([100+y_dim, 128]))
    G_b1 = tf.Variable(tf.zeros(shape=[128]))

    G_W2 = tf.Variable(xavier_init([128, 784]))
    G_b2 = tf.Variable(tf.zeros(shape=[784]))

my_array = [1,2]
my_array[1]
tf.summary.histogram('g_1_weights', G_W1)
# tf.summary.histogram('g_2_weights', G_W2)
theta_G = [G_W1, G_W2, G_b1, G_b2]

def generator(z, y):
    '''takes 100-dimensional vector and
        returns 784-dimensional vector,
        which is MNIST image (28x28)'''
    new_inputs = tf.concat(values=[z,y], axis=1)
    G_h1 = tf.nn.relu(tf.matmul(new_inputs, G_W1) + G_b1)

    output_var = tf.matmul(G_h1, G_W2) + G_b2
    G_guess = tf.nn.sigmoid(output_var)

    return G_guess


def descriminator(x, y):
    print("meow")
    new_inputs = tf.concat(values=[x,y], axis=1)
    D_h1 = tf.nn.relu(tf.matmul(new_inputs, D_W1) + D_b1)
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



with tf.name_scope("generator_net"):
    G_sample = generator(Z, Y)

with tf.name_scope('discriminator_net'):
    D_real, D_logit_real = descriminator(X, Y)
    D_fake, D_logit_fake = descriminator(G_sample, Y)

with tf.name_scope('loss_functions'):
    D_loss = -tf.reduce_mean( tf.log(D_real) + tf.log(1. - D_fake) )
    G_loss = -tf.reduce_mean( tf.log(D_fake) )

# tf.summary.histogram('D_Loss', D_loss)
# tf.summary.histogram('D_Loss', G_loss)

with tf.name_scope("gradient_descent"):
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)




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
            y_sample[:, 4] = 1
            samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim), Y:y_sample})

            fig = plot(samples)

            plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')


            i += 1
            plt.close(fig)



        X_mb, y_mb = mnist.train.next_batch(mb_size)


        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim), Y:y_mb})
        tf.summary.histogram('d_loss', D_loss_curr)
        # merged_summary = tf.summary.merge_all()
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim),Y:y_mb})

        # writer.add_summary(sums, it)

        if it % 1000 == 0:
            print('Iter: {}'.format(it))
            print('D loss: {:.4}'. format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))

            print()
