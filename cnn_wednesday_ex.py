import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data",  one_hot = True)

# [0,0,1,0,0,0,0,0,0,0] = 2

n_classes = 10
batch_size = 128

x = tf.placehodler('float', [None, 784])
y = tf.placeholder('float')

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")


def cnn(x):
    weight = {"W_conv1": tf.Variable(tf.random_normal([5,5,1,32])),
              "W_conv2": tf.Variable(tf.random_normal([5,5,32,64])),
              "W_fc": tf.Variable(tf.random_normal([7*7*64,1024])),
              "out": tf.Variable(tf.random_normal([1024, n_classes]))
                }

    biases = {"b_conv1": tf.Variable(tf.random_normal([32])),
              "b_conv2": tf.Variable(tf.random_normal([64])),
              "b_fc": tf.Variable(tf.random_normal([1024])),
              "out": tf.Variable(tf.random_normal([n_classes]))
                }

    x = tf.reshape(x, shape=[-1,28,28,1])

    conv1 = tf.nn.relu(conv2d(x, weight["W_conv1"]) + biases["b_conv1"])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weight["W_conv2"]) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2[-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weights["W_fc"]) +biases["b_fc"])

    output = tf.matmul(fc, weights["out"]) + biases['out']


def train_neural_network(x):
    prediction = cnn(x)
    '''softmax_cross_entropy_with_logits takes output and turns them into
     probabilities (between 1 and 0) than compares how far off from the
     prediction they are. then you do the reduce mean function to get how far
     off the prediction is from the actual number i.e. the "cost"'''
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):

                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


train_neural_network(x)
