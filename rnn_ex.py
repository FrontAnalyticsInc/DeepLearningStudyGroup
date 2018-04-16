import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn
mnist = input_data.read_data_sets("/tmp/data", one_hot = True)
# print mnist


hm_epochs = 3
n_layers = 1
n_classes = 10
batch_size = 128
chunk_size = 28
n_chunks = 28
rnn_size = 128




x = tf.placeholder('float', [None, n_chunks,chunk_size])
y = tf.placeholder('float')

def make_cell(lstm_size):
    return tf.nn.rnn_cell.BasicLSTMCell(lstm_size, state_is_tuple=True)

def recurrent_neural_network(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
             'biases':tf.Variable(tf.random_normal([n_classes]))}

    # print("first x is: %s", x)
    #switches batch size with sequence size
    x = tf.transpose(x, [1,0,2])

    x = tf.reshape(x,[-1,chunk_size])
    '''splitting each image into 28 rows'''
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn.BasicLSTMCell(rnn_size, state_is_tuple=True)
    # stacked_lstm = rnn.MultiRNNCell([make_cell(rnn_size) for _ in range(n_layers)], state_is_tuple=True)
    # outputs, states = rnn.static_rnn(stacked_lstm, x, dtype=tf.float32)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

    return output

def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                epoch_x = epoch_x.reshape((batch_size,n_chunks,chunk_size))
                print(epoch_x)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch+1, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy:', accuracy.eval({x:mnist.test.images.reshape((-1,n_chunks,chunk_size)), y:mnist.test.labels}))



train_neural_network(x)
