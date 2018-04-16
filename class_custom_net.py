import tensorflow as tf
import numpy as np
from load_image_data import LoadData
from tqdm import tqdm

imported_data = LoadData("../../Downloads/the-simpsons-characters-dataset/simpsons_dataset").training_data

### Hyper Paramaters



num_classes = 48
batch_size = 100
num_input_pixels = 80*80

x = tf.placeholder(tf.float32, [None,80,80,3])
y = tf.placeholder(tf.float32, [None, num_classes])

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


##Define the Model
def neural_network_model(x):
    hidden_1 = {"weights": tf.Variable(tf.random_normal([5,5,3,64])),
                "biases": tf.Variable(tf.random_normal([64]))
                }

    hidden_2 = {"weights": tf.Variable(tf.random_normal([5,5,64,128])),
                "biases": tf.Variable(tf.random_normal([128]))
                }

    hidden_3 = {"weights":tf.Variable(tf.random_normal([5,5,128,256])),
                "biases":tf.Variable(tf.random_normal([256]))
                }

    # '''input divided by 2 for every max pooling layer 80 / (2 ** 3)'''
    fc_layer = {"weights":tf.Variable(tf.random_normal([10*10*256,1024])),
                        "biases":tf.Variable(tf.random_normal([1024]))}

    output = {'weights': tf.Variable(tf.random_normal([1024,num_classes])),
              'biases':tf.Variable(tf.random_normal([num_classes]))
                }



    x = tf.reshape(x, shape=[-1,80,80,3])

    conv1 = tf.nn.relu(conv2d(x,hidden_1["weights"])+hidden_1["biases"])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1,hidden_2["weights"])+hidden_2["biases"])
    conv2 = maxpool2d(conv2)

    conv3 = tf.nn.relu(conv2d(conv2,hidden_3["weights"])+hidden_3["biases"])
    conv3 = maxpool2d(conv3)

    fc = tf.reshape(conv3, [-1, 10*10*256])
    fc = tf.nn.relu(tf.matmul(fc,fc_layer["weights"])+fc_layer["biases"])

    output_layer = tf.matmul(fc, output['weights'])+output["biases"]

    return output_layer


#Train the model
def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer= tf.train.AdamOptimizer().minimize(cost)



    print("all imported data", len(imported_data[0]))
    train_size = int(len(imported_data[0])*0.8)
    train_x = imported_data[0][:train_size]
    train_y = imported_data[1][:train_size]
    print("batch length: ", len(train_x))
    test_x = imported_data[0][train_size:]
    test_y = imported_data[1][train_size:]

    num_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(num_epochs):
            epoch_loss = 0
            start = 0
            print("Epoch", epoch, "started")
            for batch_num in tqdm(range(int(len(train_x)/batch_size))):
                end = start+batch_size

                train_x_batch= train_x[start:end]
                train_y_batch=train_y[start:end]

                _, loss = sess.run([optimizer, cost], feed_dict={x:train_x_batch, y:train_y_batch})

                start += batch_size
                epoch_loss += loss
            print("Epoch", epoch, "of", num_epochs, "finished.\nTotal Epoch Loss: ", epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))


 # Compute the output size of the CNN2 as a 1D array.
# size = image_size // 4 * image_size // 4 * depth



train_neural_network(x)































# Train The mod
