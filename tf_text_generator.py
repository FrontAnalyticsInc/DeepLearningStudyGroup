import numpy as np, tflearn, sys, tensorflow as tf
from tensorflow.python.keras import utils
input = open("pos.txt").read().lower()
chars = sorted(list(set(input)))
charint = dict((char,ints) for ints, char in enumerate(chars))
intchar = dict((ints,char) for ints, char in enumerate(chars))
print(charint)
print("\n\n\n")
print(intchar)
filename = 'mine'
seqlen = 100
lstmhid=320
keeprate = 0.80
train = []
true = []
tf.reset_default_graph()
for i in range(0, len(input)-seqlen, 1):
    train.append([charint[char] for char in input[i:i+seqlen]])
    true.append(charint[input[i+seqlen]])

X = np.reshape(train, (len(train), seqlen, 1))/float(len(chars))

y = utils.to_categorical(true)
net = tflearn.input_data(shape=(None, X.shape[1], X.shape[2]))

net = tflearn.lstm(net, lstmhid)


net = tflearn.fully_connected(net, y.shape[1], activation='softmax')
net = tflearn.regression(net, optimizer='adam',
 learning_rate=0.005, loss ='categorical_crossentropy')
model = tflearn.DNN(net, checkpoint_path = 'charmodel/model.tfl.ckpt')
model.fit(X, y, snapshot_epoch=True,
snapshot_step=5000, n_epoch=20, batch_size = 128)
# model.save(filename)
#
# for _ in range(5):
#     p = train[np.random.randint(0,len(train)-1)]
#     print("Seed:")
#     print("\"",''.join([intchar[value] for value in p]), "\"")
#     for _ in range(100):
#         sys.stdout.write((intchar[np.argmax(model.predict((np.reshape(p, (1, len(p), 1))/float(len(chars)))))]))
#         p.append(np.argmax(model.predict((np.reshape(p, (1, len(p), 1))/float(len(chars))))))
#         p = p[1:len(p)]
#     print("\n============================\n")
