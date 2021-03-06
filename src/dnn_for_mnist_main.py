# refer http://goodtogreate.tistory.com/entry/MNIST-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%85%8B%EC%9D%84-%EC%9D%B4%EC%9A%A9%ED%95%9C-%EC%86%90%EA%B8%80%EC%94%A8-%EC%9D%B8%EC%8B%9D-Deep-Neural-Network-%EA%B5%AC%ED%98%84

import tensorflow as tf
import input_data

learning_rate = 0.01
training_epochs = 15
batch_size = 100
display_step = 1

mnist = input_data.read_data_sets("./MNIST_DATA" , ont_hot=True)
# tensorflow graph input
X = tf.placeholder('float', [None, 784])
Y = tf.placeholder('float', [None, 10 ]) # 0-9 digits recognition -> 10 classes

# set model weights
W1 = tf.Variable(tf.random_normal([784, 256]))
W2 = tf.Variable(tf.random_normal([256, 256]))
W3 = tf.Variable(tf.random_normal([256, 10]))

B1 = tf.Variable(tf.random_normal([256]))
B2 = tf.Variable(tf.random_normal([256]))
B3 = tf.Variable(tf.random_normal([10]))

# Consturct model
L1 = tf.nn.relu(tf.add(tf.matmul(X,  W1) , B1) ) 
L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2) , B2) )
hypothesis = tf.add(tf.matmul(L2, W3), B3) # No need to use softmax here   

# Define loss and optimizer
       