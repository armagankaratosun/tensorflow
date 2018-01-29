# Trainable linear regression model with TensorFlow.

import tensorflow as tf

# Model Parameters: linear_model = W*x + b
W= tf.Variable([.3], dtype=tf.float32)
b= tf.Variable([-.3], dtype=tf.float32)
# Model input and output
x=tf.placeholder(tf.float32)
linear_model = W*x + b
y=tf.placeholder(tf.float32)

# loss function
squared_deltas= tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
# optimizer
optimizer=tf.train.GradientDescentOptimizer(0.01)
train=optimizer.minimize(loss)
# training session
x_train= [1,2,3,4]
y_train= [0,-1,-2,-3]

# lets train!
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
for i in range(1000):
    sess.run(train, {x:x_train, y:y_train})

curr_W, curr_b, curr_loss = sess.run([W,b,loss], {x: x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
