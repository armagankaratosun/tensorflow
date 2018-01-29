from __future__ import print_function
import tensorflow as tf

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)
print(node1, node2) # does not print the values 3 and 4,
# instead this prints nodes would produce 3.0 and 4.0

## For printing 3 and 4,

sess = tf.Session()
print (sess.run([node1,node2]))

## adding two nodes, (values)

node3 = tf.add(node1, node2)
print ("node3:", node3)
print("sess.run(node3):", sess.run(node3))

## now use placeholders to make this more interesting

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a+b # + provides a shortcut for tf.add(a,b)

print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1,3], b:[2,4]}))

############################################
# We generated an adder node to add values #
#          - -      - -      -  -          #
#        -  a  - + - b - =  - R -          #
#        - - - -   - - -    - - -          #
############################################

## Now add another operation,

add_and_triple = adder_node*3.
print(sess.run(add_and_triple, {a:3,b:4.5}))

# now lets see the differance btw. tf.constant and tf.variable

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b

init = tf.global_variables_initializer()# we have to initialize the variables
sess.run(init)
print(sess.run(linear_model, {x:[1,2,3,4]}))

# Now lets write a loss funciton for our linear model
y=tf.placeholder(tf.float32)
squared_deltas= tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

# loss funciton value is too big, lets reduce it with gradient descent
optimizer=tf.train.GradientDescentOptimizer(0.01)
train=optimizer.minimize(loss)

for i in range(1000): # 1000 iterations
    sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

print(sess.run([W,b]))
