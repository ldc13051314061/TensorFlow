#draw
#2018.03.14
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs,in_size,out_size,activation_function = None):
    Weight = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b = tf.matmul(inputs,Weight) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

#make data

x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) -0.5+ noise

# plt.figure(1)
# plt.scatter(x_data,y_data)
# plt.show(1)

xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

#add layer

L1 = add_layer(xs,1,10,activation_function=tf.nn.relu)

prediction = add_layer(L1,10,1,activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(prediction - ys),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()


for i in range(1001):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i % 50 == 0:
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass

        prediction_value = sess.run(prediction,feed_dict={xs:x_data})
        lines = ax.plot(x_data,prediction_value,'r-',lw=3)
        plt.pause(1)

        loss_value = sess.run(loss,feed_dict={xs:x_data,ys:y_data})
        print(i,'step','loss=',loss_value)
plt.savefig('train.png')



'''
0 step loss= 0.0891868
50 step loss= 0.01055
100 step loss= 0.00787424
150 step loss= 0.00711802
200 step loss= 0.00669106
250 step loss= 0.00632066
300 step loss= 0.00598536
350 step loss= 0.00563981
400 step loss= 0.00533036
450 step loss= 0.00505407
500 step loss= 0.00481931
550 step loss= 0.00459252
600 step loss= 0.00441038
650 step loss= 0.00424548
700 step loss= 0.00409879
750 step loss= 0.00396881
800 step loss= 0.00383594
850 step loss= 0.003729
900 step loss= 0.00362888
950 step loss= 0.00354343
1000 step loss= 0.00345957
'''


