# Import
import tensorflow as tf, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Import data
data = pd.read_csv('data_stocks.csv'); data = data.drop(['DATE'], 1)
n = data.shape[0]; p = data.shape[1]; data = data.values

# Training and test data
train_start = 0; train_end = int(np.floor(0.8*n))
test_start = train_end + 1; test_end = n
data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]

# Scale data
scaler = MinMaxScaler(feature_range=(-1, 1)); scaler.fit(data_train)
data_train = scaler.transform(data_train); data_test = scaler.transform(data_test)

# Build X and y
X_train = data_train[:, 1:]; y_train = data_train[:, 0]
X_test = data_test[:, 1:]; y_test = data_test[:, 0]

#level-sizes
levels = [X_train.shape[1], 1024, 512, 256, 128, 1]

# Session
net = tf.InteractiveSession()

# Placeholder
X = tf.placeholder(dtype=tf.float32, shape=[None, levels[0]])
Y = tf.placeholder(dtype=tf.float32, shape=[None])

# Initializers
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)

# Hidden layer
actn_fn = tf.nn.relu
hidden_1 = tf.layers.dense(inputs=X, units=levels[1], activation=actn_fn, kernel_initializer = weight_initializer)
hidden_2 = tf.layers.dense(inputs=hidden_1, units=levels[2], activation=actn_fn, kernel_initializer = weight_initializer)
hidden_3 = tf.layers.dense(inputs=hidden_2, units=levels[3], activation=actn_fn, kernel_initializer = weight_initializer)
hidden_4 = tf.layers.dense(inputs=hidden_3, units=levels[4], activation=actn_fn, kernel_initializer = weight_initializer)
out      = tf.transpose(tf.layers.dense(inputs=hidden_4, units=levels[5], activation=actn_fn, kernel_initializer = weight_initializer))
#hidden_1 = actn_fn(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
#hidden_2 = actn_fn(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
#hidden_3 = actn_fn(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
#hidden_4 = actn_fn(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

# Output layer (transpose!)
#out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))

# Cost function
mse = tf.reduce_mean(tf.squared_difference(out, Y))

# Optimizer
opt = tf.train.AdamOptimizer().minimize(mse)

# Init
net.run(tf.global_variables_initializer())

# Setup plot
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(y_test)
line2, = ax1.plot(y_test * 0.5)
plt.show()

# Fit neural net
batch_size = 1024
mse_train = []
mse_test = []

# Run
epochs = 20
for e in range(epochs):

    # Shuffle training data
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    # Minibatch training
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = X_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]
        # Run optimizer with batch
        net.run(opt, feed_dict={X: batch_x, Y: batch_y})

        # Show progress
        if np.mod(i, 50) == 0:
            # MSE train and test
            mse_train.append(net.run(mse, feed_dict={X: X_train, Y: y_train}))
            mse_test.append(net.run(mse, feed_dict={X: X_test, Y: y_test}))
            print('MSE Train: ', mse_train[-1])
            print('MSE Test: ', mse_test[-1])
            # Prediction
            pred = net.run(out, feed_dict={X: X_test})
            line2.set_ydata(pred)
            plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
            plt.pause(0.01)
