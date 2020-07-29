import tensorflow as tf
import sonnet as snt
import matplotlib.pyplot as plt

print("TensorFlow version {}".format(tf.__version__))
print("Sonnet version {}".format(snt.__version__))

batch_size = 100

mnist = tf.keras.datasets.mnist.load_data()

(x_train, y_train), (x_test, y_test) = mnist

x_train, x_test = (x_train.reshape(-1, 28, 28, 1)/255 - .5)*2, (x_test.reshape(-1, 28, 28, 1)/255 - .5)*2

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
print(y_train[0])
plt.imshow(x_train[0])
plt.show()


train_dataset = tf.data.Dataset.from_tensor_slices((tf.constant(x_train), tf.constant(y_train)))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices((tf.constant(x_test), tf.constant(y_test)))
test_dataset = test_dataset.batch(batch_size)

class MLP(snt.Module):

  def __init__(self):
    super(MLP, self).__init__()
    self.flatten = snt.Flatten()
    self.hidden1 = snt.Linear(1024, name="hidden1")
    self.hidden2 = snt.Linear(1024, name="hidden2")
    self.logits = snt.Linear(10, name="logits")

  def __call__(self, images):
    output = self.flatten(images)
    output = tf.nn.relu(self.hidden1(output))
    output = tf.nn.relu(self.hidden2(output))
    output = self.logits(output)
    return output

def fit(model, num_epochs, binary_gradients):

    optimizer = snt.optimizers.SGD(0.1)

    for epoch in range(num_epochs):
        accuracy_avg = 0
        loss_avg = 0
        n = 0

        for x, y in train_dataset:
            y = tf.cast(y, tf.int64)

            with tf.GradientTape() as tape:
                logits = model(x)
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
                loss = tf.reduce_mean(loss)
            variables = model.trainable_variables
            gradients = tape.gradient(loss, variables)

            if binary_gradients:
                grads = []
                for g in gradients:
                    g = tf.cast(tf.where(g > 0, 0.001, -0.001), tf.float64)
                    grads.append(g)

                gradients = tuple(grads)

            optimizer.apply(gradients, variables)

            prediction = tf.argmax(logits, axis=-1)
            accuracy = tf.reduce_mean(tf.cast(prediction == y, tf.float32))

            accuracy_avg += accuracy
            loss_avg += loss
            n += 1

        print('Epoch:', epoch, ', loss:', loss_avg/n, ', acc:', accuracy_avg/n)

def test_accuracy(model):
    accuracy_avg = 0
    n = 0

    for x, y in test_dataset:
        y = tf.cast(y, tf.int64)

        logits = model(x)

        prediction = tf.argmax(logits, axis=-1)
        accuracy = tf.reduce_mean(tf.cast(prediction == y, tf.float32))

        accuracy_avg += accuracy
        n += 1

    print('The test accuracy is:', accuracy_avg/n)

control_model = MLP()
fit(control_model, 10, False)
test_accuracy(control_model)

print('Now with binary gradients:')

binary_model = MLP()
fit(binary_model, 10, True)
test_accuracy(binary_model)

# There seems to be 1% test difference between the models, but it's still quite an
# impressive result for a binary gradient network
