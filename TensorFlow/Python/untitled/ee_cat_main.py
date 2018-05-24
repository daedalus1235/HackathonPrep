from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from functions import parse_csv, loss, grad


tf.enable_eager_execution()

print("TensorFlow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

# Dataset location
train_dataset_fp = '/home/charles/Documents/git-repos/HackathonPrep/TensorFlow/Python/untitled/era_training.csv'
print("Location of dataset file: {}".format(train_dataset_fp))

# Read dataset
train_dataset = tf.data.TextLineDataset(train_dataset_fp)
train_dataset = train_dataset.skip(1)                       # skip the first header row
train_dataset = train_dataset.map(parse_csv)                # parse each row
train_dataset = train_dataset.shuffle(buffer_size=1000)     # randomize
train_dataset = train_dataset.batch(8)


# View a single example entry from a batch
features, label = iter(train_dataset).next()
print("example features:", features[0])
print("example label:", label[0])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation="relu", input_shape=(1,)),  # input shape required
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(120)
])

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

# Train
train_loss_results = []
train_accuracy_results = []

num_epochs = 200

for epoch in range(num_epochs):
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()

    # Training loop - using batches of 8
    for x, y in train_dataset:
        # Optimize the model
        grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.variables),
                                  global_step=tf.train.get_or_create_global_step())

        # Track progress
        epoch_loss_avg(loss(model, x, y))  # add current batch loss
        # compare predicted label to actual label
        epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

    # end epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 50 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result()))

# Plot Loss and Accuracy
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)

plt.show()

# Test
test_fp = '/home/charles/Documents/git-repos/HackathonPrep/TensorFlow/Python/untitled/era_testing.csv'

test_dataset = tf.data.TextLineDataset(test_fp)
test_dataset = test_dataset.skip(1)             # skip header row
test_dataset = test_dataset.map(parse_csv)      # parse each row with the function created earlier
test_dataset = test_dataset.shuffle(1000)       # randomize
test_dataset = test_dataset.batch(8)           # use the same batch size as the training set

test_accuracy = tfe.metrics.Accuracy()

for (x, y) in test_dataset:
  prediction = tf.argmax(model(x), axis=1, output_type=tf.int32)
  test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))



predict_dataset = tf.convert_to_tensor([[5.36]])

predictions = model(predict_dataset)

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  print("Example {} prediction: {}".format(i, class_idx))


