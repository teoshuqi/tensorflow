import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Parameters
UPDATE_COUNT = 4  # no. of times to update loss for each EPOCHS
BUFFER_SIZE = 60000
BATCH_SIZE = 32
CLASSES = 10
INPUT_DIM = 28
VAL_SIZE = 0.2
EPOCHS = 200
LR = 9e-5
DECAY_STEPS = 10000
DECAY_RATE = 0.95
DELTA = 0.1
PATIENCE = 30


# Variables
wait = 0  # Early Stopping
wait_loss = []
history = {'train_acc': [], 'train_loss': [], 'val_loss': [], 'val_acc': [], 'optimizer': []}
model_dir = './'
MODEL_FILENAME = model_dir + 'fcnn_model.h5'
METRICS_FILENAME = model_dir + 'fcnn_metrics.png'
RESULTS_FILENAME = model_dir + 'fcnn_results.png'


# Instantiate regularisation layer
class L1RegularizationLayer(keras.layers.Layer):
    def __init__(self, l1=0.):
        super(L1RegularizationLayer, self).__init__()
        self.l1 = l1

    def call(self, inputs):
        self.add_loss(self.l1 * tf.reduce_sum(abs(inputs)))
        return inputs


class L2RegularizationLayer(keras.layers.Layer):
    def __init__(self, l2=0.):
        super(L2RegularizationLayer, self).__init__()
        self.l2 = l2

    def call(self, inputs):
        self.add_loss(self.l2 * tf.reduce_sum(np.square(inputs)))
        return inputs


# Create Fully Connected Neural Network
def createFCNN():
    fmodel = keras.Sequential(
        [
            keras.Input(shape=(INPUT_DIM**2,), name="digits"),
            keras.layers.Rescaling(scale=1.0 / 255),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(10, activation='softmax', name="predictions")
        ]
    )
    fmodel.summary()
    return fmodel


# Process numpy array data into tensors
def getTensorData(val_size, x, y):
    # Reserve % for validation.
    valid = round(val_size * len(x))
    x_val = x[-valid:]
    y_val = y[-valid:]
    x_train = x[:-valid]
    y_train = y[:-valid]

    # Prepare the training dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

    # Prepare the validation dataset.
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    return train_dataset, val_dataset


# tensorflow training step for single train batch
@tf.function
def train_step(x, y):
    # Open a GradientTape to record the operations run
    # during the forward pass, which enables auto-differentiation.
    with tf.GradientTape() as tape:
        # Run the forward pass of the layer. The operations that the layer applies to its inputs are going to be
        # recorded on the GradientTape.
        logits = model(x, training=True)
        # Compute the loss value for this minibatch.
        loss_value = loss_fn(y, logits)
        # # Add any extra losses created during the forward pass.
        # loss_value += sum(model.losses)
    # Use the gradient tape to automatically retrieve the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(loss_value, model.trainable_weights)
    # Run one step of gradient descent by updating the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    # Update training metric.
    train_acc_metric.update_state(y, logits)
    train_loss_metric.update_state(loss_value)
    return loss_value


# tensorflow testing step for single val batch
@tf.function
def test_step(x, y):
    val_logits = model(x, training=False)
    loss_value = loss_fn(y, val_logits)
    # Update val metrics
    val_acc_metric.update_state(y, val_logits)
    val_loss_metric.update_state(loss_value)
    return


# Early Stop Logic for loss
def earlyStopLoss(losses, delta=0.15, patience=20):
    losses_to_analyse = losses[-(patience + 1):]
    diff = np.diff(losses_to_analyse)
    diff_delta = [1 if i > -delta * 0.999999 else 0 for i in diff]
    if sum(diff_delta) >= patience:
        return True


# Plot loss, accuracy and optimiser metrics
def plotMetrics(history, filename=''):
    plt.style.use('ggplot')
    train_loss, val_loss = history['train_loss'], history['val_loss']
    train_acc, val_acc = history['train_acc'], history['val_acc']
    opt_lr = history['optimizer']

    fig, axes = plt.subplots(3, sharex=True, figsize=(12, 8))
    fig.suptitle('Metrics')

    axes[0].set_ylabel("Optimiser", fontsize=14)
    axes[0].plot(opt_lr)

    axes[1].set_ylabel("Loss", fontsize=14)
    axes[1].plot(train_loss)
    axes[1].plot(val_loss)

    axes[2].set_ylim([0, 100])
    axes[2].set_ylabel("Accuracy", fontsize=14)
    axes[2].set_xlabel("Epoch", fontsize=14)
    axes[2].plot(train_acc)
    axes[2].plot(val_acc)
    # plt.show()
    if filename != '':
        plt.savefig(filename, bbox_inches='tight')
    return


# Plot confusion matrix
def predictResults(model, x, y, filename='', class_names=[]):
    y_pred = model.predict(x)
    y_pred = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_pred, y)
    accuracy = accuracy_score(y_pred, y) * 100
    fig, ax = plt.subplots(1, sharex=True, figsize=(12, 8))
    ax = sns.heatmap(cm, annot=True, fmt='g',
                     cmap='Blues', ax=ax)  # annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(f'Accuracy: {accuracy:.2f}%', fontsize=10)
    if len(class_names) > 0:
        ax.xaxis.set_ticklabels(class_names)
        ax.yaxis.set_ticklabels(class_names)

    if filename != '':
        plt.savefig(filename, dpi=400, bbox_inches='tight')
    return


# Instantiate an optimizer.
lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=LR, decay_steps=DECAY_STEPS,
                                                          decay_rate=DECAY_RATE)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
# Instantiate a loss function.
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
# Instantiate a model
model = createFCNN()
# Metrics for this epoch
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()
train_loss_metric = keras.metrics.Mean()
val_loss_metric = keras.metrics.Mean()

print('Prepare Data')
# Prepare the training dataset.
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784))  # Flatten dataset for fcnn model
x_test = np.reshape(x_test, (-1, 784))
train_dataset, val_dataset = getTensorData(VAL_SIZE, x_train, y_train)

# Derived Parameters
train_batches = len(train_dataset)
update_size = list(range(0, train_batches, round(train_batches / UPDATE_COUNT)))

# Training
for epoch in range(EPOCHS):
    print("\nStart of epoch %d" % (epoch + 1,))
    start_time = time.time()

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        loss_value = train_step(x_batch_train, y_batch_train)

        # Log every 20 batches.
        if step in update_size:
            print("%d/%d: Batch Train Loss- %.4f" % (step, train_batches, float(loss_value)))
            # print("Seen so far: %s samples" % ((step + 1) * batch_size))

    # Save final LR for epoch
    history['optimizer'].append(optimizer._decayed_lr('float32').numpy())

    # Display and save metrics at the end of each epoch.
    train_acc = train_acc_metric.result().numpy() * 100
    train_loss = train_loss_metric.result().numpy()
    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()
    train_loss_metric.reset_states()

    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in val_dataset:
        test_step(x_batch_val, y_batch_val)

    # Display and save metrics at the end of each epoch.
    val_acc = val_acc_metric.result().numpy() * 100
    val_loss = val_loss_metric.result().numpy()
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)
    # Reset val metrics at the end of each epoch
    val_acc_metric.reset_states()
    val_loss_metric.reset_states()

    # Print current metric status at end of epoch
    result1 = f"Epoch {epoch}: Train Accuracy {train_acc:.2f}%  Val Accuracy {val_acc:.2f}% "
    result2 = f"Epoch {epoch}: Train Loss {train_loss:.4f}  Val Loss {val_loss:.4f} "
    print(result1)
    print(result2)
    print("Time taken: %.2fs" % (time.time() - start_time))

    # Early Stopping
    earlyStop = earlyStopLoss(history['val_loss'], delta=DELTA, patience=PATIENCE)
    if earlyStop:
        print('Loss not decreasing. Stopping training')
        break
model.save(MODEL_FILENAME)

# Results and Metrics
plotMetrics(history, METRICS_FILENAME)
predictResults(model, x_test, y_test, RESULTS_FILENAME)
