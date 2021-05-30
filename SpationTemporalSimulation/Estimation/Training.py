from Conf.Settings import FEATURES_PATH
from SpationTemporalSimulation.Estimation.DataGazeFeaturesFetch import DataGenerator
from SpationTemporalSimulation.NNModels.SiameseModel import SiameseModel
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import math
from sklearn.mixture import BayesianGaussianMixture

#make TF reproducible
seed = 0
tf.random.set_seed(seed)
np.random.seed(seed)

fold = 1
N = 13
k = 1
def angular_similarity(cosine_sim):
   angular_dist = tf.acos(cosine_sim) / math.pi
   angular_sim = 1 - angular_dist
   return angular_sim

training_features_file = FEATURES_PATH + "gaze_significant\\" + "data_train" + str(fold) + ".npy"
training_labels_file = FEATURES_PATH + "gaze_significant\\" + "label_train" + str(fold) + ".npy"
#testing
testing_features_file = FEATURES_PATH + "gaze_significant\\" + "data_test" + str(fold) + ".npy"
testing_labels_file = FEATURES_PATH + "gaze_significant\\" + "label_test" + str(fold) + ".npy"


data_generator = DataGenerator(training_features_file, training_labels_file, offline=True, triplet=True)
generator = data_generator.fetch_triplet_offline


# train_generator = tf.data.Dataset.from_generator(
#     lambda: generator(),
#     output_types=( tf.float32, tf.float32, tf.float32),
#     output_shapes=(tf.TensorShape([N, ]), tf.TensorShape([N, ]), tf.TensorShape([N, ])))

train_generator = tf.data.Dataset.from_generator(
    lambda: generator(),
    output_types=( tf.float32, tf.float32, tf.float32, tf.float32),
    output_shapes=(tf.TensorShape([N, ]), tf.TensorShape([N, ]), tf.TensorShape([N, ]),  tf.TensorShape([N, ])))

batch_size = 32
train_data = train_generator.shuffle(data_generator.train_n).batch(batch_size)


model = SiameseModel()
optimizer = tf.optimizers.Adamax(learning_rate=0.001)
epochs = 5
for epoch in range(epochs):
    loss_avg = []
    for step, (x_batch_train,  x_batch_positive_train, x_batch_negative_anc_train, x_batch_negative_train) in enumerate(train_data):

        with tf.GradientTape() as tape:
            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            anchor = model(x_batch_train, training=True)  # Logits for this minibatch
            positive = model(x_batch_positive_train, training=True)  # Logits for this minibatch
            negative_anc = model(x_batch_negative_anc_train, training=True)  # Logits for this minibatch
            negative = model(x_batch_negative_train, training=True)  # Logits for this minibatch
            # loss_val = model.tripletOffline(anchor, positive, negative, margin=0.1)
            loss_val = model.quadrupletOffline(anchor, positive, negative_anc, negative, margin_1=0.1,margin_2=0.1)
            loss_avg.append(loss_val)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_val, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # Log every 16 batches.

    print(
            "Training loss (for one batch) at epoch %d: %.4f"
            % (epoch, float(np.average(loss_avg)))
    )
    print("Seen so far: %s samples" % (epoch))


data_generator = DataGenerator(training_features_file, training_labels_file, offline=False)
generator = data_generator.fetch
train_generator = tf.data.Dataset.from_generator(
    lambda: generator(),
    output_types=( tf.float32,  tf.int32),
    output_shapes=(tf.TensorShape([N, ]), ()))


embed = []
label = []
for step, (x_batch_train, y_batch_train) in enumerate(train_generator.batch(1)):
    embed.append(model(x_batch_train, training=False))
    label.append(y_batch_train.numpy())



data_generator = DataGenerator(testing_features_file, testing_labels_file)
generator = data_generator.fetch


test_generator = tf.data.Dataset.from_generator(
    lambda: generator(),
    output_types=( tf.float32, tf.int32),
    output_shapes=(tf.TensorShape([N, ]), ()))

test_data = test_generator.shuffle(data_generator.train_n).batch(1)

embed = tf.concat(embed, 0)
label = tf.concat(label, 0)
typical_anchor = tf.random.shuffle(embed[label== 0][:k])
asd_anchor = tf.random.shuffle(embed[label== 1][:k])
acc = []
cosine_similarity = tf.keras.metrics.CosineSimilarity()
for step, (x_batch_test, y_batch_test) in enumerate(test_data):

    z = model(x_batch_test, training=False)


    dist_typical =  angular_similarity(tf.keras.losses.cosine_similarity(z, typical_anchor)).numpy()
    dist_asd = angular_similarity(tf.keras.losses.cosine_similarity(z, asd_anchor)).numpy()

    acc.append(y_batch_test.numpy() == np.argmin(np.array([dist_typical, dist_asd]).transpose(), 1))

    # if y_batch_test.numpy() == np.argmin(np.array([dist_typical, dist_asd]).transpose()):
    #     acc.append(1)
    # else:
    #     acc.append(0)
    print(
        "label = %f, typical = %f, asd = %f"
        % (y_batch_test.numpy(), np.median(dist_typical), np.median(dist_asd))
    )

print(np.average(acc))
