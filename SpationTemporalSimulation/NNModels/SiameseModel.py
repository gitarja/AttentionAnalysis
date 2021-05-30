from tensorflow import keras as K
import tensorflow as tf
class SiameseModel(K.models.Model):


    def __init__(self, output=2):
        super(SiameseModel, self).__init__()

        self.dense1 = K.layers.Dense(units=32, activation="elu")
        self.dense2 = K.layers.Dense(units=32, activation="elu")
        self.dense3 = K.layers.Dense(units=64, activation=None)
        self.normalize = K.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))


        #dropout
        self.dropout = K.layers.Dropout(0.75)



    def call(self, inputs, training=None, mask=None):

        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        x = self.dense3(x)
        x = self.normalize(x)

        return x


    def tripletOffline(self, anchor_output, positive_output, negative_output, margin=0.5):
        d_pos = tf.reduce_sum(tf.square(anchor_output - positive_output), 1)
        d_neg = tf.reduce_sum(tf.square(anchor_output - negative_output), 1)


        loss = tf.maximum(0.0, margin + d_pos - d_neg)
        loss = tf.reduce_mean(loss)

        return loss

    def quadrupletOffline(self, anchor_output, positive_output, negative_anchor, negative_output, margin_1=0.5, margin_2=0.5):
        d_pos = tf.reduce_sum(tf.square(anchor_output - positive_output), 1)
        d_neg = tf.reduce_sum(tf.square(anchor_output - negative_anchor), 1)

        d_pos_neg = tf.reduce_sum(tf.square(anchor_output - negative_output), 1)
        d_neg_neg =  tf.reduce_sum(tf.square(negative_anchor - negative_output), 1)


        loss_pos = tf.maximum(0.0, margin_1 + d_pos - d_neg)
        loss_neg = tf.maximum(0.0, margin_2 + d_neg_neg - d_pos_neg)
        loss = tf.reduce_mean(loss_pos + loss_neg)

        return loss