#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 05:01:56 2022

@author: filipe
"""

# import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score

result_batch = dict()
current_epoch = None


class Iteration_monitor(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        global current_epoch
        current_epoch = epoch + 1

    def on_batch_end(self, batch, logs=None):
        global result_batch
        global current_epoch
        result_batch[f'{current_epoch}_{batch}'] = [
            current_epoch, batch+1, logs['loss'], logs['accuracy']]


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

for i in range(5):
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.show()

x_train = tf.keras.utils.normalize(x_train, axis=1)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, tf.nn.relu))
model.add(tf.keras.layers.Dense(128, tf.nn.relu))
model.add(tf.keras.layers.Dense(10, tf.nn.softmax))

model.compile(

    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]

)

result = model.fit(

    x_train,
    y_train,
    batch_size=200,
    epochs=100,
    callbacks=[Iteration_monitor()]

)

result = {'epoch': [x+1 for x in result.epoch], **result.history}


# val_loss , val_acc = model.evaluate(x_test,y_test)
# print ( "Loss = " , val_loss , " ; Accuracy = " , val_acc )

# predictions = model.predict(x_test)
# y_pred = np.argmax(predictions,axis=1)

# print ( "Accuracy = ", 100 * accuracy_score(y_test,y_pred) )

df_result = pd.DataFrame(result)
df_result.to_csv(
    '/home/filipe/Documents/Aulas/IC2/Experimento_2/df_result_exp_2.csv', index=False)

df_result_batch = pd.DataFrame.from_dict(result_batch, orient='index', columns=[
                                         'epoch', 'batch', 'loss', 'accuracy']).reset_index(drop=True)
df_result_batch.to_csv(
    '/home/filipe/Documents/Aulas/IC2/Experimento_2/df_result_batch_exp_2.csv', index=False)
