
import os
import numpy as np
import tensorflow as tf

tf.distribute.MultiWorkerMirroredStrategy()

def main():

    # samples, batch, epochs
    samples = 2560
    batch = 256
    epochs = 10

    # data
    X = np.random.uniform(size=[samples, 224, 224, 3])
    y = np.random.uniform(size=[samples, 1], low=0, high=999).astype(int)
    data = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch)

    # strategy
    implementation = tf.distribute.experimental.CommunicationImplementation.NCCL
    communication_options = tf.distribute.experimental.CommunicationOptions(implementation=implementation)
    strategy = tf.distribute.MultiWorkerMirroredStrategy(communication_options=communication_options)

    # model
    with strategy.scope():
        model = tf.keras.applications.ResNet50(weights=None)
        model.compile(optimizer=tf.keras.optimizers.SGD(),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy())

    # fit
    verbose = 1 if os.environ['PMI_RANK'] == '0' else 0
    model.fit(data,  epochs=epochs, verbose=verbose)

if __name__ == '__main__':
    main()
