
import numpy as np
import tensorflow as tf

def main():

    # samples, batch, epochs
    samples = 5120
    batch = 256
    epochs = 3

    # data
    X = np.random.uniform(size=[samples, 224, 224, 3])
    y = np.random.uniform(size=[samples, 1], low=0, high=999).astype(int)
    data = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch)

    # model
    model = tf.keras.applications.ResNet50(weights=None)
    model.compile(optimizer=tf.keras.optimizers.SGD(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy())

    # fit
    model.fit(data, verbose=1)

if __name__ == '__main__':
    main()
