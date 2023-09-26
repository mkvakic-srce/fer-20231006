
import numpy as np
import tensorrt
import tensorflow as tf

import os
import ray
import ray.air
import ray.tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.integration.keras import TuneReportCallback

def trainable_func(config, samples, batch_size, epochs):

    # config
    lr = config['lr']
    momentum = config['momentum']

    # data
    X = np.random.uniform(size=[samples, 224, 224, 3])
    y = np.random.uniform(size=[samples, 1], low=0, high=999).astype(int)
    data = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size)

    # model
    model = tf.keras.applications.ResNet50(weights=None)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.optimizers.SGD(lr=config["lr"],
                                  momentum=config["momentum"])
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=["accuracy"])

    # fit
    model.fit(data,
              batch_size = batch_size,
              epochs = epochs,
              verbose = 0,
              callbacks = [TuneReportCallback({"mean_accuracy": "accuracy"})])

def main():

    # samples, batch, epochs
    samples = 256*20
    batch_size = 256
    epochs = 3

    # resources
    resources = ray.cluster_resources()
    gpus = int(resources['GPU'])
    cpus = int(resources['CPU'])
    resources_per_worker = {'GPU': 1,
                            'CPU': (cpus-1)//gpus}

    # tuner
    trainable = ray.tune.with_resources(trainable=trainable_func,
                                        resources=resources_per_worker)

    scheduler = AsyncHyperBandScheduler(time_attr="training_iteration",
                                        max_t=60,
                                        grace_period=20)

    tune_config = ray.tune.TuneConfig(metric="mean_accuracy",
                                      mode="max",
                                      scheduler=scheduler,
                                      num_samples=1)

    param_space = {"lr": ray.tune.grid_search([1e-3, 1e-1]),
                   "momentum": ray.tune.grid_search([0.1, 0.9])}

    trainable_with_parameters = ray.tune.with_parameters(trainable,
                                                         samples=samples,
                                                         batch_size=batch_size,
                                                         epochs=epochs)

    tuner = ray.tune.Tuner(trainable=trainable_with_parameters,
                           param_space=param_space,
                           tune_config=tune_config)

    results = tuner.fit()
    print("Best hyperparameters found were: ", results.get_best_result().config)

if __name__ == '__main__':
    ray.init(address='auto',
             _node_ip_address=os.environ['NODE_IP_ADDRESS'])
    main()
