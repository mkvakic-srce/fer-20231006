
import numpy as np
import torch
import torchvision

import os
import ray
import ray.train
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig

def train_loop_per_worker(train_loop_config):

    # train_loop_config
    lr = train_loop_config['lr']
    epochs= train_loop_config['epochs']
    batch_size = train_loop_config['batch_size']
    dataloader = train_loop_config['dataloader']

    # model
    model = torchvision.models.resnet50()
    model = ray.train.torch.prepare_model(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    # dataloader
    dataloader_prepared = ray.train.torch.prepare_data_loader(dataloader)

    # fit
    for epoch in range(epochs):
        for batch, (X, y) in enumerate(dataloader_prepared):
            optimizer.zero_grad()
            predicted = model(X)
            loss = loss_fn(predicted, y.to(torch.int64))
            loss.backward()
            optimizer.step()
            ray.air.session.report({"epoch": epoch,
                                    "batch": batch,
                                    "loss": loss.item()})

def main():

    # samples, batch, epochs
    samples = 256*20
    batch_size = 256
    epochs = 3

    # data
    X = np.random.uniform(size=[samples, 3, 224, 224])
    y = np.random.uniform(size=[samples], low=0, high=999).astype(int)
    X, y = torch.Tensor(X), torch.Tensor(y)
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size)

    # resources
    resources = ray.cluster_resources()
    gpus = int(resources['GPU'])
    cpus = int(resources['CPU'])
    resources_per_worker = {'GPU': 1,
                            'CPU': (cpus-1)//gpus}

    # trainer
    scaling_config = ScalingConfig(use_gpu=True,
                                   num_workers=gpus,
                                   resources_per_worker=resources_per_worker)
    train_loop_config = {'lr': 1e-3,
                         'epochs': epochs,
                         'batch_size': batch_size,
                         'dataloader': dataloader}
    trainer = TorchTrainer(train_loop_per_worker=train_loop_per_worker,
                           train_loop_config=train_loop_config,
                           scaling_config=scaling_config)
    trainer.fit()

if __name__ == '__main__':
    ray.init(address='auto',
             _node_ip_address=os.environ['NODE_IP_ADDRESS'])
    main()
