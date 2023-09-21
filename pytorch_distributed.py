
import numpy as np
import torch
import torchvision
import os

def main():

    # torchrun
    torch.distributed.init_process_group("nccl")
    local_rank = int(os.environ['LOCAL_RANK'])
    global_rank = int(os.environ['RANK'])

    # samples, batch, epochs
    samples = 5120
    batch_size = 256
    epochs = 3

    # data
    X = np.random.uniform(size=[samples, 3, 224, 224])
    y = np.random.uniform(size=[samples], low=0, high=999).astype(int)
    X, y = torch.Tensor(X), torch.Tensor(y)

    dataset = torch.utils.data.TensorDataset(X, y)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             sampler=sampler,
                                             num_workers=2,
                                             pin_memory=True)

    # model
    model = torchvision.models.resnet50()
    model = model.to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank])
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    # fit
    for epoch in range(epochs):
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(local_rank)
            y = y.to(local_rank)
            optimizer.zero_grad()
            predicted = model(X)
            loss = loss_fn(predicted, y.to(torch.int64))
            loss.backward()
            optimizer.step()
            if global_rank == 0:
                print('--- epoch %2i, batch %2i, loss %0.2f ---' % (epoch,
                                                                    batch,
                                                                    loss.item()))

if __name__ == '__main__':
    main()
