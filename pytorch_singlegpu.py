
import numpy as np
import torch
import torchvision

def main():

    # samples, batch, epochs
    samples = 2560
    batch_size = 256
    epochs = 3

    # data
    X = np.random.uniform(size=[samples, 3, 224, 224])
    y = np.random.uniform(size=[samples], low=0, high=999).astype(int)
    X, y = torch.Tensor(X), torch.Tensor(y)

    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size)

    # model
    model = torchvision.models.resnet50()
    model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    # fit
    for epoch in range(epochs):
        for batch, (X, y) in enumerate(dataloader):
            X = X.cuda()
            y = y.cuda()
            optimizer.zero_grad()
            predicted = model(X)
            loss = loss_fn(predicted, y.to(torch.int64))
            loss.backward()
            optimizer.step()
            print('--- epoch %2i, batch %2i, loss %0.2f ---' % (epoch,
                                                                batch,
                                                                loss.item()))

if __name__ == '__main__':
    main()
