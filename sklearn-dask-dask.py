
import os

from xgboost.dask import DaskXGBClassifier
from dask_ml.datasets import make_classification
from dask_ml.model_selection import train_test_split
from dask_ml.metrics import accuracy_score

from distributed import Client, progress, wait

def main():

    # client
    client = Client(os.environ['SCHEDULER_ADDRESS'])

    # vars
    n_samples = 10**8
    n_features = 50
    n_workers = len(client.scheduler_info()['workers'])

    # data
    X, y = make_classification(n_samples=n_samples,
                               n_features=n_features,
                               n_informative=n_features//5,
                               chunks=(n_samples//n_workers, n_features))

    # train, test
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2)

    # fit
    estimator = DaskXGBClassifier()
    estimator.fit(X_train,
                  y_train)

    # predict
    p_train = estimator.predict(X_train)
    p_test = estimator.predict(X_test)

    # evaluate
    train_acc = accuracy_score(y_train,
                               p_train)
    test_acc = accuracy_score(y_test,
                              p_test)
    print('train accuracy is %0.2f' % train_acc)
    print('test accuracy  is %0.2f' % test_acc)

    # shutdown
    client.shutdown()

if __name__ == '__main__':
    main()
