
import os

from xgboost import XGBClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV

def main():

    # vars
    n_samples = 10**4
    n_features = 50

    # data
    X, y = make_classification(n_samples=n_samples,
                               n_features=n_features,
                               n_informative=n_features//5)

    # grid search
    estimator = XGBClassifier(n_jobs=1)
    param_grid = {'n_estimators': [10, 40, 80],
                  'max_depth': [2, 4, 8],
                  'eta': [0.1, 0.3, 0.9]}
    classifier = GridSearchCV(estimator=estimator,
                              param_grid=param_grid,
                              scoring='f1',
                              cv=5,
                              n_jobs=int(os.environ['NCPUS']),
                              verbose=3)
    classifier.fit(X, y)

if __name__ == '__main__':
    main()
