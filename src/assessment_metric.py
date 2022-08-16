import src
import torch
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE
from sklearn.ensemble import RandomForestClassifier

TRADITIONAL_METHODS = [
    RandomOverSampler,
    SMOTE,
    ADASYN,
    BorderlineSMOTE,
]

# test traditional methods
def vs_traditional_methods(dataset_name):
    train_samples, test_samples, train_labels, test_labels = src.process_datasets.prepare_dataset(dataset_name)

    for METHOD in TRADITIONAL_METHODS:
            x, y = train_samples, train_labels
            x = x.numpy()
            y = y.numpy()
            x, y = METHOD(random_state=src.config.seed).fit_resample(x, y)
            
            balanced_dataset = src.datasets.BasicDataset()
            balanced_dataset.samples = torch.from_numpy(x)
            balanced_dataset.labels = torch.from_numpy(y)

            # tm_classifier = src.classifier.Classifier(METHOD.__name__)
            src.regression.RandomForest(balanced_dataset.samples, balanced_dataset.labels, test_samples, test_labels)