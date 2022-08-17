from math import sqrt
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score, confusion_matrix

def RandomForest(x_train, y_train, x_test, y_test):
    # modeling
    # Add GBM LGBM ... etc
    model_rf = RandomForestClassifier(n_estimators = 15)
    # train
    model_rf.fit(x_train, y_train)

    # predict
    y_pred = model_rf.predict(x_test) 
    # validation
    y_real = y_test

    print("y_real len : ", len(y_real))
    print("y_real len : ", len(y_test))
 
    tn, fp, fn, tp = confusion_matrix(
        y_true=y_real,
        y_pred=y_pred,
    ).ravel()

    accuracy = round(sum(y_pred == y_real) / len(y_pred), 4)
    precision = round(tp / (tp + fp) if tp + fp != 0 else 0 , 4)
    recall = round(tp / (tp + fn) if tp + fn != 0 else 0 , 4)
    specificity = round(tn / (tn + fp) if tn + fp != 0 else 0 , 4)

    f1 = round(2 * recall * precision / (recall + precision) if recall + precision != 0 else 0 , 4)
    g_mean = round(sqrt(recall * specificity), 4)

    auc = round(roc_auc_score(
        y_true=y_real,
        y_score=y_pred,
    ),4)

    print('Accuracy : ', accuracy)
    print('Precision : ', precision)
    print('Recall : ', recall)
    print('f1-score : ', f1)
    print('g_mean : ', g_mean)
    print('auc : ', auc)

    return