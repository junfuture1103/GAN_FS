from sklearn.ensemble import RandomForestClassifier

def RandomForest(x_train, y_train, x_test, y_test):
    # modeling
    model_rf = RandomForestClassifier(n_estimators = 15)
    # train
    model_rf.fit(x_train, y_train)
    # predict
    y_pred = model_rf.predict(x_test) 
    # validation
    y_real = y_test
    
    accuracy = round(sum(y_pred == y_real) / len(y_pred), 4)
    precision = round(sum([p == 1 & r == 1 for p, r in zip(y_pred, y_real)]) / sum(y_pred == 1), 4)
    recall = round(sum([p == 1 & r == 1 for p, r in zip(y_pred, y_real)]) / sum(y_real == 1), 4)
    f1 = round(2 / ((1/precision) + (1/recall)), 4)

    print('Accuracy : ', accuracy)
    print('Precision : ', precision)
    print('Recall : ', recall)
    print('f1-score : ', f1)

    return