# This is a sample Python script.

import pandas as pd
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score


def grid_search():

    df = pd.read_csv('./datasets/monk/monks-1.train', sep="\s+", header=None)
    df_test = pd.read_csv('./datasets/monk/monks-1.test', sep="\s+", header=None)
    # Controlla la struttura del DataFrame
    #print("Dimensioni del DataFrame originale:", df.shape)

    y_train = df.iloc[:, 0]
    y_test = df_test.iloc[:,0]

    x_test = df_test.iloc[:, 1:]
    x_train = df.iloc[:, 1:]
    #print(df.head())
    #print(df.columns)
    #print(df.dtypes)

    # Converti in liste di liste
    x_train_list = x_train.select_dtypes(include=[np.number]).to_numpy()
    y_train_list = y_train.to_numpy()

    x_test_list = x_test.select_dtypes(include=[np.number]).to_numpy()
    y_test_list = y_test.to_numpy()

    scaler = StandardScaler()
    #standardized_data = scaler.fit_transform(x_train_list)
    parameters = {'kernel':('linear', 'rbf'), 'C':[0.1, 1, 5, 10], "gamma":[0.001, 0.01, 0.1, 1]}
    #x_train_n=normalize(x_train_list, norm="l2")
    # Implementiamo la cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    # Create a kernel support vector machine model
    model = SVC()
    gs_cv = GridSearchCV(estimator=model,
            param_grid= parameters,
            cv=kfold,
            scoring="accuracy",
            n_jobs=-1,
            return_train_score = True,
            verbose = 1)

    # Stampiamo i parametri selezionati

    gs_cv.fit(x_train_list, y_train_list)

    param = gs_cv.cv_results_
    print(param)
    # Train the model on the training data
    #
    #X_train, X_val, y_train, y_val = train_test_split(
    #x_train_list, y_train_list, test_size=0.2, random_state=42)
    #ksvm.fit(x_train_list, y_train_list)
    # Evaluate the model on the test data
    print("Migliori parametri: ", gs_cv.best_params_)
    print("Miglior punteggio (cross-validation): ", gs_cv.best_score_)

    # Addestriamo con i migliori parametri
    best_model=gs_cv.best_estimator_
    best_model.fit(x_train_list, y_train_list)

    # Predici sul set di test
    y_pred = best_model.predict(x_test_list)

    # Valutazione modello
    print("\n Report di classificazione:")
    print(classification_report(y_test, y_pred))
    print("Accuratezza sul set di test: ", accuracy_score(y_test, y_pred))


def nested_grid_search():

    df = pd.read_csv('./datasets/monk/monks-1.train', sep="\s+", header=None)
    df_test = pd.read_csv('./datasets/monk/monks-1.test', sep="\s+", header=None)

    y_train = df.iloc[:, 0]
    y_test = df_test.iloc[:, 0]

    x_train = df.iloc[:, 1:]
    x_test = df_test.iloc[:, 1:]

    x_train_list = x_train.select_dtypes(include=[np.number]).to_numpy()
    y_train_list = y_train.to_numpy()

    x_test_list = x_test.select_dtypes(include=[np.number]).to_numpy()
    y_test_list = y_test.to_numpy()

    # Parametri per GridSearch
    parameters = {
        'kernel': ('linear', 'rbf'),
        'C': [0.1, 1, 5, 10],
        'gamma': [0.001, 0.01, 0.1, 1]
    }

    # Nested Cross-Validation
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

    # Modello di base
    model = SVC()

    # Risultati della nested cross-validation
    nested_scores = []

    for train_idx, val_idx in outer_cv.split(x_train_list, y_train_list):
        # Dati di training e test per l'outer loop
        X_train, X_val = x_train_list[train_idx], x_train_list[val_idx]
        Y_train, Y_val = y_train_list[train_idx], y_train_list[val_idx]
        
        # Grid Search nel loop interno
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=parameters,
            cv=inner_cv,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X_train, Y_train)
        
        # Valuta il modello con i migliori parametri sul test set dell'outer loop
        best_model = grid_search.best_estimator_
        test_score = best_model.score(X_val, Y_val)
        
        # Memorizza il punteggio
        nested_scores.append(test_score)

        # Mostra i risultati
        print("Nested CV Accuracy Scores:", nested_scores)
        print("Media Nested CV Accuracy:", np.mean(nested_scores))

    # Addestriamo con il miglior parametro
    best_model.fit(X_val, Y_val)

    # Predici sul set di test
    y_pred = best_model.predict(x_test_list)

    # Valutazione modello
    print("\n Report di classificazione:")
    print(classification_report(y_test, y_pred))
    print("Accuratezza sul set di test: ", accuracy_score(y_test, y_pred))



    """# Grid Search CV per il ciclo interno
    gs_cv = GridSearchCV(estimator=model,
                         param_grid=parameters,
                         cv=inner_cv,
                         scoring="accuracy",
                         n_jobs=-1)

    # Cross-validation esterna per valutare il modello
    nested_scores = cross_val_score(gs_cv,
                                    X=x_train_list,
                                    y=y_train_list,
                                    cv=outer_cv,
                                    scoring='accuracy')

    print(f"Accuratezza media nel ciclo esterno: {nested_scores.mean():.4f}")
    print(f"Deviazione standard nel ciclo esterno: {nested_scores.std():.4f}")

    # Addestriamo il miglior modello sull'intero set di addestramento
    gs_cv.fit(x_train_list, y_train_list)
    print("Migliori parametri (dopo Nested Grid Search): ", gs_cv.best_params_)

    best_model = gs_cv.best_estimator_
    best_model.fit(x_train_list, y_train_list)

    # Predici sul set di test
    y_pred = best_model.predict(x_test_list)

    # Valutazione sul set di test
    print("\n Report di classificazione sul set di test:")
    print(classification_report(y_test, y_pred))
    print("Accuratezza sul set di test: ", accuracy_score(y_test, y_pred))"""



grid_search()


#load_data_with_nested_grid_search()

#nested_grid_search()
