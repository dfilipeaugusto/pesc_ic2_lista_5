#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 13:01:07 2022

@author: filipe
"""

# Experimento 1
# Este experimento computacional tem o objetivo de avaliar a capacidade de classificação do SVM na base de
# dados sintética da KEEL conhecida como ”Banana dataset” (assim chamada pois as instâncias pertencem
# a vários clusters que se apresentam no formato de uma banana, veja a Figura 1). A base possui 5300 observações, 2 atributos, e 2 classes, e ela pode ser acessada em https://sci2s.ugr.es/keel/dataset.php?cod=182

# ------------------------------------------------------------------------------
#  Importar o conjunto de dados Iris em um dataframe do pandas
# ------------------------------------------------------------------------------

from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
import pandas as pd

dataframe = pd.read_csv(
    '/home/filipe/Documents/Aulas/IC2/Experimento_1/banana.csv')


# ------------------------------------------------------------------------------
#  Separar em dataframes distintos os atributos e o alvo
#    - os atributos são todas as colunas menos a última
#    - o alvo é a última coluna
# ------------------------------------------------------------------------------

attributes = dataframe.iloc[:, :-1]
target = dataframe.iloc[:, -1]

# ------------------------------------------------------------------------------
#  Criar os arrays numéricos correspondentes aos atributos e ao alvo
# ------------------------------------------------------------------------------

X = attributes.to_numpy()
y = target.to_numpy()

# ------------------------------------------------------------------------------
#  Visualizar a mariz de dispersão dos 2 atributos
# 1. Gere o gráfico de dispersão da base de dados e discuta;
# ------------------------------------------------------------------------------


foo = pd.plotting.scatter_matrix(
    attributes,
    c=y,
    figsize=(11, 11),
    marker='o',
    hist_kwds={'bins': 40},
    s=30,
    alpha=0.5,
    # cmap=ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    cmap=ListedColormap(['red', 'blue'])
)

# ------------------------------------------------------------------------------
#  Dividir o conjunto de dados em conjunto de treinamento e conjunto de teste
# ------------------------------------------------------------------------------


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.15  # , random_state = 352019
)


# ------------------------------------------------------------------------------
# SUPPORT VECTOR MACHINE COM KERNEL AND RANDOM FOREST
# ------------------------------------------------------------------------------

# 2. Teste o SVM com os kernels sigmoide, linear, RBF e polinomial. Para sigmoide e RBF use γ como
# 1, 0.5 e 0.01; para o polinomial use grau 3. Para cada kernel execute uma validação cruzada k-fold
# com três valores para k: 2, 5 e 10. Monte uma tabela com o valor do erro fora-de-amostra Eout e, por
# conseguinte, a taxa de acerto do SVM para cada caso. Discuta os resultados obtidos;

# 3. Após sua análise pessoal e discussão sobre os resultados obtidos, apresente gráficos que indiquem os
# vetores de suporte para o modelo kernel/k-fold de melhor e pior desempenho (se baseie no exemplo
# apresentado na Figura 1b - sinta-se livre para gerar gráficos mais elaborados). Discuta sobre os resultados gráficos obtidos;

# 4. Escolha outro modelo de classificação (com exceção de redes neurais ou deep learning) e o apresente
# brevemente, justificando sua escolha. Faça uma implementação computacional deste modelo e compare o resultado obtido com a melhor solução dentre os modelos SVM. Discuta os resultados


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


results = dict()
for current_kernel in ['rbf', 'sigmoid', 'linear', 'poly', 'tree']:

    for counter, g in enumerate([1, 0.5, 0.01]):

        current_result = dict()

        c = 0 if current_kernel == 'tree' else 1.0  # Default

        if current_kernel == 'poly' or current_kernel == 'linear':
            g = 1 / (len(X[0]) * X.var())
        elif current_kernel == 'tree':
            g = 0

        if current_kernel == 'tree':
            svc_model = DecisionTreeClassifier(
                criterion='gini',  # Default
                max_features=None,  # Number of features
                max_depth=None  # Until all leaves are pure or until all leaves contain less than 2 samples
            )
        else:
            svc_model = SVC(kernel=current_kernel, gamma=g, C=c, degree=3)

        svc_model = svc_model.fit(X_train, y_train)

        y_train_pred = svc_model.predict(X_train)
        y_test_pred = svc_model.predict(X_test)

        acc_in = accuracy_score(y_train, y_train_pred)
        acc_out = accuracy_score(y_test, y_test_pred)

        for cv in [2, 5, 10]:
            current_result[f'accuracy_cv_{cv}'] = np.mean(
                cross_val_score(svc_model, X, y, cv=cv, scoring='accuracy')
            )

        if counter == 0:
            print('-----------------------------------------------------')
            print(f"Kernel: {current_kernel}")
            print(
                '    g           C     Acc. IN    Acc. OUT    Acc. (k=2)    Acc. (k=5)   Acc. (k=10)')
            print(
                ' ----     -------     -------    --------    ----------    ----------   -----------')

        print(str(' %4.2f' % g) + '  ' +
              str('%10.4f' % c) + '  ' +
              str('%10.4f' % acc_in) + ' ' +
              str('%10.4f' % acc_out) + '   ' +
              str('%10.4f' % current_result['accuracy_cv_2']) + '    ' +
              str('%10.4f' % current_result['accuracy_cv_5']) + '    ' +
              str('%10.4f' % current_result['accuracy_cv_10'])
              )

        results[f'{current_kernel}_{g}'] = [
            current_kernel, g, c, acc_in, acc_out,
            current_result['accuracy_cv_2'], current_result['accuracy_cv_5'], current_result['accuracy_cv_10']
        ]

        # # Get support vectors themselves
        # support_vectors = svc_model.support_vectors_

        # # Visualize support vectors
        # plt.scatter(X_train[:,0], X_train[:,1])
        # plt.scatter(support_vectors[:,0], support_vectors[:,1], color='red')
        # plt.title('Linearly separable data with support vectors')
        # plt.xlabel('X1')
        # plt.ylabel('X2')
        # plt.show()

        fig, ax = plt.subplots()
        # title for the plots
        title = 'Decision Tree' if current_kernel == 'tree' else f'Kernel: {current_kernel}'
        title += f' - Gamma: {round(g, 3)}' if current_kernel not in (
            'linear', 'tree') else ''
        # Set-up grid for plotting.
        X0, X1 = X[:, 0], X[:, 1]
        xx, yy = make_meshgrid(X0, X1)

        plot_contours(ax, svc_model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_ylabel('Aty')
        ax.set_xlabel('Atx')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
        ax.legend()
        plt.show()

        if current_kernel == 'poly' or current_kernel == 'linear' or current_kernel == 'tree':
            break


columns = ['kernel', 'gamma', 'regularization',
           'Acc. IN (85%)', 'Acc. OUT (15%)', 'Acc. K2', 'Acc. K5', 'Acc. K10']
df_result = pd.DataFrame.from_dict(
    results, orient='index', columns=columns).reset_index(drop=True)
df_result.to_csv(
    '/home/filipe/Documents/Aulas/IC2/Experimento_1/df_result_exp_1.csv', index=False)
