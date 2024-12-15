English below:


Este repositório contém implementações de Support Vector Machines (SVM) para a resolução de problemas de classificação e regressão em diferentes conjuntos de dados. 

O repositório está dividido em três arquivos: SVM, SVM-Wine, e SVM-Housing. A seguir, detalho o que foi feito em cada um desses arquivos.

SVM

No arquivo SVM, trabalhei com o conjunto de dados MNIST, que contém imagens de dígitos manuscritos. 
O objetivo aqui foi treinar um modelo de SVM para classificar um dígito específico, o número "3", em um problema de classificação binária.
Usei o modelo LinearSVC para essa tarefa.

O que foi feito:

Carregamento dos dados:

Utilizei o conjunto de dados MNIST.
O conjunto de dados contém 70.000 imagens de 28x28 pixels de dígitos manuscritos, cada uma rotulada de 0 a 9.

Pré-processamento:

Os dados foram divididos em conjunto de treinamento (60.000 imagens) e conjunto de teste (10.000 imagens).

As imagens foram convertidas para o formato float64 para garantir que o modelo pudesse processá-las corretamente.

Usei o StandardScaler para normalizar os dados de entrada, garantindo que todas as características estivessem na mesma escala.

Modelo:

O modelo utilizado foi o LinearSVC, que é uma implementação de SVM para classificação binária. No caso, o modelo foi treinado para distinguir se o número era "3" ou não.

O modelo foi ajustado para máximo de 10.000 iterações e sem o uso da fórmula dual.

Avaliação:

Avaliei o modelo utilizando as métricas de acurácia, precisão, recall e F1-score, fornecidas pela função classification_report.

A matriz de confusão foi gerada para visualizar os erros do modelo.

Também mostrei alguns exemplos de predições incorretas, exibindo as imagens dos dígitos onde o modelo cometeu erro.

Resultados:

O modelo obteve uma acurácia de 97.77%, o que mostra um bom desempenho na tarefa de distinguir o dígito "3" de outros dígitos.


SVM-Wine

No arquivo SVM-Wine, utilizei o conjunto de dados Wine do sklearn.datasets.load_wine().
Esse conjunto contém informações sobre a composição química de vinhos produzidos por três diferentes cultivadores. 
O objetivo foi treinar um modelo de SVM para classificar os vinhos com base nessas características químicas, em um problema de classificação multiclasse.

O que foi feito:

Carregamento dos dados:

Carreguei o conjunto de dados de vinhos, que contém 178 amostras com 13 características químicas diferentes.

O conjunto de dados também inclui as classes dos vinhos (3 tipos de cultivadores), e os rótulos são armazenados em y.

Exploração inicial dos dados:

Imprimi as informações sobre o conjunto de dados, como as características dos vinhos (feature_names) e as classes possíveis (target_names).

Realizei uma breve análise exploratória para entender melhor o que estava sendo classificado.

Divisão dos dados:

Dividi o conjunto de dados em treinamento e teste utilizando a função train_test_split, com 20% dos dados reservados para o teste e o restante para o treinamento.

Modelo e Ajuste de Hiperparâmetros:

Utilizei o One-vs-Rest (OvR), que é uma estratégia para transformar problemas de classificação multiclasse em vários problemas de classificação binária.

Para a base de classificação, usei o LinearSVC para uma abordagem mais simples e o SVC com kernel RBF para um modelo mais complexo.

Ajustei os hiperparâmetros dos modelos usando o RandomizedSearchCV, para encontrar a melhor combinação de parâmetros, como o grau do polinômio para o SVC, 
e os parâmetros de regularização e penalização para o LinearSVC.

Avaliação:

Avaliei os modelos utilizando o erro quadrático médio (MSE) e o relatório de classificação, que inclui métricas como precisão, recall e F1-score para cada classe.

A matriz de confusão foi gerada para comparar as predições com os rótulos verdadeiros.

Resultados:

O modelo LinearSVC obteve um MSE de 0.055 e uma boa performance em termos de classificação.

O modelo SVC com kernel RBF obteve uma acurácia de 100% nos dados de teste, mostrando um ótimo desempenho na tarefa de classificação multiclasse.

Melhores Parâmetros:

Para o modelo LinearSVC, os melhores parâmetros encontrados foram:

Grau do polinômio: 5

Penalização: L1

Valor de C: 0.92

Para o modelo SVC, os melhores parâmetros foram:

Kernel: RBF

Grau do polinômio: 4

Valor de C: 0.80

Valor de gama: 0.12

SVM-Housing

No arquivo SVM-Housing, trabalhei com o conjunto de dados California Housing, que contém informações sobre os preços de imóveis na Califórnia. 
O objetivo foi treinar um modelo de regressão SVM para prever os preços dos imóveis com base em características como localização e dados demográficos.

O que foi feito:

Carregamento dos dados:

Utilizei o conjunto de dados fetch_california_housing(), que contém informações sobre mais de 20.000 imóveis.

Os dados foram divididos em características (X) e alvos (y), sendo os alvos os preços dos imóveis.

Pré-processamento:

Os dados foram divididos em conjunto de treinamento e teste utilizando a função train_test_split.

A normalização foi realizada utilizando o StandardScaler para ajustar as variáveis ao mesmo intervalo.

Modelos Utilizados:

Utilizei dois modelos principais:

Pipeline com LinearSVR:

Um pipeline foi criado utilizando PolynomialFeatures, StandardScaler e o regressor LinearSVR.

Os hiperparâmetros foram ajustados usando RandomizedSearchCV em um espaço amplo de valores.

O melhor modelo teve um erro médio quadrático (MSE) de 0.554.

Pipeline com SVR:

Um pipeline foi criado utilizando StandardScaler e o regressor SVR com suporte a diferentes kernels.

Os hiperparâmetros testados incluíram C, gamma, epsilon, e o tipo de kernel.

O melhor modelo, usando kernel RBF, obteve um erro médio quadrático (MSE) de 0.300.

Avaliação:

O desempenho dos modelos foi avaliado utilizando o erro médio quadrático (MSE), que mede a precisão das previsões de preços dos imóveis.

Os melhores modelos foram comparados com base no MSE e na simplicidade do pipeline.

Resultados:

LinearSVR:

Melhor MSE: 0.554

Melhores parâmetros:

polynomialfeatures__degree: 1

linearsvr__epsilon: 0.943

linearsvr__tol: 0.01415

linearsvr__C: 1.66

linearsvr__loss: epsilon_insensitive

SVR:

Melhor MSE: 0.300

Melhores parâmetros:

svr__kernel: RBF

svr__C: 7.19

svr__gamma: 0.316

svr__epsilon: 0.427







This repository contains implementations of Support Vector Machines (SVM) for solving classification and regression problems on various datasets.

The repository is divided into three files: 
SVM, SVM-Wine, and SVM-Housing. 

Below is a detailed description of what was done in each file.

SVM

In the SVM file, the MNIST dataset, which contains images of handwritten digits, was used.
My goal was to train an SVM model to classify a specific digit, the number "3," in a binary classification problem.
The LinearSVC model was used for this task.

Key Steps:

Data Loading:

The MNIST dataset was utilized.
It contains 70,000 images of 28x28 pixels of handwritten digits, each labeled from 0 to 9.
Preprocessing:

The data was split into a training set (60,000 images) and a test set (10,000 images).
Images were converted to float64 to ensure compatibility with the model.
The StandardScaler was used to normalize input features, ensuring all were on the same scale.

Model:

The LinearSVC model, designed for binary classification, was used.
The model was trained to identify whether the digit was "3" or not.
It was configured with a maximum of 10,000 iterations and without the dual formulation.


Evaluation:

Metrics such as accuracy, precision, recall, and F1-score were calculated using the classification_report function.
A confusion matrix was generated to visualize model errors.
Examples of incorrect predictions were displayed, showing images where the model made mistakes.

Results:

The model achieved an accuracy of 97.77%, demonstrating excellent performance in distinguishing the digit "3" from other digits.




SVM-Wine
In the SVM-Wine file, the Wine dataset from sklearn.datasets.load_wine() was used.
This dataset contains information on the chemical composition of wines produced by three different cultivators.
The goal was to train an SVM model to classify wines based on their chemical properties in a multi-class classification problem.

Key Steps:

Data Loading:

The dataset contains 178 samples with 13 different chemical features.
It includes wine classes (three cultivators), stored in y.


Initial Data Exploration:

Information such as wine features (feature_names) and classes (target_names) was printed.
A brief exploratory analysis was conducted to understand the classification problem.


Data Splitting:

The data was split into training and test sets using train_test_split, with 20% reserved for testing.


Modeling and Hyperparameter Tuning:

One-vs-Rest (OvR) was used to transform the multi-class problem into multiple binary classification problems.


Two base models were used:


LinearSVC for simplicity.
SVC with an RBF kernel for a more complex approach.
Hyperparameters were tuned using RandomizedSearchCV, exploring parameters like polynomial degree, regularization, and penalty for LinearSVC, and RBF kernel parameters for SVC.


Evaluation:

The models were evaluated using the mean squared error (MSE) and classification reports with metrics such as precision, recall, and F1-score for each class.
Confusion matrices were generated to compare predictions with true labels.


Results:

LinearSVC:
MSE: 0.055
Delivered good classification performance.
SVC with RBF Kernel:
Achieved 100% accuracy on the test data, indicating outstanding multi-class classification performance.
Best Parameters:

LinearSVC:

Polynomial Degree: 5
Penalty: L1
C: 0.92
SVC:

Kernel: RBF
Polynomial Degree: 4
C: 0.80
Gamma: 0.12





SVM-Housing
In the SVM-Housing file, the California Housing dataset was used, which contains information about housing prices in California.
The goal was to train an SVM regression model to predict house prices based on features like location and demographics.

Key Steps:

Data Loading:

The fetch_california_housing() dataset was used, containing information on over 20,000 houses.
Features (X) and targets (y) were extracted, where y represents house prices.


Preprocessing:

Data was split into training and test sets using train_test_split.
Normalization was applied using StandardScaler to scale features to the same range.


Models Used:

Pipeline with LinearSVR:

A pipeline was created with PolynomialFeatures, StandardScaler, and LinearSVR.
Hyperparameters were tuned using RandomizedSearchCV over a wide parameter space.
The best model achieved an MSE of 0.554.



Pipeline with SVR:

A pipeline was created with StandardScaler and SVR, supporting different kernels.
Hyperparameters like C, gamma, epsilon, and kernel type were tuned.
The best model, using an RBF kernel, achieved an MSE of 0.300.


Evaluation:

Models were evaluated based on mean squared error (MSE), measuring the accuracy of house price predictions.
The best models were compared based on MSE and pipeline simplicity.


Results:

LinearSVR:

Best MSE: 0.554
Best Parameters:
polynomialfeatures__degree: 1
linearsvr__epsilon: 0.943
linearsvr__tol: 0.01415
linearsvr__C: 1.66
linearsvr__loss: epsilon_insensitive


SVR:

Best MSE: 0.300
Best Parameters:
svr__kernel: RBF
svr__C: 7.19
svr__gamma: 0.316
svr__epsilon: 0.427
