# classificacao_churn_streaming

Nesse projeto utilizarei um modelo de classificação para mapear o perfil de usuários e prever quais tem mais chances de deixar a plataforma de streaming. 
A base de dados foi retirada do Kaggle, possui informações sobre as contas dos clientes na plataforma, divididos entre contas Basic, Standard e Premium onde
cada uma oferece uma gama maior de serviços do que a anterior.

Vamos testar dois modelos de classificação, Logistic Regression e Random Forest Classifier, avaliar as métricas para escolher o melhor modelo.

### Importando Bibliotecas
```
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
```
### Importando o dataset 
```
df = pd.read_excel('churn_data.xlsx')
df.head()
```
Após a importação fiz uma verificação por nulos e dados duplicados e gerei alguns gráficos para entender melhor os dados

```
sns.barplot(data=df_agg, x='Churn', y='customerID')
```
![image](https://github.com/user-attachments/assets/96f1d79e-cdfc-43b2-aacf-85f5fad1aa24)

```
df_payments = df.groupby("PaymentMethod")['customerID'].count().reset_index()
plt.figure(figsize=[10,7])

sns.barplot(df_payments, x='PaymentMethod', y='customerID')
```
![image](https://github.com/user-attachments/assets/9c935fe3-465a-4672-ab6b-eee5e4580068)

## Separando a base em X e Y, sendo X todas as colunas exceto CustomerID e Churn e Y (target) a coluna Churn

```
X = df.drop(columns=['customerID', 'Churn'], axis=1)
y = df[['Churn']]
```
Fazendo a transformação de Churn para 0 e 1 com o LabelEncoder
```
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(y.Churn)
y.Churn = le.transform(y.Churn)

```
Aplicando getdummies em dados categóricos para a base X
```
X = pd.get_dummies(X, dtype=int)
```
Aplicando normalização com MinMaxScaler
```
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()
X = pd.DataFrame(mm.fit_transform(X))
```
Separando a base em conjunto de treino e teste com Train Test Split
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
```
## Modelagem utilizando Regressão Logística
```
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
lr = model.fit(X_train, y_train)
lr.predict(X_test)
```
## Utilizando Confusion Matrix Display para avaliação do modelo
```
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(lr, X_test, y_test)
```
![image](https://github.com/user-attachments/assets/d43e169b-7b3c-42bd-b29d-f9348f05aefa)

## Métricas de avaliação para bases de treino e teste
```
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
print(f'Acurácia (Treino): {accuracy_score(y_train, lr.predict(X_train))}')
print(f'Acurácia (Teste): {accuracy_score(y_test, lr.predict(X_test))}')

print(f'Acurácia Balanceada (Treino): {balanced_accuracy_score(y_train, lr.predict(X_train))}')
print(f'Acurácia Balanceada (Teste): {balanced_accuracy_score(y_test, lr.predict(X_test))}')

print(f'Precision Score (Treino): {precision_score(y_train, lr.predict(X_train))}')
print(f'Precision Score(Teste): {precision_score(y_test, lr.predict(X_test))}')

print(f'Recall (Treino): {recall_score(y_train, lr.predict(X_train))}')
print(f'Recall (Teste): {recall_score(y_test, lr.predict(X_test))}')

print(f'F1 Score (Treino): {f1_score(y_train, lr.predict(X_train))}')
print(f'F1 Score(Teste): {f1_score(y_test, lr.predict(X_test))}')

print(f'ROC AUC (Treino): {roc_auc_score(y_train, lr.predict_proba(X_train)[:,1])}')
print(f'ROC AUC(Teste): {roc_auc_score(y_test, lr.predict_proba(X_test)[:,1])}')
```
Resultado:
Acurácia (Treino): 0.8113386423966629
Acurácia (Teste): 0.78839590443686
Acurácia Balanceada (Treino): 0.7308087390927813
Acurácia Balanceada (Teste): 0.7038316705472236
Precision Score (Treino): 0.6683848797250859
Precision Score(Teste): 0.6417525773195877
Recall (Treino): 0.5609228550829127
Recall (Teste): 0.516597510373444
F1 Score (Treino): 0.6099568796550372
F1 Score(Teste): 0.5724137931034483
ROC AUC (Treino): 0.8508943812671933
ROC AUC(Teste): 0.8339598589992065

## Modelagem com RandomForestClassifier
```
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()

rf.fit(X_train, y_train)
rf.predict(X_test)
```
ConfusionMatrixDisplay do RandomForest:

![image](https://github.com/user-attachments/assets/4363e33c-cd1e-45e2-a1ea-75c5d0d50bb7)

## Métricas de Avaliação do RF

Acurácia (Treino): 0.9981039059537353
Acurácia (Teste): 0.7804323094425484

Acurácia Balanceada (Treino): 0.997090666408966
Acurácia Balanceada (Teste): 0.6860813746276616

Precision Score (Treino): 0.9978308026030369
Precision Score(Teste): 0.6318681318681318

Recall (Treino): 0.9949531362653208
Recall (Teste): 0.47717842323651455

F1 Score (Treino): 0.9963898916967509
F1 Score(Teste): 0.5437352245862884

ROC AUC (Treino): 0.9998737773982341
ROC AUC(Teste): 0.8132910157520259

Pelas métricas percebemos um Overfitting do modelo, a acurácia está muito alta na base de treino e cai muito na base de teste. Por isso faremos um GridSearch para otimizar os hiperparâmetros do modelo

## GridSearch
```
from sklearn.model_selection import GridSearchCV
parameters = {"max_depth": [1,2,3,4,5,6,7,8,9,10],
              "n_estimators": [100,300,500]}
grid_search = GridSearchCV(rf, parameters, scoring='accuracy', cv=5, n_jobs=1)
grid_search.fit(X_train, y_train)
grid_search.best_estimator_.get_params()
```
Aplicação dos resultados do get_params no modelo RandomForest

```
rf_tunned = RandomForestClassifier(bootstrap = True,
 ccp_alpha = 0.0,
 class_weight = None,
 criterion = 'gini',
 max_depth = 9,
 max_features = 'sqrt',
 max_leaf_nodes = None,
 max_samples = None,
 min_impurity_decrease = 0.0,
 min_samples_leaf = 1,
 min_samples_split = 2,
 min_weight_fraction_leaf = 0.0,
 monotonic_cst = None,
 n_estimators = 300,
 n_jobs = None,
 oob_score = False,
 random_state = None,
 verbose = 0,
 warm_start = False)

rf_tunned.fit(X_train, y_train)
rf_tunned.predict(X_test)

```
Plotando ConfusionMatrixDisplay novamente para o RF otimizado
```
ConfusionMatrixDisplay.from_estimator(rf_tunned, X_test, y_test)
```
![image](https://github.com/user-attachments/assets/2ba0ef6f-db77-497e-988e-55e26ae7508c)

## Métricas de avaliação do RandomForest otimizado:

Acurácia (Treino): 0.8528631020098597
Acurácia (Teste): 0.7901023890784983

Acurácia Balanceada (Treino): 0.7777599299905087
Acurácia Balanceada (Teste): 0.69532479610817

Precision Score (Treino): 0.7759710930442638
Precision Score(Teste): 0.6591549295774648

Recall (Treino): 0.619322278298486
Recall (Teste): 0.4854771784232365

F1 Score (Treino): 0.6888532477947072
F1 Score(Teste): 0.5591397849462365

ROC AUC (Treino): 0.9344962197211825
ROC AUC(Teste): 0.8342118784063268

## Conclusão

O Random Forest Classifier depois de utilizado o Grid Search para otimizar os hiperparâmetros, se mostrou o melhor modelo para previsão de Churn nesse caso






