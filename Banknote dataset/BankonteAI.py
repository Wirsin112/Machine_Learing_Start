import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

dict = {'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        'kernel': ['poly', 'rbf', 'linear']
        }
df = pd.read_csv('data_banknote_authentication.csv')
print(df.dtypes)
print(df.head().to_string())

clas = df['Class']
del df['Class']
a = []
for i in range(200):
    scaler = MinMaxScaler()
    xtrain, xtest, ytrain, ytest = train_test_split(df.values, clas.values, train_size=0.6, shuffle=False)
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)
    model = SVC(kernel='poly',C=0.9)
    model.fit(xtrain, ytrain)
    # grid = GridSearchCV(model, param_grid=dict, scoring="accuracy")
    # grid.fit(xtrain, ytrain)
    # print(grid.best_params_)
    b = model.score(xtest, ytest)
    a.append(b)
    print(i)
print('mean', sum(a) / len(a), 'max', max(a), 'min', min(a))


