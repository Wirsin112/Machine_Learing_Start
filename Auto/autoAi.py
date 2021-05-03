import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR # Support Vector Regressor (do bardziej skomplikowanych zadan)
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsRegressor # KNN do regresji
from sklearn.linear_model import LinearRegression # Liniowa regresja (bardzo proste problemy z liniowymy zaleznosciami)
from sklearn.tree import DecisionTreeRegressor


def swap_to_nums(x):
    return int(x.split(',')[0])


df = pd.read_csv('auta.csv', sep=';')
print(df)
df['Y'] = df['Y'].map(swap_to_nums)
cost = df['Y']
del df['Y']
print(df.dtypes)
print(cost.dtypes)
a = []

scaler = MinMaxScaler()
for i in range(1):
    xtrain, xtest, ytrain, ytest = train_test_split(df.values, cost.values, train_size=0.80)
    print(xtrain)
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)
    #model = DecisionTreeRegressor()
    model = LinearRegression()
    model.fit(xtrain, ytrain)
    b = model.score(xtest, ytest)
    a.append(b)
    print(i)
print('mean', sum(a) / len(a), 'max', max(a), 'min', min(a))
