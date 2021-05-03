import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def change_RM(x):
    if x == 'R':
        return 1
    else:
        return 0


dict = {'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        'kernel': ['poly', 'rbf', 'linear']
        }
df = pd.read_csv('sonar.csv', header=None)
print(df)

df[60] = df[60].map(change_RM)
R = df[60]
print(df)
figure = plt.figure(figsize=(12, 8))
# for x in range(59):
#     print(x)
#     sns.barplot(x=60, y=x, data=df)
#     plt.show()
delum = [6, 14, 15, 16, 17, 22, 23, 24, 25, 26, 27, 28, 29, 31, 37, 38, 39, 40, 56, 60]
for x in range(len(delum)):
    del df[delum[x]]
a = []

for i in range(200):
    scaler = MinMaxScaler()
    xtrain, xtest, ytrain, ytest = train_test_split(df.values, R.values, train_size=0.8)
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)
    model = SVC(kernel='poly', C=0.9)
    model.fit(xtrain, ytrain)
    # grid = GridSearchCV(model, param_grid=dict, scoring="accuracy")
    # grid.fit(xtrain, ytrain)
    # print(grid.best_params_)
    b = model.score(xtest, ytest)
    a.append(b)
    print(i)
print('mean', sum(a) / len(a), 'max', max(a), 'min', min(a))
