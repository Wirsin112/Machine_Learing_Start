import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def spec_qual(x):
    if x > 6:
        return 1
    else:
        return 0


dict = {'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        'kernel': ['poly', 'rbf', 'linear']
        }
df = pd.read_csv('winequality-white.csv', sep=';')
df.drop_duplicates()
print(df.info())
df = df[df.quality != 9]
df = df[df.quality != 3]
print(df.info())
df['quality'] = df['quality'].map(spec_qual)
quality = df['quality']
del df['quality']
# del df['alcohol']
del df['fixed acidity']
del df['sulphates']
del df['pH']
del df['density']
print(df.head(2).to_string())
a = []
# print(df['quality'].unique())
# figure = plt.figure(figsize=(12, 8))
# sns.barplot(x='quality', y='fixed acidity',data=df)
# plt.show()
for i in range(50):
    scaler = StandardScaler()
    xtrain, xtest, ytrain, ytest = train_test_split(df.values, quality.values, train_size=0.80)
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)
    model = SVC(C = 0.9, kernel='rbf')
    model.fit(xtrain, ytrain)
    # grid = GridSearchCV(model, param_grid=dict, scoring="accuracy")
    # grid.fit(xtrain, ytrain)
    # print(grid.best_params_)
    b = model.score(xtest, ytest)
    a.append(b)
    print(i)
print(sum(a)/len(a))
print("max",max(a))
