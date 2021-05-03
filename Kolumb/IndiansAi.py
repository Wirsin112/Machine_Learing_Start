import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from WlasnaOptuna import EveningTuna
from Duzo_neuronow import two_neural

def gluco_change_zeros(x):
    if x == 0:
        x = df['Gluco Test'].mean()
    return x


def blood_change_zeros(x):
    if x == 0:
        x = df['Blood pressure'].mean()
    return x


def thick_change_zeros(x):
    if x == 0:
        x = df['Thick'].mean()
    return x


def age_change_zeros(x):
    if x == 0:
        a = []
        a = df[df['Pregnat'] == x]
        x = a['Age'].mean
    return x


def body_change_zeros(x):
    if x == 0:
        x = df['Body mass'].mean()
    return x

dict = {'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        'kernel': ['poly', 'rbf', 'linear']
        }
df = pd.read_csv('Indians.csv')
# print(df.dtypes)
# print(df.isna().any())
print(df.head(2).to_string())
# print(df['Age'].unique())
answer = df['Class']
# df['Gluco Test'] = df['Gluco Test'].map(gluco_change_zeros)
# df['Blood pressure'] = df['Blood pressure'].map(blood_change_zeros)
df['Thick'] = df['Thick'].map(thick_change_zeros)
# df['Body mass'] = df['Body mass'].map(body_change_zeros)
#df['Age'] = df['Age'].map(age_change_zeros)
# # figure = plt.figure(figsize=(12, 8))
# #Pregnat  Gluco Test  Blood pressure  Thick  Serum  Body mass  Function  Age  Class
# sns.barplot(x='Class', y='Pregnat', data=df)
# plt.show()
del df['Class']
a = []
# print(df)
for i in range(1):
    scaler = MinMaxScaler()
    xtrain, xtest, ytrain, ytest = train_test_split(df.values, answer.values, train_size=0.80)
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)
    Tuna = EveningTuna(xtrain,ytrain,xtest,ytest)
    Tuna.Eveninglution()
    # model = two_neural(layers=[5,1],learing=0.08,threshold = 0.6)
    # model2 = SVC()
    # model2.fit(xtrain,ytrain)
    # model.train(xtrain, ytrain, epoch= 50)
    # win2 = model2.score(xtest,ytest)
    # win = model.evaluate(xtest,ytest)
    # model.show_plot_acc()
    # model.show_plot_mse()
    # print("SVC = ", win2)
    # print("Evining = ", win)
    # #b = model.score(xtest, ytest)
    # #a.append(b)
    # # grid = GridSearchCV(model, param_grid=dict, scoring="accuracy")
    # # grid.fit(xtrain, ytrain)
    # # print(grid.best_params_)
