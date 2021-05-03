import pandas as pd
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
def change_sex(x):
    if pd.isna(x):
        return 0
    elif x == 'male':
        return 1
    else:
        return 0
def change_age(x):
    if pd.isna(x):
        return x
    else:
        return x//10
def specify_cabin(x):
    if pd.isna(x):
        return x
    else:
        return x[0]
xtrain = pd.read_csv("train.csv")
del xtrain['Name']
a = xtrain['PassengerId']
del xtrain['PassengerId']
dumies = pd.get_dummies(xtrain['Embarked'])
xtrain = pd.concat([xtrain, dumies], axis=1)
del xtrain['Embarked']
xtrain['Sex'] = xtrain['Sex'].map(change_sex)
xtrain['Age'] = xtrain['Age'].map(change_age)
median = xtrain['Age'].median()
xtrain['Age'].fillna(median)
dumies = pd.get_dummies(xtrain['Age'])
xtrain = pd.concat([xtrain, dumies], axis=1)
del xtrain['Age']
xtrain['Cabin'] = xtrain['Cabin'].map(specify_cabin)
dumies = pd.get_dummies(xtrain['Cabin'])
xtrain = pd.concat([xtrain,dumies],axis=1)
del xtrain['Cabin']
ytrain = xtrain['Survived']
del xtrain['Survived']
del xtrain['Ticket']
# print(xtest.isna().any())

xtest = pd.read_csv("test.csv")
del xtest['Name']
a = xtest['PassengerId']
del xtest['PassengerId']
dumies = pd.get_dummies(xtest['Embarked'])
xtest = pd.concat([xtest, dumies], axis=1)
del xtest['Embarked']
xtest['Sex'] = xtest['Sex'].map(change_sex)
xtest['Age'] = xtest['Age'].map(change_age)
median = xtest['Age'].median()
xtest['Age'].fillna(median)
dumies = pd.get_dummies(xtest['Age'])
xtest = pd.concat([xtest, dumies], axis=1)
xtest['8.0'] = 0
del xtest['Age']
xtest['Cabin'] = xtest['Cabin'].map(specify_cabin)
dumies = pd.get_dummies(xtest['Cabin'])
xtest = pd.concat([xtest, dumies], axis=1)
xtest['T'] = 0
del xtest['Cabin']
del xtest['Ticket']
xtest.fillna(xtest['Fare'].median(),inplace=True)
# print(xtest.head(2).to_string())
# print(xtrain.head(2).to_string())
# print(xtest.dtypes)
# print(xtest['Fare'])
# print(xtest.head(1).to_string())
scaler = MinMaxScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)
model = SVC(kernel='poly',C = 0.8, cache_size=8000)
model.fit(xtrain, ytrain)
b = model.predict(xtest)

final = pd.DataFrame({'PassengerId':a,'Survived':b})
print(final)
final.to_csv('final.csv',index=False)