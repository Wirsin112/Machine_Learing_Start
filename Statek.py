import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import random
from Duzo_neuronow import two_neural

def change_sex(x):
    if x == "male":
        return 1
    elif x == "Yeti":
        return 37
    else:
        return 0


def change_cabin(x):
    if pd.isna(x):
        return x
    else:
        return x[0]

def change_ticket(x):
    if x[0].isalpha():
        return 0
    else:
        return 1


def change_Age(x):
    if pd.isna(x):
        a = random.randrange(-2, 2)
        return df['Age'].median()+a
    else:
        return x//5


def change_Fare(x):
    return x//25


df = pd.read_csv("data.csv")
del df['Name']
print(df.head(2).to_string())
# x = df['Age'].median()
# df['Age'] = df['Age'].fillna(x)
df['Sex'] = df['Sex'].map(change_sex)
a = pd.get_dummies(df['Embarked'])
df = pd.concat([df, a], axis=1)
df['Cabin'] = df['Cabin'].map(change_cabin)
b = pd.get_dummies(df['Cabin'])
df = pd.concat([df, b], axis=1)
df.drop(["Embarked", "Cabin"], inplace=True, axis=1)
df["Ticket"] = df["Ticket"].map(change_ticket)
survied = df['Survived']
df.drop(["Survived"], inplace=True, axis=1)
scaler = MinMaxScaler()
df['Age'] = df['Age'].map(change_Age)
x = df['Age'].mean()
df['Age'] = df['Age'].fillna(x)
a = pd.get_dummies(df['Age'],prefix='Age')
df = pd.concat([df, a], axis=1)
df.drop("Age", inplace=True, axis=1)
df['Fare'] = df['Fare'].map(change_Fare)
a = pd.get_dummies(df['Fare'])
df = pd.concat([df, a], axis=1)
df.drop(["Fare", "Ticket"], inplace=True, axis=1)
# print(df.head(2).to_string())
# #
# print(df)
a = []
b = []
for i in range(1):
    xtrain, xtest, ytrain, ytest = train_test_split(df.values, survied.values, train_size=0.8, shuffle= False)
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)
    model = SVC(kernel='poly', cache_size = 8000)
    modelEvning = two_neural(layers=[5,1], learing = 0.15, dropout_rate=0.1)
    modelEvning.train(xtrain,ytrain,epoch=50)
    model.fit(xtrain, ytrain)

    print("Evening",modelEvning.evaluate(xtest,ytest))
    print("SVC",model.score(xtest,ytest))
    b.append(model.score(xtest,ytest))
    a.append(modelEvning.evaluate(xtest, ytest))
    modelEvning.show_plot_acc()
    modelEvning.show_plot_mse()
# print(i)
# print("SVC")
# print("max", max(a), "\n","min", min(a), "\n","mediana", sum(a)/len(a))
# print("Evnining")
# print("max", max(b), "\n","min", min(b), "\n","mediana", sum(b)/len(b))
