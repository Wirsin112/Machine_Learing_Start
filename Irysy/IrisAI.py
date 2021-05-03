import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
df = pd.read_csv('iris.csv',header=None)
results = df[4]
del df[4]
scaler = StandardScaler()
a = []
for i in range(200):
    xtrain, xtest,ytrain,ytest = train_test_split(df.values,results.values,train_size=.80)
    model = SVC(kernel='poly',C=0.9)
    model.fit(xtrain,ytrain)
    b = model.score(xtest,ytest)
    a.append(b)
print('mean', sum(a) / len(a), 'max', max(a), 'min', min(a))
