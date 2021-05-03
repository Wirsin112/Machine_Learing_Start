from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from tqdm import tqdm
class EveningTuna():
    def __init__(self,xtrain,ytrain,xtest,ytest,size=[16,16,16,1],activation="relu",final_activation="sigmoid",learing_rate=[0.4,0.04,0.004,0.0004],beta=1,epochs=1000):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.ytest = ytest
        self.size = size
        self.activation = activation
        self.final_activation = final_activation
        self.learing_rate = learing_rate
        self.beta = beta
        self.epochs = epochs
        self.size2 = len(xtrain[0])

    def Eveninglution(self):
        best_stats_jpg = ""
        best_val = 0
        for i in tqdm(range(1,self.size[0])):
            for j in range(1,self.size[1]):
                for k in range(1,self.size[2]):
                    for l in self.learing_rate:
                        classif = Sequential()
                        classif.add(Dense(i,activation=self.activation,input_shape=(self.size2,)))
                        classif.add(Dropout(0.15))
                        classif.add(Dense(j, activation=self.activation))
                        classif.add(Dropout(0.15))
                        classif.add(Dense(k, activation=self.activation))
                        classif.add(Dropout(0.15))
                        classif.add(Dense(1, activation=self.final_activation))
                        classif.compile(optimizer=SGD(learning_rate=l),loss="mean_squared_error",metrics=["accuracy"])
                        classif.fit(self.xtrain,self.ytrain,batch_size=34,epochs=self.epochs,callbacks=[EarlyStopping(monitor="loss")],verbose=False)
                        siema = classif.evaluate(self.xtest,self.ytest,verbose=False)
                        if siema[1] > best_val:
                            best_val = siema[1]
                            best_stats_jpg = "Size = ["+str(i)+","+str(j)+","+str(k)+",1] | activation = "+str(self.activation)+" | learing rate:"+str(l)+" Acc = "+str(best_val)
        print(best_stats_jpg)
if __name__ == "__main__":
    pass