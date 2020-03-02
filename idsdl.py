
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import classification_report, confusion_matrix


dataset = open("/home/robotumbel/PycharmProjects/2019/IDS IOT/hasil_normal.csv",'r')
reader = dataset.readlines()
dataset.close()
col = len(reader[0])

X = []
y = []

for i in reader:
    data = i.split(',')

#
    try:
        ad = np.array(data[:62],dtype='f')
        X.append(ad)
        y.append(data[len(data)-1].strip('\n'))
    except:
        pass

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)
print('Jumlah Data Testing :',len(X_test))
print('Jumlah Data Training :',len(X_train))

X_train_x =[]
X_test_X =[]
time_X_train =[]
time_X_test =[]
for i in range(len(X_train)):
    X_train_x.append(X_train[i][1:])

for i in range(len(X_test)):
    X_test_X.append(X_test[i][1:])
    time_X_test.append(X_test[i][0])
    #print(X_test[i][0], X_test[i][1:])

#print(y_train,y_test)
y_train1 = []
y_test1 = []
for i in range(len(y_train)):
    if y_train[i] == 'Attack':
        y_train1.append(1)
    else:
        y_train1.append(0)

for i in range(len(y_test)):
    if y_test[i] == 'Attack':
        y_test1.append(1)
    else:
        y_test1.append(0)

"""Proses Normalisasi data dalam rang 0-1 or -1,1"""
data_test = X_test_X
sc = StandardScaler()
X_train_x = sc.fit_transform(X_train_x)
X_test_X = sc.transform(X_test_X)

#print(len(X_train_x[0]))
model = Sequential()
model.add(Dense(12, input_dim=len(X_train_x[0]), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(X_train_x, y_train1, epochs=1, batch_size=10)
aa = model.predict(X_test_X)

aaa = []
for i in range(len(aa)):
    #print(aa[i][0])
    aaa.append(round(aa[i][0]))

print('Confusion Matrix :\n',confusion_matrix(y_test1,aaa))
print('classification_report :\n',classification_report(y_test1,aaa))
print('Akurasi :',accuracy_score(y_test1, aaa)*100,'%')
