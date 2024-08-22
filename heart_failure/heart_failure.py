import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Dense, InputLayer #type:ignore
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical #type:ignore
import numpy as np

#import csv and create data frame, get rid of categorical death_event column
df = pd.read_csv('heart_failure.csv')

#set features to death_event (0 or 1)
y = df['DEATH_EVENT']

#set labels to all other columns of df
x = df.iloc[:, :-1]

#one-hot-encode 'anemia', 'diabetes', 'sex', 'smoking'
x = pd.get_dummies(x)

# split data into test and train 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# create column transformer for to scale numeric x columns
ct = ColumnTransformer( [ ('numeric', StandardScaler(), ['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time'] ) ] )

# use column transformer to train on the features (x) train data and the use the trained ct to scale the test features
x_train = ct.fit_transform(x_train)
x_test = ct.transform(x_test)

# intialize label encoder
le = LabelEncoder()

# use label emcoder to train on labels (y) train data and then use the trained le to scale the test labels
y_train = le.fit_transform(y_train.astype(str))
y_test = le.transform(y_test.astype(str))

# transforming encoded training labels into binary vectors 
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# initialze a sequential tensorflow model
my_model = Sequential()

# add input layer
my_model.add(InputLayer(shape=(x_train.shape[1],)))

# create hidden layer
my_model.add(Dense(12, activation='relu'))

# create and output layer
my_model.add(Dense(2, activation='softmax'))

my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

my_model.fit(x_train, y_train, epochs=100, batch_size=16, verbose=1)

loss, acc = my_model.evaluate(x_test, y_test, verbose=1)

y_estimate = my_model.predict(x_test, verbose=1)

# select indices of true classes for each label encoding
y_estimate = np.argmax(y_estimate, axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_estimate))