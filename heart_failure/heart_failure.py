import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
import numpy as np

#import csv and create data frame, get rid of categorical death_event column
df = pd.read_csv('heart_failure.csv')
df = df.drop(['death_event'], axis=1)
print(f"\n \n")
print(df.info())

print(f"\n \n")
print(Counter(df['DEATH_EVENT']))

#set features to death_event (0 or 1)
y = df['DEATH_EVENT']
print(f"\n \n features")
print(y)

#set labels to all other columns of df
x = df.iloc[:, :-1]
print(f"\n \n LABLES")
print(x)

#one-hot-encode 'anemia', 'diabetes', 'sex', 'smoking'


#build model function
def build_model(feat):
    model = Sequential()

    #add input layer and hidden layer with 16 neurons
    model.add(layers.Dense(16, activation='relu', input_shape=(feat.shape[1],)))

    #output layer with one neuron
    model.add(layers.Dense(1))

    #compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    #print summary and return 
    print(model.summary())
    return model

my_model = build_model(x)
my_model.fit(x,y, batch_size=30, epochs=40, verbose=1)
