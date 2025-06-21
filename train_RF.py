import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np

# opens the pickle file made in data_creation.py
df = pickle.load(open('./data.pickle', 'rb'))


# unwrapping data and labels
data = np.asarray(df['data'])
labels = np.asarray(df['labels'])


# train/test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels) # stratify splits locally per category


model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)


# model preformance
score = accuracy_score(y_predict, y_test)
print('{}% of classification acc'.format(score * 100))


# save model
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()