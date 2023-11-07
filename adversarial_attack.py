import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

with open('./RF.pkl', 'rb') as file:
    s = pickle.load(file)
with open('./y_test.pkl', 'rb') as file:
    y = pickle.load(file)

rf2 = pickle.loads(s)
y_test = pickle.loads(y)
X_test = pd.read_csv('X_test.csv')


# Attack
X_test['pkt_len_min'] = np.random.rand(len(X_test['pkt_len_min']))
y_adv_pred = rf2.predict(X_test)
adversarial_accuracy = accuracy_score(y_test, y_adv_pred)

print(f"Adversarial prediction: {y_adv_pred[3]}")
print(f"Adversarial Accuracy: {adversarial_accuracy}")


