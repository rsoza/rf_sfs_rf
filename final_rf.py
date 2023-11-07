import joblib
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Apply data preprocessing
with open('./featureSelection.pkl', 'rb') as file:
    feature_selection = pickle.load(file)
feature_selection = np.append(feature_selection, ' Label')

df = pd.read_csv('./TER20.csv')
sample = df.sample(n=100)
sample = sample[feature_selection]
# Minority removal - merging labels to four cats
target = ' Label'
classes_to_merge = {
            # four minority classes
            "DDos": ["NetBIOS_DDoS", "Portmap_DDoS", "MSSQL_DDoS", "LDAP_DDoS"],
            # 3 remainder
            "Botnet": ["Botnet", "Web Attack", "Backdoor"]
        #  other two classes are Syn_DDoS and UDP_DDoS 
        }
sample['merged_label'] = sample[target].apply(lambda x: next((key for key, value in classes_to_merge.items() if x in value), x))
sample.drop(columns=[target], inplace=True)
sample.rename(columns={'merged_label': target}, inplace=True)

label_counts = sample[target].value_counts()
label_counts_df = pd.DataFrame({'Label': label_counts.index, 'Count': label_counts.values})

# features
X = sample.drop(columns=target)
# labels (4)
y = sample[target].to_numpy()
rf = RandomForestClassifier(max_depth=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

s = pickle.dumps(rf)
joblib.dump(s, "./RF.pkl")
X_test.to_csv('X_test.csv', index=False)
y = pickle.dumps(y_test)
joblib.dump(y, "./y_test.pkl")

original_accuracy = accuracy_score(y_test, y_pred)
print(f"Original prediction: {y_pred[3]}")
print(f"Original Accuracy: {original_accuracy}")