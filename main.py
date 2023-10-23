import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, classification_report

# Tree Visualisation
from sklearn.tree import export_graphviz
import graphviz

class RFModel:
    def __init__(self, dataset):
        self.dataset = dataset.drop(columns=['id'])
        self.target = ' Label'
        self.X = pd.DataFrame()
        self.y = pd.DataFrame()
        self.labels = ['DDos', 'Botnet', 'UDP_DDoS', 'Syn_DDoS']
        self.rf = RandomForestClassifier(max_depth=20, random_state=42)

    def dataPreprocessing(self):
        self.__cleaning()
        self.__minorityRemoval()
        self.__labelEncoder()
        self.__normalization()
        self.__featureSelection()

    def finalRF(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.rf.fit(X_train, y_train)
        y_pred = self.rf.predict(X_test)

        self.__stats(y_test,y_pred)
        confusion = confusion_matrix(y_test, y_pred)

        print('Confusion Matrix:')
        print(confusion)
        print('Classification Report:')
        print(classification_report(y_test, y_pred))
        self.__jobLib()

    def __cleaning(self):
        self.dataset = self.dataset.sample(n=100)
        self.dataset = self.dataset.dropna()
        print("Step 1: Cleaning Dataset", self.dataset.shape)
        
    def __minorityRemoval(self):
        classes_to_merge = {
            # four minority classes
            "DDos": ["NetBIOS_DDoS", "Portmap_DDoS", "MSSQL_DDoS", "LDAP_DDoS"],
            # 3 remainder
            "Botnet": ["Botnet", "Web Attack", "Backdoor"]
        #  other two classes are Syn_DDoS and UDP_DDoS 
        }
        self.dataset['merged_label'] = self.dataset[self.target].apply(lambda x: next((key for key, value in classes_to_merge.items() if x in value), x))
        self.dataset.drop(columns=[self.target], inplace=True)
        self.dataset.rename(columns={'merged_label': self.target}, inplace=True)

        label_counts = self.dataset[self.target].value_counts()
        label_counts_df = pd.DataFrame({'Label': label_counts.index, 'Count': label_counts.values})
        print("Step 2: Four different cats in label: ") 
        print(label_counts_df)

    def __labelEncoder(self):
        label_encoder = LabelEncoder()
        self.dataset[self.target] = label_encoder.fit_transform(self.dataset[self.target])
        print("Step 3: Label Encoder")
        print(self.dataset[self.target])

    def __normalization(self):
        features_to_normalize = self.dataset.columns[:-1]
        scaler = MinMaxScaler()
        self.dataset[features_to_normalize] = scaler.fit_transform(self.dataset[features_to_normalize])
        print('Step 4: Normalized', self.dataset)

    def __featureSelection(self):
        self.X = self.dataset.drop(columns=self.target)
        self.y = self.dataset[self.target].to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.rf.fit(X_train, y_train)

        # self.__tree(X_train=X_train)
      
        sfs = SequentialFeatureSelector(self.rf, n_features_to_select=10, direction='forward', cv=5)
        sfs.fit(self.X, self.y)

        
        selected_feature_indices = sfs.get_support(indices=True)
        final_selected_features = self.X.columns[selected_feature_indices]
        self.dataset = self.dataset[final_selected_features]
        print('Step 5: FS', final_selected_features, final_selected_features.shape)

        # y_pred = self.rf.predict(X_test)
        # cf = confusion_matrix(y_test, y_pred)
        # self.__plot(cf)
        # self.__stats(y_test, y_pred)


    def __plot(self, cf):
        plt.figure(figsize=(10,10))
        sns.set(font_scale=1)
        sns.heatmap(cf, annot=True, annot_kws={'size': 12}, linewidths=0.2)
        sns.color_palette("crest", as_cmap=True)
        tick_marks = np.arange(len(self.labels))
        tick_marks2 = tick_marks + 0.5
        plt.xticks(tick_marks, self.labels, rotation=25)
        plt.yticks(tick_marks2, self.labels, rotation=0)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix for Random Forest Model')
        plt.show()

    def __jobLib(self):
        joblib.dump(self.rf, "./randomForest.pkl")
        load_rf = joblib.load("./randomForest.pkl")
        load_rf.predict(self.X)

    def __stats(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='micro')
        recall = recall_score(y_test, y_pred, average='micro')
        f1 = f1_score(y_test, y_pred, average='micro')

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1: ", f1)

    def __tree(self, rf, X_train):
        for i in range(2):
            tree = rf.estimators_[i]
            dot_data = export_graphviz(tree,
                                    feature_names=X_train.columns,  
                                    filled=True,  
                                    max_depth=2, 
                                    impurity=False, 
                                    proportion=True)
            graph = graphviz.Source(dot_data)
            graph.render('tree', cleanup=True)




def main():
    data = pd.read_csv('./TER20.csv')
    rf = RFModel(data)
    rf.dataPreprocessing()
    rf.finalRF()

main()