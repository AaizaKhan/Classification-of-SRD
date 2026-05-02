from lightgbm import LGBMClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix,mean_squared_error,precision_score,recall_score,f1_score
from sklearn.metrics import classification_report , roc_curve, f1_score, accuracy_score, recall_score , roc_auc_score,make_scorer
import re
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
#loading dataset
df = pd.read_csv('.csv', encoding_errors= 'replace')

data = df.values

# Let's scale the data first
scaler = StandardScaler()
feature_df_scaled = scaler.fit_transform(df)
print(feature_df_scaled.shape)
print(feature_df_scaled)

#number of clusters selected 
kmeans = KMeans(n_clusters=2)
kmeans.fit(feature_df_scaled)
labels = kmeans.labels_
print(kmeans.cluster_centers_.shape)

X = feature_df_scaled
y = labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=2)

# Initialize Stratified K-Fold
skf = StratifiedKFold(n_splits=10)

# Initialize base classifiers
classifiers = {
    'GradientBoosting': GradientBoostingClassifier(),
    'RandomForest': RandomForestClassifier(),
    'ExtraTrees': ExtraTreesClassifier(),
    'AdaBoost': AdaBoostClassifier(n_estimators=50, learning_rate=1),
    'LGBM': LGBMClassifier()
}

for name, clf in classifiers.items():
    print(f"Training {name}...")
    accuracy_scores = []
    losses=[]
    for train_index, val_index in skf.split(X_train, y_train):
        # Split the data into training and validation sets
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        
        # Apply SMOTE to the training fold
        sm = SMOTE(random_state=0)
        X_train_smote, y_train_smote = sm.fit_resample(X_train_fold, y_train_fold)
        
        # Train the classifier on the SMOTE-balanced training fold
        clf.fit(X_train_smote, y_train_smote)
        
        # Validate the classifier on the validation fold
        y_val_pred = clf.predict(X_val_fold)
        
        # Evaluate the classifier and store the accuracy
        accuracy = metrics.accuracy_score(y_val_fold, y_val_pred)
        accuracy_scores.append(accuracy)
        
    print(f'Cross-validated accuracy scores for {name}: {accuracy_scores}')
    print(f'Mean accuracy for {name}: {np.mean(accuracy_scores)}')
    
    # Train the classifier on the full training set with SMOTE and evaluate on the test set
    X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)
    clf.fit(X_train_smote, y_train_smote)
    y_test_pred = clf.predict(X_test)
    print(f"Accuracy for {name}:", metrics.accuracy_score(y_test, y_test_pred))
    loss = mean_squared_error(y_test, y_test_pred)
    print(f"Loss for {name}:", loss)
    print(classification_report(y_test, y_test_pred))
    
    # Store the accuracy and loss for plotting
    accuracy_scores.append(accuracy)
    losses.append(loss)


    # Plotting accuracy and loss
    plt.figure(figsize=(12, 8))
    models_list = list(classifiers.keys())
    plt.plot(models_list, accuracy_scores, label='Accuracy', marker='o')
    plt.plot(models_list, losses, label='Loss', marker='o')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Model Performance Metrics')
    plt.legend()
    plt.grid(True)
    plt.xticks(models_list)  # Ensure all model names are displayed on the x-axis
    plt.show()

