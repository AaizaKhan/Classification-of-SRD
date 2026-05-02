import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix,mean_squared_error,precision_score,recall_score,f1_score
from sklearn.metrics import classification_report , roc_curve, f1_score, accuracy_score, recall_score , roc_auc_score,make_scorer
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier

#loading dataset
df = pd.read_csv('.csv', encoding_errors= 'replace')

data = df.values

scaler = StandardScaler()
feature_df_scaled = scaler.fit_transform(df)
print(feature_df_scaled.shape)
print(feature_df_scaled)

print (df.shape)

#x, y variables
X = df.iloc[:,0:1518]
print(X.shape)
y = df.Class



"""
#number of clusters selected 
kmeans = KMeans(n_clusters=2)
kmeans.fit(feature_df_scaled)
labels = kmeans.labels_
#print(kmeans.cluster_centers_.shape)

#dataset visualization

X = feature_df_scaled
y = labels"""


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=2)

skf = StratifiedKFold(n_splits=10)

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=0)
X_train, y_train = sm.fit_resample(X_train, y_train)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)

sns.set(font_scale=1.7)


#GradientBoostingClassifier

GB = GradientBoostingClassifier().fit(X_train, y_train)
y_pred = GB.predict(X_test)
print("Accuracy for GB",metrics.accuracy_score(y_test, (y_pred)))
print(classification_report(y_test,((y_pred))))

# compute the confusion matrix
cm = confusion_matrix(y_test,(y_pred))
#Plot the confusion matrix.
# Get the unique classes from the target variable
df_again = pd.read_csv('3-tfidf-12.csv', encoding_errors='replace')
classes = sorted(df_again['_class_'].unique())
plt.figure(figsize=(12, 8))
plot_ = sns.heatmap(cm/np.sum(cm), annot=True, annot_kws={'size': 15}, fmt= '0.2%', xticklabels=classes, yticklabels=classes)
plt.ylabel('Prediction',fontsize=17)
plt.xlabel('Actual',fontsize=17)
plt.title('Confusion Matrix for GB',fontsize=17)
plt.show()


#RandomForestClassifier
RF = RandomForestClassifier().fit(X_train, y_train)
y_pred = RF.predict(X_test)
print(confusion_matrix(y_test, (y_pred)))
acc_train_log = metrics.accuracy_score(y_test, (y_pred))
print("Accuracy for RF:",metrics.accuracy_score(y_test, (y_pred)))
print(classification_report(y_test,((y_pred))))

# compute the confusion matrix
cm = confusion_matrix(y_test,(y_pred))
#Plot the confusion matrix.
# Get the unique classes from the target variable
df_again = pd.read_csv('tfidf-12.csv', encoding_errors='replace')
classes = sorted(df_again['_class_'].unique())
plt.figure(figsize=(12, 8))
plot_ = sns.heatmap(cm/np.sum(cm), annot=True, annot_kws={'size': 15}, fmt= '0.2%', xticklabels=classes, yticklabels=classes)

plt.ylabel('Prediction',fontsize=17)
plt.xlabel('Actual',fontsize=17)
plt.title('Confusion Matrix for ET',fontsize=17)
plt.show()

#ExtraTreesClassifier

extra = ExtraTreesClassifier().fit(X_train, y_train)
y_pred = extra.predict(X_test)
print(confusion_matrix(y_test, (y_pred)))
acc_train_log = metrics.accuracy_score(y_test, (y_pred))
print("Accuracy for ET:",metrics.accuracy_score(y_test, (y_pred)))
print(classification_report(y_test,((y_pred))))

# compute the confusion matrix
cm = confusion_matrix(y_test,(y_pred))
#Plot the confusion matrix.
# Get the unique classes from the target variable
df_again = pd.read_csv('tfidf-12.csv', encoding_errors='replace')
classes = sorted(df_again['_class_'].unique())
plt.figure(figsize=(12, 8))
plot_ = sns.heatmap(cm/np.sum(cm), annot=True, annot_kws={'size': 15}, fmt= '0.2%', xticklabels=classes, yticklabels=classes)

plt.ylabel('Prediction',fontsize=17)
plt.xlabel('Actual',fontsize=17)
plt.title('Confusion Matrix for ET',fontsize=17)
plt.show()

#plotting all models ghraph in one ghraph
plt.figure()

#adaboost
abc = AdaBoostClassifier(n_estimators=50, learning_rate=1).fit(X_train, y_train)

# Train Adaboost Classifer
y_pred = abc.predict(X_test)
print(confusion_matrix(y_test, (y_pred)))
acc_train_log = metrics.accuracy_score(y_test, (y_pred))
print("Accuracy for AdaBoost:",metrics.accuracy_score(y_test, (y_pred)))
print(classification_report(y_test,((y_pred))))

# compute the confusion matrix
cm = confusion_matrix(y_test,(y_pred))
#Plot the confusion matrix.
# Get the unique classes from the target variable
df_again = pd.read_csv('tfidf-12.csv', encoding_errors='replace')
classes = sorted(df_again['_class_'].unique())
plt.figure(figsize=(12, 8))
plot_ = sns.heatmap(cm/np.sum(cm), annot=True, annot_kws={'size': 15}, fmt= '0.2%', xticklabels=classes, yticklabels=classes)

plt.ylabel('Prediction',fontsize=17)
plt.xlabel('Actual',fontsize=17)
plt.title('Confusion Matrix for AdaBoost',fontsize=17)
plt.show()

from lightgbm import LGBMClassifier
#LGBMClassifier
lg = LGBMClassifier().fit(X_train, y_train)

#Predict the response for test dataset
y_pred = lg.predict(X_test)
acc_train_log = metrics.accuracy_score(y_test, (y_pred))
print("Accuracy for LGBM:",metrics.accuracy_score(y_test, (y_pred)))
print(classification_report(y_test,((y_pred))))

# compute the confusion matrix
cm = confusion_matrix(y_test,(y_pred))
#Plot the confusion matrix.
# Get the unique classes from the target variable
df_again = pd.read_csv('tfidf-12.csv', encoding_errors='replace')
classes = sorted(df_again['_class_'].unique())
plt.figure(figsize=(12, 8))
plot_ = sns.heatmap(cm/np.sum(cm), annot=True, annot_kws={'size': 15}, fmt= '0.2%', xticklabels=classes, yticklabels=classes)
plt.ylabel('Prediction',fontsize=17)
plt.xlabel('Actual',fontsize=17)
plt.title('Confusion Matrix for GradientBoosting',fontsize=17)
plt.show()

# Stack all Learning Models
models = [
    {'label': 'GB', 'model': GradientBoostingClassifier()},
    {'label': 'RF', 'model': RandomForestClassifier()},
    {'label': 'AdaBoost', 'model': AdaBoostClassifier()},
    {'label': 'ET', 'model': ExtraTreesClassifier()},
    {'label': 'LGBM', 'model': LGBMClassifier()}
]

plt.figure()

# Below for loop iterates through the models
for m in models:
    model = m['model']  # select the model
    model.fit(X_train, y_train)  # train the model
    y_pred = model.predict(X_test)  # predict the test data

    # Compute Precision-Recall and plot curve
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
    else:  # use decision function for models without predict_proba
        y_prob = model.decision_function(X_test)
        
    # Get the scores for the positive class
    if y_prob.ndim == 2:
        y_scores = y_prob[:, 1]
    else:
        y_scores = y_prob

    precision, recall, _ = precision_recall_curve(y_test, y_scores, pos_label=1)

    plt.plot(recall, precision, label='%s' % m['label'])

# Custom settings for the plot
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve for 5-ML Models', fontsize=12)
plt.legend(loc="lower left", fontsize=12)
plt.show()  # Display
