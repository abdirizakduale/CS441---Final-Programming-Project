## Import packages
import pandas as pd
from pandas import set_option
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import normalize, Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, 
    BaggingClassifier, 
    AdaBoostClassifier, 
    GradientBoostingClassifier, 
    ExtraTreesClassifier
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error as mse
from sklearn.feature_selection import RFECV, SelectKBest
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn import datasets

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib import rcParams
rcParams['xtick.major.pad'] = 1
rcParams['ytick.major.pad'] = 1

# Load data from a csv file
bcdf = pd.read_csv('data.csv')

# Numerize diagnosis "M" malignant; "B" benign using a dictionary and map function
diagnosis_coder = {'M':1, 'B':0}
bcdf.diagnosis = bcdf.diagnosis.map(diagnosis_coder)

# Drop unecessary columns
bcdf.drop(['id','Unnamed: 32'], axis = 1, inplace = True)

# Reorder columsn so diagnosis is right-most
# First define a diagnosis series object
diagnosis = bcdf.diagnosis

# Then drop diagnosis from dataframe
bcdf.drop('diagnosis', axis = 1, inplace = True)

# Then append diagnsis to end of dataframe
bcdf['Diagnosis'] = diagnosis

# Take a quick glimpse of the dataset
bcdf.head()

# Quick glimpse of tumor features (mean values) in relation to diagnosis
bcdf.groupby('Diagnosis').mean()

# For visual comparisons of differential diagnosis...
# create to dataframes - one for benign, one for malignant tumor data

bcdf_n = bcdf[bcdf['Diagnosis'] == 0]
bcdf_y = bcdf[bcdf['Diagnosis'] == 1]

# Create list of features related to mean tumor characteristics
features_means =list(bcdf.columns[0:10])

outcome_count = bcdf['Diagnosis'].value_counts()
# Rename the series so the dataframe will have a 'Diagnosis' column
outcome_count = outcome_count.rename('Diagnosis')
outcome_count = pd.DataFrame(outcome_count)

# Now that 'Diagnosis' is the column name, these lines will work
outcome_count.index = ['Benign', 'Malignant']
outcome_count['Percent'] = 100 * outcome_count['Diagnosis'] / sum(outcome_count['Diagnosis'])
outcome_count['Percent'] = outcome_count['Percent'].round().astype('int')

print('The Perecentage of tumors classified as \'malignant\' in this data set is: {}'.format(100*float(bcdf.Diagnosis.value_counts()[1])/float((len(bcdf)))))
print('\nA good classifier should therefore outperform blind guessing knowing the proportions i.e. > 62% accuracy')
outcome_count

'''
# Visualize frequency of diagnoses in dataset
sns.barplot(x = ['Benign', 'Malignant'], y = 'Diagnosis', data = outcome_count, alpha = .8)
plt.title('Frequency of Diagnostic Outcomes in Dataset')
plt.ylabel('Frequency')
plt.savefig('graph_1.png')

# Visualize frequency of diagnoses in dataset
sns.barplot(x = ['Benign', 'Malignant'], y = 'Percent', data = outcome_count, alpha = .8)
plt.title('Percentage of Diagnostic Outcomes in Dataset')
plt.ylabel('Percentage')
plt.ylim(0,100)
plt.savefig('graph_2.png')

'''

# Split data into testing and training set. Use 80% for training
X_train, X_test, y_train, y_test = train_test_split(bcdf.iloc[:,:-1], bcdf['Diagnosis'], train_size = .8 )

# The normalize features to account for feature scaling

# Instantiate 
norm = Normalizer()

# Fit
norm.fit(X_train)

# Transform both training and testing sets
X_train_norm = norm.transform(X_train)
X_test_norm = norm.transform(X_test)

# Define parameters for optimization using dictionaries {parameter name: parameter list}

SVM_params = {'C':[0.001, 0.1, 10, 100], 'kernel':['rbf' ,'linear', 'poly', 'sigmoid']}
LR_params = {'C':[0.001, 0.1, 1, 10, 100]}
LDA_params = {'n_components': [None, 1], 'solver': ['svd'], 'shrinkage': [None]}
KNN_params = {'n_neighbors':[1,5,10,20, 50], 'p':[2], 'metric':['minkowski']}
RF_params = {'n_estimators':[10,50,100]}
DTC_params = {'criterion':['entropy', 'gini'], 'max_depth':[10, 50, 100]}

# Append list of models with parameter dictionaries

models_opt = []

models_opt.append(('LR', LogisticRegression(), LR_params))
models_opt.append(('LDA', LinearDiscriminantAnalysis(), LDA_params))
models_opt.append(('KNN', KNeighborsClassifier(),KNN_params))
models_opt.append(('DTC', DecisionTreeClassifier(), DTC_params))
models_opt.append(('RFC', RandomForestClassifier(), RF_params))
models_opt.append(('SVM', SVC(), SVM_params))

results = []
names = []


def estimator_function(parameter_dictionary, scoring='accuracy'):
    for name, model, params in models_opt:
        kfold = KFold(n_splits=5, random_state=2, shuffle=True)
        model_grid = GridSearchCV(model, params)
        cv_results = cross_val_score(model_grid, X_train_norm, y_train, cv=kfold, scoring=scoring)
        
        results.append(cv_results)
        names.append(name)
        
        msg = "Cross Validation Accuracy %s: Accuracy: %f SD: %f" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        
estimator_function(models_opt, scoring = 'accuracy')

# Guassian Naive Bayes does not require optimization so we will run it separately without
# gridsearch and append the performance results to the results and names lists.

# Instantiate model
GNB =  GaussianNB()

# Define kfold - this was done above but not as a global variable 
kfold = KFold(n_splits=5, random_state=2, shuffle=True)

# Run cross validation
cv_results_GNB= cross_val_score(GNB,X_train_norm, y_train, cv = kfold, scoring = 'accuracy')

# Append results and names lists
results.append(cv_results_GNB)
names.append('GNB')

# Ensemble Voting

from sklearn.ensemble import VotingClassifier

# Create list for estimatators
estimators = []

# Create estimator object
model1 = LogisticRegression()

# Append list with estimator name and object
estimators.append(("logistic", model1))
model2 = DecisionTreeClassifier()
estimators.append(("cart", model2))
model3 = SVC()
estimators.append(("svm", model3))
model4 = KNeighborsClassifier()
estimators.append(("KNN", model4))
model5 = RandomForestClassifier()
estimators.append(("RFC", model5))
model6 = GaussianNB()
estimators.append(("GB", model6))
model7 = LinearDiscriminantAnalysis()
estimators.append(("LDA", model7))


voting = VotingClassifier(estimators)


results_voting = cross_val_score(voting, X_train_norm, y_train, cv=kfold)

results.append(results_voting)
names.append('Voting')

print('Accuracy: {} SD: {}'.format(results_voting.mean(), results_voting.std()))

# Visualize model accuracies for comparision - boxplots will be appropriate to visualize 
# data variation
'''

plt.boxplot(results, tick_labels=names)
plt.title('Breast Cancer Diagnosis Accuracy using Various Machine Learning Models')
plt.ylabel('Model Accuracy %')
sns.set_style("whitegrid")
plt.ylim(0.8,1)
plt.savefig('graph_3.png')

'''

# Instantiate a new LDA model
lda_2 = LinearDiscriminantAnalysis()

# Fit LDA model to the entire training data
lda_2.fit(X_train_norm, y_train)

# Test LDA model on test data
lda_2_predicted = lda_2.predict(X_test_norm)

# Use sklearn's 'accuracy_score' method to check model accuracy during testing
print('Linear discriminant model analyis Accuracy is: {}'.format(accuracy_score(y_test,lda_2_predicted )))

confusion_matrix_lda = pd.DataFrame(confusion_matrix(y_test, lda_2_predicted), index = ['Actual Negative','Actual Positive'], columns = ['Predicted Negative','Predicted Postive'] )

print('Linear discriminant Model Confusion Matrix')
print(confusion_matrix_lda)

# Parameters
RF_params = {'n_estimators':[10,50,100, 200]}

# Instantiate RFC
RFC_2 = RandomForestClassifier(random_state=42)

# Instantiate gridsearch using RFC model and dictated parameters
RFC_2_grid = GridSearchCV(RFC_2, RF_params, scoring='accuracy', cv=5)

# Fit model to training data
RFC_2_grid.fit(X_train_norm, y_train)

# Extract the best number of estimators
best_estimators = RFC_2_grid.best_params_['n_estimators']

# Print best parameters
print('Best number of estimators: [{}]'.format(best_estimators))

# Train RFC on whole training set

# Instantiate RFC with optimal parameters
RFC_3 = RandomForestClassifier(n_estimators=50, random_state=42)

# Fit RFC to training data
RFC_3.fit(X_train_norm, y_train)

# Predict on training data using fitted RFC

# Evalaute RFC with test data
RFC_3_predicted = RFC_3.predict(X_test_norm)
print('Model accuracy on test data: {}'.format(accuracy_score(y_test, RFC_3_predicted)))

# Create dataframe by zipping RFC feature importances and column names
rfc_features = pd.DataFrame(zip(RFC_3.feature_importances_, bcdf.columns[:-1]), columns = ['Importance', 'Features'])

# Sort in descending order for easy organization and visualization
rfc_features = rfc_features.sort_values(['Importance'], ascending=False)# Create dataframe by zipping RFC feature importances and column names
rfc_features = pd.DataFrame(zip(RFC_3.feature_importances_, bcdf.columns[:-1]), columns = ['Importance', 'Features'])

# Sort in descending order for easy organization and visualization
rfc_features = rfc_features.sort_values(['Importance'], ascending=False)

# Visualize RFC feature importances
sns.barplot(x='Importance', y='Features', data=rfc_features)
plt.title('Feature Importance for Breast Cancer Diagnosis')
sns.set_style("whitegrid")

plt.subplots_adjust(left=0.35)  # Adjust left to give more room for y-labels, for instance

plt.savefig('graph_4.png')
