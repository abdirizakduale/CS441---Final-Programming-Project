## Import packages
import pandas as pd
from pandas import set_option
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from matplotlib import rcParams
rcParams['xtick.major.pad'] = 1
rcParams['ytick.major.pad'] = 1

# Load data from a csv file
bcdf = pd.read_csv('data.csv')

# Numerize diagnosis "M" malignant; "B" benign
diagnosis_coder = {'M':1, 'B':0}
bcdf['diagnosis'] = bcdf['diagnosis'].map(diagnosis_coder)

# Drop unnecessary columns
bcdf.drop(['id','Unnamed: 32'], axis = 1, inplace = True)

# Reorder columns so diagnosis is right-most
diagnosis = bcdf['diagnosis']
bcdf.drop('diagnosis', axis=1, inplace=True)
bcdf['Diagnosis'] = diagnosis

# Quick glimpse of the dataset
print(bcdf.head())

# Quick glimpse of tumor features (mean values) in relation to diagnosis
print(bcdf.groupby('Diagnosis').mean())

# Create list of features related to mean tumor characteristics
features_means = list(bcdf.columns[0:10])

# Compute percentage of malignant tumors
malignant_count = bcdf['Diagnosis'].value_counts()[1]
malignant_percent = 100 * malignant_count / len(bcdf)

print("The Percentage of tumors classified as 'malignant' in this data set is: {}".format(malignant_percent))
print('\nA good classifier should therefore outperform blind guessing knowing the proportions i.e. > 62% accuracy')

'''
# Visualize the frequency of diagnoses in the dataset
# Shows how many benign vs malignant tumors are present
outcome_count = bcdf['Diagnosis'].value_counts().rename('Diagnosis').to_frame()
outcome_count.index = ['Benign', 'Malignant']
outcome_count['Percent'] = 100 * outcome_count['Diagnosis'] / sum(outcome_count['Diagnosis'])
outcome_count['Percent'] = outcome_count['Percent'].round().astype('int')

sns.barplot(x = ['Benign', 'Malignant'], y = 'Diagnosis', data = outcome_count, alpha = .8)
plt.title('Frequency of Diagnostic Outcomes in Dataset')
plt.ylabel('Number of Cases')
plt.savefig('diagnosis_frequency.png')

outcome_count = bcdf['Diagnosis'].value_counts()
outcome_count = outcome_count.rename('Diagnosis')
outcome_count = pd.DataFrame(outcome_count)

# Rename the index for clarity
outcome_count.index = ['Benign', 'Malignant']
outcome_count['Percent'] = 100 * outcome_count['Diagnosis'] / sum(outcome_count['Diagnosis'])
outcome_count['Percent'] = outcome_count['Percent'].round().astype('int')

# Visualize the percentage breakdown of diagnoses
sns.barplot(x = ['Benign', 'Malignant'], y = 'Percent', data = outcome_count, alpha = .8, errorbar=None)
plt.title('Percentage of Diagnostic Outcomes in Dataset')
plt.ylabel('Percentage (%)')
plt.ylim(0,100)
plt.savefig('diagnosis_percentage.png')

# Create subsets of the data for benign (Diagnosis=0) and malignant (Diagnosis=1)
bcdf_n = bcdf[bcdf['Diagnosis'] == 0]
bcdf_y = bcdf[bcdf['Diagnosis'] == 1]

# Instantiate a figure object for OOP figure manipulation.
# Adjust figsize as desired to ensure subplots are clearly visible.
fig = plt.figure(figsize=(12, 8))

# Plot histograms in a 3x4 grid for the first 10 features in 'features_means'
for i, b in enumerate(features_means, start=1):
    ax = fig.add_subplot(3, 4, i)
    
    # Plot histograms with KDE for both benign and malignant distributions
    sns.histplot(data=bcdf_n, x=b, kde=True, label='Benign', color='blue', alpha=0.6, ax=ax)
    sns.histplot(data=bcdf_y, x=b, kde=True, label='Malignant', color='red', alpha=0.6, ax=ax)

    ax.set_title(b)
    # Remove individual legends from each subplot to avoid repetition

sns.set_style("whitegrid")
plt.tight_layout()

# Create a single legend for the entire figure
# The handles and labels from the last subplot may not be reliable,
# so we can just create a custom legend.
fig.legend(['Benign', 'Malignant'], loc='upper right', title='Diagnosis')

plt.savefig('distrubtion_variance.png')

# Create subsets of the data for benign (Diagnosis=0) and malignant (Diagnosis=1)
bcdf_n = bcdf[bcdf['Diagnosis'] == 0]
bcdf_y = bcdf[bcdf['Diagnosis'] == 1]

# Instantiate figure object, adjust size as needed
fig = plt.figure(figsize=(12, 8))
fig.suptitle('Comparing Tumor Characteristics by Diagnosis (Mean Features)', fontsize=16)

# Create a 'for loop' to iterate through tumor features and compare with histograms
for i, b in enumerate(features_means, start=1):
    ax = fig.add_subplot(3, 4, i)
    # Plot histograms for benign and malignant distributions
    ax.hist(bcdf_n[b], label='Benign', stacked=True, alpha=0.5, color='blue')
    ax.hist(bcdf_y[b], label='Malignant', stacked=True, alpha=0.5, color='red')
    
    ax.set_title(b)

sns.set_style("whitegrid")

# Adjust layout so that titles and labels fit nicely
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Add a single legend for all subplots
# We can get the handles and labels from the last axis
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', title='Diagnosis')

# Save the figure to a file
plt.savefig('tumor_characteristics_by_diagnosis.png')

# Create subsets of the data for benign (Diagnosis=0) and malignant (Diagnosis=1)
bcdf_n = bcdf[bcdf['Diagnosis'] == 0]
bcdf_y = bcdf[bcdf['Diagnosis'] == 1]

# Instantiate a figure object, adjusting the size as needed
fig = plt.figure(figsize=(12, 8))
fig.suptitle('Distribution of Tumor Characteristics (Features 11-20)', fontsize=16)

# We'll visualize features from columns 10 to 20 (indexing starts at 0)
features_subset = list(bcdf.columns[10:20])

# Create a loop to generate histograms comparing benign vs malignant distributions
for i, b in enumerate(features_subset, start=1):
    ax = fig.add_subplot(3, 4, i)
    # Plot histograms with KDE for both benign and malignant
    sns.histplot(data=bcdf_n, x=b, kde=True, label='Benign', color='blue', alpha=0.6, ax=ax)
    sns.histplot(data=bcdf_y, x=b, kde=True, label='Malignant', color='red', alpha=0.6, ax=ax)
    
    ax.set_title(b)
    # Remove individual legends from each subplot to avoid multiple legends
    if ax.get_legend() is not None:
        ax.get_legend().remove()

sns.set_style("whitegrid")
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Create a single legend for the entire figure
fig.legend(['Benign', 'Malignant'], loc='upper right', title='Diagnosis')

# Save the figure to a file with a descriptive name
plt.savefig('tumor_features_11_to_20_distribution.png')

# Quick visualization of relationships between features and diagnoses

sns.heatmap(bcdf.corr())
sns.set_style("whitegrid")
plt.subplots_adjust(left=0.35)
plt.subplots_adjust(bottom=0.35)
plt.savefig('heatmap_relationships')

# Create subsets of the data for benign (Diagnosis=0) and malignant (Diagnosis=1)
bcdf_n = bcdf[bcdf['Diagnosis'] == 0]
bcdf_y = bcdf[bcdf['Diagnosis'] == 1]

# Columns from 20 to the second last
features_subset = list(bcdf.columns[20:-1])

# Instantiate figure object and add a title
fig = plt.figure(figsize=(12, 8))
fig.suptitle('Distribution of Tumor Characteristics (Features ~21 to Second Last)', fontsize=16)

# Create a loop to visualize these features
for i, b in enumerate(features_subset, start=1):
    ax = fig.add_subplot(3, 4, i)
    sns.histplot(data=bcdf_n, x=b, kde=True, label='Benign', color='blue', alpha=0.6, ax=ax)
    sns.histplot(data=bcdf_y, x=b, kde=True, label='Malignant', color='red', alpha=0.6, ax=ax)
    ax.set_title(b)
    # Remove individual legends from each subplot
    if ax.get_legend() is not None:
        ax.get_legend().remove()

sns.set_style("whitegrid")
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Create a single legend for all subplots
fig.legend(['Benign', 'Malignant'], loc='upper right', title='Diagnosis')

# Save the figure to a file
plt.savefig('tumor_features_21_to_second_last_distribution.png')

'''

# Split data into testing and training set (80% for training)
X_train, X_test, y_train, y_test = train_test_split(bcdf.iloc[:,:-1], bcdf['Diagnosis'], train_size=0.8, random_state=42)

# Normalize features
norm = Normalizer()
norm.fit(X_train)
X_train_norm = norm.transform(X_train)
X_test_norm = norm.transform(X_test)

# Define parameters for optimization
LR_params = {'C': [0.001, 0.1, 1, 10, 100]}
DTC_params = {'criterion': ['entropy', 'gini'], 'max_depth': [10, 50, 100]}
RF_params = {'n_estimators': [10, 50, 100]}

# Models list
models_opt = [
    ('LR', LogisticRegression(), LR_params),
    ('DTC', DecisionTreeClassifier(), DTC_params),
    ('RFC', RandomForestClassifier(), RF_params)
]

results = []
names = []

def estimator_function(parameter_dictionary, scoring='accuracy'):
    for name, model, params in models_opt:
        kfold = KFold(n_splits=5, random_state=2, shuffle=True)
        model_grid = GridSearchCV(model, params, scoring=scoring)
        cv_results = cross_val_score(model_grid, X_train_norm, y_train, cv=kfold, scoring=scoring)
        
        results.append(cv_results)
        names.append(name)
        
        msg = "Cross Validation Accuracy %s: Accuracy: %f SD: %f" % (name, cv_results.mean(), cv_results.std())
        print(msg)

estimator_function(models_opt, scoring='accuracy')

# Identify the best-performing model
mean_scores = [r.mean() for r in results]
best_index = np.argmax(mean_scores)
best_model_name = names[best_index]
print("\nBest model based on cross-validation: {} (Mean Accuracy: {:.4f})".format(best_model_name, mean_scores[best_index]))

# Retrain best model on entire training data
if best_model_name == 'LR':
    best_grid = GridSearchCV(LogisticRegression(), LR_params, scoring='accuracy', cv=5)
    best_grid.fit(X_train_norm, y_train)
    best_estimator = best_grid.best_estimator_
elif best_model_name == 'DTC':
    best_grid = GridSearchCV(DecisionTreeClassifier(), DTC_params, scoring='accuracy', cv=5)
    best_grid.fit(X_train_norm, y_train)
    best_estimator = best_grid.best_estimator_
else: # 'RFC'
    best_grid = GridSearchCV(RandomForestClassifier(), RF_params, scoring='accuracy', cv=5)
    best_grid.fit(X_train_norm, y_train)
    best_estimator = best_grid.best_estimator_

y_test_pred = best_estimator.predict(X_test_norm)
print('\n{} model accuracy on test data: {}'.format(best_model_name, accuracy_score(y_test, y_test_pred)))

conf_matrix = confusion_matrix(y_test, y_test_pred)
confusion_df = pd.DataFrame(conf_matrix, index=['Actual Negative','Actual Positive'], columns=['Predicted Negative','Predicted Positive'])
print('\nConfusion Matrix for {}:'.format(best_model_name))
print(confusion_df)

print('\nClassification Report for {}:'.format(best_model_name))
print(classification_report(y_test, y_test_pred))

'''
# Visualize model accuracies for comparison (Boxplot)
# Compares the cross-validation accuracy distributions of all models side by side
plt.boxplot(results, tick_labels=names)
plt.title('Comparison of Model Accuracies (Boxplot)')
plt.ylabel('Model Accuracy')
sns.set_style("whitegrid")
plt.ylim(0.8,1)
plt.savefig('model_accuracy_boxplot.png')
'''

# Convert results to DataFrame for plotting
all_results_df = pd.DataFrame({name: res for name, res in zip(names, results)})

# 1. Individual Model Performance Distributions
# Histograms + KDE for each model's CV accuracy
for name, cv_res in zip(names, results):
    plt.figure()
    sns.histplot(cv_res, kde=True)
    plt.title(f'Distribution of Cross-Validation Accuracies for {name}')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'graph_{name}_distribution.png')
    plt.close()

# 2. Compare All Models Together Using a Boxplot
plt.figure()
plt.boxplot(results, tick_labels=names)
plt.title('Comparison of Model Accuracies (Boxplot)')
plt.ylabel('Accuracy')
sns.set_style("whitegrid")
plt.ylim(0.8,1)
plt.tight_layout()
plt.savefig('graph_comparison_box.png')
plt.close()

# 3. Compare All Models Using a Violin Plot
# Visualizes the distribution and spread of each model’s accuracy scores
plt.figure()
sns.violinplot(data=all_results_df)
plt.title('Comparison of Model Accuracies (Violin Plot)')
plt.ylabel('Accuracy')
sns.set_style("whitegrid")
plt.ylim(0.8,1)
plt.tight_layout()
plt.savefig('graph_comparison_violin.png')
plt.close()

# 4. Barplot of Mean Accuracies with Standard Deviations
# Summarizes each model's mean accuracy and variability
model_means = [r.mean() for r in results]
model_stds = [r.std() for r in results]

comparison_df = pd.DataFrame({
    'Model': names,
    'MeanAccuracy': model_means,
    'StdDev': model_stds
})

plt.figure()
# With one data point per category, default aggregator mean is fine, just disable errorbars from seaborn:
sns.barplot(data=comparison_df, x='Model', y='MeanAccuracy', errorbar=None)

# Add error bars manually
x_positions = np.arange(len(model_means))
plt.errorbar(x_positions, model_means, yerr=model_stds, fmt='none', ecolor='black', capsize=5)

plt.title('Mean Cross-Validation Accuracy by Model with Std Dev')
plt.ylabel('Accuracy')
sns.set_style("whitegrid")
plt.ylim(0.8,1)
plt.tight_layout()
plt.savefig('model_mean_accuracy_with_std.png')
plt.close()

# If the best model is RFC, visualize feature importances
if best_model_name == 'RFC':
    feature_importances = pd.DataFrame(zip(best_estimator.feature_importances_, bcdf.columns[:-1]), columns=['Importance', 'Features'])
    feature_importances = feature_importances.sort_values(['Importance'], ascending=False)
    plt.figure()
    sns.barplot(x='Importance', y='Features', data=feature_importances)
    plt.title('Feature Importance for Breast Cancer Diagnosis (RFC)')
    sns.set_style("whitegrid")
    plt.subplots_adjust(left=0.35)
    plt.savefig('rfc_feature_importances.png')
    plt.close()