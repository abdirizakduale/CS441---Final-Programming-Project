# CS441-Final-Programming-Project

# README

## Overview
This script provides a comprehensive analysis of a breast cancer dataset. It loads and preprocesses the data, explores the relationships between tumor characteristics and diagnoses, trains multiple machine learning models, and visualizes performance metrics and distributions. The workflow helps users understand how various features differ between benign and malignant tumors, and which classifiers perform best at predicting malignant cases.

## Features of the Analysis
1. **Data Loading and Cleaning**:  
   The dataset is loaded from `data.csv`. Diagnosis values are converted to numeric form (`M` to `1` for malignant, `B` to `0` for benign), and unnecessary columns are removed. The diagnosis column is then positioned as the target variable on the far right.

2. **Exploratory Data Analysis**:  
   The code displays the first few rows of the dataset and calculates mean values of tumor characteristics grouped by diagnosis. It also computes the proportion of malignant tumors. Various visualizations show the distribution of tumor characteristics for different subsets of features, allowing a clear comparison between benign and malignant tumors.

3. **Visualizations**:  
   - Bar plots show the frequency and percentage of benign vs. malignant diagnoses.  
   - Histograms and KDE plots illustrate how tumor characteristics differ between benign and malignant groups across various sets of features. This reveals patterns such as feature distributions where malignant tumors tend to have higher or lower values compared to benign ones.  
   - Heatmaps show correlations between features and the diagnosis, indicating which features might be more predictive.  
   - Stacked histograms and boxplots offer different perspectives on data distributions, highlighting variation and outliers.  
   - Additional visual comparisons break down tumor characteristics into different feature ranges, ensuring a full understanding of the entire feature set.

4. **Machine Learning Modeling**:  
   The data is split into training and testing sets, and features are normalized. Three models—Logistic Regression, Decision Tree, and Random Forest—are tuned using grid search and cross-validation. Their mean accuracies and standard deviations are computed.

   After determining the best model based on cross-validation performance, that model is refitted on the entire training set and evaluated on the test set. The code prints the confusion matrix and classification report, showing precision, recall, and f1-score, which provide detailed insights into classification performance.

5. **Model Performance Comparisons**:  
   The script visualizes model performance in multiple ways:
   - Distributions of cross-validation accuracies for each model, revealing consistency and reliability.
   - Boxplots, violin plots, and bar charts with error bars compare the performance of all models side-by-side. These summaries highlight which model is the most accurate and stable.
   - If the Random Forest is the best model, a feature importance plot is generated, showing which attributes contribute most strongly to classification.

## Getting Started

### Prerequisites
Ensure you have Python 3.7+ installed. You will also need the following Python libraries:
- pandas
- matplotlib
- seaborn
- numpy
- scipy
- scikit-learn

Install them via:
```bash
pip install pandas matplotlib seaborn numpy scipy scikit-learn
```

### Cloning the Repository
To get the code:
```bash
git clone https://github.com/abdirizakduale/CS441-Final-Programming-Project.git
cd CS441-Final-Programming-Project
```

### Running the Code
After installing the necessary libraries, you can run:
```bash
python3 Final_Project.py
```

The script will process the data, run the analyses, and generate various saved figures (Please comment out the graphs that you want one at a time). The terminal output will show summary statistics, cross-validation accuracies, the best model identified, test-set performance metrics, and a classification report. The created graphs will be saved as PNG files in the current directory.

This setup provides a complete, end-to-end approach for understanding breast cancer tumor characteristics, evaluating different machine learning models, and presenting results through comprehensive visualizations.
