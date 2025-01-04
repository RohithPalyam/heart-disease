import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

# Load and process the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/heart-disease.data"
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
data = pd.read_csv(url, names=columns)

# Streamlit app setup
st.title("Heart Disease Classification")

# Task 1: Data Analysis
st.header("1. Data Analysis")
with st.echo():
    # Show basic information about the dataset
    st.write("Dataset Description:")
    data_info = data.describe()
    st.write(data_info)

    # Find correlation between features
    st.write("Correlation Matrix:")
    correlation = data.corr()
    st.write(correlation)

# Task 2: Data Visualization
st.header("2. Data Visualization")
with st.echo():
    # Visualize the number of patients with and without heart disease
    st.write("Visualization of Patients with and without Heart Disease:")
    sns.countplot(x='target', data=data)
    plt.title('Patients with/without Heart Disease')
    st.pyplot()

    # Visualize the relationship between age and heart disease
    st.write("Visualization of Age vs Heart Disease:")
    sns.boxplot(x='target', y='age', data=data)
    plt.title('Age vs Heart Disease')
    st.pyplot()

    # Visualize correlation heatmap
    st.write("Correlation Heatmap:")
    plt.figure(figsize=(12,8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    st.pyplot()

# Task 3: Logistic Regression
st.header("3. Logistic Regression")
with st.echo():
    # Split the data into features and target
    X = data.drop('target', axis=1)
    y = data['target']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Build and train the logistic regression model
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)

    # Make predictions
    y_pred = logreg.predict(X_test)

    # Confusion matrix and accuracy score
    cm_logreg = confusion_matrix(y_test, y_pred)
    acc_logreg = accuracy_score(y_test, y_pred)
    
    st.write("Logistic Regression Accuracy:", acc_logreg)
    st.write("Logistic Regression Confusion Matrix:")
    st.write(cm_logreg)
    sns.heatmap(cm_logreg, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
    plt.title('Logistic Regression Confusion Matrix')
    st.pyplot()

# Task 4: Decision Tree
st.header("4. Decision Tree")
with st.echo():
    # Build and train the decision tree model
    dtree = DecisionTreeClassifier(random_state=42)
    dtree.fit(X_train, y_train)

    # Make predictions
    y_pred_dtree = dtree.predict(X_test)

    # Confusion matrix and accuracy score
    cm_dtree = confusion_matrix(y_test, y_pred_dtree)
    acc_dtree = accuracy_score(y_test, y_pred_dtree)

    st.write("Decision Tree Accuracy:", acc_dtree)
    st.write("Decision Tree Confusion Matrix:")
    st.write(cm_dtree)
    sns.heatmap(cm_dtree, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
    plt.title('Decision Tree Confusion Matrix')
    st.pyplot()

    # Visualize the decision tree
    st.write("Decision Tree Visualization:")
    plt.figure(figsize=(12,8))
    plot_tree(dtree, filled=True, feature_names=X.columns, class_names=['No Disease', 'Disease'])
    plt.title('Decision Tree Visualization')
    st.pyplot()

# Task 5: Random Forest
st.header("5. Random Forest")
with st.echo():
    # Build and train the random forest model
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    # Make predictions
    y_pred_rf = rf.predict(X_test)

    # Confusion matrix and accuracy score
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    acc_rf = accuracy_score(y_test, y_pred_rf)

    st.write("Random Forest Accuracy:", acc_rf)
    st.write("Random Forest Confusion Matrix:")
    st.write(cm_rf)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
    plt.title('Random Forest Confusion Matrix')
    st.pyplot()

# Task 6: Model Comparison
st.header("6. Model Comparison")
with st.echo():
    # Print the classification reports for all models
    st.write("Logistic Regression Classification Report:")
    st.write(classification_report(y_test, y_pred))
    st.write("Decision Tree Classification Report:")
    st.write(classification_report(y_test, y_pred_dtree))
    st.write("Random Forest Classification Report:")
    st.write(classification_report(y_test, y_pred_rf))

    # Compare precision, recall, and F1 score
    st.write(f"Logistic Regression - Precision: {precision_score(y_test, y_pred)}, Recall: {recall_score(y_test, y_pred)}, F1: {f1_score(y_test, y_pred)}")
    st.write(f"Decision Tree - Precision: {precision_score(y_test, y_pred_dtree)}, Recall: {recall_score(y_test, y_pred_dtree)}, F1: {f1_score(y_test, y_pred_dtree)}")
    st.write(f"Random Forest - Precision: {precision_score(y_test, y_pred_rf)}, Recall: {recall_score(y_test, y_pred_rf)}, F1: {f1_score(y_test, y_pred_rf)}")

    # Select the best model based on accuracy
    best_model = max([(acc_logreg, 'Logistic Regression'), (acc_dtree, 'Decision Tree'), (acc_rf, 'Random Forest')], key=lambda x: x[0])
    st.write(f"The best model is {best_model[1]} with accuracy {best_model[0]}")
