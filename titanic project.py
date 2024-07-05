#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# Load the Titanic dataset
titanic_data = pd.read_csv(r"C:\Users\ACER\Downloads\archive (6)\titanic.csv")
titanic_data


# 

# In[55]:


# Fill missing values
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)
titanic_data['Fare'].fillna(titanic_data['Fare'].median(), inplace=True)


# In[56]:


# Drop columns that won't be used in the model
titanic_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Convert categorical variables to numeric
titanic_data = pd.get_dummies(titanic_data, columns=['Sex', 'Embarked'])
titanic_data['Sex_female'] = titanic_data['Sex_female'].astype(int)
titanic_data['Sex_male'] = titanic_data['Sex_male'].astype(int)
titanic_data['Embarked_C'] = titanic_data['Embarked_C'].astype(int)
titanic_data['Embarked_Q'] = titanic_data['Embarked_Q'].astype(int)
titanic_data['Embarked_S'] = titanic_data['Embarked_S'].astype(int)


# In[57]:


# Create a new feature for family size
titanic_data['FamilySize'] = titanic_data['SibSp'] + titanic_data['Parch'] + 1
titanic_data.drop(['SibSp', 'Parch'], axis=1, inplace=True)


# In[58]:


# Split the data into features and target
X = titanic_data.drop('Survived', axis=1)
y = titanic_data['Survived']

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[60]:


# Define the logistic regression model
model = KNeighborsClassifier(n_neighbors=10)

# Perform K-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=kfold)
# Print cross-validation scores
print("Cross-validation scores:", cv_scores)
print("Mean CV score:", np.mean(cv_scores))


# In[61]:


# Split the data into training and validation sets for final evaluation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)

# Train the logistic regression model on the training set
model.fit(X_train, y_train)

# Evaluate the model on the validation set
y_pred = model.predict(X_val)


# In[62]:


# Print accuracy
accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)

# Print classification report
print("Classification Report:\n", classification_report(y_val, y_pred))


# In[63]:


# Print confusion matrix
conf_matrix = confusion_matrix(y_val, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[ ]:




