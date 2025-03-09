import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset

from google.colab import files
uploaded = files.upload()

# Load training and testing datasets

train_data = pd.read_csv('training.csv')
test_data = pd.read_csv('testing.csv')

# Step 2: Data Preprocessing //////

# Encoding the target variable

label_encoder = LabelEncoder()
train_data['prognosis'] = label_encoder.fit_transform(train_data['prognosis'])
test_data['prognosis'] = label_encoder.transform(test_data['prognosis'])

# Splitting data into features and labels

X_train = train_data.drop('prognosis', axis=1)
y_train = train_data['prognosis']
X_test = test_data.drop('prognosis', axis=1)
y_test = test_data['prognosis']

# Step 3: Train the model

model = SVC()
model.fit(X_train, y_train)

# Step 4: Make predictions

y_pred = model.predict(X_test)

# Step 5: Evaluate the model

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Step 6: Create a function for user input prediction

def predict_disease(symptom_vector):
prediction = model.predict([symptom_vector])
return label_encoder.inverse_transform(prediction)[0]

# Example usage

sample_input = X_test.iloc[0].values  # Taking a sample test case
predicted_disease = predict_disease(sample_input)
print(f'Predicted Disease: {predicted_disease}')
