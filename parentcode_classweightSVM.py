import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report

# Load training and testing datasets
train_data = pd.read_csv('training.csv')
test_data = pd.read_csv('testing.csv')

# Encode the target variable
label_encoder = LabelEncoder()
train_data['prognosis'] = label_encoder.fit_transform(train_data['prognosis'])
test_data['prognosis'] = label_encoder.transform(test_data['prognosis'])

# Split data into features and labels
X_train = train_data.drop('prognosis', axis=1)
y_train = train_data['prognosis']
X_test = test_data.drop('prognosis', axis=1)
y_test = test_data['prognosis']

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Train the SVM model with class weighting
model = SVC(class_weight=class_weight_dict)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Function for user input prediction
def predict_disease(symptom_vector):
    prediction = model.predict([symptom_vector])
    return label_encoder.inverse_transform(prediction)[0]

# Example usage
sample_input = X_test.iloc[0].values  # Taking a sample test case
predicted_disease = predict_disease(sample_input)
print(f'Predicted Disease: {predicted_disease}')
