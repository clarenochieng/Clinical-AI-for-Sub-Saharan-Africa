import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

symptomsData = pd.read_csv("dataset.csv")
symptomColumns = [f'Symptom_{i}' for i in range(1, 18)]
symptomsData['Symptoms'] = symptomsData[symptomColumns].fillna('').agg(' '.join, axis=1)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(symptomsData['Symptoms'])

y = symptomsData["Disease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

def predict_disease(symptoms, model, vectorizer):
    symptoms_vectorized = vectorizer.transform([symptoms])
    prediction = model.predict(symptoms_vectorized)
    return prediction[0]
