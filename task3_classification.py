import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

df = pd.read_csv("Dataset.csv")
df = df[['Restaurant Name', 'Cuisines', 'City', 'Price range', 'Votes', 'Aggregate rating']]
df = df.dropna()
df['Cuisines'] = df['Cuisines'].apply(lambda x: x.split(',')[0].strip())

top3 = df['Cuisines'].value_counts().nlargest(3).index
df = df[df['Cuisines'].isin(top3)]
print("Cuisines:\n", df['Cuisines'].value_counts())
tfidf = TfidfVectorizer(max_features=500)
name_features = tfidf.fit_transform(df['Restaurant Name'])

df['rating_votes'] = df['Aggregate rating'] * df['Votes']
numeric_features = df[['Price range', 'Votes', 'Aggregate rating', 'rating_votes']]
X = hstack([name_features, numeric_features])
le = LabelEncoder()
y = le.fit_transform(df['Cuisines'])
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=25,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\n Predict Cuisine ")

name = input("Enter Restaurant Name: ")
price = int(input("Enter Price Range (1–4): "))
votes = int(input("Enter Votes: "))
rating = float(input("Enter Rating: "))
name_vec = tfidf.transform([name])
rating_votes = rating * votes
numeric_input = np.array([[price, votes, rating, rating_votes]])
input_data = hstack([name_vec, numeric_input])
pred = model.predict(input_data)
print("\nPredicted Cuisine:", le.inverse_transform(pred)[0])