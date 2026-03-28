import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("Dataset.csv")
df = df.dropna(subset=['Cuisines','City'])
df = df[['Restaurant Name',
         'City',
         'Cuisines',
         'Price range',
         'Aggregate rating',
         'Votes',
         'Average Cost for two']]
le = LabelEncoder()
df['City_encoded'] = le.fit_transform(df['City'])
df['Cuisine_encoded'] = le.fit_transform(df['Cuisines'])
scaler = MinMaxScaler()
numeric_features = df[['Price range',
                       'Aggregate rating',
                       'Votes',
                       'Average Cost for two']]

numeric_scaled = scaler.fit_transform(numeric_features)
features = np.hstack((numeric_scaled,
                      df[['City_encoded','Cuisine_encoded']].values))
similarity = cosine_similarity(features)

def recommend_restaurants(restaurant_name, n=5):
    if restaurant_name not in df['Restaurant Name'].values:
        return "Restaurant not found in dataset"
    index = df[df['Restaurant Name'] == restaurant_name].index[0]
    similarity_scores = list(enumerate(similarity[index]))
    similarity_scores = sorted(similarity_scores,
                               key=lambda x: x[1],
                               reverse=True)
    similarity_scores = similarity_scores[1:n+1]
    restaurant_indices = [i[0] for i in similarity_scores]
    return df[['Restaurant Name',
               'City',
               'Cuisines',
               'Price range',
               'Aggregate rating']].iloc[restaurant_indices]
print("Restaurant Recommendation System")
user_input = input("Enter your favorite restaurant: ")
result = recommend_restaurants(user_input)
print("\nRecommended Restaurants:\n")
print(result)