import pandas as pd
import matplotlib.pyplot as plt
import folium
df = pd.read_csv("Dataset.csv")
df = df[['Restaurant Name', 'City', 'Locality',
         'Latitude', 'Longitude',
         'Cuisines', 'Price range', 'Aggregate rating']]
df = df.dropna()
print("\nCreating Map...")
map_center = [df['Latitude'].mean(), df['Longitude'].mean()]
restaurant_map = folium.Map(location=map_center, zoom_start=5)
for i in range(len(df)):
    folium.CircleMarker(
        location=[df.iloc[i]['Latitude'], df.iloc[i]['Longitude']],
        radius=3,
        popup=df.iloc[i]['Restaurant Name'],
        color='blue',
        fill=True,
        fill_opacity=0.7
    ).add_to(restaurant_map)
restaurant_map.save("restaurant_map.html")
print(" Map saved as 'restaurant_map.html' (Open in browser)")
print("\nTop Cities with Most Restaurants:\n")
city_counts = df['City'].value_counts()
print(city_counts.head(10))

plt.figure()
city_counts.head(10).plot(kind='bar')
plt.title("Top Cities by Number of Restaurants")
plt.xlabel("City")
plt.ylabel("Number of Restaurants")
plt.xticks(rotation=45)
plt.show()
print("\nTop Localities:\n")
locality_counts = df['Locality'].value_counts()
print(locality_counts.head(10))

plt.figure()
locality_counts.head(10).plot(kind='bar')
plt.title("Top Localities by Number of Restaurants")
plt.xlabel("Locality")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()


print("\nAverage Rating by City:\n")
avg_rating_city = df.groupby('City')['Aggregate rating'].mean().sort_values(ascending=False)
print(avg_rating_city.head(10))

plt.figure()
avg_rating_city.head(10).plot(kind='bar')
plt.title("Top Cities by Average Rating")
plt.xlabel("City")
plt.ylabel("Average Rating")
plt.xticks(rotation=45)
plt.show()
print("\nAverage Price Range by City:\n")
avg_price_city = df.groupby('City')['Price range'].mean().sort_values(ascending=False)
print(avg_price_city.head(10))


plt.figure()
avg_price_city.head(10).plot(kind='bar')
plt.title("Top Cities by Price Range")
plt.xlabel("City")
plt.ylabel("Average Price Range")
plt.xticks(rotation=45)
plt.show()

print("\nMost Popular Cuisine by City:\n")

top_cuisine_city = df.groupby('City')['Cuisines'] \
                    .agg(lambda x: x.value_counts().index[0])

print(top_cuisine_city.head(10))
