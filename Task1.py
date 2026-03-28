import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("Dataset.csv")
df = df.dropna()
df = df.drop(["Restaurant Name", "Restaurant ID"], axis=1, errors='ignore')
label = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = label.fit_transform(df[col])

important_features = [
    "Votes",
    "Price range",
    "Average Cost for two",
    "Has Online delivery",
    "Has Table booking",
    "City",
    "Cuisines"
]

X = df[important_features]
y = df["Aggregate rating"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("\nRandom Forest Model Performance")
print("MSE:", mean_squared_error(y_test, pred))
print("R2 Score:", r2_score(y_test, pred))
plt.figure(figsize=(6,6))
plt.scatter(y_test, pred)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linestyle='--')
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs Predicted Ratings")
plt.show()
print("\nEnter Restaurant Details:\n")
user_data = {}
user_data["Votes"] = float(input("Enter number of votes (e.g., 500): "))
user_data["Price range"] = float(input("Enter price range (1-4): "))
user_data["Average Cost for two"] = float(input("Enter average cost for two: "))
user_data["Has Online delivery"] = float(input("Online delivery (1 = Yes, 0 = No): "))
user_data["Has Table booking"] = float(input("Table booking (1 = Yes, 0 = No): "))
user_data["City"] = float(input("Enter city code (e.g., 0,1,2): "))
user_data["Cuisines"] = float(input("Enter cuisine code (e.g., 0,1,2): "))
user_df = pd.DataFrame([user_data])[important_features]
user_pred = model.predict(user_df)
print("\nPredicted Restaurant Rating:", round(user_pred[0], 2))