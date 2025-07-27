import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("books.csv")

# Ensure 'pages' column is numeric, fill missing with median
df["pages"] = pd.to_numeric(df["pages"], errors="coerce")
df["pages"].fillna(df["pages"].median(), inplace=True)

# Normalize the 'pages' column
scaler = MinMaxScaler()
df["pages_normalized"] = scaler.fit_transform(df[["pages"]])

# Save to CSV for your app to load
df.to_csv("books_cleaned.csv", index=False)
