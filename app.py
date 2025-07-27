import pandas as pd
from flask import Flask, render_template, request
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)
df = pd.read_csv("books_cleaned.csv")

knn = NearestNeighbors(n_neighbors=5, metric="euclidean")
knn.fit(df[["pages_normalized"]])

def recommend_books(pages_per_minute, available_time_hours):
    total_pages = pages_per_minute * available_time_hours * 60
    min_p, max_p = df["pages"].min(), df["pages"].max()
    normalized_pages = (total_pages - min_p) / (max_p - min_p)
    distances, indices = knn.kneighbors([[normalized_pages]])
    return df.iloc[indices[0]]["title"].tolist()

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    if request.method == "POST":
        pages = int(request.form["pages"])
        time_spent = int(request.form["time"])
        available_time = int(request.form["duration"])
        pages_per_minute = pages / time_spent
        recommendations = recommend_books(pages_per_minute, available_time)
    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
