from flask import Flask, request, jsonify
import pandas as pd
from sklearn.cluster import KMeans
from io import StringIO
import os

app = Flask(__name__)

@app.route("/predict/", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    content = file.stream.read().decode("utf-8")  # Read file contents
    df = pd.read_csv(StringIO(content))  # Convert to DataFrame

    # Ensure necessary columns exist
    required_columns = ["sales_per_customer", "order_item_total_amount"]  # Use appropriate numeric features

    for col in required_columns:
        if col not in df.columns:
            return jsonify({"error": f"Missing required column: {col}"}), 400

    # Apply KMeans clustering (3 clusters as an example)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["Segment"] = kmeans.fit_predict(df[required_columns])

    return jsonify(df.to_dict(orient="records"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
