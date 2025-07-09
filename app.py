from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import subprocess

app = Flask(__name__)

# Load model
try:
    print("🔄 Loading model...")
    model_data = joblib.load("content_based_model.joblib")
    hotel_features = model_data['features']
    raw_hotels = model_data['raw_hotels']
    user_orders = model_data['user_order']
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Failed to load model:", str(e))
    hotel_features = None


@app.route("/refresh_model", methods=["POST"])
def refresh_model():
    try:
        subprocess.run(["python", "train_model_script.py"], check=True)
        global hotel_features, raw_hotels, user_orders
        model_data = joblib.load("content_based_model.joblib")
        hotel_features = model_data['features']
        raw_hotels = model_data['raw_hotels']
        user_orders = model_data['user_order']
        return jsonify({"message": "Model refreshed successfully!"})
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Training failed: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Hàm đánh giá mức độ phù hợp từ cosine similarity
def get_suitability_label(score):
    if score >= 0.9:
        return "rất phù hợp"
    elif score >= 0.7:
        return "phù hợp"
    elif score >= 0.5:
        return "trung bình"
    else:
        return "ít phù hợp"


@app.route("/recommend", methods=["GET"])
def recommend():
    try:
        user_id = request.args.get("user_id", type=int)
        if user_id is None:
            return jsonify({"error": "Missing user_id"}), 400
        if hotel_features is None:
            return jsonify({"error": "Model not ready"}), 500

        user_history = user_orders[user_orders["user_id"] == user_id]["hotel_id"].tolist()
        if not user_history:
            return jsonify({"user_id": user_id, "recommendations": [], "message": "User has no orders."})

        valid_history = [hid for hid in user_history if hid in hotel_features.index]
        if not valid_history:
            return jsonify({"user_id": user_id, "recommendations": [], "message": "No valid hotels found."})

        user_vector = hotel_features.loc[valid_history].mean().values.reshape(1, -1)
        candidate_hotels = hotel_features.drop(index=valid_history)
        sims = cosine_similarity(user_vector, candidate_hotels)[0]

        sim_df = pd.DataFrame({
            'hotel_id': candidate_hotels.index,
            'score': sims
        }).sort_values(by="score", ascending=False).head(5)

        result = raw_hotels.set_index('hotel_id').loc[sim_df['hotel_id']].copy()
        result['score'] = sim_df.set_index('hotel_id')['score']
        result['suitability'] = result['score'].apply(get_suitability_label)

        return jsonify(result.reset_index().to_dict(orient='records'))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
