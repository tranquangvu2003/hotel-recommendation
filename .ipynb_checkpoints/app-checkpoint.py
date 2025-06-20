from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import mysql.connector

app = Flask(__name__)

# Kết nối database MySQL
def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="hotel_db"
    )

# Huấn luyện mô hình từ dữ liệu DB
def train_model():
    conn = connect_db()
    df_ratings = pd.read_sql("SELECT user_id, hotel_id, rating FROM history_rating", conn)

    df_hotels = pd.read_sql("""
        SELECT h.id AS hotel_id,
               h.location_score AS location_rating,
               h.rating AS overall_rating,
               h.class AS hotel_class,
               GROUP_CONCAT(ha.amenity_name) AS amenities
        FROM hotel h
        LEFT JOIN hotel_amenities ha ON h.id = ha.hotel_id
        GROUP BY h.id
    """, conn)

    df_hotels['amenities'] = df_hotels['amenities'].fillna('').apply(lambda x: x.split(',') if x else [])

    mlb = MultiLabelBinarizer()
    amenities_encoded = mlb.fit_transform(df_hotels['amenities'])
    df_amenities = pd.DataFrame(amenities_encoded, columns=mlb.classes_)
    df_hotels_encoded = df_hotels.drop(columns='amenities').join(df_amenities)
    hotel_features = df_hotels_encoded.set_index('hotel_id')

    joblib.dump({
        'features': hotel_features,
        'mlb': mlb,
        'raw_hotels': df_hotels
    }, "content_based_model.joblib")

    return hotel_features

# Load mô hình khi khởi động server
hotel_features = train_model()

@app.route("/recommend", methods=["GET"])
def recommend():
    user_id = int(request.args.get("user_id"))
    history = {
        1: [2, 3],
        2: [3],
        3: [5],
    }
    liked_hotels = history.get(user_id)
    if not liked_hotels:
        return jsonify({"user_id": user_id, "recommendations": []})
    
    liked_vectors = hotel_features.loc[liked_hotels]
    user_profile = liked_vectors.mean().values.reshape(1, -1)
    sims = cosine_similarity(user_profile, hotel_features)[0]
    sim_df = pd.DataFrame({'hotel_id': hotel_features.index, 'score': sims})
    sim_df = sim_df[~sim_df['hotel_id'].isin(liked_hotels)]
    top_hotels = sim_df.sort_values(by="score", ascending=False).head(3)['hotel_id'].tolist()
    return jsonify({"user_id": user_id, "recommendations": top_hotels})

@app.route("/train", methods=["POST"])
def retrain():
    global hotel_features
    hotel_features = train_model()
    return jsonify({"status": "success", "message": "Model retrained from database."})

if __name__ == "__main__":
    app.run(debug=True)
