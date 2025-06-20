import pandas as pd
import joblib
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
from sklearn.preprocessing import MultiLabelBinarizer

def connect_sqlalchemy():
    print("📡 Connecting to MySQL...")
    connection_url = URL.create(
        drivername="mysql+pymysql",
        username="root",
        password="",
        host="127.0.0.1",
        database="hotel_db"
    )
    engine = create_engine(connection_url, connect_args={"connect_timeout": 5})
    print("✅ SQLAlchemy engine created.")
    return engine

def train_model():
    engine = connect_sqlalchemy()

    try:
        print("📥 Reading user_order data...")
        df_orders = pd.read_sql("SELECT user_id, hotel_id FROM user_order", con=engine)
        print(f"✅ Orders loaded: {len(df_orders)} rows")
    except Exception as e:
        print("❌ Failed to read user_order:", e)
        return

    try:
        print("📥 Reading hotel data...")
        df_hotels = pd.read_sql("""
            SELECT h.id AS hotel_id,
                   h.location_rating,
                   h.overall_rating,
                   h.hotel_class,
                   GROUP_CONCAT(ha.amenity) AS amenities
            FROM hotel h
            LEFT JOIN hotel_amenities ha ON h.id = ha.hotel_id
            GROUP BY h.id
        """, con=engine)
        print(f"✅ Hotels loaded: {len(df_hotels)} rows")
    except Exception as e:
        print("❌ Failed to read hotels:", e)
        return

    try:
        print("🧹 Processing amenities...")
        df_hotels['amenities'] = df_hotels['amenities'].fillna('').apply(lambda x: x.split(',') if x else [])
        mlb = MultiLabelBinarizer()
        amenities_encoded = mlb.fit_transform(df_hotels['amenities'])
        df_amenities = pd.DataFrame(amenities_encoded, columns=mlb.classes_)

        df_hotels_encoded = df_hotels.drop(columns='amenities').join(df_amenities)
        hotel_features = df_hotels_encoded.set_index('hotel_id')
        hotel_features = hotel_features.apply(pd.to_numeric, errors='coerce').fillna(0)

        print("💾 Saving model to content_based_model.joblib...")
        joblib.dump({
            'features': hotel_features,
            'raw_hotels': df_hotels,
            'user_order': df_orders
        }, "content_based_model.joblib")
        print("✅ Model trained and saved to content_based_model.joblib")
    except Exception as e:
        print("❌ Error during processing/saving model:", e)

if __name__ == "__main__":
    train_model()
