from sqlalchemy import create_engine
from sqlalchemy.engine import URL

url = URL.create(
    "mysql+pymysql",
    username="root",
    password="",
    host="127.0.0.1",
    database="hotel_db"
)
engine = create_engine(url)
with engine.connect() as conn:
    result = conn.exec_driver_sql("SELECT 1")
    print(result.fetchone())