import sqlite3

# Connect to database (or create one if it doesn't exist)
conn = sqlite3.connect("face_database.db")
cursor = conn.cursor()

# Create table to store name and face encoding
cursor.execute("""
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    encoding TEXT
)
""")

conn.commit()
conn.close()
