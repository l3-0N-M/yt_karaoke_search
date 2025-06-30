import sqlite3

conn = sqlite3.connect('karaoke_videos.db')
cursor = conn.cursor()

# Get table schema
cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='videos'")
schema = cursor.fetchone()
if schema:
    print("Videos table schema:")
    print(schema[0])
    print("\n")

# Get column info
cursor.execute("PRAGMA table_info(videos)")
columns = cursor.fetchall()
print("Columns in videos table:")
for col in columns:
    print(f"  {col[1]} ({col[2]})")

conn.close()