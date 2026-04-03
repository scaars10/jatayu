import sqlite3

def check_lock():
    conn = sqlite3.connect("data/storage/jatayu.sqlite3", timeout=1.0)
    try:
        conn.execute("BEGIN EXCLUSIVE")
        print("Got exclusive lock!")
    except Exception as e:
        print("Error getting lock:", e)

check_lock()
