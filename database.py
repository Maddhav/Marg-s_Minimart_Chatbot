import sqlite3
from datetime import datetime

DB_PATH = "chat_history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            timestamp TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            customer_name TEXT,
            contact TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_message(session_id, role, content):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (session_id, role, content, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()

def save_lead(session_id, customer_name, contact):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO leads (session_id, customer_name, contact, timestamp) VALUES (?, ?, ?, ?)",
        (session_id, customer_name, contact, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()

def get_all_leads():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT customer_name, contact, timestamp FROM leads ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    return rows

def get_all_conversations():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT session_id, role, content, timestamp 
        FROM messages 
        ORDER BY timestamp DESC
    """)
    rows = c.fetchall()
    conn.close()
    return rows