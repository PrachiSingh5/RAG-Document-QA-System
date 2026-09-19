import sqlite3
import os
import json


DB_DIR = "database"
DB_PATH = os.path.join(DB_DIR, "app.db")


def get_connection():
    """Open a connection to the SQLite database, creating the
    database folder if it doesn't exist yet."""

    os.makedirs(DB_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    return conn


def init_db():
    """Create the users and chat_history tables if they don't
    already exist. Safe to call every app run."""

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            document_names TEXT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    conn.commit()
    conn.close()


def create_user(name, email, password_hash):
    """Insert a new user. Returns (success, message)."""

    conn = get_connection()

    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
            (name, email, password_hash)
        )
        conn.commit()
        return True, "Account created successfully."

    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."

    finally:
        conn.close()


def get_user_by_email(email):
    """Return a user row as a dict, or None if not found."""

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT * FROM users WHERE email = ?", (email,))
    row = cur.fetchone()

    conn.close()

    return dict(row) if row else None


def save_message(user_id, document_names, role, content, sources=None):
    """Persist one chat message (user question or assistant answer)."""

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """INSERT INTO chat_history
           (user_id, document_names, role, content, sources)
           VALUES (?, ?, ?, ?, ?)""",
        (
            user_id,
            document_names,
            role,
            content,
            json.dumps(sources) if sources else None
        )
    )

    conn.commit()
    conn.close()


def delete_user_history(user_id):
    """Delete all saved chat history for this user. Returns the
    number of rows deleted."""

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("DELETE FROM chat_history WHERE user_id = ?", (user_id,))
    deleted_count = cur.rowcount

    conn.commit()
    conn.close()

    return deleted_count


def get_user_history(user_id, limit=30):
    """Return this user's most recent chat messages, newest first."""

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """SELECT * FROM chat_history
           WHERE user_id = ?
           ORDER BY created_at DESC
           LIMIT ?""",
        (user_id, limit)
    )

    rows = cur.fetchall()
    conn.close()

    history = []
    for row in rows:
        item = dict(row)
        if item["sources"]:
            item["sources"] = json.loads(item["sources"])
        history.append(item)

    return history