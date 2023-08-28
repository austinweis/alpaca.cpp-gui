import os, sys, sqlite3, click
from flask import current_app, g

def get_db():
    db = sqlite3.connect(current_app.config['DATABASE'])
    print(current_app.config['DATABASE'])
    db.row_factory = sqlite3.Row

    return db

def init_db():
    db = get_db()

    with current_app.open_resource('schema.sql') as f:
        db.executescript(f.read().decode())

    print("Initialized the database...")

def delete_db():
    os.remove(current_app.config['DATABASE'])

