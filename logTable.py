import sqlite3 as sl

con = sl.connect('parking-log.db')

with con:
    con.execute("""
        CREATE TABLE LOG (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            datetime DATETIME,
            message TEXT
        );
    """)