import sqlite3
import json
from pprint import pprint

# 连接到SQLite数据库（如果不存在则会创建）
conn = sqlite3.connect('example.db')
cursor = conn.cursor()

# 创建表
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    age INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')

# 插入数据
users_to_insert = [
    ('Alice', 'alice@example.com', 30),
    ('Bob', 'bob@example.com', 25),
    ('Charlie', 'charlie@example.com', 35),
    ('Diana', 'diana@example.com', 28)
]

cursor.executemany('INSERT INTO users (name, email, age) VALUES (?, ?, ?)', users_to_insert)
conn.commit()

print("=== SQLite 结构化数据检索 ===")

# 1. 检索所有数据
print("\n1. 所有用户:")
cursor.execute("SELECT * FROM users")
for row in cursor.fetchall():
    print(row)

# 2. 条件查询
print("\n2. 年龄大于28的用户:")
cursor.execute("SELECT * FROM users WHERE age > 28")
for row in cursor.fetchall():
    print(row)

# 3. 模糊查询
print("\n3. 名字中包含 'a' 的用户:")
cursor.execute("SELECT * FROM users WHERE name LIKE '%a%'")
for row in cursor.fetchall():
    print(row)

# 关闭连接
conn.close()
