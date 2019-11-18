import pymssql
from collections.abc import Iterable

conn = pymssql.connect(server="localhost", user="bit", password="1234", database="bitdb")

cursor = conn.cursor()

# cursor.execute("SELECT TOP(1000)* FROM train;")

# # row = cursor.fetchall()
# row = cursor.fetchone()

# while row:
#     print(row[0], row[1], row[2], row[3], row[4])
#     row = cursor.fetchone()


sql = "INSERT INTO iris2 (SepalLength, SepalWidth, PetalLength, PetalWidth, na) values (1.9, 1.9, 1.9, 1.9);"
cursor.execute(sql)
conn.commit()   # insert 후 commit 필수

conn.close()