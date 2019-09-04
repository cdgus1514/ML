import pymssql
from collections.abc import Iterable
import numpy as np

conn = pymssql.connect(server="localhost", user="bit", password="1234", database="bitdb")

cursor = conn.cursor()

# cursor.execute("SELECT TOP(1000)* FROM train;")
cursor.execute("SELECT * FROM iris2;")

# row = cursor.fetchall()
row = cursor.fetchall()
print(row)

test = np.array(row)
print(test)
print(type(test))
print(test.shape)
# while row:
#     print(row[0], row[1], row[2], row[3], row[4])
#     row = cursor.fetchone()

# sql = "INSERT INTO iris2 (SepalLength, SepalWidth, PetalLength, PetalWidth) values (18, 18, 18, 18);"
# cursor.execute(sql)

conn.close()