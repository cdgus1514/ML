import pyodbc as pyo


server = "localhost"
database = "bitdb"
username = "bit"
password = "1234"

cnxn = pyo.connect("DRIVER={ODBC DRIVER 13 for SQL SERVER}; SERVER=" +server+"; PORT=1433; DATABASE="+database+";UID="+username+";PWD="+password)

cursor = cnxn.cursor()

tsql = "SELECT * FROM iris2;"


from flask import Flask, render_template

app = Flask(__name__)

@app.route("/sqltable")
def showsql():
    cursor.execute(tsql)

    return render_template("myweb.html", rows=cursor.fetchall())

if __name__ == "__main__":
    app.run(host="127.0.0.1")

# with cursor.execute(tsql):
#     row = cursor.fetchone()

#     while row:
#         print(str(row[0])+" "+str(row[1])+" "+str(row[2])+" "+str(row[3])+" "+str(row[4]))
#         row = cursor.fetchone()

cnxn.close()