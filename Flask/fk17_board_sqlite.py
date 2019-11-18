from flask import Flask, render_template, request
import sqlite3

app = Flask(__name__)

conn = sqlite3.connect("C:/Study/ML/Flask/wanggun.db")
cursor = conn.cursor()
cursor.execute("SELECT * FROM general")
print(cursor.fetchall())

@app.route("/")
def run():
    conn = sqlite3.connect("C:/Study/ML/Flask/wanggun.db")
    c = conn.cursor()
    c.execute("SELECT * FROM general")
    rows = c.fetchall()

    return render_template("board_index.html", rows=rows)

@app.route("/moid")
def modi():
    id = request.args.get("id")
    conn = sqlite3.connect("C:/Study/ML/Flask/wanggun.db")
    c = conn.cursor()
    c.execute("SELECT * FROM general where id="+str(id))
    rows = c.fetchall()

    return render_template("board_modi.html", rows=rows)


@app.route("/addrec", methods=["POST", "GET"])
def addrec():
    if request.method == "POST":
        try:
            war = request.form["war"]
            id = request.form["id"]

            with sqlite3.connect("wanggun.db") as con:
                cur = con.cursor()

                cur.execute("update general set war ="+str(war)+"where id"+str(id))

                con.commit()
                msg = "정상입력"
        except:
            con.rollback()
            msg = "에러"
        
        finally:
            return render_template("result.html", msg=msg)
            con.close()

app.run(host="0.0.0.0", port=8080, debug=False)