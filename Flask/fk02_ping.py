from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello1818():
    return "<h1>hello world 1818181818181818</h1>"


@app.route("/ping", methods=["GET"])
def hello18182():
    return "<h1>181818181818181818</h1>"


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=False)
    # app.run(host="192.168.0.147", port=8080, debug=False)
