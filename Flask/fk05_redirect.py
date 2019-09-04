from flask import Flask

app = Flask(__name__)


from flask import redirect

@app.route("/")
def index():

    return redirect("http://cdgus1514.github.io")


@app.route("/dl")
def index2():

    return redirect("https://cdgus1514.github.io/categories/%EB%94%A5%EB%9F%AC%EB%8B%9D/")



if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=False)
