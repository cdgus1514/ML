from flask import Flask

app = Flask(__name__)

from flask import Response, make_response

@app.route("/")
def response_test():
    custom_response = Response("Custom Response", 200, {"Program": "Flask Web App"})

    return make_response(custom_response)


@app.before_first_request
def before_first_request():
    print("앱이 기동되고 나서 첫번째 http요청에만 응답")


@app.before_request
def before_request():
    print("매 http요청이 처리되기 전에 실행")


@app.after_request
def after_request(response):
    print("매 http요청이 처리되고 나서 실행")

    return response


@app.teardown_request
def teardown_request(exception):
    print("매 http요청의 결과가 브라우저에 응답하고 나서 호출")


@app.teardown_appcontext
def teardown_appcontext(exception):
    print("http요청의 애플리케이션 컨텍스트가 종료될 때 실행")


if __name__ == "__main__":
    app.run(host="127.0.0.1")