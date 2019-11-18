from wsgiref.simple_server import make_server

def my_app(environ, start_response):
    status = "200 OK"
    headers = [("Content-Type", "text/plain")]
    start_response(status, headers)

    response = [b"This is a sample WSGI Application."]

    return response

if __name__ == "__main__":
    print("Started WSGI Server on port 8080")
    server = make_server("", 8080, my_app)
    server.serve_forever()