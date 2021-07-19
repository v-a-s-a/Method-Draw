from flask import Flask, jsonify, request, render_template

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/svg", methods=["GET", "POST"])
def svg():
    # GET request
    if request.method == "GET":
        message = {"greeting": "hi from the server"}
        return jsonify(message)  # serialize and use JSON headers

    # POST request
    if request.method == "POST":
        print(request.get_json())  # parse as JSON
        return "Sucesss", 200

