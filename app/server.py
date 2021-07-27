from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import json
import xml.etree.ElementTree as etree

from refine_svg import refine_svg

app = Flask(__name__)
socketio = SocketIO(app)


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("svg")
def handle_message(data):
    print("received message: " + str(data))
    root = etree.fromstring(data)
    refined_svg = refine_svg(root)
    # emit("svg", refined_svg)
    # print("sent svg")


if __name__ == "__main__":
    socketio.run(app)
