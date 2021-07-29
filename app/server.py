from flask import Flask, render_template
from flask_socketio import SocketIO, emit

from refine_svg import refine_svg

app = Flask(__name__)
socketio = SocketIO(app)


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("svg")
def handle_message(data):
    print("received svg")
    refined_svg = refine_svg(data.decode())
    emit("svg", refined_svg)
    print("sent svg")


if __name__ == "__main__":
    socketio.run(app)
