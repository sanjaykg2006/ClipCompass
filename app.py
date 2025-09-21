import os
import subprocess
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from processing import transcribe

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_video():
    # Match the HTML input name
    if "videoFile" not in request.files:
        return "No file uploaded", 400

    file = request.files["videoFile"]
    if file.filename == "":
        return "No file selected", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Extract audio in 16kHz mono for Whisper
    audio_path = os.path.join(UPLOAD_FOLDER, "audio.wav")
    subprocess.run([
        "ffmpeg", "-y", "-i", filepath, "-ar", "16000", "-ac", "1", audio_path
    ])

    # Transcribe with timestamps
    full_text, raw_segments = transcribe(audio_path)
    print("DEBUG segments:", raw_segments)

    return render_template(
        "index.html",
        video_file=filename,
        audio_file="audio.wav",
        transcript=full_text,
        segments=raw_segments
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True)
