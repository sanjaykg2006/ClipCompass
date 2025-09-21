import os
import subprocess
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from processing import transcribe, generate_highlight_clips

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
CLIPS_FOLDER = os.path.join(os.getcwd(), "clips")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CLIPS_FOLDER, exist_ok=True)


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

    # Generate highlight clips (first 3 segments longer than 5s)
    clips = generate_highlight_clips(filepath, CLIPS_FOLDER, raw_segments, max_clips=3)

    return render_template(
        "index.html",
        video_file=filename,
        audio_file="audio.wav",
        transcript=full_text,
        segments=raw_segments,
        clips=clips  # list of clip filenames
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/clips/<path:filename>")
def serve_clip(filename):
    # Ensure we only use the basename of the filename for security
    safe_filename = os.path.basename(filename)
    return send_from_directory(CLIPS_FOLDER, safe_filename)


if __name__ == "__main__":
    app.run(debug=True)
