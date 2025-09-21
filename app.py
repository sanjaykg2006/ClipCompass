from flask import Flask, request, jsonify, render_template, send_from_directory
import os, subprocess

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_video():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    video_filename = "input.mp4"
    audio_filename = "audio.wav"

    video_path = os.path.join(UPLOAD_FOLDER, video_filename)
    audio_path = os.path.join(UPLOAD_FOLDER, audio_filename)

    # Save uploaded video
    file.save(video_path)

    # Extract audio using FFmpeg
    subprocess.run(["ffmpeg", "-y", "-i", video_path, audio_path])

    # Return HTML page with video and audio players
    return f"""
    <h3>Video Uploaded Successfully!</h3>
    <p>Video:</p>
    <video width="480" controls>
        <source src="/uploads/{video_filename}" type="video/mp4">
    </video>
    <p>Extracted Audio:</p>
    <audio controls>
        <source src="/uploads/{audio_filename}" type="audio/wav">
    </audio>
    <br><br>
    <a href="/">Upload another video</a>
    """

# Serve uploaded files
@app.route("/uploads/<filename>")
def serve_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
