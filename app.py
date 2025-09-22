import os
import subprocess
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from processing import transcribe, generate_highlight_clips, combine_clips_into_reel

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
    if "videoFile" not in request.files:
        return "No file uploaded", 400

    file = request.files["videoFile"]
    if file.filename == "":
        return "No file selected", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Extract audio for Whisper
    audio_path = os.path.join(UPLOAD_FOLDER, "audio.wav")
    subprocess.run([
        "ffmpeg", "-y", "-i", filepath, "-ar", "16000", "-ac", "1", audio_path
    ], capture_output=True)

    # Transcribe audio with timestamps
    full_text, raw_segments = transcribe(audio_path)

    # Generate highlight clips
    clips = generate_highlight_clips(filepath, CLIPS_FOLDER, raw_segments, max_clips=3)

    return render_template(
        "index.html",
        video_file=filename,
        audio_file="audio.wav",
        transcript=full_text,
        segments=raw_segments,
        clips=clips
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/clips/<path:filename>")
def serve_clip(filename):
    safe_filename = os.path.basename(filename)
    return send_from_directory(CLIPS_FOLDER, safe_filename)


@app.route("/combine", methods=["POST"])
def combine_clips():
    clip_orders = {}

    # Extract clip orders from form
    for key, value in request.form.items():
        if key.startswith("clipOrder_"):
            clip_name = key[len("clipOrder_"):]
            order = int(value)
            clip_orders[clip_name] = order

    # Sort by user-specified order
    ordered_clips = sorted(clip_orders.keys(), key=lambda x: clip_orders[x])
    print("Clips will be combined in this order:", ordered_clips)

    # Create highlight reel with intro & outro
    highlight_reel = combine_clips_into_reel(ordered_clips, CLIPS_FOLDER)

    # Find original uploaded video (for context display)
    video_file = None
    for filename in os.listdir(UPLOAD_FOLDER):
        if filename.endswith((".mp4", ".mov", ".avi", ".mkv")):
            video_file = filename
            break

    return render_template(
        "index.html",
        video_file=video_file,
        audio_file="audio.wav",
        clips=ordered_clips,
        highlight_reel=highlight_reel
    )


if __name__ == "__main__":
    app.run(debug=True)
