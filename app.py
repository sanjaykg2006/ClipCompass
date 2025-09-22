import os
import subprocess
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from processing import transcribe, generate_highlight_clips, combine_clips_into_reel

app = Flask(__name__, static_folder=os.getcwd())
app.config['SECRET_KEY'] = 'your-secret-key-here'

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
CLIPS_FOLDER = os.path.join(os.getcwd(), "clips")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CLIPS_FOLDER, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_video():
    file = request.files.get("videoFile")
    if not file or file.filename == "":
        return "No file selected", 400

    model_size = request.form.get("model", "base")
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Extract audio for Whisper transcription
    audio_path = os.path.join(UPLOAD_FOLDER, "audio.wav")
    subprocess.run([
        "ffmpeg", "-y", "-i", filepath, "-ar", "16000", "-ac", "1", audio_path
    ], capture_output=True)

    # Transcribe audio
    full_text, raw_segments = transcribe(audio_path, model_size)

    # Generate highlight clips
    clips = generate_highlight_clips(filepath, CLIPS_FOLDER, raw_segments, max_clips=3)

    return render_template(
        "index.html",
        video_file=filename,
        audio_file="audio.wav",
        transcript=full_text,
        segments=raw_segments,
        clips=clips,
        model=model_size
    )


@app.route("/combine", methods=["POST"])
def combine_clips_route():
    # Collect clip order
    clip_orders = {k[len("clipOrder_"):]: int(v) 
                   for k, v in request.form.items() if k.startswith("clipOrder_")}
    ordered_clips = sorted(clip_orders.keys(), key=lambda x: clip_orders[x])

    # Intro/outro customization
    intro_text = request.form.get("introText", "ClipCompass Highlights")
    outro_text = request.form.get("outroText", "Thanks for Watching!")
    font_color_intro = request.form.get("introColor", "white")
    font_color_outro = request.form.get("outroColor", "white")
    font_size_intro = int(request.form.get("introFontSize", 48))
    font_size_outro = int(request.form.get("outroFontSize", 42))
    intro_font = request.form.get("introFont", "arial")
    outro_font = request.form.get("outroFont", "arial")
    fade_duration = float(request.form.get("fadeDuration", 0.5))

    # Combine clips
    highlight_reel = combine_clips_into_reel(
        ordered_clips,
        CLIPS_FOLDER,
        intro_text=intro_text,
        outro_text=outro_text,
        font_color_intro=font_color_intro,
        font_color_outro=font_color_outro,
        font_size_intro=font_size_intro,
        font_size_outro=font_size_outro,
        intro_font=intro_font,
        outro_font=outro_font,
        fade_duration=fade_duration
    )

    video_file = next((f for f in os.listdir(UPLOAD_FOLDER)
                       if f.endswith((".mp4", ".mov", ".avi", ".mkv"))), None)

    return render_template(
        "index.html",
        video_file=video_file,
        audio_file="audio.wav",
        clips=ordered_clips,
        highlight_reel=highlight_reel,
        intro_text=intro_text,
        outro_text=outro_text,
        font_color_intro=font_color_intro,
        font_color_outro=font_color_outro,
        font_size_intro=font_size_intro,
        font_size_outro=font_size_outro,
        intro_font=intro_font,
        outro_font=outro_font,
        fade_duration=fade_duration
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/clips/<path:filename>")
def serve_clip(filename):
    return send_from_directory(CLIPS_FOLDER, os.path.basename(filename))


if __name__ == "__main__":
    try:
        # Try port 5000 first
        port = 5000
        try:
            app.run(host='127.0.0.1', port=port, debug=True)
        except OSError:
            # If port 5000 is in use, try port 5001
            port = 5001
            print(f"Port 5000 is in use, trying port {port}")
            app.run(host='127.0.0.1', port=port, debug=True)
    except Exception as e:
        print(f"Error starting server: {str(e)}")
