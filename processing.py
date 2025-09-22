import os
import uuid
from faster_whisper import WhisperModel
import ffmpeg
import subprocess

# ---------------- Whisper Model (CPU only) ----------------
def load_model(model_size="base"):
    return WhisperModel(model_size, device="cpu")

# Keywords to detect "interesting" segments
HIGHLIGHT_KEYWORDS = ["important", "wow", "amazing", "note", "key", "attention", "highlight"]

def transcribe(audio_path, model_size="base", progress_callback=None):
    """
    Transcribe audio and return full text + segments with keyword scoring.
    Reports progress via progress_callback(percent)
    """
    model = load_model(model_size)
    segments_gen, info = model.transcribe(audio_path, beam_size=5)
    segments = list(segments_gen)

    full_text = ""
    segment_list = []

    total_segments = len(segments)
    for idx, seg in enumerate(segments, start=1):
        text = seg.text.strip()
        if text:
            keywords = sum(1 for kw in HIGHLIGHT_KEYWORDS if kw.lower() in text.lower())
            segment_list.append({
                "start": float(seg.start),
                "end": float(seg.end),
                "text": text,
                "duration": float(seg.end - seg.start),
                "keywords": keywords
            })
            full_text += text + " "

        # Report transcription progress (0-40%)
        if progress_callback:
            progress_callback(min(40, idx / total_segments * 40))

    return full_text.strip(), segment_list


def generate_highlight_clips(video_path, output_folder, segments, max_clips=3, max_total_duration=90, progress_callback=None):
    """
    Generate highlight clips with burned-in captions and fade-in/out.
    Limits total duration of clips to max_total_duration seconds.
    Reports progress via progress_callback(percent)
    """
    os.makedirs(output_folder, exist_ok=True)

    # Filter and sort by keyword count & duration
    highlights = sorted(
        [s for s in segments if s["duration"] > 5],
        key=lambda x: (x["keywords"], x["duration"]),
        reverse=True
    )

    clip_files = []
    total_duration = 0
    total_highlights = len(highlights)

    for idx, s in enumerate(highlights):
        if len(clip_files) >= max_clips or total_duration >= max_total_duration:
            break

        start, end, text = s["start"], s["end"], s["text"]
        clip_duration = end - start

        if total_duration + clip_duration > max_total_duration:
            clip_duration = max_total_duration - total_duration
            end = start + clip_duration

        clip_filename = f"{uuid.uuid4()}.mp4"
        clip_path = os.path.join(output_folder, clip_filename)

        # Wrap text at 60 chars
        words, lines, current_line = text.split(), [], []
        for word in words:
            if sum(len(w) for w in current_line) + len(current_line) + len(word) <= 60:
                current_line.append(word)
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
        if current_line:
            lines.append(" ".join(current_line))
        wrapped_text = "\n".join(lines)

        safe_text = wrapped_text.replace("'", "").replace(":", "\\:").replace(",", "\\,")

        fade_duration = 0.5
        draw_text = (
            f"drawtext=fontfile=/Windows/Fonts/arial.ttf:"
            f"fontsize=28:fontcolor=white:x=(w-text_w)/2:y=h-th-100:"
            f"box=1:boxcolor=black@0.25:text='{safe_text}'"
        )
        fade_filter = f"fade=t=in:st=0:d={fade_duration},fade=t=out:st={clip_duration-fade_duration}:d={fade_duration}"
        vf_filter = f"{draw_text},{fade_filter},scale=1280:720,fps=30"

        try:
            stream = ffmpeg.input(video_path, ss=start, t=clip_duration)
            stream = ffmpeg.output(
                stream,
                clip_path,
                vf=vf_filter,
                acodec="aac", ar="44100", ac=2,
                vcodec="libx264", preset="ultrafast", crf=23,
                movflags="faststart"
            )
            stream.run(overwrite_output=True, capture_stderr=True)
            clip_files.append(os.path.basename(clip_path))
            total_duration += clip_duration

            # Report clip generation progress (40-70%)
            if progress_callback and total_highlights:
                progress_callback(40 + idx / total_highlights * 30)

        except ffmpeg.Error as e:
            stderr = e.stderr.decode() if e.stderr else str(e)
            print(f"FFmpeg error for segment {start}-{end}s: {stderr}")

    # Finalize progress for clips
    if progress_callback:
        progress_callback(70)

    return clip_files


def combine_clips_into_reel(
        clip_files, clips_folder, output_file="highlight_reel.mp4",
        intro_text="ClipCompass Highlights", outro_text="Thanks for Watching!",
        font_size_intro=48, font_size_outro=42,
        font_color_intro="white", font_color_outro="white",
        intro_font="arial", outro_font="arial",
        fade_duration=0.5, progress_callback=None
    ):
    """
    Combine multiple highlight clips into a single reel with intro/outro.
    Accepts a progress_callback(percent) to report progress.
    Reports progress from 70-100%.
    """
    if not clip_files:
        print("No clips to combine.")
        return None

    print(f"Starting to combine {len(clip_files)} clips...")
    output_path = os.path.join(clips_folder, output_file)

    # --- Create intro & outro cards ---
    intro_path = os.path.join(clips_folder, "intro.mp4")
    outro_path = os.path.join(clips_folder, "outro.mp4")

    # Font mapping for Windows fonts
    FONT_MAP = {
        "arial": "/Windows/Fonts/arial.ttf",
        "times": "/Windows/Fonts/times.ttf",
        "georgia": "/Windows/Fonts/georgia.ttf",
        "verdana": "/Windows/Fonts/verdana.ttf",
        "impact": "/Windows/Fonts/impact.ttf",
        "comic": "/Windows/Fonts/comic.ttf"
    }

    for text, path, font_size, font_color, font_name, step in [
        (intro_text, intro_path, font_size_intro, font_color_intro, intro_font, 0.71),
        (outro_text, outro_path, font_size_outro, font_color_outro, outro_font, 0.72)
    ]:
        print(f"\nCreating {'intro' if 'intro' in path else 'outro'} card...")
        print(f"Text: {text}")
        print(f"Font size: {font_size}")
        print(f"Font color: {font_color}")
        
        # Replace problematic characters
        safe_text = text.replace("'", "").replace(":", "\\:").replace(",", "\\,")
        
        # Fixed fade timing logic
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=1280x720:d=3:r=30",
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-vf", 
            f"drawtext=fontfile={FONT_MAP.get(font_name, FONT_MAP['arial'])}:text='{safe_text}':fontcolor={font_color}:fontsize={font_size}:"
            f"x=(w-text_w)/2:y=(h-text_h)/2:box=1:boxcolor=black@0.5,"
            f"fade=t=in:st=0:d={fade_duration},"
            f"fade=t=out:st={3-fade_duration}:d={fade_duration}",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-c:a", "aac", "-ar", "44100", "-ac", "2",
            "-shortest", path
        ]
        
        print("Executing FFmpeg command:")
        print(" ".join(cmd))
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"Successfully created {path}")
        except subprocess.CalledProcessError as e:
            print(f"Error creating {'intro' if 'intro' in path else 'outro'} card:")
            print(f"Command failed with return code {e.returncode}")
            print("Error output:")
            print(e.stderr)
            raise
        if progress_callback:
            progress_callback(int(step*100))

    # --- Concatenate clips ---
    print("\nPreparing to concatenate clips...")
    all_files = [intro_path] + [os.path.join(clips_folder, f) for f in clip_files] + [outro_path]
    inputs, filter_parts = [], []

    print("Files to concatenate:")
    for idx, file in enumerate(all_files):
        print(f"{idx + 1}. {file}")
        if not os.path.exists(file):
            raise FileNotFoundError(f"File not found: {file}")
        inputs.extend(["-i", file])
        filter_parts.append(f"[{idx}:v:0][{idx}:a:0]")

    filter_complex = "".join(filter_parts) + f"concat=n={len(all_files)}:v=1:a=1[outv][outa]"

    concat_cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "faststart",
        output_path
    ]

    print("\nExecuting final concatenation command:")
    print(" ".join(concat_cmd))

    try:
        result = subprocess.run(concat_cmd, check=True, capture_output=True, text=True)
        print("Concatenation successful!")
        print("FFmpeg output:")
        print(result.stderr)
    except subprocess.CalledProcessError as e:
        print("Error during concatenation:")
        print(f"Command failed with return code {e.returncode}")
        print("Error output:")
        print(e.stderr)
        raise

    if progress_callback:
        progress_callback(100)

    print(f"\n✅ Highlight reel created: {output_path}")
    return output_file
