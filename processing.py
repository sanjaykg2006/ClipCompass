import os
import uuid
from faster_whisper import WhisperModel
import ffmpeg
import subprocess
import tempfile

# Load Whisper base model on CPU
model = WhisperModel("base", device="cpu")

# Keywords to detect "interesting" segments
HIGHLIGHT_KEYWORDS = ["important", "wow", "amazing", "note", "key", "attention", "highlight"]

def transcribe(audio_path):
    """
    Transcribe audio and return full text + segments.
    Adds keyword count for highlight scoring.
    """
    segments_gen, info = model.transcribe(audio_path, beam_size=5)
    segments = list(segments_gen)

    full_text = ""
    segment_list = []

    for seg in segments:
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

    return full_text.strip(), segment_list


def generate_highlight_clips(video_path, output_folder, segments, max_clips=3):
    """
    Generate highlight clips with burned-in captions and fade-in/out.
    Select segments by keyword count, then duration.
    """
    os.makedirs(output_folder, exist_ok=True)

    highlights = sorted(
        [s for s in segments if s["duration"] > 5],
        key=lambda x: (x["keywords"], x["duration"]),
        reverse=True
    )[:max_clips]

    clip_files = []

    for s in highlights:
        start, end, text = s["start"], s["end"], s["text"]
        clip_filename = f"{uuid.uuid4()}.mp4"
        clip_path = os.path.join(output_folder, clip_filename)

        # Wrap text at 60 chars
        words = text.split()
        lines, current_line = [], []
        for word in words:
            if sum(len(w) for w in current_line) + len(current_line) + len(word) <= 60:
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
        if current_line:
            lines.append(' '.join(current_line))
        wrapped_text = '\n'.join(lines)

        # Escape for FFmpeg drawtext
        safe_text = wrapped_text.replace("'", "").replace(":", "\\:").replace(",", "\\,")

        fade_duration = 0.5
        clip_duration = end - start
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
                acodec="aac", ar="44100", ac="2",
                vcodec="libx264", preset="ultrafast", crf=23,
                movflags="faststart"
            )
            stream.run(overwrite_output=True, capture_stderr=True)
            print(f"Created clip: {clip_path}")
            clip_files.append(os.path.basename(clip_path))

        except ffmpeg.Error as e:
            stderr = e.stderr.decode() if e.stderr else str(e)
            print(f"FFmpeg error for segment {start}-{end}s: {stderr}")

    return clip_files


def combine_clips_into_reel(clip_files, clips_folder, output_file="highlight_reel.mp4"):
    """
    Combine multiple highlight clips into a single video highlight reel with
    intro, outro, and consistent encoding (prevents frame drops).
    """
    if not clip_files:
        print("No clips to combine.")
        return None

    output_path = os.path.join(clips_folder, output_file)

    # --- Create intro & outro cards with silent audio ---
    intro_path = os.path.join(clips_folder, "intro.mp4")
    outro_path = os.path.join(clips_folder, "outro.mp4")

    intro_cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=1280x720:d=3:r=30",
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-vf", "drawtext=text='ClipCompass Highlights':fontcolor=white:fontsize=48:x=(w-text_w)/2:y=(h-text_h)/2",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-c:a", "aac", "-ar", "44100", "-ac", "2",
        "-shortest", intro_path
    ]
    subprocess.run(intro_cmd, check=True)

    outro_cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=1280x720:d=3:r=30",
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-vf", "drawtext=text='Thanks for Watching!':fontcolor=white:fontsize=42:x=(w-text_w)/2:y=(h-text_h)/2",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-c:a", "aac", "-ar", "44100", "-ac", "2",
        "-shortest", outro_path
    ]
    subprocess.run(outro_cmd, check=True)

    # --- Build ffmpeg concat filter ---
    inputs = []
    filter_complex_parts = []
    all_files = [intro_path] + [os.path.join(clips_folder, f) for f in clip_files] + [outro_path]

    for i, file in enumerate(all_files):
        inputs.extend(["-i", file])
        filter_complex_parts.append(f"[{i}:v:0][{i}:a:0]")

    filter_complex = "".join(filter_complex_parts) + f"concat=n={len(all_files)}:v=1:a=1[outv][outa]"

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
    subprocess.run(concat_cmd, check=True)

    print(f"\n✅ Smooth Highlight reel created: {output_path}")
    return output_file
