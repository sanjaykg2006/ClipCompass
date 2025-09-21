import os
import uuid
from faster_whisper import WhisperModel
import ffmpeg

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

    # Pick segments longer than 5s, sort by keyword count then duration
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

        # Filters
        fade_duration = 0.5
        clip_duration = end - start
        draw_text = (
            f"drawtext=fontfile=/Windows/Fonts/arial.ttf:"
            f"fontsize=28:fontcolor=white:x=(w-text_w)/2:y=h-th-100:"
            f"box=1:boxcolor=black@0.25:text='{safe_text}'"
        )
        fade_filter = f"fade=t=in:st=0:d={fade_duration},fade=t=out:st={clip_duration-fade_duration}:d={fade_duration}"
        vf_filter = f"{draw_text},{fade_filter}"

        try:
            stream = ffmpeg.input(video_path, ss=start, t=clip_duration)
            stream = ffmpeg.output(
                stream,
                clip_path,
                vf=vf_filter,
                acodec="aac",
                vcodec="libx264",
                video_bitrate="2M",
                preset="ultrafast",
                movflags="faststart"
            )

            # Debug: print FFmpeg command
            cmd = ffmpeg.compile(stream)
            print("FFmpeg command:", " ".join(cmd))

            stream.run(overwrite_output=True, capture_stderr=True)
            print(f"Created clip: {clip_path}")
            clip_files.append(os.path.basename(clip_path))

        except ffmpeg.Error as e:
            stderr = e.stderr.decode() if e.stderr else str(e)
            print(f"FFmpeg error for segment {start}-{end}s: {stderr}")

    return clip_files
