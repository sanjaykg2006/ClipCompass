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

            cmd = ffmpeg.compile(stream)
            print("FFmpeg command:", " ".join(cmd))
            stream.run(overwrite_output=True, capture_stderr=True)
            print(f"Created clip: {clip_path}")
            clip_files.append(os.path.basename(clip_path))

        except ffmpeg.Error as e:
            stderr = e.stderr.decode() if e.stderr else str(e)
            print(f"FFmpeg error for segment {start}-{end}s: {stderr}")

    return clip_files


def combine_clips_into_reel(clip_files, clips_folder, output_file="highlight_reel.mp4"):
    """
    Combine multiple highlight clips into a single video highlight reel with fade transitions.
    """
    if not clip_files:
        print("No clips to combine.")
        return None

    print("\nStarting clip combination process...")
    print(f"Number of clips to combine: {len(clip_files)}")
    
    # Full paths of clips
    clip_paths = [os.path.join(clips_folder, f) for f in clip_files]
    
    # Validate all clips exist and are readable
    print("\nValidating input clips:")
    for clip_path in clip_paths:
        if not os.path.exists(clip_path):
            print(f"ERROR: Clip not found: {clip_path}")
            return None
        try:
            size = os.path.getsize(clip_path)
            if size == 0:
                print(f"ERROR: Clip is empty: {clip_path}")
                return None
            print(f"✓ Validated {os.path.basename(clip_path)}: {size/1024/1024:.2f} MB")
        except OSError as e:
            print(f"ERROR: Cannot access clip {clip_path}: {str(e)}")
            return None

    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt") as tf:
        for clip_path in clip_paths:
            tf.write(f"file '{clip_path.replace('\\', '/')}'\n")
        concat_list_file = tf.name

    output_path = os.path.join(clips_folder, output_file)

    try:
        print("\nCombining clips into highlight reel:")
        print("=" * 80)
        
        # Print input clips information and verify order
        print("\n1. Input Clips:")
        for i, clip_path in enumerate(clip_paths):
            clip_size = os.path.getsize(clip_path) / (1024*1024)  # Convert to MB
            print(f"Clip {i+1}: {os.path.basename(clip_path)} ({clip_size:.2f} MB)")
            
        # Verify correct clip order based on file names
        clip_paths = sorted(clip_paths, key=lambda x: x.lower())  # Sort clips by name
        print("\nClips after sorting:")
        for i, clip_path in enumerate(clip_paths):
            print(f"Clip {i+1}: {os.path.basename(clip_path)}")
        
        # Build the filter complex chain
        filter_complex = []
        inputs = []
        
        print("\n2. Building Filter Complex Chain:")
        print(f"Building chain for {len(clip_paths)} clips...")
        
        # First add the input references
        for i, clip_path in enumerate(clip_paths):
            inputs.extend(["-i", clip_path])
            print(f"\nAdding clip {i+1}: {os.path.basename(clip_path)}")
            # Split audio and video streams for each clip
            filter_complex.append(f"[{i}:v]setpts=PTS-STARTPTS[v{i}]")
            filter_complex.append(f"[{i}:a]asetpts=PTS-STARTPTS[a{i}]")
        
        # Build concatenation chain in the correct order
        video_inputs = ''.join(f'[v{i}]' for i in range(len(clip_paths)))
        audio_inputs = ''.join(f'[a{i}]' for i in range(len(clip_paths)))
        
        # Add concat filter for both video and audio
        filter_complex.append(
            f"{video_inputs}concat=n={len(clip_paths)}:v=1:a=0[vout];"
            f"{audio_inputs}concat=n={len(clip_paths)}:v=0:a=1[aout]"
        )
        
        print(f"Created concatenation chain for {len(clip_paths)} clips")
        
        # Join all filters with semicolon
        filter_str = ";".join(filter_complex)
        
        if not filter_complex:
            print("WARNING: No transitions were created in filter chain")
        
        print("\n3. Final Filter Chain:")
        print(filter_str)
        
        # Build final command
        cmd = ["ffmpeg", "-y"] + inputs
        
        # Use consistent output mapping for concatenated streams
        video_map = "[vout]"
        audio_map = "[aout]"
        
        cmd.extend([
            "-filter_complex", filter_str,
            "-map", video_map,
            "-map", audio_map,  # Map the concatenated audio
            "-c:v", "libx264",
            "-c:a", "aac",  # Explicitly set audio codec
            "-crf", "23",
            "-preset", "veryfast",
            output_path
        ])
        
        print("\n4. FFmpeg Command:")
        print("Command components:")
        print("  Input files:", inputs)
        print("  Filter complex:", filter_str)
        print("  Video output mapping:", video_map)
        print("  Audio output mapping:", audio_map)
        print("\nFull command:")
        print(" ".join(cmd))
        print("\n5. Executing command...")
        
        try:
            # Run FFmpeg with both stdout and stderr captured
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
        except subprocess.CalledProcessError as e:
            print("\nFFmpeg Error:")
            print("Return code:", e.returncode)
            print("\nError output:")
            print(e.stderr if e.stderr else "No error output available")
            print("\nStandard output:")
            print(e.stdout if e.stdout else "No standard output available")
            raise
        
        print("\nFFmpeg Output:")
        print(result.stdout)
        if result.stderr:
            print("\nFFmpeg Errors/Warnings:")
            print(result.stderr)
        
        print(f"\nHighlight reel created: {output_path}")
        print("=" * 80)
        return output_file
    finally:
        if os.path.exists(concat_list_file):
            os.remove(concat_list_file)
