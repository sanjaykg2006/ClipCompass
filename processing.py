import os
import uuid
import tempfile
from faster_whisper import WhisperModel
import ffmpeg

# Load a more robust model (base) on CPU
model = WhisperModel("base", device="cpu")


def transcribe(audio_path):
    """
    Transcribe audio and return full text + segments.
    """
    segments_gen, info = model.transcribe(audio_path, beam_size=5)

    # Convert generator to list
    segments = list(segments_gen)

    full_text = ""
    segment_list = []

    for seg in segments:
        text = seg.text.strip()
        if text:
            segment_list.append({
                "start": float(seg.start),
                "end": float(seg.end),
                "text": text
            })
            full_text += text + " "

    return full_text.strip(), segment_list


def generate_highlight_clips(video_path, output_folder, segments, max_clips=3):
    """
    Generate highlight clips with burned-in captions.
    Picks the first few segments longer than 5s.
    """
    os.makedirs(output_folder, exist_ok=True)
    highlights = [s for s in segments if s["end"] - s["start"] > 5][:max_clips]

    clip_files = []

    for s in highlights:
        start, end, text = s["start"], s["end"], s["text"]
        clip_filename = f"{uuid.uuid4()}.mp4"
        clip_path = os.path.join(output_folder, clip_filename)

        # No temporary file needed anymore since we're using direct text in the filter

        # Let's create the filter with proper text escaping for FFmpeg
        # Escape all special characters that could interfere with FFmpeg filter syntax
        def escape_ffmpeg_text(t):
            # Replace common special characters
            replacements = {
                '\\': '\\\\\\\\',  # Backslashes need extra escaping
                "'": '',           # Remove single quotes
                ':': '\\:',        # Escape colons
                ',': '\\,',        # Escape commas
                '[': '\\[',        # Escape brackets
                ']': '\\]',
                '=': '\\='         # Escape equals
            }
            for old, new in replacements.items():
                t = t.replace(old, new)
            return t
        
        # Escape the text and create the filter
        text_escaped = escape_ffmpeg_text(text)
        vf_filter = f'drawtext=fontsize=24:fontcolor=white:x=10:y=h-40:box=1:boxcolor=black@0.5:text={text_escaped}'

        try:
            stream = ffmpeg.input(video_path, ss=start, t=end-start)
            stream = ffmpeg.output(stream, clip_path, vf=vf_filter, acodec='aac', vcodec='libx264', strict='experimental')
            
            # Get the command for debugging
            cmd = ffmpeg.compile(stream)
            cmd_str = ' '.join(cmd)
            print("\nFFmpeg command:", cmd_str)
            
            # Run the command
            try:
                stream.run(overwrite_output=True, capture_stderr=True)
                print(f"Successfully created clip: {clip_path}")
                # Store just the filename instead of the full path
                clip_files.append(os.path.basename(clip_path))
            except ffmpeg.Error as e:
                stderr = e.stderr.decode() if e.stderr else "No error output"
                print(f"\nFFmpeg error details:")
                print(f"Command was: {cmd_str}")
                print(f"Error output: {stderr}")
                raise RuntimeError(f"FFmpeg failed: {stderr}") from e
        finally:
            # No cleanup needed anymore since we're not using temporary files
            pass
    
    return clip_files  # Return the list of successfully generated clips
