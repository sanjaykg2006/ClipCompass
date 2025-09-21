import os
import uuid
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

    # Helper functions are defined inside the loop for clarity

    for s in highlights:
        start, end, text = s["start"], s["end"], s["text"]
        clip_filename = f"{uuid.uuid4()}.mp4"
        clip_path = os.path.join(output_folder, clip_filename)

        # Handle text wrapping and file creation
        temp_text_file = os.path.join(output_folder, f"temp_text_{uuid.uuid4()}.txt")
        
        # Simple text wrapping at 40 characters
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            if sum(len(w) for w in current_line) + len(current_line) + len(word) <= 40:
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        processed_text = '\n'.join(lines)  # Simple newline
        
        # Write the text to a temporary file
        with open(temp_text_file, 'w', encoding='utf-8') as f:
            f.write(processed_text)
        
        # Create drawtext filter with timing control
        clip_duration = end - start
        fade_duration = 0.5  # Duration of fade in/out in seconds
        
        # Debug path handling
        print("\nDebug - Path handling:")
        print(f"Original temp file path: {temp_text_file}")
        
        # Clean and normalize the path for FFmpeg
        # Use just the filename instead of full path
        text_filename = os.path.basename(temp_text_file)
        text_dir = os.path.dirname(temp_text_file)
        
        print(f"Text filename: {text_filename}")
        print(f"Text directory: {text_dir}")
        
        # Change working directory to where the text file is
        os.chdir(text_dir)
        
        # Create a minimal filter string with relative path
        vf_filter = (
            f'drawtext=fontfile=/Windows/Fonts/arial.ttf:'
            f'fontsize=28:'
            f'fontcolor=white:'
            f'x=(w-text_w)/2:'
            f'y=h-th-100:'
            f'box=1:'
            f'boxcolor=black@0.25:'
            f'textfile={text_filename}'
        )
        
        print(f"Final filter string: {vf_filter}")
        print(f"Working directory: {os.getcwd()}")
        print(f"Text file exists: {os.path.exists(text_filename)}")
        print(f"Text file content:")
        try:
            with open(text_filename, 'r', encoding='utf-8') as f:
                print(f.read())
        except Exception as e:
            print(f"Error reading text file: {e}")

        try:
            stream = ffmpeg.input(video_path, ss=start, t=end-start)
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

            # Debug information
            cmd = ffmpeg.compile(stream)
            cmd_str = " ".join(cmd)
            print("\nFFmpeg Command Details:")
            print("=" * 80)
            print(f"Working directory: {os.getcwd()}")
            print(f"Input video path: {video_path}")
            print(f"Output clip path: {clip_path}")
            print(f"Clip duration: {end-start:.2f} seconds")
            print(f"Text file: {text_filename}")
            print(f"Text content:\n{text}")
            print("\nFilter chain:")
            print("-" * 40)
            print(f"VF filter: {vf_filter}")
            print("-" * 40)
            print(f"Full FFmpeg command:\n{cmd_str}")
            print("=" * 80)

            try:
                # Run FFmpeg
                stream.run(overwrite_output=True, capture_stderr=True)
                print(f"Successfully created clip: {clip_path}")
                clip_files.append(os.path.basename(clip_path))
            except ffmpeg.Error as e:
                print(f"\nFFmpeg error details:")
                print(f"Error output: {e.stderr.decode() if e.stderr else str(e)}")
                raise
            finally:
                # Clean up the temporary text file
                try:
                    if os.path.exists(text_filename):
                        os.remove(text_filename)
                        print(f"Cleaned up temporary file: {text_filename}")
                except Exception as e:
                    print(f"Warning: Could not remove temporary file {text_filename}: {e}")
                
                # Restore original working directory
                os.chdir(output_folder)

        except ffmpeg.Error as e:
            stderr = e.stderr.decode() if e.stderr else "No error output"
            print(f"\nFFmpeg error details:\nCommand: {' '.join(cmd)}\nError output: {stderr}")
            raise RuntimeError(f"FFmpeg failed: {stderr}") from e

    return clip_files
