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
        
        # Broader text wrapping at 60 characters
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            if sum(len(w) for w in current_line) + len(current_line) + len(word) <= 60:
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        processed_text = '\n'.join(lines)
        
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
        
        # Add fade effects with timing
        fade_duration = 0.5  # Duration of fade in/out in seconds
        clip_duration = end - start
        
        print("\nFade Effect Parameters:")
        print(f"Fade duration: {fade_duration} seconds")
        print(f"Clip duration: {clip_duration} seconds")
        print(f"Fade in period: 0 to {fade_duration} seconds")
        print(f"Full opacity period: {fade_duration} to {clip_duration - fade_duration} seconds")
        print(f"Fade out period: {clip_duration - fade_duration} to {clip_duration} seconds")

        # Calculate fade timings in frames (assuming 24fps)
        fps = 24
        fade_frames = int(fade_duration * fps)
        end_fade_start = int((clip_duration - fade_duration) * fps)
        
        # Create separate filters for text and fading
        print("\nFilter Construction:")
        print(f"FPS: {fps}")
        print(f"Fade frames: {fade_frames}")
        print(f"End fade starts at frame: {end_fade_start}")
        
        # Build the filter string in parts for better debugging
        draw_text = (
            f'drawtext=fontfile=/Windows/Fonts/arial.ttf:'
            f'fontsize=28:'
            f'fontcolor=white:'
            f'x=(w-text_w)/2:'
            f'y=h-th-100:'
            f'box=1:'
            f'boxcolor=black@0.25:'
            f'textfile={text_filename}'
        )
        
        fade_filter = (
            f'fade=t=in:st=0:d={fade_duration},'
            f'fade=t=out:st={clip_duration-fade_duration}:d={fade_duration}'
        )
        
        # Combine the filters
        vf_filter = f"{draw_text},{fade_filter}"
        
        print("\nFilter Components:")
        print("1. Draw Text Filter:")
        print(draw_text)
        print("\n2. Fade Filter:")
        print(fade_filter)
        print("\nFinal Combined Filter:")
        print(vf_filter)
        
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

            # Debug information and validation
            cmd = ffmpeg.compile(stream)
            cmd_str = " ".join(cmd)
            
            print("\nPre-execution Validation:")
            print("=" * 80)
            
            print("1. File System Check:")
            print(f"- Working directory: {os.getcwd()}")
            print(f"- Working directory exists: {os.path.exists(os.getcwd())}")
            print(f"- Input video exists: {os.path.exists(video_path)}")
            print(f"- Input video size: {os.path.getsize(video_path) / (1024*1024):.2f} MB")
            print(f"- Text file exists: {os.path.exists(text_filename)}")
            print(f"- Output directory exists: {os.path.exists(os.path.dirname(clip_path))}")
            
            print("\n2. Text File Validation:")
            try:
                with open(text_filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print(f"- Content length: {len(content)} characters")
                    print(f"- Number of lines: {len(content.splitlines())}")
                    print(f"- Content:\n{content}")
            except Exception as e:
                print(f"Error reading text file: {e}")
            
            print("\n3. Timing Parameters:")
            print(f"- Clip start time: {start:.2f}s")
            print(f"- Clip end time: {end:.2f}s")
            print(f"- Clip duration: {end-start:.2f}s")
            print(f"- Fade duration: {fade_duration:.2f}s")
            
            print("\n4. Filter Chain Components:")
            print("Draw Text Filter:")
            print(draw_text)
            print("\nFade Filter:")
            print(fade_filter)
            print("\nCombined Filter:")
            print(vf_filter)
            
            print("\n5. Full FFmpeg Command:")
            print("-" * 40)
            print(cmd_str)
            print("-" * 40)
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
