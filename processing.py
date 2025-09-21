import os
import uuid
from faster_whisper import WhisperModel
import ffmpeg

# Load a more robust model (base) on CPU
model = WhisperModel("base", device="cpu")


def transcribe(audio_path):
    """
    Transcribe audio and return full text + segments.
    Includes debug prints.
    """
    segments_gen, info = model.transcribe(audio_path, beam_size=5)

    # Convert generator to list
    segments = list(segments_gen)

    full_text = ""
    segment_list = []

    for i, seg in enumerate(segments):
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

        (
            ffmpeg.input(video_path, ss=start, to=end)
                  .output(
                      clip_path,
                      vf=f"drawtext=text='{text}':fontcolor=white:fontsize=24:x=10:y=h-40:box=1:boxcolor=black@0.5",
                      vcodec="mpeg4",
                      acodec="copy"
                  )
                  .run(overwrite_output=True)
        )
        clip_files.append(clip_filename)

    return clip_files
