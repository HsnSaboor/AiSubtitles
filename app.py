import os
import subprocess
import streamlit as st
from pytube import YouTube
from pydub import AudioSegment
from groq import Groq
from googletrans import Translator
from youtube_transcript_api import YouTubeTranscriptApi

# Suppress SyntaxWarnings
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Initialize Groq client and Google Translator
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
translator = Translator()

# Helper function to convert seconds to SRT time format
def seconds_to_srt_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

# Function to download and convert audio
def download_and_convert_audio(url, output_path):
    yt = YouTube(url)
    yt.bypass_age_gate = True  # Fix: Set the property, don't call it
    audio_stream = yt.streams.filter(only_audio=True).first()
    default_filename = audio_stream.default_filename
    audio_path = os.path.join(output_path, default_filename)
    audio_stream.download(output_path=output_path, filename=default_filename)
    
    flac_path = os.path.join(output_path, "audio.flac")
    subprocess.run([
        "ffmpeg",
        "-i", audio_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "flac",
        flac_path
    ], check=True)
    
    os.remove(audio_path)
    return flac_path

# Function to calculate average bitrate
def calculate_bitrate(audio_path):
    file_size_bytes = os.path.getsize(audio_path)
    audio = AudioSegment.from_file(audio_path)
    duration_seconds = len(audio) / 1000.0
    bitrate_bps = (file_size_bytes * 8) / duration_seconds
    return bitrate_bps, duration_seconds

# Function to determine chunk duration for 24 MB
def calculate_chunk_duration(bitrate_bps, target_size_mb=24):
    target_size_bits = target_size_mb * 1024 * 1024 * 8
    chunk_duration_seconds = target_size_bits / bitrate_bps
    return chunk_duration_seconds

# Function to split audio into chunks
def split_audio_into_chunks(audio_path, chunk_duration_seconds, output_path):
    audio = AudioSegment.from_file(audio_path)
    chunk_duration_ms = int(chunk_duration_seconds * 1000)
    chunks = []
    for i in range(0, len(audio), chunk_duration_ms):
        chunk = audio[i:i + chunk_duration_ms]
        chunk_path = os.path.join(output_path, f"chunk_{len(chunks)}.flac")
        chunk.export(chunk_path, format="flac")
        chunks.append((chunk_path, i / 1000.0))
    return chunks

# Function to adjust chunks if any exceed 25 MB
def adjust_chunks(chunks, max_size_mb=25):
    adjusted_chunks = []
    for chunk_path, start_time in chunks:
        file_size_bytes = os.path.getsize(chunk_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        if file_size_mb > max_size_mb:
            audio = AudioSegment.from_file(chunk_path)
            half_duration = len(audio) / 2
            chunk1 = audio[:half_duration]
            chunk2 = audio[half_duration:]
            chunk1_path = os.path.join(os.path.dirname(chunk_path), f"chunk_split_{len(adjusted_chunks)}.flac")
            chunk2_path = os.path.join(os.path.dirname(chunk_path), f"chunk_split_{len(adjusted_chunks)+1}.flac")
            chunk1.export(chunk1_path, format="flac")
            chunk2.export(chunk2_path, format="flac")
            adjusted_chunks.append((chunk1_path, start_time))
            adjusted_chunks.append((chunk2_path, start_time + half_duration / 1000.0))
            os.remove(chunk_path)
        else:
            adjusted_chunks.append((chunk_path, start_time))
    return adjusted_chunks

# Function to transcribe each chunk
def transcribe_chunk(chunk_path, start_time, language="tr"):
    with open(chunk_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(chunk_path, file.read()),
            model="whisper-large-v3",
            language=language,
            response_format="verbose_json"
        )
    adjusted_segments = []
    for segment in transcription["segments"]:
        segment_start = segment["start"] + start_time
        segment_end = segment["end"] + start_time
        adjusted_segments.append({
            "start": segment_start,
            "end": segment_end,
            "text": segment["text"]
        })
    return adjusted_segments

# Function to generate SRT file
def generate_srt_file(transcriptions, output_path):
    with open(output_path, "w", encoding="utf-8") as srt_file:
        for idx, segment in enumerate(transcriptions, start=1):
            start_time = segment["start"]
            end_time = segment["end"]
            text = segment["text"]
            srt_start = seconds_to_srt_time(start_time)
            srt_end = seconds_to_srt_time(end_time)
            srt_file.write(f"{idx}\n")
            srt_file.write(f"{srt_start} --> {srt_end}\n")
            srt_file.write(f"{text}\n\n")

# Function to fetch and translate subtitles
# Function to fetch and translate subtitles
def fetch_and_translate_subtitles(video_id, target_language="ur"):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        non_auto_transcript = None
        for transcript in transcript_list:
            if not transcript.is_generated:
                non_auto_transcript = transcript
                break
        if non_auto_transcript:
            subtitles = non_auto_transcript.fetch()
            translated_subtitles = []
            for subtitle in subtitles:
                # Use translator.translate synchronously
                translated_text = translator.translate(subtitle["text"], dest=target_language).text
                translated_subtitles.append({
                    "start": subtitle["start"],
                    "end": subtitle["start"] + subtitle["duration"],
                    "text": translated_text
                })
            return translated_subtitles
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching subtitles: {e}")
        return None

# Streamlit app
def main():
    st.title("YouTube Video to Urdu SRT Converter")
    yt_url = st.text_input("Enter YouTube Video URL:")
    if yt_url:
        st.video(yt_url)
        if st.button("Process Video"):
            try:
                if not os.path.exists("temp"):
                    os.makedirs("temp")
                
                video_id = YouTube(yt_url).video_id
                subtitles = fetch_and_translate_subtitles(video_id)
                if subtitles:
                    st.success("Non-auto-generated subtitles found! Translating to Urdu...")
                    generate_srt_file(subtitles, "output.srt")
                    st.success("SRT file generated successfully!")
                    st.download_button(
                        label="Download SRT File",
                        data=open("output.srt", "rb").read(),
                        file_name="output.srt",
                        mime="text/srt"
                    )
                else:
                    st.warning("No non-auto-generated subtitles found. Falling back to audio transcription...")
                    
                    audio_path = download_and_convert_audio(yt_url, "temp")
                    bitrate_bps, total_duration = calculate_bitrate(audio_path)
                    chunk_duration = calculate_chunk_duration(bitrate_bps)
                    chunks = split_audio_into_chunks(audio_path, chunk_duration, "temp")
                    chunks = adjust_chunks(chunks, max_size_mb=25)
                    
                    transcriptions = []
                    for chunk_path, start_time in chunks:
                        with st.spinner(f"Transcribing {chunk_path}..."):
                            transcription = transcribe_chunk(chunk_path, start_time)
                            transcriptions.extend(transcription)
                    
                    # Translate all transcription texts to Urdu
                    translated_transcriptions = []
                    for segment in transcriptions:
                        translated_text = translator.translate(segment["text"], dest="ur").text
                        translated_transcriptions.append({
                            "start": segment["start"],
                            "end": segment["end"],
                            "text": translated_text
                        })
                    
                    # Generate SRT file with translated text
                    generate_srt_file(translated_transcriptions, "output.srt")
                    st.success("SRT file generated successfully!")
                    st.download_button(
                        label="Download SRT File",
                        data=open("output.srt", "rb").read(),
                        file_name="output.srt",
                        mime="text/srt"
                    )
            except Exception as e:
                st.error(f"An error occurred: {e}")
                
if __name__ == "__main__":
    main()
