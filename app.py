import streamlit as st
import os
from groq import Groq
from deep_translator import GoogleTranslator
from langdetect import detect
import yt_dlp
from pydub import AudioSegment

# Initialize Groq client with your API key
client = Groq(api_key=os.environ.get('GROQ_API_KEY'))

def main():
    st.title("YouTube Video to Urdu SRT Converter")
    yt_url = st.text_input("Enter YouTube Video URL:")
    if yt_url:
        st.video(yt_url)
        if st.button("Process Video"):
            try:
                video_id = yt_url.split("v=")[1]
                st.write("Checking for subtitles...")
                # Fetch subtitles if available
                try:
                    transcript = fetch_subtitles(video_id)
                    subtitles_available = True
                    st.write("Subtitles found. Translating to Urdu...")
                    # Translate subtitles to Urdu
                    urdu_subtitles = translate_subtitles_to_urdu(transcript)
                    # Generate SRT with timestamps
                    srt_content = generate_srt(urdu_subtitles)
                except:
                    subtitles_available = False
                    st.write("Subtitles not found. Downloading audio...")
                    # Download audio
                    audio_path = download_audio(yt_url)
                    if not audio_path:
                        st.error("Failed to download audio. Please try again later.")
                        return
                    # Calculate bitrate and chunk duration
                    bitrate_bps, duration_seconds = calculate_bitrate(audio_path)
                    chunk_duration = calculate_chunk_duration(bitrate_bps)
                    # Split audio into chunks
                    chunks = split_audio_into_chunks(audio_path, chunk_duration, "chunks")
                    # Adjust chunks to ensure no chunk exceeds 25MB
                    adjusted_chunks = adjust_chunks(chunks)
                    # Transcribe each chunk
                    transcription_segments = []
                    for chunk_path, start_time in adjusted_chunks:
                        segments = transcribe_chunk(chunk_path, start_time, language="tr")
                        transcription_segments.extend(segments)
                    # Translate transcription to Urdu
                    translated_segments = translate_transcription_to_urdu(transcription_segments)
                    # Generate SRT with timestamps
                    srt_content = generate_srt(translated_segments)
                # Provide download link for SRT file
                st.write("Generating SRT file...")
                st.download_button(
                    label="Download SRT File",
                    data=srt_content,
                    file_name="urdu_subtitles.srt",
                    mime="application/x-subrip; charset=utf-8"
                )
                # Cleanup temporary files
                cleanup_files(audio_path, adjusted_chunks)
            except Exception as e:
                st.error(f"An error occurred: {e}")

def fetch_subtitles(video_id):
    # Implement subtitle fetching logic here
    # For demonstration, return dummy data
    return []

def translate_subtitles_to_urdu(subtitles):
    # Implement subtitle translation logic here
    # For demonstration, return dummy data
    return []

def download_audio(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'audio.flac',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'flac',
            'preferredquality': '192',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    audio_path = [f for f in os.listdir('.') if f.endswith('.flac')][0]
    return audio_path

def calculate_bitrate(audio_path):
    file_size_bytes = os.path.getsize(audio_path)
    audio = AudioSegment.from_file(audio_path)
    duration_seconds = len(audio) / 1000.0
    bitrate_bps = (file_size_bytes * 8) / duration_seconds
    return bitrate_bps, duration_seconds

def calculate_chunk_duration(bitrate_bps, target_size_mb=24):
    target_size_bits = target_size_mb * 1024 * 1024 * 8
    chunk_duration_seconds = target_size_bits / bitrate_bps
    return chunk_duration_seconds

def split_audio_into_chunks(audio_path, chunk_duration_seconds, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    audio = AudioSegment.from_file(audio_path)
    chunk_duration_ms = int(chunk_duration_seconds * 1000)
    chunks = []
    for i in range(0, len(audio), chunk_duration_ms):
        chunk = audio[i:i + chunk_duration_ms]
        chunk_path = os.path.join(output_path, f"chunk_{len(chunks)}.flac")
        chunk.export(chunk_path, format="flac")
        chunks.append((chunk_path, i / 1000.0))
    return chunks

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

def transcribe_chunk(chunk_path, start_time, language="tr"):
    with open(chunk_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(chunk_path, file.read()),
            model="whisper-large-v3-turbo",
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

def translate_transcription_to_urdu(transcription_segments):
    source_lang = detect(' '.join([seg['text'] for seg in transcription_segments]))
    translator = GoogleTranslator(source=source_lang, target='urdu')
    translated_segments = []
    for segment in transcription_segments:
        translated_text = translator.translate(segment['text'])
        translated_segments.append({
            "start": segment["start"],
            "end": segment["end"],
            "text": translated_text
        })
    return translated_segments

def generate_srt(translated_segments):
    srt_content = ""
    for i, segment in enumerate(translated_segments, start=1):
        start_time = seconds_to_srt_time(segment['start'])
        end_time = seconds_to_srt_time(segment['end'])
        srt_content += f"{i}\n{start_time} --> {end_time}\n{segment['text']}\n\n"
    return srt_content

def seconds_to_srt_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

def cleanup_files(audio_path, adjusted_chunks):
    os.remove(audio_path)
    for chunk_path, _ in adjusted_chunks:
        os.remove(chunk_path)

if __name__ == "__main__":
    main()
