import streamlit as st
import os
from pytubefix import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from async_google_trans_new import AsyncTranslator
import asyncio
from groq import Groq
from pydub import AudioSegment

# Initialize Groq client with your API key
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def main():
    st.title("YouTube Video to Urdu SRT Converter")
    yt_url = st.text_input("Enter YouTube Video URL:")
    if yt_url:
        st.video(yt_url)
        if st.button("Process Video"):
            try:
                video_id = YouTube(yt_url).video_id
                st.write("Fetching non-auto-generated subtitles...")
                # Step 1: Fetch non-auto-generated subtitles
                try:
                    transcript = fetch_non_auto_subtitles(video_id)
                    if transcript:
                        st.write("Non-auto-generated subtitles found. Translating to Urdu...")
                        # Step 2: Translate subtitles to Urdu
                        translated_subtitles = asyncio.run(translate_subtitles_to_urdu(transcript))
                        # Step 3: Generate SRT file
                        srt_content = generate_srt(translated_subtitles)
                    else:
                        st.write("No non-auto-generated subtitles found. Downloading audio...")
                        # Step 4: Download audio using pytubefix
                        audio_path = download_audio(yt_url)
                        if not audio_path:
                            st.error("Failed to download audio. Please try again later.")
                            return
                        # Step 5: Split audio into chunks
                        st.write("Splitting audio into chunks...")
                        chunks = split_audio_into_chunks(audio_path)
                        # Step 6: Transcribe each chunk using Groq API
                        st.write("Transcribing audio...")
                        transcription_segments = []
                        for chunk_path, start_time in chunks:
                            segments = transcribe_chunk(chunk_path, start_time)
                            transcription_segments.extend(segments)
                        # Step 7: Translate transcription to Urdu
                        st.write("Translating to Urdu...")
                        translated_segments = asyncio.run(translate_transcription_to_urdu(transcription_segments))
                        # Step 8: Generate SRT file
                        srt_content = generate_srt(translated_segments)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    return
                # Step 9: Provide download link for SRT file
                st.write("Generating SRT file...")
                st.download_button(
                    label="Download SRT File",
                    data=srt_content,
                    file_name="urdu_subtitles.srt",
                    mime="application/x-subrip; charset=utf-8",
                )
                # Step 10: Cleanup temporary files
                cleanup_files(audio_path, chunks)
            except Exception as e:
                st.error(f"An error occurred: {e}")

def fetch_non_auto_subtitles(video_id):
    """Fetch non-auto-generated subtitles."""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        for transcript in transcript_list:
            if not transcript.is_generated:
                return transcript.fetch()
        return None
    except (TranscriptsDisabled, NoTranscriptFound):
        return None

def download_audio(url):
    """Download audio from YouTube using pytubefix."""
    try:
        yt = YouTube(url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        audio_path = audio_stream.download(filename="audio.mp4")
        return audio_path
    except Exception as e:
        st.error(f"Failed to download audio: {e}")
        return None

def split_audio_into_chunks(audio_path, target_size_mb=24):
    """Split audio into chunks less than 25MB."""
    audio = AudioSegment.from_file(audio_path)
    bitrate_bps, duration_seconds = calculate_bitrate(audio_path)
    chunk_duration_seconds = calculate_chunk_duration(bitrate_bps, target_size_mb)
    chunk_duration_ms = int(chunk_duration_seconds * 1000)
    chunks = []
    for i in range(0, len(audio), chunk_duration_ms):
        chunk = audio[i:i + chunk_duration_ms]
        chunk_path = f"chunk_{len(chunks)}.flac"
        chunk.export(chunk_path, format="flac")
        chunks.append((chunk_path, i / 1000.0))
    return chunks

def calculate_bitrate(audio_path):
    """Calculate the bitrate and duration of the audio file."""
    file_size_bytes = os.path.getsize(audio_path)
    audio = AudioSegment.from_file(audio_path)
    duration_seconds = len(audio) / 1000.0
    bitrate_bps = (file_size_bytes * 8) / duration_seconds
    return bitrate_bps, duration_seconds

def calculate_chunk_duration(bitrate_bps, target_size_mb=24):
    """Calculate the duration of each chunk to fit within the target size."""
    target_size_bits = target_size_mb * 1024 * 1024 * 8
    chunk_duration_seconds = target_size_bits / bitrate_bps
    return chunk_duration_seconds

def transcribe_chunk(chunk_path, start_time):
    """Transcribe a single audio chunk using Groq API."""
    with open(chunk_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(chunk_path, file.read()),
            model="whisper-large-v3",
            response_format="verbose_json",
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

async def translate_subtitles_to_urdu(subtitles):
    """Translate subtitles to Urdu using async-google-trans-new."""
    translator = AsyncTranslator()
    translated_subtitles = []
    for subtitle in subtitles:
        translated_text = await translator.translate(subtitle['text'], 'ur')
        translated_subtitles.append({
            "start": subtitle['start'],
            "end": subtitle['start'] + subtitle['duration'],
            "text": translated_text
        })
    return translated_subtitles

async def translate_transcription_to_urdu(transcription_segments):
    """Translate transcription segments to Urdu."""
    translator = AsyncTranslator()
    translated_segments = []
    for segment in transcription_segments:
        translated_text = await translator.translate(segment['text'], 'ur')
        translated_segments.append({
            "start": segment['start'],
            "end": segment['end'],
            "text": translated_text
        })
    return translated_segments

def generate_srt(subtitles):
    """Generate SRT content from translated subtitles."""
    srt_content = ""
    for i, subtitle in enumerate(subtitles, start=1):
        start_time = seconds_to_srt_time(subtitle['start'])
        end_time = seconds_to_srt_time(subtitle['end'])
        srt_content += f"{i}\n{start_time} --> {end_time}\n{subtitle['text']}\n\n"
    return srt_content

def seconds_to_srt_time(seconds):
    """Convert seconds to SRT timestamp format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

def cleanup_files(audio_path, chunks):
    """Remove temporary audio and chunk files."""
    if audio_path and os.path.exists(audio_path):
        os.remove(audio_path)
    for chunk_path, _ in chunks:
        if os.path.exists(chunk_path):
            os.remove(chunk_path)

if __name__ == "__main__":
    main()
