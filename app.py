import os
import subprocess
import streamlit as st
from pytube import YouTube
from yt_dlp import YoutubeDL
from pydub import AudioSegment
from groq import Groq
from deep_translator import GoogleTranslator
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

# Suppress SyntaxWarnings
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Initialize Groq client and Google Translator
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
translator = GoogleTranslator(source='auto', target='ur')  # Set target language to Urdu

# Helper function to convert seconds to SRT time format
def seconds_to_srt_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

# Method 1: Download audio using pytube
def download_audio_pytube(url, output_path):
    try:
        yt = YouTube(url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        default_filename = audio_stream.default_filename
        audio_path = os.path.join(output_path, default_filename)
        audio_stream.download(output_path=output_path, filename=default_filename)
        
        # Convert to FLAC
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
    except Exception as e:
        st.warning(f"pytube failed: {e}")
        return None

# Method 2: Download audio using yt-dlp
def download_audio_ytdlp(url, output_path):
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'flac',
                'preferredquality': '192',
            }],
            'cookiefile': 'cookies.txt',  # Use cookies to avoid 403 errors
            'proxy': '',  # Remove proxy if not needed
        }
        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            audio_filename = ydl.prepare_filename(info_dict).replace('.webm', '.flac').replace('.mp4', '.flac')
            return audio_filename
    except Exception as e:
        st.warning(f"yt-dlp failed: {e}")
        return None

# Method 3: Download audio using youtube-dl (fallback)
def download_audio_youtubedl(url, output_path):
    try:
        command = [
            "youtube-dl",
            "-f", "bestaudio",
            "--extract-audio",
            "--audio-format", "flac",
            "--audio-quality", "0",
            "-o", os.path.join(output_path, "%(title)s.%(ext)s"),
            url
        ]
        subprocess.run(command, check=True)
        for file in os.listdir(output_path):
            if file.endswith(".flac"):
                return os.path.join(output_path, file)
        return None
    except Exception as e:
        st.warning(f"youtube-dl failed: {e}")
        return None

# Function to download audio using multiple methods with fallbacks
def download_audio(url, output_path):
    # Try pytube first
    audio_path = download_audio_pytube(url, output_path)
    if audio_path:
        return audio_path
    
    # Try yt-dlp if pytube fails
    audio_path = download_audio_ytdlp(url, output_path)
    if audio_path:
        return audio_path
    
    # Try youtube-dl if yt-dlp fails
    audio_path = download_audio_youtubedl(url, output_path)
    if audio_path:
        return audio_path
    
    # If all methods fail, raise an error
    raise Exception("All audio download methods failed.")

# Function to fetch and translate subtitles
def fetch_and_translate_subtitles(video_id, target_language="tr"):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        non_auto_transcript = None
        for transcript in transcript_list:
            if not transcript.is_generated and transcript.language_code == target_language:
                non_auto_transcript = transcript
                break
        if non_auto_transcript:
            subtitles = non_auto_transcript.fetch()
            translated_subtitles = []
            for subtitle in subtitles:
                translated_text = translator.translate(subtitle["text"])
                translated_subtitles.append({
                    "start": subtitle["start"],
                    "end": subtitle["start"] + subtitle["duration"],
                    "text": translated_text
                })
            return translated_subtitles
        else:
            return None
    except TranscriptsDisabled:
        st.error("Subtitles are disabled for this video.")
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
                
                video_id = yt_url.split("v=")[1].split("&")[0]  # Extract video ID from URL
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
                    
                    audio_path = download_audio(yt_url, "temp")
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
                        translated_text = translator.translate(segment["text"])
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
