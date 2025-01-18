import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import re
import google.generativeai as genai
import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import random
import yt_dlp
from google.api_core.exceptions import ResourceExhausted
import httpx
from groq import Groq
from huggingface_hub import InferenceClient

# --- Setup APIs ---
# Google AI Studio (Gemini)
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Groq
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Hugging Face
huggingface_client = InferenceClient(token=st.secrets["HUGGINGFACE_TOKEN"])

# --- Utility Functions ---
def extract_video_id(url):
    """Extract the YouTube video ID."""
    video_id_match = re.match(
        r'(?:https?://)?(?:www\.)?(?:youtube\.com/(?:watch\?v=|embed/|v/|.+\?v=)|youtu\.be/)([^&=%\?]{11})', url)
    if video_id_match:
        return video_id_match.group(1)
    else:
        st.error("Invalid YouTube URL. Please ensure the URL is correct.")
        return None

def fetch_transcript(video_id):
    """Fetches transcript data from YouTube with fallback for 'TranscriptsDisabled'."""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        # Try to get a manually created Turkish transcript
        if 'tr' in transcript_list._manually_created_transcripts:
            transcript = transcript_list.find_manually_created_transcript(['tr'])
            st.success("Turkish manually created transcript found.")
            transcript_data = transcript.fetch()
            return transcript_data
        # Try to get a manually created English transcript
        elif 'en' in transcript_list._manually_created_transcripts:
            transcript = transcript_list.find_manually_created_transcript(['en'])
            st.success("English manually created transcript found.")
            transcript_data = transcript.fetch()
            return transcript_data
        # Try to get auto-generated Turkish transcript
        elif 'tr' in transcript_list._generated_transcripts:
            transcript = transcript_list.find_generated_transcript(['tr'])
            st.success("Turkish auto-generated transcript found.")
            transcript_data = transcript.fetch()
            return transcript_data
        else:
            st.error("No manually created or auto-generated transcripts found in Turkish or English.")
            return None

    except TranscriptsDisabled:
        st.warning("Transcripts are disabled using list_transcripts method, attempting fallback method...")
        return fetch_transcript_fallback(video_id)
    except NoTranscriptFound:
        st.error("No transcripts found for this video.")
        return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

def fetch_transcript_fallback(video_id):
    """Fallback method to fetch transcript using get_transcript."""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        if 'tr' in transcript_list._manually_created_transcripts:
            transcript = transcript_list.find_manually_created_transcript(['tr'])
            st.success("Turkish manually created transcript found using fallback method.")
            transcript_data = transcript.fetch()
            return transcript_data
        elif 'en' in transcript_list._manually_created_transcripts:
            transcript = transcript_list.find_manually_created_transcript(['en'])
            st.success("English manually created transcript found using fallback method.")
            transcript_data = transcript.fetch()
            return transcript_data
        else:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['tr'])
            st.success("Turkish transcript fetched using fallback method.")
            return transcript

    except TranscriptsDisabled:
        st.warning("Transcripts are disabled even using fallback method, attempting yt-dlp fallback method...")
        return fetch_transcript_yt_dlp(video_id)
    except NoTranscriptFound:
        st.error("No transcripts found using fallback method.")
        return None
    except Exception as e:
        st.error(f"An error occurred in fallback method: {e}")
        return None

def fetch_transcript_yt_dlp(video_id):
    """Fallback method to fetch transcript using yt-dlp."""
    try:
        ydl_opts = {
            'writesubtitles': True,
            'subtitleslangs': ['tr'],
            'skip_download': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)

            if 'subtitles' in info_dict and 'tr' in info_dict['subtitles']:
                subtitles = info_dict['subtitles']['tr']
                srt_content = yt_dlp.utils.get_first(subtitles, {}).get('data', '')
                if srt_content:
                    st.success("Turkish transcript fetched using yt_dlp fallback method.")
                    return parse_srt(srt_content)
                else:
                    st.error("No subtitles found in yt_dlp output.")
                    return None
            else:
                st.error("No Turkish subtitles found using yt_dlp")
                return None

    except Exception as e:
        st.error(f"An error occurred in yt_dlp fallback method: {e}")
        return None

def parse_srt(srt_content):
    """Parses the SRT content and returns the list of transcript dictionaries."""
    lines = srt_content.strip().split("\n")
    entries = []
    i = 0
    while i < len(lines) - 2:
        try:
            index = int(lines[i])
            time_line = lines[i + 1]
            text = lines[i + 2]

            time_match = re.match(r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})", time_line)
            if time_match:
                start_time_str, end_time_str = time_match.groups()
                start_time = time_to_seconds(start_time_str)
                end_time = time_to_seconds(end_time_str)
                duration = end_time - start_time

                entry = {
                    'start': start_time,
                    'duration': duration,
                    'text': text
                }
                entries.append(entry)
                i += 4
            else:
                i += 1  # Skip this line
        except ValueError:
            i += 1  # Skip this line if index can't be converted into int

    return entries

def time_to_seconds(time_str):
    """Converts an SRT time string to seconds."""
    hours, minutes, seconds_milliseconds = time_str.split(":")
    seconds, milliseconds = seconds_milliseconds.split(",")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000

def convert_to_srt(transcript_data):
    """Converts transcript data to SRT format."""
    srt_content = ""
    for i, entry in enumerate(transcript_data):
        start = entry['start']
        duration = entry['duration']
        end = start + duration
        text = entry['text']
        srt_content += f"{i + 1}\n"
        srt_content += f"{format_time(start)} --> {format_time(end)}\n"
        srt_content += f"{text}\n\n"
    return srt_content

def format_time(seconds):
    """Converts seconds to SRT time format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

# --- Translation Functions ---
def translate_text(text, api, request_no):
    """Translates text using the selected API."""
    try:
        st.write(f"Request {request_no}: Translating line: {text}")
        if api == "Google AI Studio":
            response = gemini_model.generate_content(f"Translate this Turkish text to Urdu: {text}")
            return response.text
        elif api == "Groq":
            response = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "{system_promptYou are an advanced AI translator specializing in converting Turkish historical drama subtitles into Urdu for a Pakistani audience. The input and output will be in JSON format, and your task is to:

        Translate all dialogues and narration from Turkish to Urdu.
        Ensure ranks, idioms, poetry, and cultural references are appropriately translated into Urdu.
        Account for potential spelling errors in the Turkish input.
        The JSON object you will be translating is:
        {input_json}
    Respond with a JSON object in the same format that has the translated subtitles as lines.

    Detailed Instructions:

    Translate Ranks and Titles:
    Replace Turkish ranks with culturally relevant Urdu equivalents:
    "Bey" → "سردار"
    "Sultan" → "سلطان"
    "Alp" → "سپاہی" or "مجاہد"
    "Şeyh" → "شیخ"

    Poetry and Idioms:
    Translate poetry, idiomatic expressions, and figures of speech in a way that preserves their emotional and poetic impact.

    Handle Spelling Errors:
    Correct common spelling errors in Turkish input. For example:
        "Osmalı" → "Osmanlı"
        "By" → "Bey"

    Examples of Turkish Input and Urdu Output:
    Example 1:

    Turkish SRT Input:

    1  
    00:00:01,000 --> 00:00:04,000  
    Bugün savaş meydanında kanımızı akıtacağız!  

    2  
    00:00:05,000 --> 00:00:08,000  
    Osmanlı'nın adını yaşatmak için öleceğiz.  

    Urdu SRT Output:

    1  
    00:00:01,000 --> 00:00:04,000  
    آج ہم جنگ کے میدان میں اپنا خون بہائیں گے!  

    2  
    00:00:05,000 --> 00:00:08,000  
    عثمانی کے نام کو زندہ رکھنے کے لئے جان دیں گے۔  

    Example 2:

    Turkish SRT Input (with spelling errors):

    3  
    00:00:09,000 --> 00:00:12,000  
    Byler, zafere giden yol buradan geçer!  

    4  
    00:00:13,000 --> 00:00:16,000  
    Şimdi savaşmaya hazır olun!  

    Urdu SRT Output:

    3  
    00:00:09,000 --> 00:00:12,000  
    سرداروں، فتح کا راستہ یہیں سے گزرتا ہے!  

    4  
    00:00:13,000 --> 00:00:16,000  
    اب جنگ کے لئے تیار ہو جاؤ!  

    Example 3:

    Turkish SRT Input (with poetry):

    5  
    00:00:17,000 --> 00:00:21,000  
    Adaletin ağacı kanla beslenir, ama zulüm de bir gün düşer.  

    6  
    00:00:22,000 --> 00:00:26,000  
    Herkes, Osman Bey’in adaletine şahit olacak!  

    Urdu SRT Output:

    5  
    00:00:17,000 --> 00:00:21,000  
    انصاف کا درخت خون سے سینچا جاتا ہے، لیکن ظلم بھی ایک دن گر جاتا ہے۔  

    6  
    00:00:22,000 --> 00:00:26,000  
    ہر کوئی عثمان سردار کے انصاف کا گواہ بنے گا!  

    Example 4:

    Turkish SRT Input (with cultural references):

    7  
    00:00:27,000 --> 00:00:30,000  
    Şeyh Edebali: “Sabır, zaferin anahtarıdır.”  

    8  
    00:00:31,000 --> 00:00:35,000  
    Osman Bey: “Bu topraklar bizim kanımızla yeşerecek!”  

    Urdu SRT Output:

    7  
    00:00:27,000 --> 00:00:30,000  
    شیخ ایدبالی: "صبر فتح کی کنجی ہے۔"  

    8  
    00:00:31,000 --> 00:00:35,000  
    عثمان سردار: "یہ زمینیں ہمارے خون سے سرسبز ہوں گی!"}."},
                    {"role": "user", "content": f"Translate this Turkish text to Urdu: {text}"},
                ],
                model="llama-3.3-70b-versatile",
            )
            return response.choices[0].message.content
        elif api == "Hugging Face":
            response = huggingface_client.text_generation(
                prompt=f"Translate this Turkish text to Urdu: {text}",
                model="meta-llama/Meta-Llama-3.1-70B-Instruct",
            )
            return response
    except Exception as e:
        st.error(f"Translation failed with {api}: {e}")
        return None

def translate_srt(transcript_data, api):
    """Translates all the SRT data to Urdu using the selected API."""
    translated_srt = ""
    total_lines = len(transcript_data)
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, entry in enumerate(transcript_data):
        request_no = i + 1
        status_text.text(f"Processing line {request_no} of {total_lines}...")
        translated_text = translate_text(entry['text'], api, request_no)
        if translated_text:
            translated_srt += f"{entry['start']} --> {entry['start'] + entry['duration']}\n{translated_text}\n\n"
        else:
            st.error(f"Failed to translate line: {entry['text']}")
        progress_bar.progress((i + 1) / total_lines)

    return translated_srt

# --- Main Function ---
def main():
    st.title("YouTube Turkish to Urdu Subtitle Translator")

    # Input for YouTube URL or file upload
    option = st.radio("Choose input method:", ("YouTube URL", "Upload Subtitle File (SRT)"))

    if option == "YouTube URL":
        youtube_url = st.text_input("Enter YouTube Video URL:")
        if youtube_url:
            video_id = extract_video_id(youtube_url)
            if video_id:
                st.info(f"Extracted Video ID: {video_id}")

                # Fetch transcript
                transcript_data = fetch_transcript(video_id)
                if transcript_data:
                    st.success("Transcript fetched successfully!")

                    # Display original transcript
                    if st.checkbox("Show Original Transcript"):
                        st.text_area("Original Transcript", convert_to_srt(transcript_data), height=300)

                    # Select translation API
                    selected_api = st.selectbox("Select Translation API", ["Google AI Studio", "Groq", "Hugging Face"])

                    # Translate transcript
                    if st.button("Translate to Urdu"):
                        with st.spinner("Translating..."):
                            translated_srt = translate_srt(transcript_data, selected_api)
                            if translated_srt:
                                st.success("Translation completed!")
                                st.text_area("Translated Urdu Subtitles", translated_srt, height=300)

                                # Download translated SRT file
                                st.download_button(
                                    label="Download Translated SRT",
                                    data=translated_srt,
                                    file_name="translated_subtitles.srt",
                                    mime="text/srt"
                                )
                            else:
                                st.error("Translation failed. Please try again.")
                else:
                    st.error("Failed to fetch transcript.")
            else:
                st.error("Invalid YouTube URL.")
    else:
        uploaded_file = st.file_uploader("Upload Subtitle File (SRT)", type=["srt"])
        if uploaded_file:
            srt_content = uploaded_file.read().decode("utf-8")
            transcript_data = parse_srt(srt_content)
            if transcript_data:
                st.success("Subtitle file uploaded and parsed successfully!")

                # Display original transcript
                if st.checkbox("Show Original Transcript"):
                    st.text_area("Original Transcript", convert_to_srt(transcript_data), height=300)

                # Select translation API
                selected_api = st.selectbox("Select Translation API", ["Google AI Studio", "Groq", "Hugging Face"])

                # Translate transcript
                if st.button("Translate to Urdu"):
                    with st.spinner("Translating..."):
                        translated_srt = translate_srt(transcript_data, selected_api)
                        if translated_srt:
                            st.success("Translation completed!")
                            st.text_area("Translated Urdu Subtitles", translated_srt, height=300)

                            # Download translated SRT file
                            st.download_button(
                                label="Download Translated SRT",
                                data=translated_srt,
                                file_name="translated_subtitles.srt",
                                mime="text/srt"
                            )
                        else:
                            st.error("Translation failed. Please try again.")
            else:
                st.error("Failed to parse the uploaded subtitle file.")

if __name__ == "__main__":
    main()
