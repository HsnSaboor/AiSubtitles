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

# Setup Gemini API
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
model = genai.GenerativeModel('gemini-pro')

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

# --- Translation Functions ---
def chunk_transcript(transcript_data, max_tokens=7000):
    """
    Chunks the transcript into segments less than max_tokens, avoiding split lines.

    Args:
        transcript_data: List of dictionary entries, each entry has start, duration, and text.
        max_tokens: The maximum token count for each chunk.

    Returns:
        list of list of transcript lines
    """
    chunks = []
    current_chunk = []
    current_token_count = 0

    for line in transcript_data:
        line_tokens = len(line['text'].split())
        if current_token_count + line_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_token_count = 0

        current_chunk.append(line)
        current_token_count += line_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def replace_ranks_titles(text):
    """Replaces Turkish ranks and titles with Urdu equivalents."""
    replacements = {
        r"\bBey\b": "سردار",
        r"\bSultan\b": "سلطان",
        r"\bAlp\b": "سپاہی",
        r"\bŞeyh\b": "شیخ"
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    return text

def translate_chunk(chunk, retry_queue, rate_limit_info, target_language="urdu", max_retries=3, llm_provider='gemini'):
    """Translates a chunk of text to Urdu using the Gemini API with httpx."""

    input_lines = [line['text'] for line in chunk]
    input_json = json.dumps({"lines": input_lines})

    prompt = f"""
    You are an advanced AI translator specializing in converting Turkish historical drama subtitles into Urdu for a Pakistani audience. The input and output will be in JSON format, and your task is to:

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
    عثمان سردار: "یہ زمینیں ہمارے خون سے سرسبز ہوں گی!"
    """
    for attempt in range(max_retries):
        if not rate_limit_info.can_send_request(llm_provider):
            st.warning("Rate limit reached. Sleeping...")
            time.sleep(60)
            rate_limit_info.reset_rate_limits(llm_provider)
            continue
        try:
            if llm_provider == 'gemini':
                response = model.generate_content(prompt)
                rate_limit_info.update_rate_limits(len(prompt.split()) + len(str(response).split()), llm_provider)

                if response and response.text:
                    try:
                        json_response = json.loads(response.text)
                        translated_lines = json_response.get('lines')

                        if not translated_lines:
                            st.warning(f"Invalid JSON format: 'lines' key missing, retrying request (attempt {attempt + 1}/{max_retries})")
                            retry_queue.append(chunk)
                            return None

                        if len(translated_lines) != len(input_lines):
                            st.warning(f"Line length mismatch, retrying request (attempt {attempt + 1}/{max_retries})")
                            retry_queue.append(chunk)
                            return None
                        else:
                            return translated_lines
                    except json.JSONDecodeError:
                        st.warning(f"Invalid JSON Response, retrying request (attempt {attempt + 1}/{max_retries})")
                        retry_queue.append(chunk)
                        return None
                else:
                    st.warning(f"Empty or invalid response received from API, retrying request (attempt {attempt + 1}/{max_retries})")
                    retry_queue.append(chunk)
                    return None

            elif llm_provider == 'groq':
                from groq import Groq
                client = Groq(api_key=st.secrets["GROQ_API_KEY"])  # Create Groq client in each request
                try:
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": prompt},
                        ],
                        model="llama-3.3-70b-versatile",
                        temperature=0,
                        max_completion_tokens=8192,
                        response_format={"type": "json_object"},
                    )
                    api_response_text = chat_completion.choices[0].message.content

                    rate_limit_info.update_rate_limits(len(prompt.split()) + len(str(api_response_text).split()), llm_provider)

                    try:
                        json_response = json.loads(api_response_text)
                        translated_lines = json_response.get('lines')
                        if not translated_lines:
                            st.warning(f"Invalid JSON format: 'lines' key missing, retrying request (attempt {attempt + 1}/{max_retries})")
                            retry_queue.append(chunk)
                            return None
                        if len(translated_lines) != len(input_lines):
                            st.warning(f"Line length mismatch, retrying request (attempt {attempt + 1}/{max_retries})")
                            retry_queue.append(chunk)
                            return None
                        else:
                            return translated_lines
                    except json.JSONDecodeError:
                        st.warning(f"Invalid JSON Response, retrying request (attempt {attempt + 1}/{max_retries})")
                        retry_queue.append(chunk)
                        return None
                except Exception as e:
                    st.error(f"Groq API error : {e}, retrying request (attempt {attempt + 1}/{max_retries})")
                    retry_queue.append(chunk)
                    continue
            elif llm_provider == 'huggingface':
                api_url = "https://api-inference.huggingface.co/models/meta-llama/Llama-3-70B-Instruct"
                headers = {
                    "Authorization": f"Bearer {st.secrets['HUGGINGFACE_TOKEN']}",
                    "Content-Type": "application/json"
                }
                data = {
                    "inputs": prompt,
                    "options": {"wait_for_model": True},
                    "parameters": {"max_new_tokens": 8192}
                }
                try:
                    with httpx.Client() as client:
                        response = client.post(api_url, headers=headers, json=data)
                        response.raise_for_status()
                        response_json = response.json()
                        if response_json and isinstance(response_json, list) and len(response_json) > 0:
                            api_response_text = response_json[0]['generated_text']
                            rate_limit_info.update_rate_limits(len(prompt.split()) + len(str(api_response_text).split()), llm_provider)

                            try:
                                json_response = json.loads(api_response_text)
                                translated_lines = json_response.get('lines')
                                if not translated_lines:
                                    st.warning(f"Invalid JSON format: 'lines' key missing, retrying request (attempt {attempt + 1}/{max_retries})")
                                    retry_queue.append(chunk)
                                    return None
                                if len(translated_lines) != len(input_lines):
                                    st.warning(f"Line length mismatch, retrying request (attempt {attempt + 1}/{max_retries})")
                                    retry_queue.append(chunk)
                                    return None
                                else:
                                    return translated_lines
                            except json.JSONDecodeError:
                                st.warning(f"Invalid JSON Response, retrying request (attempt {attempt + 1}/{max_retries})")
                                retry_queue.append(chunk)
                                return None
                        else:
                            st.warning(f"Empty or invalid response received from API, retrying request (attempt {attempt + 1}/{max_retries})")
                            retry_queue.append(chunk)
                            return None

                except httpx.HTTPError as e:
                    st.error(f"HTTP error with huggingface : {e}")
                    retry_queue.append(chunk)
                    continue

            else:
                st.error(f"Invalid model provided : {llm_provider}")
                retry_queue.append(chunk)
                return None

        except ResourceExhausted as e:
            st.error(f"API Quota Exceeded (429): {e}, retrying request (attempt {attempt + 1}/{max_retries})")
            retry_queue.append(chunk)
            delay = (2 ** attempt) + random.uniform(0, 1)  # exponential backoff with jitter
            time.sleep(delay)
            continue
        except Exception as e:
            st.error(f"An error occurred: {e}, retrying request (attempt {attempt + 1}/{max_retries})")
            retry_queue.append(chunk)
            time.sleep(random.uniform(1, 3))
            continue

    st.error(f"Failed to translate chunk after {max_retries} retries.")
    return None

class RateLimitInfo:
    def __init__(self, max_requests_per_minute=15, max_tokens_per_minute=1_000_000, max_requests_per_day=1500):
        self.max_requests_per_minute = {
            'gemini': 15,
            'huggingface': int(1000 / 60),  # Convert to integer
            'groq': int(14400 / 60)  # Convert to integer
        }
        self.max_tokens_per_minute = {
            'gemini': 1_000_000,
            'huggingface': 100000,
            'groq': 6000
        }
        self.max_requests_per_day = {
            'gemini': 1500,
            'huggingface': 1000,
            'groq': 1000,
        }

        self.request_queue = {
            'gemini': deque(maxlen=self.max_requests_per_minute['gemini']),
            'huggingface': deque(maxlen=self.max_requests_per_minute['huggingface']),
            'groq': deque(maxlen=self.max_requests_per_minute['groq'])
        }
        self.tokens_this_minute = {
            'gemini': 0,
            'huggingface': 0,
            'groq': 0,
        }
        self.requests_today = {
            'gemini': 0,
            'huggingface': 0,
            'groq': 0
        }
        self.last_reset_time = {
            'gemini': time.time(),
            'huggingface': time.time(),
            'groq': time.time(),
        }
        self.daily_requests_made = {
            'gemini': 0,
            'huggingface': 0,
            'groq': 0
        }
        self.last_reset_day = {
            'gemini': time.time() // (24 * 3600),
            'huggingface': time.time() // (24 * 3600),
            'groq': time.time() // (24 * 3600),
        }

    def can_send_request(self, llm_provider):
        now = time.time()
        current_day = now // (24 * 3600)

        if current_day > self.last_reset_day[llm_provider]:
            self.reset_daily_request_count(llm_provider)  # Reset the daily requests when a new day starts

        if now - self.last_reset_time[llm_provider] >= 60:
            self.reset_rate_limits(llm_provider)

        if len(self.request_queue[llm_provider]) >= self.max_requests_per_minute[llm_provider]:
            return False

        if self.daily_requests_made[llm_provider] >= self.max_requests_per_day[llm_provider]:
            return False

        return True

    def update_rate_limits(self, tokens_used, llm_provider):
        self.request_queue[llm_provider].append(time.time())
        self.tokens_this_minute[llm_provider] += tokens_used
        self.requests_today[llm_provider] += 1
        self.daily_requests_made[llm_provider] += 1

    def reset_rate_limits(self, llm_provider):
        self.request_queue[llm_provider].clear()
        self.tokens_this_minute[llm_provider] = 0
        self.last_reset_time[llm_provider] = time.time()

    def reset_daily_request_count(self, llm_provider):
        self.daily_requests_made[llm_provider] = 0
        self.last_reset_day[llm_provider] = time.time() // (24 * 3600)

def translate_srt(transcript_data, rate_limit_info, selected_model='gemini'):
    """
    Translates all the srt data to Urdu using parallel requests.
    """
    chunks = chunk_transcript(transcript_data)
    st.info(f"Created {len(chunks)} chunks for translation.")

    # Display chunk info and create a progress bar
    chunk_info_text = ""
    for i, chunk in enumerate(chunks):
        chunk_info_text += f"Chunk {i + 1}: {len(chunk)} lines, {sum(len(line['text'].split()) for line in chunk)} tokens\n"

    st.text_area("Chunk Information", chunk_info_text)

    progress_bar = st.progress(0)
    translated_chunks = []
    retry_queue = deque()

    llm_providers = [selected_model, 'gemini']

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(translate_chunk, chunk, retry_queue, rate_limit_info, llm_provider=llm_provider) for chunk in chunks for llm_provider in [selected_model]]
        results = [future.result() for future in futures]

        for chunk, translated_lines in zip(chunks, results):
            if translated_lines:
                translated_chunks.append((chunk, translated_lines))
            progress_bar.progress(len(translated_chunks) / len(chunks))

    if len(translated_chunks) != len(chunks):
        st.warning("Translation failed with selected model, retrying with other models...")

        while llm_providers and len(translated_chunks) != len(chunks):
            selected_model = llm_providers.pop(0)
            st.info(f"Retrying with model {selected_model}")

            retry_queue.extend([chunk for chunk, _ in translated_chunks if not _])
            translated_chunks = [(chunk, lines) for chunk, lines in translated_chunks if lines]

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(translate_chunk, chunk, retry_queue, rate_limit_info, llm_provider=selected_model) for chunk in chunks]
                results = [future.result() for future in futures]

                for chunk, translated_lines in zip(chunks, results):
                    if translated_lines:
                        for i, (original_chunk, _) in enumerate(translated_chunks):
                            if original_chunk == chunk:
                                translated_chunks[i] = (original_chunk, translated_lines)
                                break
                        else:
                            translated_chunks.append((chunk, translated_lines))
                    progress_bar.progress(len(translated_chunks) / len(chunks))
        if len(translated_chunks) != len(chunks):
            st.error("Translation failed with all the models, Please try again.  ")

            st.text("Here are the splitted chunks and the system prompt that you can copy paste to do it manually on a model of your choice:")

            for i, chunk in enumerate(chunks):
                st.text(f"Chunk {i + 1} :")
                for line in chunk:
                    st.text(line['text'])

    # Combine translated chunks into a single SRT
    translated_srt = ""
    for chunk, translated_lines in translated_chunks:
        for line, translated_line in zip(chunk, translated_lines):
            translated_srt += f"{line['start']} --> {line['start'] + line['duration']}\n{translated_line}\n\n"

    return translated_srt

# --- Main Function ---
def main():
    st.title("YouTube Turkish to Urdu Subtitle Translator")

    # Input for YouTube URL
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

                # Select translation model
                selected_model = st.selectbox("Select Translation Model", ["gemini", "groq", "huggingface"])

                # Initialize rate limit info
                rate_limit_info = RateLimitInfo()

                # Translate transcript
                if st.button("Translate to Urdu"):
                    with st.spinner("Translating..."):
                        translated_srt = translate_srt(transcript_data, rate_limit_info, selected_model)
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

if __name__ == "__main__":
    main()
