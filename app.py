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
        #Try to get a manually created English transcript
         elif 'en' in transcript_list._manually_created_transcripts:
            transcript = transcript_list.find_manually_created_transcript(['en'])
            st.success("English manually created transcript found.")
            transcript_data = transcript.fetch()
            return transcript_data
        #Try to get auto generated turkish transcript
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
        while i < len(lines) - 2 :
            try:
              index = int(lines[i])
              time_line = lines[i + 1]
              text = lines[i + 2]
              
              time_match = re.match(r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})",time_line)
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
                i+=4
              else:
                  i+=1 #Skip this line
            except ValueError:
                i+=1 # Skip this line if index cant be converted into int

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
       transcript_data: List of dictionary entries, each entry has start, duration and text.
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
    
def translate_chunk(chunk, retry_queue, rate_limit_info, target_language="urdu", max_retries=3):
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
        if not rate_limit_info.can_send_request():
            st.warning("Rate limit reached. Sleeping...")
            time.sleep(60)
            rate_limit_info.reset_rate_limits()
            continue
        try:
            response = model.generate_content(prompt)
            rate_limit_info.update_rate_limits(len(prompt.split()) + len(str(response).split()))

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

        except Exception as e:
            st.error(f"An error occurred: {e}, retrying request (attempt {attempt + 1}/{max_retries})")
            retry_queue.append(chunk)
            time.sleep(random.uniform(1, 3))
            continue

    st.error(f"Failed to translate chunk after {max_retries} retries.")
    return None

class RateLimitInfo:
    def __init__(self, max_requests_per_minute=15, max_tokens_per_minute=1_000_000, max_requests_per_day=1500):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute
        self.max_requests_per_day = max_requests_per_day
        self.request_queue = deque(maxlen=max_requests_per_minute)
        self.tokens_this_minute = 0
        self.requests_today = 0
        self.last_reset_time = time.time()

    def can_send_request(self):
        now = time.time()
        if now - self.last_reset_time >= 60:
            self.reset_rate_limits()

        if len(self.request_queue) >= self.max_requests_per_minute:
            return False

        if self.requests_today >= self.max_requests_per_day:
           return False

        return True

    def update_rate_limits(self, tokens_used):
        self.request_queue.append(time.time())
        self.tokens_this_minute += tokens_used
        self.requests_today += 1

    def reset_rate_limits(self):
        self.request_queue.clear()
        self.tokens_this_minute = 0
        self.last_reset_time = time.time()


def translate_srt(transcript_data, rate_limit_info):
    """
    Translates all the srt data to Urdu using parallel requests.
    """

    chunks = chunk_transcript(transcript_data)
    st.info(f"Created {len(chunks)} chunks for translation.")

    # Display chunk info and create a progress bar
    chunk_info_text = ""
    for i, chunk in enumerate(chunks):
        chunk_info_text+=f"Chunk {i + 1}: {len(chunk)} lines\n"
    
    st.text_area("Chunk Information", chunk_info_text)

    progress_bar = st.progress(0)
    translated_chunks = []
    retry_queue = deque()


    with ThreadPoolExecutor(max_workers=10) as executor:
      futures = [executor.submit(translate_chunk, chunk, retry_queue, rate_limit_info) for chunk in chunks]
      results = [future.result() for future in futures]

      for chunk,translated_lines in zip(chunks,results):
          if translated_lines:
             translated_chunks.append((chunk,translated_lines))
          progress_bar.progress(len(translated_chunks)/len(chunks))

    while retry_queue:
        chunk = retry_queue.popleft()
        translated_lines = translate_chunk(chunk, retry_queue, rate_limit_info)
        if translated_lines:
            for i, (original_chunk, _) in enumerate(translated_chunks):
                if original_chunk == chunk:
                    translated_chunks[i] = (original_chunk, translated_lines)
                    break
        progress_bar.progress(len(translated_chunks)/len(chunks))

    if len(translated_chunks) != len(chunks):
        st.error("Translation failed for some chunks, Please try again.")
        return None
    
    translated_srt = ""
    line_index = 1
    for original_chunk, translated_lines in translated_chunks:
        for i, original_line in enumerate(original_chunk):
            start = original_line['start']
            duration = original_line['duration']
            end = start + duration
            translated_text = translated_lines[i]
            translated_text = replace_ranks_titles(translated_text)
            translated_srt += f"{line_index}\n"
            translated_srt += f"{format_time(start)} --> {format_time(end)}\n"
            translated_srt += f"{translated_text}\n\n"
            line_index += 1
    st.success("Translation complete.")
    return translated_srt

# --- Streamlit App ---
def main():
    st.title("YouTube Subtitle Downloader & Translator")

    video_url = st.text_input("Enter YouTube Video URL:")
    uploaded_file = st.file_uploader("Or Upload an SRT File", type=["srt"])

    if video_url or uploaded_file:
        
        if uploaded_file:
            srt_content = uploaded_file.read().decode("utf-8")
            transcript_data = parse_srt(srt_content)
            st.text_area("Uploaded SRT Content", srt_content, height=300)

            st.download_button(
                  label="Download Original SRT File",
                  data=srt_content,
                  file_name=f"uploaded_transcript.srt",
                  mime="text/plain"
              )


        elif video_url:
            video_id = extract_video_id(video_url)
            if video_id:
                 transcript_data = fetch_transcript(video_id)

                 if transcript_data:
                    srt_content = convert_to_srt(transcript_data)
                    st.text_area("Original SRT Content", srt_content, height=300)
                    st.download_button(
                         label="Download Original SRT File",
                         data=srt_content,
                         file_name=f"{video_id}_transcript.srt",
                         mime="text/plain"
                     )
        else:
            transcript_data= None
        
        if transcript_data:
             rate_limit_info = RateLimitInfo()
             translated_srt = translate_srt(transcript_data, rate_limit_info)
             if translated_srt:
                st.text_area("Translated SRT Content (Urdu)", translated_srt, height=300)
                st.download_button(
                   label="Download Translated SRT File (Urdu)",
                    data=translated_srt,
                    file_name=f"translated_transcript_urdu.srt",
                    mime="text/plain"
                )
    
if __name__ == "__main__":
    main()
