import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import re
import google.generativeai as genai
import json
import time
from collections import deque
import asyncio
import httpx
import random


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
        # Attempt to get the Turkish transcript using get_transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['tr'])
        st.success("Turkish transcript fetched using fallback method.")
        return transcript

   except TranscriptsDisabled:
        st.error("Transcripts are disabled even using fallback method.")
        return None
   except NoTranscriptFound:
        st.error("No transcripts found using fallback method.")
        return None
   except Exception as e:
        st.error(f"An error occurred in fallback method: {e}")
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
    
async def async_translate_chunk(chunk, retry_queue, rate_limit_info, client, target_language="urdu", max_retries=3):
    """Asynchronously translates a chunk of text to Urdu using the Gemini API with httpx."""

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
            await asyncio.sleep(60)
            rate_limit_info.reset_rate_limits()
            continue

        try:
            async with client.post(url="https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent", json={
            "contents": [{
              "parts": [{
                  "text": prompt
                  }]
              }]
            },headers={"x-goog-api-key":st.secrets["GOOGLE_API_KEY"]}) as response:
                response.raise_for_status()
                response_json = response.json()
                
                if response_json and "candidates" in response_json and response_json["candidates"] and 'content' in response_json["candidates"][0] and "parts" in response_json["candidates"][0]['content'] and response_json["candidates"][0]["content"]["parts"] and  'text' in response_json["candidates"][0]["content"]["parts"][0]:
                    api_response_text = response_json["candidates"][0]["content"]["parts"][0]['text']
                    rate_limit_info.update_rate_limits(len(prompt.split())+len(str(api_response_text).split()))
                    try:
                        json_response = json.loads(api_response_text)
                        translated_lines = json_response.get('lines')

                        if not translated_lines:
                            st.warning(f"Invalid JSON format: 'lines' key missing, retrying request (attempt {attempt+1}/{max_retries})")
                            retry_queue.append(chunk)
                            return None

                        if len(translated_lines) != len(input_lines):
                            st.warning(f"Line length mismatch, retrying request (attempt {attempt+1}/{max_retries})")
                            retry_queue.append(chunk)
                            return None
                        else:
                            return translated_lines
                    except json.JSONDecodeError:
                        st.warning(f"Invalid JSON Response, retrying request (attempt {attempt+1}/{max_retries})")
                        retry_queue.append(chunk)
                        return None
                else:
                     st.warning(f"Empty or invalid response received from API, retrying request (attempt {attempt+1}/{max_retries})")
                     retry_queue.append(chunk)
                     return None

        except Exception as e:
            st.error(f"An error occurred: {e}, retrying request (attempt {attempt+1}/{max_retries})")
            retry_queue.append(chunk)
            await asyncio.sleep(random.uniform(1,3))
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

async def translate_srt(transcript_data, rate_limit_info):
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

    async with httpx.AsyncClient() as client:
      tasks = [async_translate_chunk(chunk, retry_queue, rate_limit_info,client) for chunk in chunks]
      results = await asyncio.gather(*tasks)
      
      for chunk,translated_lines in zip(chunks,results):
          if translated_lines:
              translated_chunks.append((chunk,translated_lines))
          
          progress_bar.progress(len(translated_chunks)/len(chunks))


    while retry_queue:
         chunk = retry_queue.popleft()
         translated_lines = await async_translate_chunk(chunk, retry_queue, rate_limit_info,client)
         if translated_lines:
            for i,(original_chunk, _) in enumerate(translated_chunks):
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
    if video_url:
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
                rate_limit_info = RateLimitInfo()
                translated_srt = asyncio.run(translate_srt(transcript_data, rate_limit_info))
                if translated_srt:
                    st.text_area("Translated SRT Content (Urdu)", translated_srt, height=300)
                    st.download_button(
                        label="Download Translated SRT File (Urdu)",
                        data=translated_srt,
                        file_name=f"{video_id}_transcript_urdu.srt",
                        mime="text/plain"
                    )

if __name__ == "__main__":
    main()
