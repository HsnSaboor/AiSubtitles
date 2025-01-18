import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import re
import google.generativeai as genai
import json
import time
from collections import deque

# Setup Gemini API
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])  # Make sure you have the API key set in secrets.toml
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
    """Fetches transcript data from YouTube."""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        if 'tr' in transcript_list._manually_created_transcripts:
            transcript = transcript_list.find_manually_created_transcript(['tr'])
            st.success("Turkish manually created transcript found.")
        elif 'en' in transcript_list._manually_created_transcripts:
             transcript = transcript_list.find_manually_created_transcript(['en'])
             st.success("English manually created transcript found.")   
        else:
            st.error("No manually created transcripts found in Turkish or English.")
            return None

        transcript_data = transcript.fetch()
        return transcript_data

    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
        return None
    except NoTranscriptFound:
        st.error("No transcripts found for this video.")
        return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
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

def chunk_transcript(transcript_data,max_tokens=7000): # Adjusted max tokens for safety
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
            line_tokens = len(line['text'].split()) # Token estimation using whitespace split for simplicity
            if current_token_count + line_tokens > max_tokens and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_token_count= 0

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
    
def translate_chunk(chunk,retry_queue,rate_limit_info,target_language="urdu"):
    """Translates a chunk of text to Urdu using the Gemini API."""
    
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
    
    
    while True:
        if not rate_limit_info.can_send_request():
                st.warning("Rate limit reached. Sleeping...")
                time.sleep(60)  # Wait 60 sec for rate limit to reset
                rate_limit_info.reset_rate_limits()
                continue # Retry the translation request
    
        try:
             
            response = model.generate_content(prompt)
            rate_limit_info.update_rate_limits(len(prompt.split())+len(str(response).split())) # rough estimation
            try:
              json_response = json.loads(response.text)
              translated_lines = json_response['lines']
              if len(translated_lines) != len(input_lines):
                 st.warning(f"Line length mismatch, retrying request")
                 retry_queue.append(chunk)
                 return None  
              else:
                  return translated_lines
            except json.JSONDecodeError:
              st.warning(f"Invalid JSON Response, retrying request")
              retry_queue.append(chunk)
              return None

        except Exception as e:
             st.error(f"An error occurred: {e} , Retrying request")
             retry_queue.append(chunk)
             return None

class RateLimitInfo:
    def __init__(self, max_requests_per_minute=15, max_tokens_per_minute=1_000_000, max_requests_per_day=1500):
      self.max_requests_per_minute=max_requests_per_minute
      self.max_tokens_per_minute=max_tokens_per_minute
      self.max_requests_per_day = max_requests_per_day
      self.request_queue = deque(maxlen=max_requests_per_minute)  # keep track of the requests made in the current minute
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
      self.requests_today +=1

    def reset_rate_limits(self):
      self.request_queue.clear()
      self.tokens_this_minute = 0
      self.last_reset_time = time.time()

def translate_srt(transcript_data, rate_limit_info):
        """
        Translates all the srt data to urdu and gives in an srt format.
        """

        chunks = chunk_transcript(transcript_data)
        translated_chunks = []
        retry_queue = deque()

        st.info(f"Translating {len(chunks)} chunks to Urdu.This may take a while...")
        progress_bar = st.progress(0)
        chunk_idx = 0

        while chunk_idx < len(chunks):
             chunk = chunks[chunk_idx]
             translated_lines = translate_chunk(chunk, retry_queue,rate_limit_info)
             if translated_lines:
                 translated_chunks.append((chunk, translated_lines))
                 chunk_idx+=1
             else:
                 pass #Retry logic handled inside the translate_chunk

             progress_bar.progress((chunk_idx)/len(chunks))

        
        while retry_queue:
            chunk = retry_queue.popleft()
            translated_lines = translate_chunk(chunk,retry_queue,rate_limit_info)
            if translated_lines:
              for i,(original_chunk, _) in enumerate(translated_chunks):
                  if original_chunk == chunk:
                       translated_chunks[i] = (original_chunk, translated_lines)
                       break
            
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
                  line_index+=1

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
                translated_srt = translate_srt(transcript_data, rate_limit_info)
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
