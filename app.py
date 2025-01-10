import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from googletrans import Translator
import re

def extract_video_id(url):
    """
    Extract the YouTube video ID from various URL formats.
    """
    video_id_match = re.match(
        r'(?:https?://)?(?:www\.)?(?:youtube\.com/(?:watch\?v=|embed/|v/|.+\?v=)|youtu\.be/)([^&=%\?]{11})', url)
    if video_id_match:
        return video_id_match.group(1)
    else:
        st.error("Invalid YouTube URL. Please ensure the URL is correct.")
        return None

def fetch_transcript(video_id):
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

        return transcript.fetch()

    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        st.error("No transcripts found for this video.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

def translate_to_urdu(text):
    translator = Translator()
    try:
        translated = translator.translate(text, src='tr', dest='ur')
        return translated.text
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text

def convert_to_srt(transcript_data):
    srt_content = ""
    for i, entry in enumerate(transcript_data):
        start = entry['start']
        duration = entry['duration']
        end = start + duration
        text = entry['text']
        translated_text = translate_to_urdu(text)
        srt_content += f"{i + 1}\n"
        srt_content += f"{format_time(start)} --> {format_time(end)}\n"
        srt_content += f"{translated_text}\n\n"
    return srt_content

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def main():
    st.title("YouTube Subtitle Downloader and Translator")

    video_url = st.text_input("Enter YouTube Video URL:")
    if video_url:
        video_id = extract_video_id(video_url)
        if video_id:
            transcript_data = fetch_transcript(video_id)

            if transcript_data:
                srt_content = convert_to_srt(transcript_data)
                st.text_area("Translated SRT Content", srt_content, height=300)

                st.download_button(
                    label="Download Translated SRT File",
                    data=srt_content,
                    file_name=f"{video_id}_translated_transcript.srt",
                    mime="text/plain"
                )

if __name__ == "__main__":
    main()