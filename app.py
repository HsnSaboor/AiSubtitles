import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import re
from deep_translator import GoogleTranslator

def extract_video_id(url):
    """
    Extract the YouTube video ID from various URL formats.
    """
    # Regular expression to match YouTube URL patterns
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

        # Attempt to find Turkish manually created transcript
        if 'tr' in transcript_list._manually_created_transcripts:
            transcript = transcript_list.find_manually_created_transcript(['tr'])
            st.success("Turkish manually created transcript found.")
        # If not available, attempt to find English manually created transcript
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

def convert_to_srt(transcript_data):
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
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def translate_srt_to_urdu(srt_content):
    """
    Translate the content of an SRT file to Urdu.
    """
    # Split the SRT content into lines to identify subtitles
    lines = srt_content.split("\n")
    
    translated_content = []
    
    for line in lines:
        if line.strip():  # If the line is not empty
            # Translate each subtitle line to Urdu
            translated_line = GoogleTranslator(source='tr', target='ur').translate(line)
            translated_content.append(translated_line)
        else:
            translated_content.append(line)
    
    # Join the translated lines back together into SRT format
    translated_srt = "\n".join(translated_content)
    
    return translated_srt

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

                # Translate the SRT content to Urdu
                translated_srt_content = translate_srt_to_urdu(srt_content)
                st.text_area("Translated SRT Content (Urdu)", translated_srt_content, height=300)

                st.download_button(
                    label="Download Original SRT File",
                    data=srt_content,
                    file_name=f"{video_id}_transcript.srt",
                    mime="text/plain"
                )

                st.download_button(
                    label="Download Translated SRT File (Urdu)",
                    data=translated_srt_content,
                    file_name=f"{video_id}_transcript_urdu.srt",
                    mime="text/plain"
                )

if __name__ == "__main__":
    main()