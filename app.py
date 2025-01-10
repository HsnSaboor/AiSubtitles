import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import re
from translator import translate_srt_to_urdu

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
        # Fetch available transcripts for the video
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Try to get a manually created Turkish transcript
        if 'tr' in transcript_list._manually_created_transcripts:
            transcript = transcript_list.find_manually_created_transcript(['tr'])
            st.success("Turkish manually created transcript found.")
        # Try to get a manually created English transcript
        elif 'en' in transcript_list._manually_created_transcripts:
            transcript = transcript_list.find_manually_created_transcript(['en'])
            st.success("English manually created transcript found.")
        else:
            st.error("No manually created transcripts found in Turkish or English.")
            return None

        # Fetch the transcript data as a list of dictionary entries
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
    """
    Converts the transcript data (a list of dictionaries) to an SRT formatted string.
    """
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
    """
    Converts seconds into the SRT time format (hh:mm:ss,SSS).
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def main():
    st.title("YouTube Subtitle Downloader & Translator")

    video_url = st.text_input("Enter YouTube Video URL:")
    if video_url:
        video_id = extract_video_id(video_url)
        if video_id:
            transcript_data = fetch_transcript(video_id)

            if transcript_data:
                # Convert the transcript data to SRT format
                srt_content = convert_to_srt(transcript_data)
                st.text_area("Original SRT Content", srt_content, height=300)

                # Translate the SRT content to Urdu
                translated_srt_content = translate_srt_to_urdu(srt_content)
                st.text_area("Translated SRT Content (Urdu)", translated_srt_content, height=300)

                # Allow users to download both the original and translated SRT files
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