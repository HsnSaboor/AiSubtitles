import json
import re
from groq import Groq
import os

# Get the Groq API key from the environment variable
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize the Groq client with the API key from the environment
client = Groq(api_key=groq_api_key)


def chunk_srt(srt_content, chunk_size=5000):
    """
    Chunk the SRT content into manageable pieces for translation.
    Each chunk will have approximately 'chunk_size' tokens.
    """
    # Tokenize the SRT content
    tokens = srt_content.split()
    chunks = []
    current_chunk = []
    current_size = 0

    for token in tokens:
        current_chunk.append(token)
        current_size += len(token)
        if current_size >= chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def translate_chunk(chunk):
    """
    Translate a single chunk of text using the Groq API.
    """
    system_prompt = """
    You are a highly accurate and sensitive translator. Your task is to translate Turkish subtitles into Urdu.
    Please ensure that you:
    - Preserve the emotions of the sentences (e.g., excitement, sadness).
    - Keep the character names and dialogue intact without modifying them.
    - Ensure that Urdu translation closely matches the emotional tone and context of the original Turkish text.
    
    You will receive Turkish subtitles in chunks. Translate them into Urdu while maintaining the original structure of the subtitles. 

    The output format must be a JSON object with the following structure:
    [
        {
            "index": <Subtitle index number>,
            "time": "<Subtitle time range in SRT format>",
            "text": "<Translated Urdu subtitle>"
        },
        ...
    ]
    """
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chunk}
        ],
        response_format={"type": "json_object"}
    )
    
    return response.choices[0].message.content

def translate_srt_to_urdu(srt_content):
    """
    Translate the entire SRT content from Turkish to Urdu.
    """
    # Chunk the SRT content
    chunks = chunk_srt(srt_content)

    translated_chunks = []
    for chunk in chunks:
        translated_chunk = translate_chunk(chunk)
        translated_chunks.append(translated_chunk)

    # Reassemble the translated chunks into a single string
    translated_srt = '\n'.join(translated_chunks)
    return translated_srt

def convert_srt_to_json(srt_content):
    """
    Convert SRT content to JSON format.
    """
    srt_json = []
    lines = srt_content.strip().split('\n\n')
    for line in lines:
        parts = line.split('\n')
        index = parts[0]
        time = parts[1]
        text = '\n'.join(parts[2:])
        srt_json.append({
            'index': index,
            'time': time,
            'text': text
        })
    return json.dumps(srt_json, ensure_ascii=False, indent=4)

def convert_json_to_srt(srt_json):
    """
    Convert JSON format back to SRT content.
    """
    srt_content = ''
    for item in srt_json:
        srt_content += f"{item['index']}\n{item['time']}\n{item['text']}\n\n"
    return srt_content.strip()