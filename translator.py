import os
import json
from groq import Groq
from typing import List

# Get the Groq API key from the environment variable
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize the Groq client with the API key from the environment
client = Groq(api_key=groq_api_key)

# System prompt with instructions for the model
system_prompt = """
You are a translator for subtitle files. You must preserve the emotional tone of the sentences, correct character names, and handle cultural context properly.
You should also format the output as JSON with the following structure:
{
    "subtitles": [
        {
            "index": int,
            "start": str (time in "hh:mm:ss,SSS" format),
            "end": str (time in "hh:mm:ss,SSS" format),
            "text": str (translated Urdu subtitle)
        }
    ]
}
"""

def create_chunks(srt_data: List[str], chunk_size: int) -> List[List[str]]:
    """
    Split the SRT data into chunks of size `chunk_size` while preserving full lines.
    """
    chunks = []
    current_chunk = []
    current_token_count = 0
    
    for line in srt_data:
        # Estimate token count per line (simple approximation)
        token_count = len(line.split())
        
        # If adding this line exceeds the chunk size, store the current chunk and start a new one
        if current_token_count + token_count > chunk_size:
            chunks.append(current_chunk)
            current_chunk = [line]
            current_token_count = token_count
        else:
            current_chunk.append(line)
            current_token_count += token_count
    
    # Append any remaining lines as the last chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def translate_srt_to_urdu(srt_data: str) -> str:
    """
    Translates SRT file to Urdu using Groq's Llama-3.3-70B model.
    """
    # Split SRT into lines and chunks
    srt_lines = srt_data.splitlines()
    chunks = create_chunks(srt_lines, chunk_size=25000)  # Allow 25K tokens for the chunk
    
    translated_subtitles = []
    
    for chunk in chunks:
        # Prepare the request payload
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n".join(chunk)}
        ]
        
        # Send the request to Groq API for translation
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            temperature=0.5,
            max_tokens=1024,
            top_p=1,
            stream=False,
            response_format={"type": "json_object"}
        )
        
        # Parse the JSON response and append it to the result
        response = chat_completion.choices[0].message.content
        translated_subtitles.append(json.loads(response)["subtitles"])
    
    # Flatten the list of translated subtitles
    all_translated_subtitles = [item for sublist in translated_subtitles for item in sublist]
    
    # Convert the translated subtitles back to SRT format
    translated_srt = ""
    for subtitle in all_translated_subtitles:
        translated_srt += f"{subtitle['index']}\n"
        translated_srt += f"{subtitle['start']} --> {subtitle['end']}\n"
        translated_srt += f"{subtitle['text']}\n\n"
    
    return translated_srt