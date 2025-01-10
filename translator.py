import os
import json
from groq import Groq
from typing import List
import time

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

# Estimate the number of tokens in the system prompt
system_prompt_tokens = len(system_prompt.split())

# Define the maximum tokens per request
max_tokens_per_request = 6000

# Calculate the available tokens for input text
available_tokens_for_input = max_tokens_per_request - system_prompt_tokens

# Define the number of chunks to process per minute
chunks_per_minute = 3

# Calculate the maximum tokens per chunk
max_tokens_per_chunk = available_tokens_for_input // chunks_per_minute

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
    chunks = create_chunks(srt_lines, chunk_size=max_tokens_per_chunk)  # Allow chunk size based on available tokens
    
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
            max_tokens=8192,
            top_p=1,
            stream=False,
            response_format={"type": "json_object"}
        )
        
        # Parse the JSON response and append it to the result
        response = chat_completion.choices[0].message.content
        translated_subtitles.append(json.loads(response)["subtitles"])
        
        # Sleep to adhere to the rate limit of 3 requests per minute
        time.sleep(20)  # Sleep for 20 seconds between requests
    
    # Flatten the list of translated subtitles
    all_translated_subtitles = [item for sublist in translated_subtitles for item in sublist]
    
    # Convert the translated subtitles back to SRT format
    translated_srt = ""
    for subtitle in all_translated_subtitles:
        translated_srt += f"{subtitle['index']}\n"
        translated_srt += f"{subtitle['start']} --> {subtitle['end']}\n"
        translated_srt += f"{subtitle['text']}\n\n"
    
    return translated_srt