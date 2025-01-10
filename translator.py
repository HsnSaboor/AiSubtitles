from deep_translator import GoogleTranslator

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