import asyncio
from async_google_trans_new import AsyncTranslator

async def translate_srt_to_urdu_async(srt_content, progress_callback=None):
    """
    Asynchronously translate the content of an SRT file to Urdu.
    """
    translator = AsyncTranslator()
    lines = srt_content.split("\n")
    translated_lines = []
    total_lines = len(lines)

    for i, line in enumerate(lines):
        if line.strip() and not line.replace(' ', '').isdigit() and '-->' not in line:
            translated_line = await translator.translate(line, lang_tgt='ur')
            translated_lines.append(translated_line)
        else:
            translated_lines.append(line)

        # Update progress
        if progress_callback:
            progress_callback((i + 1) / total_lines)

    return "\n".join(translated_lines)

def run_async_task(coro):
    """
    Helper function to run an asynchronous task.
    """
    return asyncio.run(coro)