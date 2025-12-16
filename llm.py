import time
import os
from basereal import BaseReal
from logger import logger
import requests
import datetime
import re

# Configuration for local API service
API_URL = os.getenv("LOCAL_API_URL", "http://localhost:5002")

def debug_log(message, session_id):
    """Write debug information to a dedicated file for analysis"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    debug_file = f"/mnt/HDD4/thanglq/he110/GitSpace/TalkingHead/llm_debug_{session_id}.log"
    
    with open(debug_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")
        f.flush()

def clean_end_tokens(text):
    """Remove all END_FLAG variations from text"""
    if not text:
        return text
    
    debug_original = text
    
    # Remove END_FLAG and variations with multiple patterns
    patterns = [
        r'\bEND_FLAG\b',     # Exact word match
        r'\bEND_\w*',        # END_ + any word characters
        r'\.END_FLAG',       # .END_FLAG 
        r'\s+END_FLAG',      # whitespace + END_FLAG
        r'END_FLAG\s*',      # END_FLAG + optional whitespace
    ]
    
    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Clean up extra spaces only - preserve emotional tags and natural punctuation
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Only remove leading punctuation that appears without context
    cleaned = re.sub(r'^\s*[,.!;:，。！？：；]\s*', '', cleaned)
    
    # Don't remove trailing punctuation - it's important for natural speech flow
    # The LLM already provides proper emotional tags and punctuation
    
    # Return None if empty after cleaning
    result = cleaned if cleaned and len(cleaned.strip()) > 0 else None
    
    # Log cleaning operation if something was changed
    if result != debug_original:
        print(f"CLEANED: '{debug_original}' -> '{result}'")
    
    return result

def llm_response(message, nerfreal: BaseReal, user_description: str = None):
    """
    Send message to local RAG+LLM API and stream response to TTS pipeline.
    Maintains the same streaming logic as the original OpenAI implementation.
    
    Args:
        message: User's text message
        nerfreal: BaseReal instance for the avatar session
        user_description: Optional visual description (from Vision API or manual input)
    """
    start = time.perf_counter()
    
    # Generate unique session_id based on the nerfreal session
    session_id = f"talkinghead_session_{nerfreal.sessionid}"
    
    end = time.perf_counter()
    logger.info(f"llm Time init: {end-start}s")
    
    # Prepare endpoint and payload for local API following CLIENT_README.md specs
    endpoint = f"{API_URL}/api/rag-llm/query"
    payload = {
        "text_user_msg": message,
        "session_id": session_id,
        "convert_tone": True,
        "include_history": True
    }
    
    # Add user_description if provided (from Vision API or manual input)
    if user_description:
        payload["user_description"] = user_description
        debug_log(f"User description provided: '{user_description}'", session_id)
    
    result = ""
    first = True
    
    # Initialize debug log for this session
    debug_log(f"=== NEW REQUEST ===", session_id)
    debug_log(f"User message: '{message}'", session_id)
    debug_log(f"Session ID: {session_id}", session_id)
    debug_log(f"API URL: {endpoint}", session_id)
    debug_log(f"Convert tone enabled: {payload['convert_tone']}", session_id)
    debug_log(f"Will fetch vision context and apply dynamic tone conversion", session_id)
    
    try:
        # Stream response from local API as per CLIENT_README.md
        with requests.post(endpoint, json=payload, stream=True) as response:
            response.raise_for_status()
            debug_log(f"API Response status: {response.status_code}", session_id)
            
            # DEBUG MODE: NO FILTERING - collect ALL text to see raw responses
            all_response_text = ""
            chunk_count = 0
            
            for chunk in response.iter_content(chunk_size=1, decode_unicode=True):
                if chunk:
                    chunk_count += 1
                    debug_log(f"Chunk #{chunk_count}: '{chunk}' (ASCII: {ord(chunk) if len(chunk)==1 else 'multi-char'})", session_id)
                    all_response_text += chunk
                    
                    # REMOVED: No more early break on END_ detection
                    # Let's see the complete raw response
                        
                    # Process chunk immediately with original logic (but with debug)
                    if first:
                        end = time.perf_counter()
                        logger.info(f"llm Time to first chunk: {end-start}s")
                        debug_log(f"First chunk received in {end-start}s", session_id)
                        first = False
                    
                    # Original OpenAI streaming logic with debug
                    msg = chunk
                    if msg is None:
                        continue
                        
                    lastpos = 0
                    for i, char in enumerate(msg):
                        if char in ",.!;:，。！？：；":
                            fragment = result + msg[lastpos:i+1]
                            lastpos = i+1
                            if len(fragment) > 10:
                                # FILTER 1: Clean fragment before sending to TTS
                                cleaned_fragment = clean_end_tokens(fragment)
                                if cleaned_fragment:  # Only send if not empty after cleaning
                                    debug_log(f"SENDING TO TTS (FILTERED): '{cleaned_fragment}' (length: {len(cleaned_fragment)})", session_id)
                                    logger.info(cleaned_fragment)
                                    nerfreal.put_msg_txt(cleaned_fragment)
                                else:
                                    debug_log(f"FRAGMENT FILTERED OUT: '{fragment}' (was END_FLAG)", session_id)
                                result = ""
                            else:
                                result = fragment
                                debug_log(f"Fragment too short, accumulating: '{result}'", session_id)
                    
                    result = result + msg[lastpos:]
                    debug_log(f"Current result buffer: '{result}'", session_id)
            
            # DEBUG MODE: Show complete raw response without any filtering
            debug_log(f"COMPLETE RAW API RESPONSE: '{all_response_text}'", session_id)
            debug_log(f"Final result buffer: '{result}'", session_id)
            
    except requests.exceptions.RequestException as e:
        error_msg = f"API request error: {e}"
        logger.error(error_msg)
        debug_log(f"ERROR: {error_msg}", session_id)
        # Fallback: send error message to TTS
        nerfreal.put_msg_txt(f"Error connecting to local API: {str(e)}")
        return
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        logger.error(error_msg)
        debug_log(f"ERROR: {error_msg}", session_id)
        nerfreal.put_msg_txt(f"Unexpected error: {str(e)}")
        return
    
    end = time.perf_counter()
    logger.info(f"llm Time to last chunk: {end-start}s")
    debug_log(f"Total processing time: {end-start}s", session_id)
    
    # FILTER 2: Clean final buffer before sending to TTS
    if result:
        cleaned_final = clean_end_tokens(result)
        if cleaned_final:  # Only send if not empty after cleaning
            debug_log(f"FINAL SEND TO TTS (FILTERED): '{cleaned_final}' (length: {len(cleaned_final)})", session_id)
            logger.info(cleaned_final)
            nerfreal.put_msg_txt(cleaned_final)
        else:
            debug_log(f"FINAL BUFFER FILTERED OUT: '{result}' (was END_FLAG)", session_id)
    else:
        debug_log("No remaining text to send", session_id)
    
    debug_log("=== REQUEST COMPLETED ===", session_id)