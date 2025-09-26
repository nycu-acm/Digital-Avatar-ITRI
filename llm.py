import time
import os
from basereal import BaseReal
from logger import logger
import requests
import datetime

# Configuration for local API service
API_URL = os.getenv("LOCAL_API_URL", "http://localhost:5002")

def debug_log(message, session_id):
    """Write debug information to a dedicated file for analysis"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    debug_file = f"/mnt/HDD4/thanglq/he110/GitSpace/TalkingHead/llm_debug_{session_id}.log"
    
    with open(debug_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")
        f.flush()

def llm_response(message, nerfreal: BaseReal):
    """
    Send message to local RAG+LLM API and stream response to TTS pipeline.
    Maintains the same streaming logic as the original OpenAI implementation.
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
        "include_history": True
    }
    
    result = ""
    first = True
    
    # Initialize debug log for this session
    debug_log(f"=== NEW REQUEST ===", session_id)
    debug_log(f"User message: '{message}'", session_id)
    debug_log(f"Session ID: {session_id}", session_id)
    debug_log(f"API URL: {endpoint}", session_id)
    
    try:
        # Stream response from local API as per CLIENT_README.md
        with requests.post(endpoint, json=payload, stream=True) as response:
            response.raise_for_status()
            debug_log(f"API Response status: {response.status_code}", session_id)
            
            # Simple approach: collect ALL text first, then filter END_FLAG
            all_response_text = ""
            chunk_count = 0
            
            for chunk in response.iter_content(chunk_size=1, decode_unicode=True):
                if chunk:
                    chunk_count += 1
                    debug_log(f"Chunk #{chunk_count}: '{chunk}' (ASCII: {ord(chunk) if len(chunk)==1 else 'multi-char'})", session_id)
                    all_response_text += chunk
                    
                    # Check if we have END_FLAG in our accumulated text
                    if "END_FLAG" in all_response_text:
                        debug_log(f"END_FLAG detected! Full response so far: '{all_response_text}'", session_id)
                        break
                        
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
                                debug_log(f"SENDING TO TTS: '{fragment}' (length: {len(fragment)})", session_id)
                                logger.info(fragment)
                                nerfreal.put_msg_txt(fragment)
                                result = ""
                            else:
                                result = fragment
                                debug_log(f"Fragment too short, accumulating: '{result}'", session_id)
                    
                    result = result + msg[lastpos:]
                    debug_log(f"Current result buffer: '{result}'", session_id)
            
            # Clean the response by removing END_FLAG
            if "END_FLAG" in all_response_text:
                clean_response = all_response_text.split("END_FLAG")[0]
                debug_log(f"Response before END_FLAG: '{clean_response}'", session_id)
                debug_log(f"Final result buffer: '{result}'", session_id)
            else:
                debug_log(f"No END_FLAG found. Full response: '{all_response_text}'", session_id)
            
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
    
    # Send any remaining text (same as original code) - but filter out END_FLAG fragments
    if result and not result.startswith("END_"):
        debug_log(f"FINAL SEND TO TTS: '{result}' (length: {len(result)})", session_id)
        logger.info(result)
        nerfreal.put_msg_txt(result)
    else:
        if result.startswith("END_"):
            debug_log(f"SKIPPING END_FLAG fragment: '{result}' (length: {len(result)})", session_id)
        else:
            debug_log("No remaining text to send", session_id)
    
    debug_log("=== REQUEST COMPLETED ===", session_id)