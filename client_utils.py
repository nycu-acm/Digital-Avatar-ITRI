import requests
import json
import time

def stream_rag_llm_query(api_url: str, text_user_msg: str, session_id: str):
    """
    Stream a query to the RAG + LLM API service
    
    Args:
        api_url: Base URL of the API service (e.g., "http://localhost:5002")
        text_user_msg: The user's text message/question
        session_id: Optional session ID for maintaining chat history
    
    Returns:
        Generator yielding streaming text chunks
    """
    
    endpoint = f"{api_url}/api/rag-llm/query"
    payload = {
        "text_user_msg": text_user_msg,
        "session_id": session_id,
        "include_history": True
    }
    
    print(f"🤖 Sending query: {text_user_msg}")
    print("📡 Streaming response:")
    print("-" * 50)
    
    try:
        with requests.post(endpoint, json=payload, stream=True) as response:
            response.raise_for_status()
            
            accumulated_response = ""
            
            for chunk in response.iter_content(chunk_size=1, decode_unicode=True):
                if chunk:
                    # Check for END_FLAG
                    if chunk == "E" and accumulated_response.endswith("END_FLA"):
                        # Complete END_FLAG received
                        print("\n" + "=" * 50)
                        print("✅ Response complete (END_FLAG received)")
                        break
                    elif "END_FLAG" in chunk:
                        # Handle case where END_FLAG comes in one chunk
                        text_part = chunk.replace("END_FLAG", "")
                        if text_part:
                            accumulated_response += text_part
                            print(text_part, end="", flush=True)
                        print("\n" + "=" * 50)
                        print("✅ Response complete (END_FLAG received)")
                        break
                    else:
                        accumulated_response += chunk
                        print(chunk, end="", flush=True)
            
            return accumulated_response
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

def check_service_health(api_url: str):
    """Check if the API service is healthy"""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        response.raise_for_status()
        health_data = response.json()
        
        print(f"🔍 Service Health Check:")
        print(f"  Status: {health_data.get('status')}")
        print(f"  RAG Initialized: {health_data.get('rag_initialized')}")
        print(f"  Timestamp: {health_data.get('timestamp')}")
        
        return health_data.get('status') == 'healthy'
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def initialize_rag_system(api_url: str):
    """Initialize the RAG system via API"""
    try:
        response = requests.post(f"{api_url}/api/rag-llm/init", timeout=30)
        response.raise_for_status()
        result = response.json()
        
        print(f"🔄 RAG Initialization:")
        print(f"  Success: {result.get('success')}")
        print(f"  Message: {result.get('message')}")
        
        return result.get('success', False)
        
    except Exception as e:
        print(f"❌ RAG initialization failed: {e}")
        return False

def close_connection(api_url: str, session_id: str):
    """
    Elegant connection close function
    
    Client calls this function before program termination.
    Server will clear the history of the client based on the chat_session_id.
    
    Args:
        api_url: Base URL of the API service (e.g., "http://localhost:5002")
        session_id: Session ID to close and clean up
    
    Returns:
        bool: True if connection was closed successfully, False otherwise
    """
    try:
        endpoint = f"{api_url}/api/rag-llm/close"
        payload = {
            "session_id": session_id
        }
        
        print(f"👋 Closing connection for session: {session_id}")
        
        response = requests.post(endpoint, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        
        print(f"✅ Connection Close:")
        print(f"  Session ID: {result.get('session_id')}")
        print(f"  Success: {result.get('success')}")
        print(f"  Message: {result.get('message')}")
        print(f"  Session Existed: {result.get('session_existed')}")
        print(f"  Messages Cleared: {result.get('messages_cleared')}")
        
        return result.get('success', False)
        
    except Exception as e:
        print(f"❌ Connection close failed: {e}")
        return False
