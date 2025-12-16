#!/usr/bin/env python3
"""
Multi-Session Vision API for Digital Avatar System
Handles concurrent user sessions with individual visual perception
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image
import sys
import os
import time
import torch
import threading
import queue
import uuid
from collections import defaultdict

# Add FastVLM to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ml-fastvlm'))

from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

app = Flask(__name__)
CORS(app)

class MultiSessionVisionProcessor:
    """FastVLM processor supporting multiple concurrent user sessions"""
    
    def __init__(self):
        # FastVLM model (shared across all sessions)
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.device = None
        self.initialized = False
        
        # Per-session data structures
        self.session_queues = {}          # sessionid -> Queue(maxsize=1)
        self.session_contexts = {}        # sessionid -> latest_analysis
        self.session_workers = {}         # sessionid -> worker_thread
        self.session_locks = {}           # sessionid -> threading.Lock()
        self.session_stats = defaultdict(lambda: {
            'total_received': 0,
            'total_processed': 0,
            'total_dropped': 0,
            'last_analysis_time': None,
            'is_processing': False,
            'current_processing_id': None
        })
        
        # Global management
        self.shutdown_events = {}         # sessionid -> Event()
        self.global_lock = threading.Lock()
        
    def initialize_model(self):
        """Initialize shared FastVLM model"""
        if self.initialized:
            return True
            
        try:
            print("[MultiSessionVision] Initializing FastVLM model...")
            
            # Auto-detect device
            if torch.cuda.is_available():
                self.device = "cuda:0"
                print(f"[MultiSessionVision] Using GPU: {self.device}")
            else:
                self.device = "cpu"
                print(f"[MultiSessionVision] Using CPU")
            
            # Load FastVLM model
            disable_torch_init()
            model_path = os.path.join(os.path.dirname(__file__), 'ml-fastvlm', 'llava-fastvithd_0.5b_stage3')
            model_name = get_model_name_from_path(model_path)
            
            self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
                model_path, None, model_name, device=self.device
            )
            
            self.initialized = True
            print("[MultiSessionVision] FastVLM model loaded successfully!")
            print("[MultiSessionVision] Ready for multi-session visual perception")
            return True
            
        except Exception as e:
            print(f"[MultiSessionVision] Failed to initialize model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_session(self, sessionid):
        """Create new session with dedicated processing resources"""
        with self.global_lock:
            if sessionid in self.session_queues:
                print(f"[MultiSessionVision] Session {sessionid} already exists")
                return True
            
            try:
                # Create session-specific resources
                self.session_queues[sessionid] = queue.Queue(maxsize=1)
                self.session_contexts[sessionid] = None
                self.session_locks[sessionid] = threading.Lock()
                self.shutdown_events[sessionid] = threading.Event()
                
                # Start dedicated worker for this session
                worker = threading.Thread(
                    target=self._session_worker,
                    args=(sessionid,),
                    daemon=True,
                    name=f"VisionWorker-{sessionid}"
                )
                worker.start()
                self.session_workers[sessionid] = worker
                
                print(f"[MultiSessionVision] Created session {sessionid} with dedicated worker")
                return True
                
            except Exception as e:
                print(f"[MultiSessionVision] Failed to create session {sessionid}: {e}")
                self._cleanup_session_resources(sessionid)
                return False
    
    def _session_worker(self, sessionid):
        """Dedicated worker thread for processing images from specific session"""
        print(f"[MultiSessionVision] Worker started for session {sessionid}")
        
        session_queue = self.session_queues[sessionid]
        shutdown_event = self.shutdown_events[sessionid]
        session_lock = self.session_locks[sessionid]
        
        while not shutdown_event.is_set():
            try:
                # Wait for image from this session
                try:
                    image_data, image_id, timestamp = session_queue.get(timeout=2.0)
                    session_queue.task_done()
                except queue.Empty:
                    continue
                
                # Mark as processing
                with session_lock:
                    self.session_stats[sessionid]['is_processing'] = True
                    self.session_stats[sessionid]['current_processing_id'] = image_id
                
                print(f"[MultiSessionVision] Session {sessionid} processing image {image_id}")
                
                # Analyze image with shared FastVLM model
                analysis_start = time.time()
                analysis = self._analyze_image_internal(image_data)
                processing_time = time.time() - analysis_start
                queue_delay = time.time() - timestamp
                
                # Store result for this session
                result = {
                    'sessionid': sessionid,
                    'image_id': image_id,
                    'analysis': analysis,
                    'processing_time': processing_time,
                    'timestamp': time.time(),
                    'queue_delay': queue_delay
                }
                
                # Update session context and stats
                with session_lock:
                    self.session_contexts[sessionid] = result
                    self.session_stats[sessionid]['total_processed'] += 1
                    self.session_stats[sessionid]['last_analysis_time'] = time.time()
                    self.session_stats[sessionid]['is_processing'] = False
                    self.session_stats[sessionid]['current_processing_id'] = None
                
                # Log result
                timestamp_str = time.strftime("%H:%M:%S")
                print(f"\n{'='*80}")
                print(f"[{timestamp_str}] SESSION {sessionid} - VISUAL ANALYSIS (ID: {image_id})")
                print(f"Processing: {processing_time:.2f}s | Queue Delay: {queue_delay:.2f}s")
                print(f"Session Stats: Processed={self.session_stats[sessionid]['total_processed']}, "
                      f"Dropped={self.session_stats[sessionid]['total_dropped']}")
                print(f"{'='*80}")
                print(f"{analysis}")
                print(f"{'='*80}\n")
                
            except Exception as e:
                print(f"[MultiSessionVision] Session {sessionid} worker error: {e}")
                with session_lock:
                    self.session_stats[sessionid]['is_processing'] = False
                    self.session_stats[sessionid]['current_processing_id'] = None
                time.sleep(1)
        
        print(f"[MultiSessionVision] Worker stopped for session {sessionid}")
    
    def queue_image_for_session(self, sessionid, image_data):
        """Queue image for analysis by specific session"""
        if not self.initialized:
            return {'error': 'Model not initialized', 'success': False}
        
        # Create session if it doesn't exist
        if sessionid not in self.session_queues:
            if not self.create_session(sessionid):
                return {'error': f'Failed to create session {sessionid}', 'success': False}
        
        session_queue = self.session_queues[sessionid]
        session_lock = self.session_locks[sessionid]
        
        image_id = str(uuid.uuid4())[:8]
        timestamp = time.time()
        
        with session_lock:
            self.session_stats[sessionid]['total_received'] += 1
        
        # C1 Strategy per session: Replace if queue is full
        try:
            session_queue.put_nowait((image_data, image_id, timestamp))
            print(f"[MultiSessionVision] Session {sessionid} queued image {image_id}")
        except queue.Full:
            try:
                # Remove old image and add new one
                old_image, old_id, old_timestamp = session_queue.get_nowait()
                with session_lock:
                    self.session_stats[sessionid]['total_dropped'] += 1
                print(f"[MultiSessionVision] Session {sessionid} dropped image {old_id}, queued {image_id}")
                
                session_queue.put_nowait((image_data, image_id, timestamp))
            except queue.Empty:
                session_queue.put_nowait((image_data, image_id, timestamp))
        
        # Return status
        with session_lock:
            stats = self.session_stats[sessionid].copy()
        
        return {
            'success': True,
            'sessionid': sessionid,
            'image_id': image_id,
            'queued': True,
            'stats': stats
        }
    
    def get_session_context(self, sessionid):
        """Get latest visual context for specific session"""
        if sessionid not in self.session_contexts:
            return None
        
        with self.session_locks.get(sessionid, threading.Lock()):
            context = self.session_contexts[sessionid]
            return context.copy() if context else None
    
    def cleanup_session(self, sessionid):
        """Clean up resources for ended session"""
        with self.global_lock:
            if sessionid not in self.session_queues:
                return
            
            print(f"[MultiSessionVision] Cleaning up session {sessionid}")
            
            # Signal worker to stop
            if sessionid in self.shutdown_events:
                self.shutdown_events[sessionid].set()
            
            # Wait for worker to finish
            if sessionid in self.session_workers:
                worker = self.session_workers[sessionid]
                worker.join(timeout=3)
            
            # Clean up resources
            self._cleanup_session_resources(sessionid)
            
            print(f"[MultiSessionVision] Session {sessionid} cleaned up")
    
    def _cleanup_session_resources(self, sessionid):
        """Internal method to clean up session resources"""
        # Remove from all data structures
        self.session_queues.pop(sessionid, None)
        self.session_contexts.pop(sessionid, None)
        self.session_workers.pop(sessionid, None)
        self.session_locks.pop(sessionid, None)
        self.shutdown_events.pop(sessionid, None)
        self.session_stats.pop(sessionid, None)
    
    def _analyze_image_internal(self, image):
        """Internal method to analyze image with FastVLM (shared across sessions)"""
        if not self.initialized:
            return "Model not initialized"
        
        try:
            # Avatar-specific prompt for user analysis
            prompt = "Analyze this user for avatar interaction: 1) Person's apparent age range, gender, and current facial expression or emotion. 2) Clothing style and color. 3) Environment setting (office, home, lighting quality). 4) Overall mood and energy level. Be concise but specific for personalizing conversation responses."
            
            # Prepare conversation
            conv = conv_templates["qwen_2"].copy()
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + prompt
            
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt_formatted = conv.get_prompt()
            
            # Set pad token for generation
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
            
            # Tokenize prompt
            input_ids = tokenizer_image_token(
                prompt_formatted, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
            ).unsqueeze(0).to(torch.device(self.device))
            
            # Process image
            image_tensor = process_images([image], self.image_processor, self.model.config)[0]
            image_tensor = image_tensor.to(self.device)
            
            # Run inference
            with torch.inference_mode():
                if self.device.startswith("cuda"):
                    images_input = image_tensor.unsqueeze(0).half()
                else:
                    images_input = image_tensor.unsqueeze(0)
                
                output_ids = self.model.generate(
                    input_ids,
                    images=images_input,
                    image_sizes=[image.size],
                    do_sample=False,
                    temperature=0.1,
                    max_new_tokens=150,
                    use_cache=True
                )
                
                outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                
                # Extract response
                if conv.roles[1] in outputs:
                    analysis = outputs.split(conv.roles[1])[-1].strip()
                else:
                    analysis = outputs
                
                return analysis
                
        except Exception as e:
            print(f"[MultiSessionVision] Analysis error: {e}")
            return f"Analysis failed: {str(e)}"
    
    def get_all_sessions_status(self):
        """Get status of all active sessions"""
        with self.global_lock:
            return {
                'active_sessions': list(self.session_queues.keys()),
                'total_sessions': len(self.session_queues),
                'session_stats': dict(self.session_stats)
            }

# Global processor instance
processor = MultiSessionVisionProcessor()

@app.route('/analyze-vision', methods=['POST'])
def analyze_vision():
    """Queue image for analysis by specific session"""
    try:
        data = request.json
        if not data or 'image' not in data or 'sessionid' not in data:
            return jsonify({'error': 'Missing image data or sessionid'}), 400
        
        sessionid = str(data['sessionid'])  # Ensure string
        image_data = data['image']
        
        # Remove data URL prefix if present
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        # Convert to PIL Image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Queue for session-specific analysis
        result = processor.queue_image_for_session(sessionid, image)
        
        return jsonify({
            **result,
            'image_size': f"{image.size[0]}x{image.size[1]}"
        })
        
    except Exception as e:
        print(f"[MultiSessionVision] Endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get-visual-context/<sessionid>', methods=['GET'])
def get_visual_context(sessionid):
    """Get latest visual context for specific session"""
    try:
        context = processor.get_session_context(str(sessionid))
        
        if context:
            return jsonify({
                'success': True,
                'sessionid': sessionid,
                'visual_context': context['analysis'],
                'timestamp': context['timestamp'],
                'processing_time': context['processing_time']
            })
        else:
            return jsonify({
                'success': False,
                'sessionid': sessionid,
                'visual_context': None,
                'message': 'No visual context available'
            })
        
    except Exception as e:
        print(f"[MultiSessionVision] Get context error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/cleanup-session/<sessionid>', methods=['POST'])
def cleanup_session(sessionid):
    """Clean up resources for ended session"""
    try:
        processor.cleanup_session(str(sessionid))
        return jsonify({'success': True, 'message': f'Session {sessionid} cleaned up'})
    except Exception as e:
        print(f"[MultiSessionVision] Cleanup error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    """Get overall system status"""
    return jsonify({
        'status': 'running',
        'model_initialized': processor.initialized,
        'device': processor.device if processor.initialized else 'not_initialized',
        **processor.get_all_sessions_status()
    })

@app.route('/initialize', methods=['POST'])
def initialize_model():
    """Initialize the FastVLM model"""
    success = processor.initialize_model()
    return jsonify({'success': success, 'initialized': processor.initialized})

# ===== SIMPLE API ENDPOINTS FOR LLM TEAM =====

@app.route('/visual-context/<sessionid>', methods=['GET'])
def get_visual_context_simple(sessionid):
    """Simple endpoint for LLM team to get visual context for a session"""
    try:
        context = processor.get_session_context(str(sessionid))
        
        if context and context.get('analysis'):
            return jsonify({
                'sessionid': sessionid,
                'visual_context': context['analysis'],
                'available': True
            })
        else:
            return jsonify({
                'sessionid': sessionid,
                'visual_context': None,
                'available': False
            })
        
    except Exception as e:
        print(f"[LLM API] Error getting context for session {sessionid}: {e}")
        return jsonify({
            'sessionid': sessionid,
            'visual_context': None,
            'available': False,
            'error': str(e)
        }), 500

@app.route('/sessions', methods=['GET'])
def get_active_sessions():
    """Get list of active sessions with vision enabled"""
    try:
        with processor.global_lock:
            active_sessions = []
            for sessionid in processor.session_queues.keys():
                context = processor.get_session_context(sessionid)
                has_context = context is not None and context.get('analysis') is not None
                
                active_sessions.append({
                    'sessionid': sessionid,
                    'vision_enabled': True,
                    'has_visual_context': has_context
                })
            
            return jsonify({
                'active_sessions': active_sessions,
                'total_sessions': len(active_sessions)
            })
        
    except Exception as e:
        print(f"[LLM API] Error getting active sessions: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Multi-Session Vision API Server...")
    print("Initializing FastVLM model for concurrent user sessions...")
    processor.initialize_model()
    print("Ready to handle multiple concurrent avatar sessions!")
    app.run(host='0.0.0.0', port=5004, debug=False)  # Different port to avoid conflict