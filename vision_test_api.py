#!/usr/bin/env python3
"""
Separate API server for testing FastVLM vision analysis
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

# Add FastVLM to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ml-fastvlm'))

from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

app = Flask(__name__)
CORS(app)

class FastVLMProcessor:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.device = None
        self.initialized = False
        
        # Real-time processing queue (C1 strategy)
        self.image_queue = queue.Queue(maxsize=1)  # Only hold latest image
        self.processing_lock = threading.Lock()
        self.is_processing = False
        self.worker_thread = None
        self.shutdown_event = threading.Event()
        self.latest_result = None
        self.latest_result_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'total_received': 0,
            'total_processed': 0,
            'total_dropped': 0,
            'current_processing_id': None
        }
        
    def initialize(self):
        """Initialize FastVLM model"""
        if self.initialized:
            return True
            
        try:
            print("[VisionTestAPI] Initializing FastVLM model...")
            
            # Auto-detect device
            if torch.cuda.is_available():
                self.device = "cuda:0"
                print(f"[VisionTestAPI] Using GPU: {self.device}")
            else:
                self.device = "cpu"
                print(f"[VisionTestAPI] Using CPU")
            
            # Load FastVLM model
            disable_torch_init()
            model_path = os.path.join(os.path.dirname(__file__), 'ml-fastvlm', 'llava-fastvithd_0.5b_stage3')
            model_name = get_model_name_from_path(model_path)
            
            self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
                model_path, None, model_name, device=self.device
            )
            
            self.initialized = True
            print("[VisionTestAPI] FastVLM model loaded successfully!")
            
            # Start worker thread for C1 processing
            self.start_worker_thread()
            return True
            
        except Exception as e:
            print(f"[VisionTestAPI] Failed to initialize model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def start_worker_thread(self):
        """Start the worker thread for C1 processing strategy"""
        if self.worker_thread and self.worker_thread.is_alive():
            return
        
        self.shutdown_event.clear()
        self.worker_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.worker_thread.start()
        print("[VisionTestAPI] Worker thread started for real-time processing")
    
    def _processing_worker(self):
        """Worker thread that processes images using C1 strategy"""
        print("[VisionTestAPI] Processing worker started")
        
        while not self.shutdown_event.is_set():
            try:
                # Wait for image with timeout
                try:
                    image_data, image_id, timestamp = self.image_queue.get(timeout=1.0)
                    self.image_queue.task_done()
                except queue.Empty:
                    continue
                
                with self.processing_lock:
                    self.is_processing = True
                    self.stats['current_processing_id'] = image_id
                
                print(f"[VisionTestAPI] Processing image {image_id} (captured at {time.time() - timestamp:.2f}s ago)")
                
                # Analyze the image
                analysis_start = time.time()
                analysis = self._analyze_image_internal(image_data)
                processing_time = time.time() - analysis_start
                
                # Store result
                result = {
                    'image_id': image_id,
                    'analysis': analysis,
                    'processing_time': processing_time,
                    'timestamp': time.time(),
                    'queue_delay': time.time() - timestamp
                }
                
                with self.latest_result_lock:
                    self.latest_result = result
                
                self.stats['total_processed'] += 1
                
                # Log processing completion
                timestamp_str = time.strftime("%H:%M:%S")
                queue_delay = result['queue_delay']
                print(f"\n{'='*80}")
                print(f"[{timestamp_str}] REAL-TIME VISION ANALYSIS (ID: {image_id})")
                print(f"Processing: {processing_time:.2f}s | Queue Delay: {queue_delay:.2f}s")
                print(f"Stats: Processed={self.stats['total_processed']}, Dropped={self.stats['total_dropped']}")
                print(f"{'='*80}")
                print(f"{analysis}")
                print(f"{'='*80}\n")
                
                with self.processing_lock:
                    self.is_processing = False
                    self.stats['current_processing_id'] = None
                
            except Exception as e:
                print(f"[VisionTestAPI] Worker error: {e}")
                with self.processing_lock:
                    self.is_processing = False
                    self.stats['current_processing_id'] = None
                time.sleep(1)
        
        print("[VisionTestAPI] Processing worker stopped")
    
    def queue_image_for_analysis(self, image_data):
        """Queue image for analysis using C1 strategy (Replace & Wait)"""
        if not self.initialized:
            return {'error': 'Model not initialized', 'success': False}
        
        image_id = str(uuid.uuid4())[:8]
        timestamp = time.time()
        self.stats['total_received'] += 1
        
        # C1 Strategy: Replace if queue is full (only keep latest)
        try:
            # Try to put without blocking
            self.image_queue.put_nowait((image_data, image_id, timestamp))
            print(f"[VisionTestAPI] Queued image {image_id} for analysis")
        except queue.Full:
            # Queue is full, replace with new image
            try:
                # Remove old image
                old_image, old_id, old_timestamp = self.image_queue.get_nowait()
                self.stats['total_dropped'] += 1
                print(f"[VisionTestAPI] Dropped image {old_id} (replaced with {image_id})")
                
                # Add new image
                self.image_queue.put_nowait((image_data, image_id, timestamp))
                print(f"[VisionTestAPI] Queued image {image_id} for analysis (replaced old)")
            except queue.Empty:
                # This shouldn't happen, but handle gracefully
                self.image_queue.put_nowait((image_data, image_id, timestamp))
        
        # Return status immediately (non-blocking)
        with self.processing_lock:
            is_processing = self.is_processing
            current_id = self.stats['current_processing_id']
        
        return {
            'success': True,
            'image_id': image_id,
            'queued': True,
            'currently_processing': current_id,
            'is_processing': is_processing,
            'stats': self.stats.copy()
        }
    
    def get_latest_result(self):
        """Get the latest analysis result"""
        with self.latest_result_lock:
            return self.latest_result.copy() if self.latest_result else None
    
    def _analyze_image_internal(self, image):
        """Internal method to analyze image with FastVLM (used by worker thread)"""
        if not self.initialized:
            return "Model not initialized"
        
        try:
            start_time = time.time()
            
            # Construct prompt for detailed person and environment analysis
            prompt = "Analyze this webcam image in detail: 1) Describe the person (age range, gender, facial expression, emotions, clothing, accessories). 2) Describe the environment, background, lighting, and setting. 3) Note any changes from typical office/home setup. Be specific and concise."
            
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
                # Use appropriate precision based on device
                if self.device.startswith("cuda"):
                    images_input = image_tensor.unsqueeze(0).half()
                else:
                    images_input = image_tensor.unsqueeze(0)
                
                output_ids = self.model.generate(
                    input_ids,
                    images=images_input,
                    image_sizes=[image.size],
                    do_sample=False,  # Deterministic
                    temperature=0.1,  # Low temperature for consistent analysis
                    max_new_tokens=200,  # Detailed analysis
                    use_cache=True
                )
                
                # Decode response
                outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                
                # Extract only the response part
                if conv.roles[1] in outputs:
                    analysis = outputs.split(conv.roles[1])[-1].strip()
                else:
                    analysis = outputs
                
                processing_time = time.time() - start_time
                
                # Log to console with timestamp
                timestamp = time.strftime("%H:%M:%S")
                print(f"\n{'='*80}")
                print(f"[{timestamp}] REAL-TIME VISION ANALYSIS (Processing: {processing_time:.2f}s)")
                print(f"{'='*80}")
                print(f"{analysis}")
                print(f"{'='*80}\n")
                
                return analysis
                
        except Exception as e:
            print(f"[VisionTestAPI] Analysis error: {e}")
            import traceback
            traceback.print_exc()
            return f"Analysis failed: {str(e)}"

# Global processor instance
processor = FastVLMProcessor()

@app.route('/analyze-vision', methods=['POST'])
def analyze_vision():
    """Endpoint to analyze vision from webcam image"""
    try:
        # Get image data from request
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = data['image']
        if image_data.startswith('data:image'):
            # Remove data URL prefix
            image_data = image_data.split(',')[1]
        
        # Convert to PIL Image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Queue for analysis using C1 strategy (non-blocking)
        result = processor.queue_image_for_analysis(image)
        
        return jsonify({
            'success': result['success'],
            'image_id': result.get('image_id'),
            'queued': result.get('queued', False),
            'currently_processing': result.get('currently_processing'),
            'is_processing': result.get('is_processing', False),
            'stats': result.get('stats', {}),
            'image_size': f"{image.size[0]}x{image.size[1]}"
        })
        
    except Exception as e:
        print(f"[VisionTestAPI] Endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    """Check API status"""
    # Get latest analysis result
    latest_result = processor.get_latest_result()
    
    return jsonify({
        'status': 'running',
        'model_initialized': processor.initialized,
        'device': processor.device if processor.initialized else 'not_initialized',
        'stats': processor.stats,
        'latest_result': latest_result
    })

@app.route('/initialize', methods=['POST'])
def initialize_model():
    """Initialize the FastVLM model"""
    success = processor.initialize()
    return jsonify({'success': success, 'initialized': processor.initialized})

@app.route('/latest-result', methods=['GET'])
def get_latest_result():
    """Get the latest analysis result"""
    result = processor.get_latest_result()
    return jsonify({'latest_result': result})

if __name__ == '__main__':
    print("Starting Vision Test API Server...")
    print("Initializing FastVLM model on startup...")
    processor.initialize()
    print("Ready to accept vision analysis requests!")
    print("Dashboard should be served on a separate server")
    app.run(host='0.0.0.0', port=5003, debug=False)