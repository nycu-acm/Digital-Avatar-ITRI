import cv2
import time
import threading
import os
import sys
from PIL import Image
import torch
import numpy as np
from logger import logger
import hashlib

# Add FastVLM to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ml-fastvlm'))

from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


class VisionProcessor:
    """FastVLM-powered visual perception for avatar system"""
    
    def __init__(self):
        self.enabled = False
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.device = None
        self.last_analysis = ""
        self.last_frame_hash = ""
        self.analysis_interval = 5  # Analyze every 5 seconds
        self.last_analysis_time = 0
        self.camera = None
        self.processing_thread = None
        self.running = False
        
    def initialize_model(self):
        """Initialize FastVLM model for visual analysis"""
        try:
            logger.info("[VisionProcessor] Initializing FastVLM model...")
            
            # Auto-detect device
            if torch.cuda.is_available():
                self.device = "cuda:0"
            else:
                self.device = "cpu"
            
            # Load FastVLM model
            disable_torch_init()
            model_path = os.path.join(os.path.dirname(__file__), 'ml-fastvlm', 'llava-fastvithd_0.5b_stage3')
            model_name = get_model_name_from_path(model_path)
            
            self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
                model_path, None, model_name, device=self.device
            )
            
            logger.info(f"[VisionProcessor] FastVLM model loaded on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"[VisionProcessor] Failed to initialize model: {e}")
            return False
    
    def start_camera(self):
        """Start webcam capture"""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                logger.error("[VisionProcessor] Failed to open camera")
                return False
            
            # Set camera resolution
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            logger.info("[VisionProcessor] Camera initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"[VisionProcessor] Camera initialization failed: {e}")
            return False
    
    def capture_frame(self):
        """Capture single frame from camera"""
        if not self.camera or not self.camera.isOpened():
            return None
        
        ret, frame = self.camera.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)
        return None
    
    def get_frame_hash(self, frame):
        """Generate hash of frame to detect scene changes"""
        if frame is None:
            return ""
        
        # Resize for hash comparison (faster)
        small_frame = frame.resize((64, 64))
        frame_bytes = small_frame.tobytes()
        return hashlib.md5(frame_bytes).hexdigest()
    
    def analyze_frame(self, frame):
        """Analyze frame with FastVLM for person and environment detection"""
        if not self.model or frame is None:
            return ""
        
        try:
            # Construct prompt for person and environment analysis
            prompt = "Analyze this image and describe: 1) The person's apparent age range, gender, facial expression/emotion, and clothing. 2) The environment and setting around them. Be concise and focus on key details that would help personalize a conversation."
            
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
            image_tensor = process_images([frame], self.image_processor, self.model.config)[0]
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
                    image_sizes=[frame.size],
                    do_sample=False,  # Deterministic for consistency
                    temperature=0.2,
                    max_new_tokens=128,  # Concise analysis
                    use_cache=True
                )
                
                # Decode response
                outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                
                # Extract only the response part (after the prompt)
                if conv.roles[1] in outputs:
                    analysis = outputs.split(conv.roles[1])[-1].strip()
                else:
                    analysis = outputs
                
                logger.info(f"[VisionProcessor] Analysis: {analysis[:100]}...")
                return analysis
                
        except Exception as e:
            logger.error(f"[VisionProcessor] Frame analysis failed: {e}")
            return ""
    
    def processing_loop(self):
        """Main processing loop for continuous visual analysis"""
        logger.info("[VisionProcessor] Starting processing loop")
        
        while self.running:
            try:
                current_time = time.time()
                
                # Check if it's time for analysis
                if current_time - self.last_analysis_time < self.analysis_interval:
                    time.sleep(0.5)
                    continue
                
                # Capture frame
                frame = self.capture_frame()
                if frame is None:
                    time.sleep(0.5)
                    continue
                
                # Check if scene changed significantly
                frame_hash = self.get_frame_hash(frame)
                if frame_hash == self.last_frame_hash:
                    # Scene hasn't changed, skip analysis
                    time.sleep(0.5)
                    continue
                
                # Analyze frame
                analysis = self.analyze_frame(frame)
                if analysis:
                    self.last_analysis = analysis
                    self.last_frame_hash = frame_hash
                    self.last_analysis_time = current_time
                    
                    logger.info(f"[VisionProcessor] Updated visual context: {analysis[:50]}...")
                
            except Exception as e:
                logger.error(f"[VisionProcessor] Processing loop error: {e}")
                time.sleep(1)
    
    def enable(self):
        """Enable visual perception"""
        if self.enabled:
            return True
        
        logger.info("[VisionProcessor] Enabling visual perception...")
        
        # Initialize model if not done
        if not self.model:
            if not self.initialize_model():
                return False
        
        # Start camera if not done
        if not self.camera:
            if not self.start_camera():
                return False
        
        # Start processing thread
        self.running = True
        self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
        self.processing_thread.start()
        
        self.enabled = True
        logger.info("[VisionProcessor] Visual perception enabled successfully")
        return True
    
    def disable(self):
        """Disable visual perception"""
        if not self.enabled:
            return
        
        logger.info("[VisionProcessor] Disabling visual perception...")
        
        # Stop processing
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
        
        # Release camera
        if self.camera:
            self.camera.release()
            self.camera = None
        
        self.enabled = False
        self.last_analysis = ""
        logger.info("[VisionProcessor] Visual perception disabled")
    
    def get_visual_context(self):
        """Get current visual context for LLM"""
        if not self.enabled or not self.last_analysis:
            return ""
        
        return f"Visual Context: {self.last_analysis}"
    
    def is_enabled(self):
        """Check if visual perception is enabled"""
        return self.enabled


# Global vision processor instance
vision_processor = VisionProcessor()