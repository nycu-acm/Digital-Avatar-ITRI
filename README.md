# Avatar ITRI - Real-time Digital Avatar System

A real-time digital avatar system supporting both **Wav2Lip** and **MuseTalk** models for lip-sync video generation with live audio processing and WebRTC streaming.

![Architecture](./assets/architecture.png)

## 🎬 Demo Videos

- **[Computer Demo](https://drive.google.com/file/d/1WrXx5Y-3J9XJqXYNcAm7M0gXu_baWb_j/view?usp=sharing)** - Desktop browser experience
- **[Mobile Demo](https://drive.google.com/file/d/192REA6hSZonl06c3n_UwN0B8jWGQWO3u/view?usp=sharing)** - Mobile device experience

## 🌟 Features

- **Real-time Avatar Generation**: Live lip-sync video generation at 25 FPS
- **Dual Model Support**: 
  - **Wav2Lip**: High-quality 256x256 lip synchronization
  - **MuseTalk**: Advanced diffusion-based avatar with better facial expressions
- **Text-to-Speech**: EdgeTTS with multiple voice options
- **WebRTC Streaming**: Low-latency browser-based video streaming
- **Multi-session Support**: Concurrent client connections
- **Custom Avatar Creation**: Generate your own avatars from videos
- **Recording Capabilities**: Save avatar sessions as MP4 videos

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Web Client    │◄──►│   WebRTC     │◄──►│   Avatar App    │
│  (Browser)      │    │  Streaming   │    │   (Python)      │
└─────────────────┘    └──────────────┘    └─────────────────┘
                                                    │
                       ┌────────────────────────────┼────────────────────────┐
                       │                            │                        │
                ┌──────▼──────┐            ┌───────▼───────┐        ┌───────▼───────┐
                │     TTS     │            │      ASR      │        │   Inference   │
                │  (EdgeTTS/  │            │   (Audio      │        │  (Wav2Lip/    │
                │   Others)   │            │  Processing)  │        │  MuseTalk)    │
                └─────────────┘            └───────────────┘        └───────────────┘
```

## 📋 Requirements

- **OS**: Linux (Ubuntu 20.04+ recommended)
- **GPU**: NVIDIA GPU with CUDA support (>= 8GB VRAM recommended)
- **Python**: 3.8-3.10
- **CUDA**: 12.4 (or compatible)
- **FFmpeg**: System package required

## 🚀 Installation

### 1. Clone Repository
```bash
git clone https://github.com/your-username/Avatar-ITRI.git
cd Avatar-ITRI
```

### 2. Download Models and Data
Download the `Avatar.zip` file containing the required models and avatar data. Extract it in the project root directory:

```bash
# Extract Avatar.zip to get these folders:
unzip Avatar.zip
# This should create:
# ./models/     - Pre-trained models (wav2lip.pth, MuseTalk models, etc.)
# ./data/       - Avatar data (wav2lip256_avatar, musetalk_avatar)
```

**Required structure:**
```
Avatar-ITRI/
├── models/
│   ├── wav2lip.pth
│   ├── musetalkV15/
│   ├── sd-vae/
│   ├── whisper/
│   └── ...
├── data/
│   └── avatars/
│       ├── wav2lip256_avatar/
│       └── musetalk_avatar/
└── app.py
```

### 3. Create Conda Environment
```bash
# Create new conda environment
conda create -n avatar-itri python=3.10 -y
conda activate avatar-itri

# Install PyTorch with CUDA support
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia

# Install Python dependencies
pip install -r requirements.txt
```

### 4. System Dependencies
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg -y

# Verify CUDA installation
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 🎭 Usage

### Running with Different Models

#### Wav2Lip (Recommended)
```bash
python app.py --transport webrtc --model wav2lip --avatar_id wav2lip256_avatar --REF_FILE en-US-AriaNeural --listenport 8010
```

#### MuseTalk (Advanced)
```bash
python app.py --transport webrtc --model musetalk --avatar_id musetalk_avatar --REF_FILE en-US-AriaNeural --listenport 8010
```

### Accessing the Interface
Open your browser and navigate to:
- **Main Interface**: `http://localhost:8010/webrtcapi.html`
- **Dashboard**: `http://localhost:8010/dashboard.html` (recommended)

### Command Line Parameters
- `--model`: Choose between `wav2lip` (recommended) or `musetalk`
- `--avatar_id`: Avatar to use (e.g., `wav2lip256_avatar`, `musetalk_avatar`)
- `--REF_FILE`: TTS voice (see available voices below)
- `--listenport`: Server port (default: 8010)
- `--batch_size`: Inference batch size (default: 16)
- `--tts`: TTS engine (default: `edgetts`)

### Available TTS Voices
- **English**: `en-US-AriaNeural` (Female, neutral)
- **Chinese**: `zh-CN-XiaoxiaoNeural` (Female, Chinese)
- **Neutral**: `en-US-JennyNeural` (Female, friendly)

## 🛠️ Creating Custom Avatars

### For Wav2Lip
```bash
cd wav2lip
python genavatar.py --video_path /path/to/your/video.mp4 --img_size 256 --avatar_id your_avatar_name
```

### For MuseTalk
```bash
python genavatar_musetalk.py --file /path/to/your/video.mp4 --avatar_id your_musetalk_avatar
```

**Requirements for source videos:**
- **Duration**: 10-60 seconds
- **Quality**: HD (1080p recommended)
- **Content**: Clear frontal face view
- **Lighting**: Good, consistent lighting
- **Background**: Static or minimal movement

## 🎮 API Endpoints

### WebRTC Connection
- `POST /offer` - Establish WebRTC connection
- `POST /human` - Send text for avatar to speak
- `POST /interrupt_talk` - Stop current speech
- `POST /is_speaking` - Check if avatar is speaking

### Audio Processing
- `POST /humanaudio` - Upload audio file
- `POST /process_voice` - Process voice with Whisper ASR
- `POST /process_voice_blob` - Process audio blob

### Controls
- `POST /set_audiotype` - Switch to custom video
- `POST /record` - Start/stop recording

## ⚙️ Configuration

### TTS Configuration
The system uses EdgeTTS by default with the following voice options:

```bash
# English voice (default)
--REF_FILE en-US-AriaNeural

# Chinese voice
--REF_FILE zh-CN-XiaoxiaoNeural

# Friendly English voice
--REF_FILE en-US-JennyNeural
```

### Custom Video Integration
Create `custom_config.json` for custom video sequences:
```json
[
    {
        "audiotype": 1,
        "imgpath": "./custom_videos/video1/",
        "audiopath": "./custom_videos/audio1.wav"
    }
]
```

## 🔧 Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Reduce batch size
python app.py --batch_size 8 --model musetalk
```

**WebRTC Connection Issues**
- Check firewall settings
- Ensure port 8010 is open
- Try different STUN servers in `app.py`

**Audio Issues**
```bash
# Install audio dependencies
sudo apt install portaudio19-dev python3-pyaudio
pip install pyaudio
```

**Model Loading Errors**
- Verify `Avatar.zip` was extracted correctly
- Check file permissions: `chmod -R 755 models/ data/`
- Ensure sufficient disk space (>10GB)

### Performance Optimization

**For better performance:**
1. Use NVIDIA GPU with >8GB VRAM
2. Increase `--batch_size` (16-32)
3. Use SSD storage for models
4. Close unnecessary applications

**For lower-end hardware:**
1. Reduce `--batch_size` to 4-8
2. Use Wav2Lip instead of MuseTalk
3. Lower video resolution in WebRTC settings

## 📊 System Requirements by Model

| Model     | VRAM    | Processing | Quality | Speed | Recommended |
|-----------|---------|------------|---------|-------|-------------|
| Wav2Lip   | 4GB     | Light      | Good    | Fast  | ✅ Yes      |
| MuseTalk  | 8GB+    | Heavy      | Excellent| Medium| Advanced    |

## 📚 Technical Details

### Processing Pipeline
1. **Text Input** → TTS Engine → Audio Stream (16kHz, 20ms chunks)
2. **Audio Stream** → ASR → Audio Features (Mel/Whisper)
3. **Features + Avatar** → AI Model → Generated Frames
4. **Generated Frames** → Face Blending → Final Video
5. **Video + Audio** → WebRTC → Browser Client

### Model Specifications
- **Wav2Lip**: 256x256 resolution, mel-spectrogram features
- **MuseTalk**: Diffusion-based, Whisper features, advanced blending
- **Frame Rate**: 25 FPS video, 50 FPS audio processing
- **Latency**: <200ms total pipeline latency

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Acknowledgments

- **Wav2Lip**: [Original Paper](https://arxiv.org/abs/2008.10010)
- **MuseTalk**: [Original Repository](https://github.com/TMElyralab/MuseTalk)
- **WebRTC**: aiortc library for Python WebRTC implementation

## 💬 Support

For issues and questions:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Search existing [Issues](https://github.com/your-username/Avatar-ITRI/issues)
3. Create a new issue with detailed information

---

**Made with ❤️ by ITRI Team**