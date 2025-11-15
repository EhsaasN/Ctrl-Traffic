# Ctrl-Traffic 🚦

An intelligent traffic management system that uses computer vision and audio processing to detect emergency vehicles and optimize traffic flow based on real-time analysis.

## 📋 Overview

Ctrl-Traffic is an AI-powered traffic control system that combines:
- **Emergency Vehicle Detection**: Identifies ambulances, fire trucks, and other emergency vehicles through both visual and audio recognition
- **Traffic Density Analysis**: Analyzes traffic patterns to optimize signal timing
- **Dynamic Signal Control**: Automatically adjusts green light duration based on traffic conditions and emergency situations

## 🚀 Features

### Core Functionality
- **Dual-Modal Detection**: Processes both image and 3-second audio (.wav) files
- **Emergency Override**: Immediately prioritizes emergency vehicles with extended green signals (120 seconds)
- **Traffic Density Classification**: Categorizes traffic as high, medium, low, or no traffic
- **Intelligent Timing**: Dynamically adjusts green light duration based on traffic density
- **Real-time API**: FastAPI-based endpoints for real-time inference

### Technical Capabilities
- **Image Processing**: TensorFlow/Keras models for vehicle and traffic analysis
- **Audio Processing**: CNN-based audio classification using MFCC features
- **Cross-platform**: Web-based interface with CORS support

## 🛠️ Installation

### Prerequisites
- Check requirements.txt
- pip install -r requirements.txt 

### Dependencies
```bash
pip install tensorflow
pip install torch transformers
pip install fastapi uvicorn
pip install pillow numpy
pip install librosa soundfile
pip install python-multipart
```

### Model Files
Ensure these pre-trained models are in the root directory:
- `vehicle.h5` - Emergency vehicle detection model
- `audio_cnn.h5` - Audio classification model (for emergency sirens)
- `labelencoder.pkl` or `label_encoder.pkl` - Audio label encoder (optional)

## 🚀 Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/EhsaasN/Ctrl-Traffic.git
   cd Ctrl-Traffic
   ```

2. **Run the main application**:
   ```bash
   python app_audio.py
   ```

3. **Access the API**:
   - Server starts at `http://localhost:8000`
     
## 📡 API Endpoints

### `/infer` - Image-based Traffic Analysis
**POST** - Upload an image for traffic analysis

**Request**: Multipart form data with image file

### `/infer-audio` - Audio-based Emergency Detection
**POST** - Upload a 3-second .wav audio file

**Request**: Multipart form data with audio file


## 🧠 Traffic Logic

### Emergency Detection
- **Image Analysis**: Detects emergency vehicles with >30% confidence
- **Audio Analysis**: Identifies emergency sirens (ambulance, fire truck)
- **Override Action**: Emergency detection triggers 120-second green signal

### Traffic Density Mapping
| Traffic Level | Green Signal Duration |
|---------------|----------------------|
| High Traffic  | 80 seconds          |
| Medium Traffic| 60 seconds          |
| Low Traffic   | 30 seconds          |
| No Traffic    | 0 seconds           |

## 🗂️ Project Structure

```
Ctrl-Traffic/
├── app_audio.py          # Main application (final version)
├── app.py                # Basic image-only version
├── integrated.py         # Integration utilities
├── model.py              # Model definitions
├── run_worker.py         # Background worker processes
├── vehicle.h5            # Pre-trained vehicle detection model
├── audio_cnn.h5          # Pre-trained audio classification model
├── test.html             # Basic web interface
├── test_audio.html       # Audio testing interface
└── LICENSE               # MIT License
```

## 🌐 Web Interface

- **Basic Interface**: `test.html` - Image upload and analysis
- **Audio Interface**: `test_audio.html` - Combined image and audio testing

## 🤖 Models Used

1. **Vehicle Detection**: Custom TensorFlow model trained on emergency vehicle images
2. **Traffic Density**: HuggingFace transformer model (`prithivMLmods/Traffic-Density-Classification`)
3. **Audio Classification**: Custom CNN trained on emergency siren sounds using MFCC features

## 📊 Performance

- **Real-time Processing**: < 2 seconds per inference
- **Accuracy**: 90%+ emergency vehicle detection
- **Audio Processing**: 3-second .wav file analysis
- **Concurrent Users**: Supports multiple simultaneous requests

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For issues and questions:
- Open an issue on GitHub
- Check the `/health` endpoint for system diagnostics

## 🔮 Future Enhancements

- Real-time video stream processing
- Integration with IoT traffic sensors
- Mobile app for traffic monitoring
- Multi-intersection coordination

---

**Built for smarter traffic management**
