# 🤖 AI Depth Estimation Web App

Real-time depth estimation using AI models, accessible from any device with a web browser.

![AI Depth Vision](https://img.shields.io/badge/AI-Depth%20Vision-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.7+-green?style=for-the-badge)
![Flask](https://img.shields.io/badge/Flask-Web%20App-red?style=for-the-badge)

## ✨ Features

- 🤖 **AI-powered depth estimation** using Depth-Anything-V2-Small model
- 🔥 **Heat map visualization** with INFERNO colormap
- 📱 **Mobile-friendly** responsive design
- ⚡ **CUDA acceleration** (if available)
- 🌐 **Cross-platform** - works on any device with a browser
- 🎨 **Beautiful modern UI** with gradient backgrounds

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/ARIA_vision.git
cd ARIA_vision
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Web App
```bash
python web_app.py
```

### 4. Access from Any Device
- **Computer**: Open `http://localhost:5000`
- **Phone/Tablet**: Open `http://YOUR_IP:5000` (replace YOUR_IP with your computer's IP address)

## 📁 Project Structure

```
ARIA_vision/
├── web_app.py              # Main Flask application
├── templates/
│   └── index.html          # Web interface
├── phone_camera.html       # Direct phone camera access
├── requirements.txt        # Dependencies
├── README.md              # This file
└── .gitignore             # Git ignore rules
```

## 🎯 Usage

1. **Start the server** by running `python web_app.py`
2. **Open the web interface** in your browser
3. **Click "🎥 Start Camera"** to begin depth estimation
4. **View real-time results** with original feed and AI depth map

## 🌡️ Heat Map Legend

- **Far (Black/Purple)**: Distant objects
- **Medium (Red/Orange)**: Medium distance objects  
- **Near (Yellow/White)**: Close objects

## 📋 Requirements

- Python 3.7+
- Webcam or camera device
- CUDA-compatible GPU (optional, for better performance)

## 📱 Alternative: Direct Phone Camera

For direct phone camera access without a server:
1. Copy `phone_camera.html` to your phone
2. Open it directly in your phone's browser
3. Grant camera permissions when prompted

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) for the depth estimation model
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [OpenCV](https://opencv.org/) for computer vision utilities