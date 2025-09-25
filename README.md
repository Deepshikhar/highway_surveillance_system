# Highway Surveillance System

A comprehensive real-time highway surveillance system that uses YOLOv11 and DeepSORT for vehicle detection, tracking, counting, and speed estimation. This application provides an interactive dashboard with live video feed and analytics.

## Features

### 🚗 Vehicle Detection & Tracking
- **Real-time Detection**: Uses YOLOv11 for accurate vehicle detection
- **Multi-Object Tracking**: Implements DeepSORT algorithm for persistent tracking
- **Vehicle Classification**: Identifies bicycles, cars, motorcycles, buses, and trucks

### 📊 Analytics & Metrics
- **Vehicle Counting**: Tracks vehicles entering and exiting the surveillance area
- **Speed Estimation**: Calculates real-time vehicle speeds in km/h
- **Live Statistics**: Displays comprehensive analytics with interactive charts
- **FPS Monitoring**: Real-time performance metrics

### 🎯 Advanced Features
- **Region-based Speed Calculation**: Uses perspective transformation for accurate speed estimation
- **Dual Surveillance Zones**: Monitors both entry and exit points simultaneously
- **Interactive Dashboard**: Streamlit-based web interface with real-time updates

## Technology Stack

- **Computer Vision**: OpenCV, YOLOv11, DeepSORT
- **Machine Learning**: PyTorch, Ultralytics YOLO
- **Web Framework**: Streamlit
- **Visualization**: Plotly for interactive charts
- **Processing**: NumPy for mathematical operations

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for better performance)

### Step-by-Step Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd highway_surveillance_system
```

2. **Create virtual environment (recommended)**
```bash
python -m venv surveillance-env
source surveillance-env/bin/activate  # On Windows: surveillance-env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Required Packages

The system requires the following packages:
```python
streamlit==1.28.0
opencv-python==4.8.1.78
torch>=1.7.0
torchvision>=0.8.0
numpy>=1.21.0
ultralytics==8.0.0
deep-sort-realtime==1.3.2
plotly==5.15.0
absl-py==1.4.0
```

## Usage

### Running the Application

1. **Start the Streamlit application**
```bash
streamlit run app.py
```

2. **Access the dashboard**
   - Open your web browser
   - Navigate to `http://localhost:8501`

3. **Monitor the surveillance**
   - The system will automatically start processing the video feed
   - View real-time analytics on the dashboard

### Video Input Configuration

The system supports multiple video sources:

- **Local video files**: Update `video_input` variable with file path


Example configurations:
```python
# For local video file
video_input = "highway.mp4"
```

## Configuration

### Surveillance Regions

The system uses predefined regions for optimal performance:

**Entry/Exit Lines:**
- Entry line coordinates: (160, 558) to (708, 558)
- Exit line coordinates: (1155, 558) to (1718, 558)

**Speed Estimation Regions:**
- Left region: Perspective transformation for speed calculation
- Right region: Secondary speed monitoring zone

### Vehicle Classes

The system tracks the following vehicle types:
- **Class 1**: Bicycle
- **Class 2**: Car
- **Class 3**: Motorcycle
- **Class 5**: Bus
- **Class 7**: Truck

### Performance Settings

- **Confidence Threshold**: 0.5 (adjustable)
- **Max Tracking Age**: 5 frames
- **FPS Calculation**: Real-time performance monitoring

## Dashboard Overview

### Main Components

1. **Video Feed Section**
   - Real-time processed video with bounding boxes
   - Vehicle IDs and classifications
   - Speed estimates overlay

2. **Metrics Panel**
   - FPS: Current frames per second
   - Vehicle Enter: Count of entering vehicles
   - Vehicle Exit: Count of exiting vehicles
   - Average Speed: Mean speed of detected vehicles

3. **Statistics Section**
   - Interactive bar chart showing vehicle distribution
   - Percentage breakdown by vehicle type
   - Real-time updates

### Customization Options

**Styling:**
- CSS customization for dashboard appearance
- Color schemes for different vehicle types
- Layout adjustments for various screen sizes

**Detection Parameters:**
- Adjust confidence thresholds
- Modify tracking parameters
- Customize region boundaries

## Technical Implementation

### Architecture Overview

```
Video Input → YOLOv11 Detection → DeepSORT Tracking → Analytics → Dashboard
```

### Key Algorithms

1. **YOLOv11 Detection**
   - Real-time object detection
   - COCO dataset pre-trained weights
   - Optimized for vehicle detection

2. **DeepSORT Tracking**
   - Multi-object tracking with re-identification
   - Kalman filter for motion prediction
   - Appearance features for tracking consistency

3. **Speed Estimation**
   - Perspective transformation for accurate distance calculation
   - Time-based speed computation
   - Dual-region validation

### Performance Optimization

- GPU acceleration support
- Efficient memory management
- Batch processing capabilities
- Real-time streaming optimization

## Troubleshooting

### Common Issues

1. **Low FPS**
   - Reduce video resolution
   - Enable GPU acceleration
   - Close other resource-intensive applications

2. **Detection Accuracy Issues**
   - Adjust confidence thresholds
   - Ensure proper lighting conditions
   - Verify camera angle and positioning

### Performance Tips

- Use CUDA-enabled GPU for better performance
- Adjust video resolution based on hardware capabilities
- Monitor system resources during operation
- Use optimized YOLO model versions (s, m, l, x) based on requirements

## File Structure

```
highway_surveillance_system/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── models                 # YOLOv11 model weights 
|     └── yolo11s.pt           
├── videos                 # Sample video file
|      └──highway.mp4      
└── README.md              # Project documentation
```

## Model Weights

The application uses YOLOv11 model weights. Available options:
- `yolo11s.pt`: Small (fastest, lower accuracy)
- `yolo11m.pt`: Medium (balanced)
- `yolo11l.pt`: Large (higher accuracy)
- `yolo11x.pt`: X-Large (highest accuracy, slowest)

Download pre-trained weights from Ultralytics or train custom models for specific use cases.

## Contributing

We welcome contributions to enhance the highway surveillance system:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

### Areas for Improvement

- Additional vehicle classes
- Enhanced speed estimation algorithms
- Improved tracking accuracy
- Additional visualization options
- Export functionality for analytics

## License

This project is intended for educational and research purposes. Please ensure compliance with local regulations and privacy laws when deploying surveillance systems.

## Acknowledgments

- **YOLOv11**: Ultralytics for the object detection model
- **DeepSORT**: Object tracking algorithm implementation
- **OpenCV**: Computer vision library
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualization library

## Support

For technical support or questions:
- Create an issue in the GitHub repository
- Check existing documentation and troubleshooting guides
- Review implementation details in the code comments

---

**Note**: This system is designed for highway monitoring and traffic analysis. Always ensure compliance with privacy regulations and obtain necessary permissions when deploying surveillance systems in public or private areas.