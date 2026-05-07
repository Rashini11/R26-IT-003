# R26-IT-003

## AI-Driven Multi-Sensor Decision Support System for Autonomous Ship Surveillance and Condition Assessment

This project is a research based maritime decision support system developed to improve ship surveillance, sea state monitoring, vessel detection, radar-based object classification, and hull condition assessment using artificial intelligence and computer vision techniques.

The overall system is designed to support safer maritime navigation, improved situational awareness, and intelligent decision-making for maritime operators.


## Project Components

The system consists of four main research components:

1. **AI-Based Sea State Classification using Onboard Ship Cameras**
2. **Radar-Based Object Detection and Classification**
3. **Underwater Hull Inspection and Defect Detection**
4. **Drone-Assisted Vessel Detection and Classification**

---
## Technologies Used

### Machine Learning
- Python
- PyTorch
- Torchvision
- MobileNetV2
- ResNet18
- Scikit-learn
- Pillow
- fastapi
-uvicorn
-tensorflow
-ultralytics


### Backend
- FastAPI
- Uvicorn
- Python Multipart

### Frontend
- React
- Vite
- Axios


## Current System Workflow

User uploads sea image
        ↓
Frontend sends image to backend API
        ↓
Backend preprocesses image
        ↓
Trained model predicts sea state
        ↓
Prediction result and confidence scores are returned
        ↓
Frontend displays final prediction
