# FMS Automation - Live Movement Analysis System

🔴 Real-time Functional Movement Screen scoring with live feedback

## Features
- **Live Analysis:** WebRTC streaming from webcam with real-time pose detection
- **Real-time Feedback:** Audio coaching cues and visual overlays
- **Upload Mode:** Traditional video file analysis
- **7 FMS Tests:** Deep squat, hurdle step, lunge, shoulder mobility, leg raise, push-up, rotary stability
- **Clinical Reports:** PDF generation with scoring and recommendations

## Architecture
- **Backend:** FastAPI + MediaPipe BlazePose + WebSockets
- **Frontend:** React + Canvas pose visualization + Web Speech API
- **Deployment:** Docker Compose + Caddy + systemd

## Live at
https://fms.kitt.deploid.io

Built by KITT on deploid.io

