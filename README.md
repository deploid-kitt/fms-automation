# FMS Automation - Live Movement Analysis System

🔴 Real-time Functional Movement Screen scoring with live feedback

## Features
- **Live Analysis:** WebRTC streaming from webcam with real-time pose detection
- **Real-time Feedback:** Audio coaching cues and visual overlays
- **Upload Mode:** Traditional video file analysis
- **7 FMS Tests:** Deep squat, hurdle step, lunge, shoulder mobility, leg raise, push-up, rotary stability
- **Clinical Reports:** PDF generation with scoring and recommendations
- **Multi-Model AI:** Choose from OpenAI, Anthropic, Google, or local models for different tasks

## AI-Enhanced Features
- **Smart Coaching:** LLM-generated coaching cues that adapt to your movement
- **Enhanced Reports:** AI-written narrative summaries and personalized recommendations
- **Movement Analysis:** Detailed biomechanical insights powered by advanced models
- **Model Selection:** Choose fast models for real-time or premium models for accuracy

Supports: GPT-4o, Claude Sonnet/Opus, Gemini, Llama (local via Ollama)

See [Multi-Model Support](docs/MULTI_MODEL_SUPPORT.md) for configuration details.

## Architecture
- **Backend:** FastAPI + MediaPipe BlazePose + WebSockets + Multi-LLM Integration
- **Frontend:** React + Canvas pose visualization + Web Speech API
- **Deployment:** Docker Compose + Caddy + systemd

## Quick Start

```bash
# Copy environment file
cp .env.example .env

# Add your API keys (optional - for AI features)
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...

# Start services
docker-compose up -d
```

## Live at
https://fms.kitt.deploid.io

Built by KITT on deploid.io

