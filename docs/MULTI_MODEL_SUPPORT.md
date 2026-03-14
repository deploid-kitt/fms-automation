# Multi-Model AI Support

The FMS Automation system supports multiple AI model providers for enhanced analysis, coaching, and report generation.

## Overview

The multi-model architecture allows you to:
- Choose different models for different tasks (speed vs. quality tradeoffs)
- Use local models for privacy-sensitive deployments
- Gracefully fall back when a model is unavailable
- Cache responses for improved performance

## Supported Providers

### OpenAI
- **GPT-4o** (Premium) - Best for detailed analysis and reports
- **GPT-4o Mini** (Fast) - Great for real-time coaching
- **GPT-3.5 Turbo** (Fast) - Fastest, good fallback option

### Anthropic
- **Claude Opus 4** (Premium) - Most capable for complex analysis
- **Claude Sonnet 4** (Standard) - Balanced performance and quality
- **Claude 3.5 Haiku** (Fast) - Best for real-time feedback

### Google
- **Gemini 2.0 Flash** (Fast) - Fast multimodal model
- **Gemini 1.5 Pro** (Premium) - Best for detailed analysis

### Ollama (Local)
- **Llama 3.2** (Fast) - Local model for privacy
- **Llama 3.1 70B** (Standard) - Powerful local option
- **Mistral** (Fast) - Very fast local model

## Configuration

### Environment Variables

```bash
# Enable/disable LLM features
FMS_ENABLE_LLM_REPORTS=true
FMS_ENABLE_LLM_COACHING=true
FMS_ENABLE_LLM_ANALYSIS=true

# API Keys (set the ones you want to use)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

# Local models
FMS_OLLAMA_URL=http://localhost:11434

# Model preferences
FMS_LLM_REALTIME_MODEL=gpt-4o-mini
FMS_LLM_ANALYSIS_MODEL=claude-sonnet-4-20250514
FMS_LLM_REPORT_MODEL=claude-sonnet-4-20250514
FMS_LLM_FALLBACK_MODEL=gpt-3.5-turbo

# Performance
FMS_LLM_CACHE_ENABLED=true
FMS_LLM_CACHE_TTL=3600
FMS_LLM_REALTIME_TIMEOUT=3.0
```

### Runtime Configuration

Model preferences can be changed at runtime via the API or UI:

**API Endpoint:** `PUT /api/llm/preferences`

```json
{
  "realtime_feedback": "claude-3-5-haiku-20241022",
  "movement_analysis": "gpt-4o",
  "report_generation": "claude-opus-4-20250514",
  "fallback": "gpt-3.5-turbo",
  "enable_llm": true,
  "enable_caching": true,
  "cache_ttl_seconds": 3600
}
```

## Capabilities

Models are tagged with capabilities to help with automatic selection:

| Capability | Description | Recommended Tier |
|------------|-------------|------------------|
| `realtime_feedback` | Live coaching cues during exercise | Fast |
| `movement_analysis` | Detailed biomechanical analysis | Standard/Premium |
| `report_generation` | Comprehensive assessment reports | Premium |
| `exercise_classification` | Identifying exercises from movement | Fast |
| `coaching_cues` | Natural coaching language | Fast |

## Integration Points

### 1. Real-time Coaching (Live Analysis)

During live analysis, the system can generate AI-enhanced coaching cues:

- Uses fast models (GPT-4o Mini, Claude Haiku) with 3-second timeout
- Cues are generated asynchronously to not block frame processing
- Falls back to rule-based cues if LLM is unavailable or slow
- Indicated in UI with "AI" badge

### 2. Movement Analysis

Detailed analysis of movement patterns:

- Uses capable models for thorough biomechanical analysis
- Identifies compensation patterns and root causes
- Provides clinical insights

### 3. Report Generation

Enhanced narrative reports:

- Executive summary of assessment
- Detailed test-by-test analysis
- Prioritized recommendations
- Professional language suitable for healthcare/fitness settings

### 4. Score Verification

LLM can verify rule-based scores:

- Catches edge cases the rule-based system might miss
- Provides reasoning for score adjustments
- Increases confidence in results

## Performance Optimization

### Caching

Responses are cached based on prompt hash:
- Default TTL: 1 hour
- Similar queries return cached results instantly
- Cache can be cleared via API: `POST /api/llm/cache/clear`

### Async Processing

- LLM calls are async and non-blocking
- Real-time analysis continues even if LLM is slow
- Graceful degradation when models are unavailable

### Rate Limiting

Built-in rate limiting per provider:
- Tracks requests per minute
- Automatically waits when approaching limits
- Prevents API errors from quota exhaustion

## API Reference

### List Models
```
GET /api/llm/models
GET /api/llm/models?capability=realtime_feedback
GET /api/llm/models?tier=fast
```

### Check Provider Status
```
GET /api/llm/providers
GET /api/llm/providers/{provider}/health
```

### Manage Preferences
```
GET /api/llm/preferences
PUT /api/llm/preferences
```

### Statistics
```
GET /api/llm/stats
POST /api/llm/cache/clear
```

### Test Completion
```
POST /api/llm/test
{
  "prompt": "Test prompt",
  "model": "gpt-4o-mini",
  "capability": "realtime_feedback"
}
```

## UI Components

### Model Settings Modal

Access via the gear icon in the Live Analysis interface:
- Configure models for each capability
- View provider status
- Monitor usage statistics
- Clear cache

### Model Quick Select

Compact model selector for embedding in other views:
- Shows current model
- Quick switching between models
- Tier indicators

## Troubleshooting

### Model Not Available

If a model is not available:
1. Check API key is set in environment
2. Verify provider health: `GET /api/llm/providers`
3. Check model ID is correct

### Slow Responses

If LLM responses are slow:
1. Use a faster model tier
2. Enable caching
3. Consider local models (Ollama)

### No AI Features

If AI features aren't working:
1. Check `FMS_ENABLE_LLM_*` settings
2. Verify at least one provider is configured
3. Check logs for errors

## Local Model Setup (Ollama)

For privacy-sensitive deployments:

1. Install Ollama: https://ollama.ai
2. Pull models:
   ```bash
   ollama pull llama3.2
   ollama pull mistral
   ```
3. Start Ollama server (usually automatic)
4. Set `FMS_OLLAMA_URL=http://localhost:11434`

## Cost Optimization

Tips for reducing API costs:

1. Use caching (enabled by default)
2. Use fast/cheap models for real-time feedback
3. Reserve premium models for reports only
4. Consider local models for development
5. Monitor usage via `/api/llm/stats`
