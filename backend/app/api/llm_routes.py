"""API routes for LLM management and configuration."""
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field

from app.llm.manager import get_llm_manager, ModelPreferences
from app.llm.base import (
    LLMProvider,
    ModelCapability,
    AVAILABLE_MODELS,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/llm", tags=["llm"])


# --- Request/Response Models ---

class ModelInfo(BaseModel):
    """Information about an available model."""
    id: str
    provider: str
    display_name: str
    capabilities: list[str]
    tier: str
    description: str
    avg_latency_ms: int
    supports_streaming: bool
    supports_vision: bool


class ProviderStatus(BaseModel):
    """Status of an LLM provider."""
    provider: str
    available: bool
    healthy: bool


class PreferencesUpdate(BaseModel):
    """Request to update model preferences."""
    realtime_feedback: Optional[str] = None
    movement_analysis: Optional[str] = None
    report_generation: Optional[str] = None
    exercise_classification: Optional[str] = None
    coaching_cues: Optional[str] = None
    fallback: Optional[str] = None
    enable_llm: Optional[bool] = None
    enable_caching: Optional[bool] = None
    cache_ttl_seconds: Optional[int] = Field(None, ge=60, le=86400)


class PreferencesResponse(BaseModel):
    """Current model preferences."""
    realtime_feedback: str
    movement_analysis: str
    report_generation: str
    exercise_classification: str
    coaching_cues: str
    fallback: str
    enable_llm: bool
    enable_caching: bool
    cache_ttl_seconds: int


class LLMStatsResponse(BaseModel):
    """LLM usage statistics."""
    request_count: int
    cache_hits: int
    cache_rate: float
    avg_latency_ms: float
    cache_size: int
    available_providers: list[str]


# --- Endpoints ---

@router.get("/models")
async def list_models(
    capability: Optional[str] = None,
    tier: Optional[str] = None,
) -> list[ModelInfo]:
    """
    List available LLM models.
    
    Optionally filter by capability or tier.
    
    - **capability**: Filter by capability (realtime_feedback, movement_analysis, etc.)
    - **tier**: Filter by tier (fast, standard, premium)
    """
    manager = get_llm_manager()
    
    # Parse capability filter
    cap_filter = None
    if capability:
        try:
            cap_filter = ModelCapability(capability)
        except ValueError:
            raise HTTPException(400, f"Invalid capability: {capability}")
    
    # Get available models
    models = manager.get_available_models(capability=cap_filter, tier=tier)
    
    return [
        ModelInfo(
            id=f"{m.provider.value}/{m.model_id}",
            provider=m.provider.value,
            display_name=m.display_name,
            capabilities=[c.value for c in m.capabilities],
            tier=m.tier,
            description=m.description,
            avg_latency_ms=m.avg_latency_ms,
            supports_streaming=m.supports_streaming,
            supports_vision=m.supports_vision,
        )
        for m in models
    ]


@router.get("/models/{model_id}")
async def get_model_info(model_id: str) -> ModelInfo:
    """Get information about a specific model."""
    manager = get_llm_manager()
    
    # Handle both "provider/model" and just "model" formats
    if "/" in model_id:
        _, model_id = model_id.split("/", 1)
    
    config = manager.get_model_config(model_id)
    if not config:
        raise HTTPException(404, f"Model not found: {model_id}")
    
    return ModelInfo(
        id=f"{config.provider.value}/{config.model_id}",
        provider=config.provider.value,
        display_name=config.display_name,
        capabilities=[c.value for c in config.capabilities],
        tier=config.tier,
        description=config.description,
        avg_latency_ms=config.avg_latency_ms,
        supports_streaming=config.supports_streaming,
        supports_vision=config.supports_vision,
    )


@router.get("/providers")
async def list_providers() -> list[ProviderStatus]:
    """List all providers and their availability status."""
    manager = get_llm_manager()
    
    statuses = []
    for provider in LLMProvider:
        available = provider in manager.clients
        healthy = await manager.check_provider_health(provider) if available else False
        
        statuses.append(ProviderStatus(
            provider=provider.value,
            available=available,
            healthy=healthy,
        ))
    
    return statuses


@router.get("/providers/{provider}/health")
async def check_provider_health(provider: str, force: bool = False) -> ProviderStatus:
    """
    Check health of a specific provider.
    
    - **force**: Force a fresh health check (bypass cache)
    """
    try:
        provider_enum = LLMProvider(provider)
    except ValueError:
        raise HTTPException(400, f"Invalid provider: {provider}")
    
    manager = get_llm_manager()
    
    if provider_enum not in manager.clients:
        return ProviderStatus(
            provider=provider,
            available=False,
            healthy=False,
        )
    
    healthy = await manager.check_provider_health(provider_enum, force=force)
    
    return ProviderStatus(
        provider=provider,
        available=True,
        healthy=healthy,
    )


@router.get("/preferences")
async def get_preferences() -> PreferencesResponse:
    """Get current model preferences."""
    manager = get_llm_manager()
    prefs = manager.preferences
    
    return PreferencesResponse(
        realtime_feedback=prefs.realtime_feedback,
        movement_analysis=prefs.movement_analysis,
        report_generation=prefs.report_generation,
        exercise_classification=prefs.exercise_classification,
        coaching_cues=prefs.coaching_cues,
        fallback=prefs.fallback,
        enable_llm=prefs.enable_llm,
        enable_caching=prefs.enable_caching,
        cache_ttl_seconds=prefs.cache_ttl_seconds,
    )


@router.put("/preferences")
async def update_preferences(update: PreferencesUpdate) -> PreferencesResponse:
    """
    Update model preferences.
    
    Only provided fields will be updated.
    """
    manager = get_llm_manager()
    current = manager.preferences
    
    # Validate model IDs
    for field_name in ["realtime_feedback", "movement_analysis", "report_generation", 
                       "exercise_classification", "coaching_cues", "fallback"]:
        model_id = getattr(update, field_name)
        if model_id:
            config = manager.get_model_config(model_id)
            if not config:
                raise HTTPException(400, f"Invalid model for {field_name}: {model_id}")
    
    # Create updated preferences
    new_prefs = ModelPreferences(
        realtime_feedback=update.realtime_feedback or current.realtime_feedback,
        movement_analysis=update.movement_analysis or current.movement_analysis,
        report_generation=update.report_generation or current.report_generation,
        exercise_classification=update.exercise_classification or current.exercise_classification,
        coaching_cues=update.coaching_cues or current.coaching_cues,
        fallback=update.fallback or current.fallback,
        enable_llm=update.enable_llm if update.enable_llm is not None else current.enable_llm,
        enable_caching=update.enable_caching if update.enable_caching is not None else current.enable_caching,
        cache_ttl_seconds=update.cache_ttl_seconds or current.cache_ttl_seconds,
    )
    
    manager.set_preferences(new_prefs)
    
    return PreferencesResponse(
        realtime_feedback=new_prefs.realtime_feedback,
        movement_analysis=new_prefs.movement_analysis,
        report_generation=new_prefs.report_generation,
        exercise_classification=new_prefs.exercise_classification,
        coaching_cues=new_prefs.coaching_cues,
        fallback=new_prefs.fallback,
        enable_llm=new_prefs.enable_llm,
        enable_caching=new_prefs.enable_caching,
        cache_ttl_seconds=new_prefs.cache_ttl_seconds,
    )


@router.get("/stats")
async def get_stats() -> LLMStatsResponse:
    """Get LLM usage statistics."""
    manager = get_llm_manager()
    stats = manager.get_stats()
    
    return LLMStatsResponse(**stats)


@router.post("/cache/clear")
async def clear_cache():
    """Clear the LLM response cache."""
    manager = get_llm_manager()
    manager.clear_cache()
    
    return {"message": "Cache cleared", "success": True}


@router.get("/capabilities")
async def list_capabilities() -> list[dict]:
    """List available model capabilities with descriptions."""
    return [
        {
            "id": "realtime_feedback",
            "name": "Real-time Feedback",
            "description": "Fast models for live coaching cues during exercise",
            "recommended_tier": "fast",
        },
        {
            "id": "movement_analysis",
            "name": "Movement Analysis",
            "description": "Detailed biomechanical analysis of movement patterns",
            "recommended_tier": "standard",
        },
        {
            "id": "report_generation",
            "name": "Report Generation",
            "description": "Comprehensive assessment report writing",
            "recommended_tier": "premium",
        },
        {
            "id": "exercise_classification",
            "name": "Exercise Classification",
            "description": "Identifying exercises from movement data",
            "recommended_tier": "fast",
        },
        {
            "id": "coaching_cues",
            "name": "Coaching Cues",
            "description": "Generating natural coaching language",
            "recommended_tier": "fast",
        },
    ]


@router.post("/test")
async def test_completion(
    prompt: str = Body(..., embed=True),
    model: Optional[str] = Body(None, embed=True),
    capability: Optional[str] = Body(None, embed=True),
):
    """
    Test an LLM completion.
    
    Useful for testing model availability and response quality.
    """
    manager = get_llm_manager()
    
    cap = None
    if capability:
        try:
            cap = ModelCapability(capability)
        except ValueError:
            raise HTTPException(400, f"Invalid capability: {capability}")
    
    response = await manager.complete(
        prompt=prompt,
        model=model,
        capability=cap,
        max_tokens=500,
        timeout=30.0,
    )
    
    return {
        "success": response.success,
        "content": response.content if response.success else None,
        "error": response.error,
        "model": response.model,
        "provider": response.provider.value,
        "latency_ms": response.latency_ms,
        "from_cache": response.from_cache,
        "tokens": {
            "prompt": response.prompt_tokens,
            "completion": response.completion_tokens,
            "total": response.total_tokens,
        },
    }
