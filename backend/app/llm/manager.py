"""LLM Manager for coordinating model access and caching."""
import asyncio
import hashlib
import json
import logging
import os
import time
from typing import Optional, Callable, Awaitable
from dataclasses import dataclass, field
from functools import lru_cache

from app.llm.base import (
    LLMProvider,
    ModelConfig,
    ModelCapability,
    LLMResponse,
    BaseLLMClient,
    AVAILABLE_MODELS,
)
from app.llm.clients import (
    OpenAIClient,
    AnthropicClient,
    GoogleClient,
    OllamaClient,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelPreferences:
    """User preferences for model selection per capability."""
    realtime_feedback: str = "gpt-4o-mini"
    movement_analysis: str = "claude-sonnet-4-20250514"
    report_generation: str = "claude-sonnet-4-20250514"
    exercise_classification: str = "gpt-4o-mini"
    coaching_cues: str = "gpt-4o-mini"
    
    # Fallback model when preferred is unavailable
    fallback: str = "gpt-3.5-turbo"
    
    # Global settings
    enable_llm: bool = True
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour
    
    def get_model_for_capability(self, capability: ModelCapability) -> str:
        """Get the configured model for a capability."""
        mapping = {
            ModelCapability.REALTIME_FEEDBACK: self.realtime_feedback,
            ModelCapability.MOVEMENT_ANALYSIS: self.movement_analysis,
            ModelCapability.REPORT_GENERATION: self.report_generation,
            ModelCapability.EXERCISE_CLASSIFICATION: self.exercise_classification,
            ModelCapability.COACHING_CUES: self.coaching_cues,
        }
        return mapping.get(capability, self.fallback)


@dataclass
class CacheEntry:
    """Cache entry for LLM responses."""
    response: LLMResponse
    created_at: float
    ttl: int
    
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl


class LLMManager:
    """
    Manages LLM clients, routing, caching, and fallback logic.
    
    This is the main interface for all LLM operations in the application.
    """
    
    def __init__(
        self,
        openai_key: Optional[str] = None,
        anthropic_key: Optional[str] = None,
        google_key: Optional[str] = None,
        ollama_url: Optional[str] = None,
    ):
        # Initialize clients
        self.clients: dict[LLMProvider, BaseLLMClient] = {}
        
        if openai_key:
            self.clients[LLMProvider.OPENAI] = OpenAIClient(api_key=openai_key)
            
        if anthropic_key:
            self.clients[LLMProvider.ANTHROPIC] = AnthropicClient(api_key=anthropic_key)
            
        if google_key:
            self.clients[LLMProvider.GOOGLE] = GoogleClient(api_key=google_key)
        
        # Always try to add Ollama (local, no key needed)
        self.clients[LLMProvider.OLLAMA] = OllamaClient(base_url=ollama_url)
        
        # Cache for responses
        self._cache: dict[str, CacheEntry] = {}
        self._cache_lock = asyncio.Lock()
        
        # Preferences (can be updated per-user)
        self.preferences = ModelPreferences()
        
        # Provider health status
        self._provider_health: dict[LLMProvider, bool] = {}
        self._health_check_time: dict[LLMProvider, float] = {}
        
        # Stats
        self._request_count = 0
        self._cache_hits = 0
        self._total_latency_ms = 0.0
        
        logger.info(f"LLMManager initialized with providers: {list(self.clients.keys())}")
    
    def get_available_providers(self) -> list[LLMProvider]:
        """Get list of configured providers."""
        return [p for p, c in self.clients.items() if c.is_available()]
    
    def get_available_models(
        self, 
        capability: Optional[ModelCapability] = None,
        tier: Optional[str] = None,
    ) -> list[ModelConfig]:
        """
        Get list of available models, optionally filtered.
        
        Args:
            capability: Filter by capability
            tier: Filter by tier (fast, standard, premium)
        """
        available_providers = self.get_available_providers()
        
        models = []
        for model_id, config in AVAILABLE_MODELS.items():
            # Check if provider is available
            if config.provider not in available_providers:
                continue
            
            # Check capability filter
            if capability and capability not in config.capabilities:
                continue
            
            # Check tier filter
            if tier and config.tier != tier:
                continue
            
            models.append(config)
        
        return models
    
    def get_model_config(self, model_id: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model."""
        return AVAILABLE_MODELS.get(model_id)
    
    def set_preferences(self, preferences: ModelPreferences):
        """Update model preferences."""
        self.preferences = preferences
        logger.info(f"Updated LLM preferences: {preferences}")
    
    async def check_provider_health(self, provider: LLMProvider, force: bool = False) -> bool:
        """
        Check if a provider is healthy.
        
        Results are cached for 5 minutes unless force=True.
        """
        if not force:
            last_check = self._health_check_time.get(provider, 0)
            if time.time() - last_check < 300:  # 5 minute cache
                return self._provider_health.get(provider, False)
        
        if provider not in self.clients:
            return False
        
        healthy = await self.clients[provider].health_check()
        self._provider_health[provider] = healthy
        self._health_check_time[provider] = time.time()
        
        return healthy
    
    def _get_cache_key(self, prompt: str, model: str, system_prompt: Optional[str]) -> str:
        """Generate cache key for a request."""
        content = f"{model}:{system_prompt or ''}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]
    
    async def _get_cached_response(self, cache_key: str) -> Optional[LLMResponse]:
        """Get cached response if available and not expired."""
        async with self._cache_lock:
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                if not entry.is_expired():
                    self._cache_hits += 1
                    response = entry.response
                    response.from_cache = True
                    return response
                else:
                    del self._cache[cache_key]
        return None
    
    async def _set_cached_response(self, cache_key: str, response: LLMResponse, ttl: int):
        """Cache a response."""
        async with self._cache_lock:
            self._cache[cache_key] = CacheEntry(
                response=response,
                created_at=time.time(),
                ttl=ttl,
            )
            
            # Cleanup old entries if cache is too large
            if len(self._cache) > 1000:
                expired_keys = [
                    k for k, v in self._cache.items() if v.is_expired()
                ]
                for k in expired_keys:
                    del self._cache[k]
    
    async def complete(
        self,
        prompt: str,
        capability: Optional[ModelCapability] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
        timeout: float = 30.0,
        use_cache: bool = True,
        fallback: bool = True,
    ) -> LLMResponse:
        """
        Generate a completion using the appropriate model.
        
        Args:
            prompt: The user prompt
            capability: Optional capability to select model by
            model: Optional specific model to use (overrides capability)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            system_prompt: Optional system instruction
            timeout: Request timeout
            use_cache: Whether to use caching
            fallback: Whether to try fallback models on failure
            
        Returns:
            LLMResponse with the completion
        """
        if not self.preferences.enable_llm:
            return LLMResponse(
                content="",
                model="disabled",
                provider=LLMProvider.OPENAI,
                success=False,
                error="LLM integration is disabled",
            )
        
        # Determine which model to use
        if not model:
            if capability:
                model = self.preferences.get_model_for_capability(capability)
            else:
                model = self.preferences.fallback
        
        # Get model config
        config = self.get_model_config(model)
        if not config:
            return LLMResponse(
                content="",
                model=model,
                provider=LLMProvider.OPENAI,
                success=False,
                error=f"Unknown model: {model}",
            )
        
        # Check cache
        if use_cache and self.preferences.enable_caching:
            cache_key = self._get_cache_key(prompt, model, system_prompt)
            cached = await self._get_cached_response(cache_key)
            if cached:
                logger.debug(f"Cache hit for {model}")
                return cached
        
        # Get the appropriate client
        provider = config.provider
        if provider not in self.clients:
            if fallback:
                return await self._try_fallback(
                    prompt, capability, temperature, max_tokens, 
                    system_prompt, timeout, use_cache
                )
            return LLMResponse(
                content="",
                model=model,
                provider=provider,
                success=False,
                error=f"Provider {provider} not configured",
            )
        
        client = self.clients[provider]
        
        # Make the request
        self._request_count += 1
        response = await client.complete(
            prompt=prompt,
            model=config.model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            timeout=timeout,
        )
        
        if response.success:
            self._total_latency_ms += response.latency_ms
            
            # Cache successful response
            if use_cache and self.preferences.enable_caching:
                await self._set_cached_response(
                    cache_key, response, self.preferences.cache_ttl_seconds
                )
            
            return response
        
        # Try fallback on failure
        if fallback and model != self.preferences.fallback:
            logger.warning(f"Model {model} failed, trying fallback")
            return await self._try_fallback(
                prompt, capability, temperature, max_tokens,
                system_prompt, timeout, use_cache
            )
        
        return response
    
    async def _try_fallback(
        self,
        prompt: str,
        capability: Optional[ModelCapability],
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
        timeout: float,
        use_cache: bool,
    ) -> LLMResponse:
        """Try fallback model."""
        fallback_model = self.preferences.fallback
        fallback_config = self.get_model_config(fallback_model)
        
        if not fallback_config or fallback_config.provider not in self.clients:
            # Try any available model for the capability
            available = self.get_available_models(capability=capability)
            if available:
                fallback_config = available[0]
                fallback_model = fallback_config.model_id
            else:
                return LLMResponse(
                    content="",
                    model=fallback_model,
                    provider=LLMProvider.OPENAI,
                    success=False,
                    error="No available models for fallback",
                )
        
        return await self.complete(
            prompt=prompt,
            model=fallback_model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            timeout=timeout,
            use_cache=use_cache,
            fallback=False,  # Don't recurse
        )
    
    async def complete_for_realtime(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        timeout: float = 5.0,  # Short timeout for real-time
    ) -> LLMResponse:
        """
        Completion optimized for real-time feedback.
        
        Uses fast models with short timeouts.
        """
        return await self.complete(
            prompt=prompt,
            capability=ModelCapability.REALTIME_FEEDBACK,
            temperature=0.5,  # Lower temperature for consistency
            max_tokens=150,  # Short responses
            system_prompt=system_prompt,
            timeout=timeout,
            use_cache=True,
            fallback=True,
        )
    
    async def complete_for_analysis(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        timeout: float = 30.0,
    ) -> LLMResponse:
        """
        Completion for detailed movement analysis.
        
        Uses more capable models with longer timeouts.
        """
        return await self.complete(
            prompt=prompt,
            capability=ModelCapability.MOVEMENT_ANALYSIS,
            temperature=0.7,
            max_tokens=2000,
            system_prompt=system_prompt,
            timeout=timeout,
            use_cache=True,
            fallback=True,
        )
    
    async def complete_for_report(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        timeout: float = 60.0,
    ) -> LLMResponse:
        """
        Completion for report generation.
        
        Uses capable models with generous token limits.
        """
        return await self.complete(
            prompt=prompt,
            capability=ModelCapability.REPORT_GENERATION,
            temperature=0.7,
            max_tokens=3000,
            system_prompt=system_prompt,
            timeout=timeout,
            use_cache=True,
            fallback=True,
        )
    
    def get_stats(self) -> dict:
        """Get usage statistics."""
        avg_latency = (
            self._total_latency_ms / self._request_count 
            if self._request_count > 0 else 0
        )
        cache_rate = (
            self._cache_hits / (self._request_count + self._cache_hits)
            if (self._request_count + self._cache_hits) > 0 else 0
        )
        
        return {
            "request_count": self._request_count,
            "cache_hits": self._cache_hits,
            "cache_rate": cache_rate,
            "avg_latency_ms": avg_latency,
            "cache_size": len(self._cache),
            "available_providers": [p.value for p in self.get_available_providers()],
        }
    
    def clear_cache(self):
        """Clear the response cache."""
        self._cache.clear()
        logger.info("LLM cache cleared")


# Singleton instance
_llm_manager: Optional[LLMManager] = None


def get_llm_manager() -> LLMManager:
    """Get or create the LLM manager instance."""
    global _llm_manager
    
    if _llm_manager is None:
        _llm_manager = LLMManager(
            openai_key=os.environ.get("OPENAI_API_KEY"),
            anthropic_key=os.environ.get("ANTHROPIC_API_KEY"),
            google_key=os.environ.get("GOOGLE_API_KEY"),
            ollama_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        )
    
    return _llm_manager


def reset_llm_manager():
    """Reset the LLM manager (for testing)."""
    global _llm_manager
    _llm_manager = None
