"""Base classes and types for LLM integration."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any
import asyncio
import time
import logging

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"


class ModelCapability(str, Enum):
    """Model capabilities for FMS tasks."""
    REALTIME_FEEDBACK = "realtime_feedback"  # Fast models for live coaching
    MOVEMENT_ANALYSIS = "movement_analysis"  # Detailed movement pattern analysis
    REPORT_GENERATION = "report_generation"  # Comprehensive report writing
    EXERCISE_CLASSIFICATION = "exercise_classification"  # Identify exercises
    COACHING_CUES = "coaching_cues"  # Generate coaching language


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    provider: LLMProvider
    model_id: str
    display_name: str
    
    # Capabilities this model is suited for
    capabilities: list[ModelCapability] = field(default_factory=list)
    
    # Performance characteristics
    max_tokens: int = 4096
    default_temperature: float = 0.7
    avg_latency_ms: int = 1000  # Expected average latency
    cost_per_1k_tokens: float = 0.0  # Approximate cost
    
    # Rate limiting
    requests_per_minute: int = 60
    tokens_per_minute: int = 100000
    
    # Feature flags
    supports_streaming: bool = True
    supports_function_calling: bool = False
    supports_vision: bool = False
    
    # For display
    description: str = ""
    tier: str = "standard"  # fast, standard, premium
    
    def __post_init__(self):
        if not self.capabilities:
            self.capabilities = list(ModelCapability)


# Predefined model configurations
AVAILABLE_MODELS: dict[str, ModelConfig] = {
    # OpenAI Models
    "gpt-4o": ModelConfig(
        provider=LLMProvider.OPENAI,
        model_id="gpt-4o",
        display_name="GPT-4o",
        capabilities=[
            ModelCapability.MOVEMENT_ANALYSIS,
            ModelCapability.REPORT_GENERATION,
            ModelCapability.EXERCISE_CLASSIFICATION,
            ModelCapability.COACHING_CUES,
        ],
        max_tokens=4096,
        default_temperature=0.7,
        avg_latency_ms=2000,
        cost_per_1k_tokens=0.005,
        supports_function_calling=True,
        supports_vision=True,
        description="Most capable GPT-4 model with vision",
        tier="premium",
    ),
    "gpt-4o-mini": ModelConfig(
        provider=LLMProvider.OPENAI,
        model_id="gpt-4o-mini",
        display_name="GPT-4o Mini",
        capabilities=[
            ModelCapability.REALTIME_FEEDBACK,
            ModelCapability.COACHING_CUES,
            ModelCapability.EXERCISE_CLASSIFICATION,
        ],
        max_tokens=4096,
        default_temperature=0.7,
        avg_latency_ms=500,
        cost_per_1k_tokens=0.00015,
        supports_function_calling=True,
        supports_vision=True,
        description="Fast and affordable, great for real-time feedback",
        tier="fast",
    ),
    "gpt-3.5-turbo": ModelConfig(
        provider=LLMProvider.OPENAI,
        model_id="gpt-3.5-turbo",
        display_name="GPT-3.5 Turbo",
        capabilities=[
            ModelCapability.REALTIME_FEEDBACK,
            ModelCapability.COACHING_CUES,
        ],
        max_tokens=4096,
        default_temperature=0.7,
        avg_latency_ms=300,
        cost_per_1k_tokens=0.0005,
        supports_function_calling=True,
        description="Fastest OpenAI model, best for real-time",
        tier="fast",
    ),
    
    # Anthropic Models
    "claude-sonnet-4-20250514": ModelConfig(
        provider=LLMProvider.ANTHROPIC,
        model_id="claude-sonnet-4-20250514",
        display_name="Claude Sonnet 4",
        capabilities=[
            ModelCapability.MOVEMENT_ANALYSIS,
            ModelCapability.REPORT_GENERATION,
            ModelCapability.EXERCISE_CLASSIFICATION,
            ModelCapability.COACHING_CUES,
        ],
        max_tokens=4096,
        default_temperature=0.7,
        avg_latency_ms=1500,
        cost_per_1k_tokens=0.003,
        supports_streaming=True,
        description="Balanced performance and quality",
        tier="standard",
    ),
    "claude-3-5-haiku-20241022": ModelConfig(
        provider=LLMProvider.ANTHROPIC,
        model_id="claude-3-5-haiku-20241022",
        display_name="Claude 3.5 Haiku",
        capabilities=[
            ModelCapability.REALTIME_FEEDBACK,
            ModelCapability.COACHING_CUES,
            ModelCapability.EXERCISE_CLASSIFICATION,
        ],
        max_tokens=4096,
        default_temperature=0.7,
        avg_latency_ms=400,
        cost_per_1k_tokens=0.00025,
        supports_streaming=True,
        description="Fastest Claude model for real-time use",
        tier="fast",
    ),
    "claude-opus-4-20250514": ModelConfig(
        provider=LLMProvider.ANTHROPIC,
        model_id="claude-opus-4-20250514",
        display_name="Claude Opus 4",
        capabilities=[
            ModelCapability.MOVEMENT_ANALYSIS,
            ModelCapability.REPORT_GENERATION,
        ],
        max_tokens=4096,
        default_temperature=0.7,
        avg_latency_ms=3000,
        cost_per_1k_tokens=0.015,
        supports_streaming=True,
        description="Most capable Claude for detailed analysis",
        tier="premium",
    ),
    
    # Google Models
    "gemini-2.0-flash": ModelConfig(
        provider=LLMProvider.GOOGLE,
        model_id="gemini-2.0-flash",
        display_name="Gemini 2.0 Flash",
        capabilities=[
            ModelCapability.REALTIME_FEEDBACK,
            ModelCapability.COACHING_CUES,
            ModelCapability.MOVEMENT_ANALYSIS,
        ],
        max_tokens=8192,
        default_temperature=0.7,
        avg_latency_ms=500,
        cost_per_1k_tokens=0.0001,
        supports_streaming=True,
        supports_vision=True,
        description="Fast multimodal model from Google",
        tier="fast",
    ),
    "gemini-1.5-pro": ModelConfig(
        provider=LLMProvider.GOOGLE,
        model_id="gemini-1.5-pro",
        display_name="Gemini 1.5 Pro",
        capabilities=[
            ModelCapability.MOVEMENT_ANALYSIS,
            ModelCapability.REPORT_GENERATION,
            ModelCapability.EXERCISE_CLASSIFICATION,
        ],
        max_tokens=8192,
        default_temperature=0.7,
        avg_latency_ms=2000,
        cost_per_1k_tokens=0.00125,
        supports_streaming=True,
        supports_vision=True,
        description="Best Gemini for complex analysis",
        tier="premium",
    ),
    
    # Ollama (Local) Models
    "llama3.2": ModelConfig(
        provider=LLMProvider.OLLAMA,
        model_id="llama3.2",
        display_name="Llama 3.2 (Local)",
        capabilities=[
            ModelCapability.REALTIME_FEEDBACK,
            ModelCapability.COACHING_CUES,
        ],
        max_tokens=4096,
        default_temperature=0.7,
        avg_latency_ms=200,
        cost_per_1k_tokens=0.0,
        requests_per_minute=1000,
        tokens_per_minute=1000000,
        supports_streaming=True,
        description="Fast local model, privacy-focused",
        tier="fast",
    ),
    "llama3.1:70b": ModelConfig(
        provider=LLMProvider.OLLAMA,
        model_id="llama3.1:70b",
        display_name="Llama 3.1 70B (Local)",
        capabilities=[
            ModelCapability.MOVEMENT_ANALYSIS,
            ModelCapability.REPORT_GENERATION,
            ModelCapability.COACHING_CUES,
        ],
        max_tokens=4096,
        default_temperature=0.7,
        avg_latency_ms=2000,
        cost_per_1k_tokens=0.0,
        requests_per_minute=100,
        tokens_per_minute=100000,
        supports_streaming=True,
        description="Powerful local model for analysis",
        tier="standard",
    ),
    "mistral": ModelConfig(
        provider=LLMProvider.OLLAMA,
        model_id="mistral",
        display_name="Mistral 7B (Local)",
        capabilities=[
            ModelCapability.REALTIME_FEEDBACK,
            ModelCapability.COACHING_CUES,
        ],
        max_tokens=4096,
        default_temperature=0.7,
        avg_latency_ms=150,
        cost_per_1k_tokens=0.0,
        requests_per_minute=1000,
        tokens_per_minute=1000000,
        supports_streaming=True,
        description="Very fast local model",
        tier="fast",
    ),
}


@dataclass
class LLMResponse:
    """Response from an LLM call."""
    content: str
    model: str
    provider: LLMProvider
    
    # Timing
    latency_ms: float = 0.0
    
    # Token usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    # Status
    success: bool = True
    error: Optional[str] = None
    
    # Caching
    from_cache: bool = False
    
    # Raw response for debugging
    raw_response: Optional[Any] = None


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url
        self._last_request_time: float = 0
        self._request_count: int = 0
        self._token_count: int = 0
        
    @property
    @abstractmethod
    def provider(self) -> LLMProvider:
        """Return the provider enum for this client."""
        pass
    
    @abstractmethod
    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
        timeout: float = 30.0,
    ) -> LLMResponse:
        """
        Generate a completion from the model.
        
        Args:
            prompt: The user prompt
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            system_prompt: Optional system instruction
            timeout: Request timeout in seconds
            
        Returns:
            LLMResponse with the completion
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the client can connect to the API."""
        pass
    
    async def _rate_limit_wait(self, config: ModelConfig):
        """Wait if needed to respect rate limits."""
        now = time.time()
        time_since_last = now - self._last_request_time
        
        # Simple rate limiting: ensure minimum time between requests
        min_interval = 60.0 / config.requests_per_minute
        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)
        
        self._last_request_time = time.time()
        self._request_count += 1
    
    def is_available(self) -> bool:
        """Check if the client is configured and available."""
        return self.api_key is not None or self.provider == LLMProvider.OLLAMA
