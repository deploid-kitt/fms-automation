"""LLM client implementations for different providers."""
import asyncio
import time
import logging
from typing import Optional, Any

import httpx

from app.llm.base import (
    BaseLLMClient,
    LLMProvider,
    LLMResponse,
    AVAILABLE_MODELS,
)

logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__(api_key, base_url or "https://api.openai.com/v1")
        
    @property
    def provider(self) -> LLMProvider:
        return LLMProvider.OPENAI
    
    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
        timeout: float = 30.0,
    ) -> LLMResponse:
        if not self.api_key:
            return LLMResponse(
                content="",
                model=model,
                provider=self.provider,
                success=False,
                error="OpenAI API key not configured",
            )
        
        # Rate limit check
        config = AVAILABLE_MODELS.get(model)
        if config:
            await self._rate_limit_wait(config)
        
        start_time = time.perf_counter()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                )
                
                if response.status_code != 200:
                    error_detail = response.text
                    logger.error(f"OpenAI API error: {response.status_code} - {error_detail}")
                    return LLMResponse(
                        content="",
                        model=model,
                        provider=self.provider,
                        success=False,
                        error=f"API error: {response.status_code}",
                        latency_ms=(time.perf_counter() - start_time) * 1000,
                    )
                
                data = response.json()
                latency_ms = (time.perf_counter() - start_time) * 1000
                
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                
                return LLMResponse(
                    content=content,
                    model=model,
                    provider=self.provider,
                    latency_ms=latency_ms,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                    success=True,
                    raw_response=data,
                )
                
        except httpx.TimeoutException:
            return LLMResponse(
                content="",
                model=model,
                provider=self.provider,
                success=False,
                error="Request timeout",
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )
        except Exception as e:
            logger.exception(f"OpenAI API exception: {e}")
            return LLMResponse(
                content="",
                model=model,
                provider=self.provider,
                success=False,
                error=str(e),
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )
    
    async def health_check(self) -> bool:
        if not self.api_key:
            return False
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                return response.status_code == 200
        except Exception:
            return False


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude API client."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__(api_key, base_url or "https://api.anthropic.com/v1")
        
    @property
    def provider(self) -> LLMProvider:
        return LLMProvider.ANTHROPIC
    
    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
        timeout: float = 30.0,
    ) -> LLMResponse:
        if not self.api_key:
            return LLMResponse(
                content="",
                model=model,
                provider=self.provider,
                success=False,
                error="Anthropic API key not configured",
            )
        
        config = AVAILABLE_MODELS.get(model)
        if config:
            await self._rate_limit_wait(config)
        
        start_time = time.perf_counter()
        
        request_body = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        
        if system_prompt:
            request_body["system"] = system_prompt
        
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self.base_url}/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "Content-Type": "application/json",
                        "anthropic-version": "2023-06-01",
                    },
                    json=request_body,
                )
                
                if response.status_code != 200:
                    error_detail = response.text
                    logger.error(f"Anthropic API error: {response.status_code} - {error_detail}")
                    return LLMResponse(
                        content="",
                        model=model,
                        provider=self.provider,
                        success=False,
                        error=f"API error: {response.status_code}",
                        latency_ms=(time.perf_counter() - start_time) * 1000,
                    )
                
                data = response.json()
                latency_ms = (time.perf_counter() - start_time) * 1000
                
                # Extract content from Anthropic response format
                content = ""
                for block in data.get("content", []):
                    if block.get("type") == "text":
                        content += block.get("text", "")
                
                usage = data.get("usage", {})
                
                return LLMResponse(
                    content=content,
                    model=model,
                    provider=self.provider,
                    latency_ms=latency_ms,
                    prompt_tokens=usage.get("input_tokens", 0),
                    completion_tokens=usage.get("output_tokens", 0),
                    total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                    success=True,
                    raw_response=data,
                )
                
        except httpx.TimeoutException:
            return LLMResponse(
                content="",
                model=model,
                provider=self.provider,
                success=False,
                error="Request timeout",
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )
        except Exception as e:
            logger.exception(f"Anthropic API exception: {e}")
            return LLMResponse(
                content="",
                model=model,
                provider=self.provider,
                success=False,
                error=str(e),
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )
    
    async def health_check(self) -> bool:
        if not self.api_key:
            return False
        # Anthropic doesn't have a lightweight health endpoint, so we just verify the key format
        return len(self.api_key) > 20


class GoogleClient(BaseLLMClient):
    """Google Gemini API client."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__(api_key, base_url or "https://generativelanguage.googleapis.com/v1beta")
        
    @property
    def provider(self) -> LLMProvider:
        return LLMProvider.GOOGLE
    
    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
        timeout: float = 30.0,
    ) -> LLMResponse:
        if not self.api_key:
            return LLMResponse(
                content="",
                model=model,
                provider=self.provider,
                success=False,
                error="Google API key not configured",
            )
        
        config = AVAILABLE_MODELS.get(model)
        if config:
            await self._rate_limit_wait(config)
        
        start_time = time.perf_counter()
        
        # Build the request
        contents = []
        if system_prompt:
            contents.append({
                "role": "user",
                "parts": [{"text": f"System instructions: {system_prompt}"}]
            })
            contents.append({
                "role": "model",
                "parts": [{"text": "I understand and will follow these instructions."}]
            })
        
        contents.append({
            "role": "user",
            "parts": [{"text": prompt}]
        })
        
        request_body = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        
        try:
            url = f"{self.base_url}/models/{model}:generateContent?key={self.api_key}"
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=request_body,
                )
                
                if response.status_code != 200:
                    error_detail = response.text
                    logger.error(f"Google API error: {response.status_code} - {error_detail}")
                    return LLMResponse(
                        content="",
                        model=model,
                        provider=self.provider,
                        success=False,
                        error=f"API error: {response.status_code}",
                        latency_ms=(time.perf_counter() - start_time) * 1000,
                    )
                
                data = response.json()
                latency_ms = (time.perf_counter() - start_time) * 1000
                
                # Extract content from Gemini response
                content = ""
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    for part in parts:
                        content += part.get("text", "")
                
                # Gemini token counts
                usage = data.get("usageMetadata", {})
                
                return LLMResponse(
                    content=content,
                    model=model,
                    provider=self.provider,
                    latency_ms=latency_ms,
                    prompt_tokens=usage.get("promptTokenCount", 0),
                    completion_tokens=usage.get("candidatesTokenCount", 0),
                    total_tokens=usage.get("totalTokenCount", 0),
                    success=True,
                    raw_response=data,
                )
                
        except httpx.TimeoutException:
            return LLMResponse(
                content="",
                model=model,
                provider=self.provider,
                success=False,
                error="Request timeout",
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )
        except Exception as e:
            logger.exception(f"Google API exception: {e}")
            return LLMResponse(
                content="",
                model=model,
                provider=self.provider,
                success=False,
                error=str(e),
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )
    
    async def health_check(self) -> bool:
        if not self.api_key:
            return False
        try:
            url = f"{self.base_url}/models?key={self.api_key}"
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url)
                return response.status_code == 200
        except Exception:
            return False


class OllamaClient(BaseLLMClient):
    """Ollama local model client."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__(api_key, base_url or "http://localhost:11434")
        
    @property
    def provider(self) -> LLMProvider:
        return LLMProvider.OLLAMA
    
    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
        timeout: float = 60.0,  # Local models may need more time
    ) -> LLMResponse:
        start_time = time.perf_counter()
        
        request_body = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        
        if system_prompt:
            request_body["system"] = system_prompt
        
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    headers={"Content-Type": "application/json"},
                    json=request_body,
                )
                
                if response.status_code != 200:
                    error_detail = response.text
                    logger.error(f"Ollama API error: {response.status_code} - {error_detail}")
                    return LLMResponse(
                        content="",
                        model=model,
                        provider=self.provider,
                        success=False,
                        error=f"API error: {response.status_code}",
                        latency_ms=(time.perf_counter() - start_time) * 1000,
                    )
                
                data = response.json()
                latency_ms = (time.perf_counter() - start_time) * 1000
                
                content = data.get("response", "")
                
                # Ollama provides eval_count for completion tokens
                prompt_tokens = data.get("prompt_eval_count", 0)
                completion_tokens = data.get("eval_count", 0)
                
                return LLMResponse(
                    content=content,
                    model=model,
                    provider=self.provider,
                    latency_ms=latency_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    success=True,
                    raw_response=data,
                )
                
        except httpx.ConnectError:
            return LLMResponse(
                content="",
                model=model,
                provider=self.provider,
                success=False,
                error="Cannot connect to Ollama - is it running?",
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )
        except httpx.TimeoutException:
            return LLMResponse(
                content="",
                model=model,
                provider=self.provider,
                success=False,
                error="Request timeout",
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )
        except Exception as e:
            logger.exception(f"Ollama API exception: {e}")
            return LLMResponse(
                content="",
                model=model,
                provider=self.provider,
                success=False,
                error=str(e),
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )
    
    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False
    
    async def list_models(self) -> list[str]:
        """List available models in Ollama."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        return []
