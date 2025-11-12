"""
LLM Service - Large Language Model integration and content generation
"""
import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

# Groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

# OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

# Ollama
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ollama = None

logger = logging.getLogger(__name__)


class LLMService:
    """Service for interacting with Large Language Models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        
        # Configuration
        self.groq_api_key = self.config.get("groq_api_key")
        self.groq_model = self.config.get("groq_model", "openai/gpt-oss-20b")  # Updated default model
        
        self.openai_api_key = self.config.get("openai_api_key")
        self.openai_model = self.config.get("openai_model", "gpt-3.5-turbo")
        
        self.local_llm_enabled = self.config.get("local_llm_enabled", False)
        self.ollama_base_url = self.config.get("ollama_base_url", "http://localhost:11434")
        self.ollama_model = self.config.get("ollama_model", "llama2")
        
        # Provider priority: groq > openai > ollama
        self.provider = None
        self.client = None
        
        # Initialize provider
        self._initialize_provider()
        
        logger.info(f"LLM Service initialized with provider: {self.provider}")
    
    def _initialize_provider(self):
        """Initialize the LLM provider"""
        
        # Try Groq first (preferred)
        if GROQ_AVAILABLE and self.groq_api_key:
            try:
                self.client = Groq(api_key=self.groq_api_key)
                self.provider = "groq"
                logger.info(f"Initialized Groq with model: {self.groq_model}")
                return
            except Exception as e:
                logger.error(f"Failed to initialize Groq: {str(e)}")
        
        # Try OpenAI second
        if OPENAI_AVAILABLE and self.openai_api_key:
            try:
                openai.api_key = self.openai_api_key
                self.provider = "openai"
                logger.info(f"Initialized OpenAI with model: {self.openai_model}")
                return
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI: {str(e)}")
        
        # Try Ollama last (local)
        if OLLAMA_AVAILABLE and self.local_llm_enabled:
            try:
                self.provider = "ollama"
                logger.info(f"Initialized Ollama with model: {self.ollama_model}")
                return
            except Exception as e:
                logger.error(f"Failed to initialize Ollama: {str(e)}")
        
        # No provider available
        logger.warning("No LLM provider available. Service will have limited functionality.")
        self.provider = "none"
    
    async def generate_content(
        self,
        content: str = "",
        prompt: str = "",
        max_tokens: int = 8192,
        temperature: float = 1.0,
        stream: bool = False,
        **kwargs
    ) -> str:
        """Generate content using the configured LLM"""
        
        if self.provider == "none":
            return "LLM service not available"
        
        # Combine content and prompt
        full_prompt = f"{content}\n\n{prompt}".strip() if content else prompt
        
        try:
            if self.provider == "groq":
                return await self._generate_groq(full_prompt, max_tokens, temperature, stream, **kwargs)
            elif self.provider == "openai":
                return await self._generate_openai(full_prompt, max_tokens, temperature, **kwargs)
            elif self.provider == "ollama":
                return await self._generate_ollama(full_prompt, max_tokens, temperature, **kwargs)
            else:
                return "No LLM provider configured"
                
        except Exception as e:
            logger.error(f"Content generation failed with {self.provider}: {str(e)}")
            
            # Try fallback to groq if not already using it
            if self.provider != "groq" and GROQ_AVAILABLE and self.groq_api_key:
                logger.info("Attempting fallback to groq")
                try:
                    if not self.client:
                        self.client = Groq(api_key=self.groq_api_key)
                    return await self._generate_groq(full_prompt, max_tokens, temperature, stream, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"Fallback to groq failed: {str(fallback_error)}")
            
            raise
    
    async def _generate_groq(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        stream: bool = True,
        **kwargs
    ) -> str:
        """Generate content using Groq API with new model and streaming"""
        
        try:
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Use the new Groq model with specified parameters
            completion = self.client.chat.completions.create(
                model=self.groq_model,  # openai/gpt-oss-20b
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                stream=stream,
                stop=None
            )
            
            # Handle streaming response
            if stream:
                full_response = ""
                for chunk in completion:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                return full_response
            else:
                return completion.choices[0].message.content
                
        except Exception as e:
            logger.error(f"Groq generation failed: {str(e)}")
            raise
    
    async def _generate_openai(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> str:
        """Generate content using OpenAI API"""
        
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.openai_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {str(e)}")
            raise
    
    async def _generate_ollama(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> str:
        """Generate content using Ollama (local)"""
        
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            response = await loop.run_in_executor(
                None,
                lambda: ollama.generate(
                    model=self.ollama_model,
                    prompt=prompt,
                    options={
                        "num_predict": max_tokens,
                        "temperature": temperature,
                    }
                )
            )
            
            return response.get("response", "")
            
        except Exception as e:
            logger.error(f"Ollama generation failed: {str(e)}")
            raise
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Simple generate method for backward compatibility"""
        return await self.generate_content(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
    
    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        max_tokens: int = 2000,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """Generate structured output (JSON)"""
        
        structured_prompt = f"""{prompt}

Please respond with a valid JSON object matching this schema:
{schema}

Response:"""
        
        try:
            response = await self.generate_content(
                prompt=structured_prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Try to parse JSON
            import json
            
            # Extract JSON from response
            response = response.strip()
            
            # Find JSON object
            start = response.find("{")
            end = response.rfind("}") + 1
            
            if start != -1 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
            
            # If no JSON found, try to parse entire response
            return json.loads(response)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            logger.error(f"Response was: {response}")
            return {}
        except Exception as e:
            logger.error(f"Structured generation failed: {str(e)}")
            return {}
    
    async def generate_with_context(
        self,
        prompt: str,
        context: List[Dict[str, str]],
        max_tokens: int = 2000,
        temperature: float = 0.7
    ) -> str:
        """Generate content with conversation context"""
        
        messages = context + [{"role": "user", "content": prompt}]
        
        try:
            if self.provider == "groq":
                completion = self.client.chat.completions.create(
                    model=self.groq_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=1,
                    stream=True,
                    stop=None
                )
                
                full_response = ""
                for chunk in completion:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                return full_response
                
            elif self.provider == "openai":
                response = await openai.ChatCompletion.acreate(
                    model=self.openai_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
            
            else:
                # Fallback for ollama
                context_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
                return await self._generate_ollama(context_str, max_tokens, temperature)
                
        except Exception as e:
            logger.error(f"Context-based generation failed: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        
        return {
            "provider": self.provider,
            "model": self._get_current_model(),
            "available_providers": self._get_available_providers(),
            "configuration": {
                "groq_model": self.groq_model,
                "openai_model": self.openai_model,
                "ollama_model": self.ollama_model,
            }
        }
    
    def _get_current_model(self) -> str:
        """Get the current model name"""
        if self.provider == "groq":
            return self.groq_model
        elif self.provider == "openai":
            return self.openai_model
        elif self.provider == "ollama":
            return self.ollama_model
        return "none"
    
    def _get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        providers = []
        
        if GROQ_AVAILABLE and self.groq_api_key:
            providers.append("groq")
        if OPENAI_AVAILABLE and self.openai_api_key:
            providers.append("openai")
        if OLLAMA_AVAILABLE and self.local_llm_enabled:
            providers.append("ollama")
        
        return providers
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of LLM service"""
        
        health_status = {
            "status": "healthy" if self.provider != "none" else "unhealthy",
            "provider": self.provider,
            "model": self._get_current_model(),
            "available_providers": self._get_available_providers()
        }
        
        # Test generation
        try:
            test_response = await self.generate(
                prompt="Say 'OK' if you can respond.",
                max_tokens=10,
                temperature=0.1
            )
            
            if test_response and len(test_response) > 0:
                health_status["generation_test"] = "passed"
            else:
                health_status["generation_test"] = "failed"
                health_status["status"] = "degraded"
                
        except Exception as e:
            health_status["generation_test"] = "failed"
            health_status["error"] = str(e)
            health_status["status"] = "unhealthy"
        
        return health_status
    
    async def switch_provider(self, provider: str) -> bool:
        """Switch to a different provider"""
        
        old_provider = self.provider
        
        try:
            if provider == "groq" and GROQ_AVAILABLE and self.groq_api_key:
                self.client = Groq(api_key=self.groq_api_key)
                self.provider = "groq"
            elif provider == "openai" and OPENAI_AVAILABLE and self.openai_api_key:
                openai.api_key = self.openai_api_key
                self.provider = "openai"
            elif provider == "ollama" and OLLAMA_AVAILABLE and self.local_llm_enabled:
                self.provider = "ollama"
            else:
                logger.error(f"Provider {provider} not available")
                return False
            
            # Test new provider
            test_result = await self.health_check()
            
            if test_result["status"] in ["healthy", "degraded"]:
                logger.info(f"Successfully switched from {old_provider} to {provider}")
                return True
            else:
                # Rollback
                self.provider = old_provider
                logger.error(f"Failed to switch to {provider}, rolled back to {old_provider}")
                return False
                
        except Exception as e:
            self.provider = old_provider
            logger.error(f"Provider switch failed: {str(e)}")
            return False
    
    def get_token_count_estimate(self, text: str) -> int:
        """Estimate token count for text"""
        # Rough estimation: ~4 characters per token
        return len(text) // 4
    
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to approximate token count"""
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."