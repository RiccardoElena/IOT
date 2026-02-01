"""
Gemini AI Assistant Module for IoT Financial Data Analytics.

This module provides integration with Google's Gemini API for intelligent
chat assistance. It sends structured data context (not images) for reliable
LLM analysis of financial metrics.

Key Features:
- Automatic API key detection from environment
- Mock mode fallback for testing without API key
- Rich context builders for each page type
- Conversation history management
- Error handling with user-friendly messages

Usage:
    from src.gemini_assistant import get_assistant, build_single_asset_context
    
    assistant = get_assistant()
    context = build_single_asset_context(...)
    response = assistant.send_message("Analyze the anomalies", context)
"""

import os
from typing import Any, Dict, List, Optional
from PIL import Image


# Import logger
from utils.logger import logger
from utils.conversions import to_json_string, fig_to_png

# Try to import kaleido for chart export
try:
    import kaleido
    KALEIDO_AVAILABLE = True
except ImportError:
    KALEIDO_AVAILABLE = False
    kaleido = None

# Try to import python-dotenv for .env file support
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Try to import config - only AI settings needed here
from config import ai as config


# Try to import Gemini library
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


# =============================================================================
# GEMINI ASSISTANT CLASS
# =============================================================================

class GeminiAssistant:
    """
    Wrapper class for Google Gemini API interactions.
    
    Handles:
    - API initialization and configuration
    - Message sending with context
    - Conversation history management
    - Error handling and fallback modes
    
    Attributes:
        client: The Gemini client instance (or None in mock mode)
        api_key_set: Whether a valid API key is configured
        history: List of conversation messages
    """
    
    def __init__(self) -> None:
        """Initialize the Gemini assistant with configuration from config.py or defaults."""
        self.client = None
        self.api_key_set = False
        self.history: List[Dict[str, str]] = []
        
        # Load configuration
        self.model_name = config.GEMINI_MODEL
        self.max_tokens = config.GEMINI_MAX_TOKENS
        self.temperature = config.GEMINI_TEMPERATURE
        self.max_history = config.GEMINI_HISTORY_LENGTH
        self.system_prompt = config.GEMINI_SYSTEM_PROMPT
        
        # Initialize API if available
        self._initialize_api()
    
    def _initialize_api(self) -> None:
        """Initialize the Gemini API with the configured API key."""
        if not GENAI_AVAILABLE:
            return
        
        # Load .env file if available
        if DOTENV_AVAILABLE:
            load_dotenv() # type: ignore
        
        api_key = os.environ.get(config.GEMINI_API_KEY_ENV, '')
        
        if not api_key:
            return
        
        try:
            # Create client with API key
            self.client = genai.Client(api_key=api_key) # type: ignore
            
            # Store model name for generate_content calls
            self.model_name_for_api = self.model_name
            
            self.api_key_set = True
            
        except Exception as e:
            logger.error(f"Gemini API initialization failed: {e}")
            self.client = None
            self.api_key_set = False
    
    def send_message(
        self,
        question: str,
        page_context: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        chart_figures: Optional[List] = None
    ) -> str:
        """
        Send a message to Gemini and get a response.
        
        Args:
            question: The user's question
            page_context: Dictionary with current page data and statistics
            history: Optional conversation history to include
            chart_figures: Optional list of Plotly Figure objects to include as images
        
        Returns:
            The assistant's response text
        """
        # Use mock mode if API not available
        if not self.api_key_set or self.client is None:
            return self._get_mock_response(question)
        
        try:
            # Build the full prompt
            prompt_text = self._build_prompt(question, page_context, history)
            content: List[str|Image.Image] = [fig_to_png(fig) for fig in (chart_figures or [])]
            content+= [prompt_text] 

            # Send to Gemini with new API
            config = types.GenerateContentConfig( # type: ignore
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                system_instruction=self.system_prompt
            )
            
            response = self.client.models.generate_content(
                model=self.model_name_for_api,
                contents=content, # type: ignore
                config=config
            )
            
            # Extract text from response
            if response and response.text:
                return response.text
            else:
                return "⚠️ Risposta vuota dal modello. Riprova."
                
        except Exception as e:
            return self._handle_error(e)
    
    def _build_prompt(
        self,
        question: str,
        page_context: Optional[Dict[str, Any]],
        history: Optional[List[Dict[str, str]]],
    ) -> str:
        """
        Build the complete prompt with system instructions, context, and history.
        
        Args:
            question: The user's question
            page_context: Current page data
            history: Conversation history
        
        Returns:
            Complete formatted prompt string
        """
        parts = []
        
        # Page context (if provided)
        if page_context:
            context_str = to_json_string(page_context)
            parts.append("# CURRENT DATA CONTEXT\n")
            parts.append(f"The data is provided in JSON format, please read it carefully before responding\n{context_str}\n")

        # Conversation history (limited)
        if history:
            recent_history = history[-self.max_history:]
            if recent_history:
                history_str = self._format_history(recent_history)
                parts.append(f"# PREVIOUS CONVERSATION\n{history_str}\n")
        
        # Current question
        parts.append(f"# USER QUESTION\n\n{question}\n")
        parts.append("# YOUR ANSWER")
        
        return "\n".join(parts)

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        """
        Format conversation history for the prompt.
        
        Args:
            history: List of message dictionaries
        
        Returns:
            Formatted history string
        """
        lines = []
        for msg in history:
            role = "**User**" if msg.get("role") == "user" else "**Assistant**"
            content = msg.get("content", "")
            # Truncate long messages
            if len(content) > 300:
                content = content[:300] + "..."
            lines.append(f"\n{role}: {content}")
        return "\n".join(lines)
    
    def _get_mock_response(self, question: str) -> str:
        """
        Get a mock response when API is not available.
        
        Args:
            question: The user's question
        
        Returns:
            Mock response string
        """
        mock_template = getattr(config, 'GEMINI_MOCK_RESPONSE', MOCK_RESPONSE) if CONFIG_AVAILABLE else MOCK_RESPONSE # type: ignore
        return mock_template.format(question=question)
    
    def _handle_error(self, error: Exception) -> str:
        """
        Handle API errors with user-friendly messages.
        
        Args:
            error: The exception that occurred
        
        Returns:
            User-friendly error message
        """
        error_str = str(error).lower()
        
        if "quota" in error_str or "rate" in error_str:
            return "⚠️ **Limite API raggiunto**\n\nHai esaurito le richieste gratuite. Attendi qualche minuto o verifica la tua quota su Google AI Studio."
        
        if "invalid" in error_str and "key" in error_str:
            return "❌ **API Key non valida**\n\nVerifica che la chiave GEMINI_API_KEY sia corretta."
        
        if "not found" in error_str or "404" in error_str:
            return f"❌ **Modello non trovato**\n\nIl modello '{self.model_name}' non è disponibile. Verifica il nome in config.py."
        
        # Generic error
        return f"❌ **Errore API**\n\n```\n{str(error)[:200]}\n```\n\nRiprova tra qualche secondo."
    
    # =========================================================================
    # HISTORY MANAGEMENT
    # =========================================================================
    
    def add_to_history(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            role: Either 'user' or 'assistant'
            content: The message content
        """
        self.history.append({"role": role, "content": content})
        
        # Trim history if too long
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.history = []
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get a copy of the conversation history."""
        return self.history.copy()
    
    def set_history(self, history: List[Dict[str, str]]) -> None:
        """
        Set the conversation history.
        
        Args:
            history: List of message dictionaries
        """
        self.history = history[-self.max_history:] if history else []

# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_assistant_instance: Optional[GeminiAssistant] = None

def get_assistant() -> GeminiAssistant:
    """
    Get the singleton GeminiAssistant instance.
    
    Returns:
        The global GeminiAssistant instance
    """
    global _assistant_instance
    if _assistant_instance is None:
        _assistant_instance = GeminiAssistant()
    return _assistant_instance

def is_gemini_available() -> bool:
    """
    Check if Gemini API is properly configured and available.
    
    Returns:
        True if API is ready, False otherwise
    """
    assistant = get_assistant()
    return assistant.api_key_set and assistant.client is not None


def get_gemini_status() -> Dict[str, Any]:
    """
    Get detailed status information about Gemini configuration.
    
    Returns:
        Dictionary with status details
    """
    assistant = get_assistant()
    return {
        "library_installed": GENAI_AVAILABLE,
        "api_key_set": assistant.api_key_set,
        "model": assistant.model_name,
        "available": is_gemini_available(),
    }
