"""
Gemini Assistant Module for IoT Financial Data Analytics.

This module handles all interactions with Google's Gemini AI API, providing:
- Client initialization and configuration
- Message sending with multimodal support (text + images)
- Conversation history management
- Page context building for contextual responses
- Chart capture and encoding for visual analysis

The module supports a "mock mode" for UI testing without an API key.

Usage:
    from src.gemini_assistant import GeminiAssistant
    
    assistant = GeminiAssistant()
    response = assistant.send_message(
        question="What does this anomaly mean?",
        page_context={"page": "Single Asset", "asset": "sp500"},
        image_base64=None  # Optional: chart image
    )
"""

import base64
import io
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Try to import Google Generative AI library
# Gracefully handle missing dependency
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None

# Try to import dotenv for .env file support
# This is optional - environment variables work without it
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if present
except ImportError:
    pass  # dotenv not installed, skip


# =============================================================================
# GEMINI ASSISTANT CLASS
# =============================================================================

class GeminiAssistant:
    """
    Main class for interacting with the Gemini AI API.
    
    Handles initialization, message sending, and conversation management.
    Supports both text-only and multimodal (text + image) interactions.
    
    Attributes:
        client: The Gemini generative model instance (or None in mock mode)
        is_mock_mode: Whether the assistant is running without a real API key
        history: List of conversation messages
    
    Example:
        >>> assistant = GeminiAssistant()
        >>> if assistant.is_mock_mode:
        ...     print("Running in mock mode - configure API key for real responses")
        >>> response = assistant.send_message("Explain Z-score")
    """
    
    def __init__(self):
        """
        Initialize the Gemini Assistant.
        
        Attempts to configure the Gemini API client using the API key from
        environment variables. Falls back to mock mode if the key is missing
        or the library is not installed.
        """
        self.client = None
        self.is_mock_mode = True
        self.history: List[Dict[str, str]] = []
        
        # Attempt to initialize the real client
        self._init_client()
    
    def _init_client(self) -> None:
        """
        Initialize the Gemini API client.
        
        Reads the API key from the environment variable specified in config.
        Sets is_mock_mode to False only if initialization succeeds.
        """
        # Check if the library is available
        if not GENAI_AVAILABLE:
            self._log_warning(
                "google-generativeai library not installed. "
                "Run: pip install google-generativeai"
            )
            return
        
        # Get API key from environment
        api_key = os.environ.get(config.GEMINI_API_KEY_ENV)
        
        if not api_key:
            self._log_warning(
                f"API key not found in environment variable '{config.GEMINI_API_KEY_ENV}'. "
                "Running in mock mode."
            )
            return
        
        try:
            # Configure the API with the key
            genai.configure(api_key=api_key)
            
            # Initialize the generative model
            self.client = genai.GenerativeModel(
                model_name=config.GEMINI_MODEL,
                generation_config={
                    "max_output_tokens": config.GEMINI_MAX_TOKENS,
                    "temperature": config.GEMINI_TEMPERATURE,
                }
            )
            
            self.is_mock_mode = False
            self._log_info(f"Gemini client initialized successfully with model: {config.GEMINI_MODEL}")
            
        except Exception as e:
            self._log_warning(f"Failed to initialize Gemini client: {str(e)}")
            self.client = None
            self.is_mock_mode = True
    
    # =========================================================================
    # MESSAGE SENDING
    # =========================================================================
    
    def send_message(
        self,
        question: str,
        page_context: Optional[Dict[str, Any]] = None,
        image_base64: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Send a message to Gemini and get a response.
        
        Builds a complete prompt including system instructions, page context,
        conversation history, and the user's question. Optionally includes
        an image for visual analysis.
        
        Args:
            question: The user's question or message
            page_context: Dictionary with current page information (asset, dates, etc.)
            image_base64: Optional base64-encoded PNG image of a chart
            history: Optional conversation history (uses internal history if not provided)
        
        Returns:
            The assistant's response as a string
        
        Example:
            >>> response = assistant.send_message(
            ...     question="Why is this point red?",
            ...     page_context={"page": "Single Asset", "asset": "gold"},
            ...     image_base64=chart_image_base64
            ... )
        """
        # Use mock mode if client not available
        if self.is_mock_mode:
            return self._get_mock_response(question)
        
        try:
            # Build the complete prompt
            prompt_parts = self._build_prompt_parts(
                question=question,
                page_context=page_context,
                image_base64=image_base64,
                history=history
            )
            
            # Send to Gemini API
            response = self.client.generate_content(prompt_parts)
            
            # Extract text from response
            if response and response.text:
                return response.text
            else:
                return "Mi dispiace, non sono riuscito a generare una risposta. Riprova."
                
        except Exception as e:
            error_message = str(e)
            self._log_warning(f"Gemini API error: {error_message}")
            
            # Return user-friendly error message
            if "quota" in error_message.lower():
                return (
                    "⚠️ **Quota API esaurita**\n\n"
                    "Hai raggiunto il limite di richieste gratuite. "
                    "Riprova tra qualche minuto o domani."
                )
            elif "invalid" in error_message.lower() and "key" in error_message.lower():
                return (
                    "⚠️ **API Key non valida**\n\n"
                    "La chiave API configurata non è valida. "
                    "Verifica la configurazione."
                )
            else:
                return (
                    f"⚠️ **Errore API**\n\n"
                    f"Si è verificato un errore: {error_message}\n\n"
                    "Riprova tra qualche istante."
                )
    
    def _build_prompt_parts(
        self,
        question: str,
        page_context: Optional[Dict[str, Any]] = None,
        image_base64: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> List[Any]:
        """
        Build the complete prompt parts for the Gemini API.
        
        Assembles all components into a single prompt:
        1. System prompt (assistant personality and knowledge)
        2. Page context (current dashboard state)
        3. Conversation history (recent messages)
        4. Current question
        5. Optional image
        
        Args:
            question: The user's current question
            page_context: Dictionary with page-specific information
            image_base64: Optional base64-encoded image
            history: Conversation history
        
        Returns:
            List of prompt parts ready for the Gemini API
        """
        parts = []
        
        # 1. System prompt
        system_prompt = self.get_system_prompt()
        parts.append(system_prompt)
        
        # 2. Page context
        if page_context:
            context_str = self.build_page_context(page_context)
            parts.append(f"\n\n## CONTESTO PAGINA CORRENTE\n{context_str}")
        
        # 3. Conversation history
        history_to_use = history if history is not None else self.history
        if history_to_use:
            history_str = self._format_history(history_to_use)
            if history_str:
                parts.append(f"\n\n## CONVERSAZIONE PRECEDENTE\n{history_str}")
        
        # 4. Current question
        parts.append(f"\n\n## DOMANDA UTENTE\n{question}")
        
        # 5. Image (if provided)
        if image_base64:
            try:
                image_data = self._decode_base64_image(image_base64)
                parts.append(image_data)
                parts.append("\n\n[L'utente ha allegato un'immagine del grafico corrente. Analizzala nel contesto della domanda.]")
            except Exception as e:
                self._log_warning(f"Failed to decode image: {str(e)}")
        
        return parts
    
    def _format_history(self, history: List[Dict[str, str]]) -> str:
        """
        Format conversation history as a string for the prompt.
        
        Takes the last N messages (defined by GEMINI_HISTORY_LENGTH) and
        formats them as a readable conversation.
        
        Args:
            history: List of message dictionaries with 'role' and 'content' keys
        
        Returns:
            Formatted history string
        """
        if not history:
            return ""
        
        # Take only the last N messages
        recent_history = history[-config.GEMINI_HISTORY_LENGTH:]
        
        formatted_lines = []
        for msg in recent_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "user":
                formatted_lines.append(f"**Utente**: {content}")
            else:
                formatted_lines.append(f"**Assistente**: {content}")
        
        return "\n\n".join(formatted_lines)
    
    def _decode_base64_image(self, image_base64: str) -> Dict[str, Any]:
        """
        Decode a base64 image string into a format suitable for Gemini.
        
        Args:
            image_base64: Base64-encoded PNG image string
        
        Returns:
            Dictionary with image data for the Gemini API
        """
        # Remove data URL prefix if present
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        
        # Decode base64 to bytes
        image_bytes = base64.b64decode(image_base64)
        
        # Return in format expected by Gemini
        return {
            "mime_type": "image/png",
            "data": image_bytes
        }
    
    def _get_mock_response(self, question: str) -> str:
        """
        Generate a mock response when running without an API key.
        
        Useful for testing the UI without consuming API quota.
        
        Args:
            question: The user's question
        
        Returns:
            Mock response string with setup instructions
        """
        return config.GEMINI_MOCK_RESPONSE.format(question=question)
    
    # =========================================================================
    # CONTEXT BUILDING
    # =========================================================================
    
    @staticmethod
    def get_system_prompt() -> str:
        """
        Get the system prompt that defines the assistant's behavior.
        
        Returns:
            The system prompt string from configuration
        """
        return config.GEMINI_SYSTEM_PROMPT
    
    @staticmethod
    def build_page_context(context_dict: Dict[str, Any]) -> str:
        """
        Build a formatted context string from a dictionary.
        
        Converts the page context dictionary into a human-readable string
        that helps Gemini understand the current state of the dashboard.
        
        Args:
            context_dict: Dictionary with page-specific information
                Expected keys vary by page:
                - "page": Current page name
                - "asset": Selected asset (if applicable)
                - "granularity": Data granularity
                - "date_range": Selected date range
                - "anomalies_count": Number of detected anomalies
                - "patterns_found": Detected patterns (for pattern page)
                - etc.
        
        Returns:
            Formatted context string
        
        Example:
            >>> context = {"page": "Single Asset", "asset": "gold", "anomalies_count": 5}
            >>> print(GeminiAssistant.build_page_context(context))
            - **Pagina**: Single Asset
            - **Asset**: gold
            - **Anomalies Count**: 5
        """
        if not context_dict:
            return "Nessun contesto disponibile."
        
        lines = []
        
        # Define friendly names for context keys (Italian)
        key_names = {
            "page": "Pagina",
            "asset": "Asset selezionato",
            "asset_display": "Asset",
            "granularity": "Granularità",
            "date_range": "Periodo",
            "start_date": "Data inizio",
            "end_date": "Data fine",
            "zscore_threshold": "Soglia Z-score",
            "anomalies_count": "Anomalie rilevate",
            "anomalies_price": "Anomalie prezzo",
            "anomalies_volume": "Anomalie volume",
            "anomalies_volatility": "Anomalie volatilità",
            "window_size": "Finestra sliding window",
            "simulation_progress": "Progresso simulazione",
            "correlation_window": "Finestra correlazione",
            "systemic_threshold": "Soglia eventi sistemici",
            "systemic_events": "Eventi sistemici rilevati",
            "selected_pair": "Coppia selezionata",
            "patterns_doji": "Pattern Doji",
            "patterns_hammer": "Pattern Hammer",
            "patterns_engulfing_bullish": "Pattern Engulfing Bullish",
            "patterns_engulfing_bearish": "Pattern Engulfing Bearish",
            "chart_patterns_count": "Chart pattern rilevati",
            "tolerance": "Tolleranza pattern",
            "prominence": "Prominenza picchi"
        }
        
        for key, value in context_dict.items():
            # Get friendly name or use the key with title case
            friendly_name = key_names.get(key, key.replace("_", " ").title())
            
            # Format value based on type
            if isinstance(value, dict):
                # Nested dictionary: format as sub-items
                value_str = ", ".join(f"{k}: {v}" for k, v in value.items())
            elif isinstance(value, (list, tuple)):
                value_str = ", ".join(str(v) for v in value)
            elif isinstance(value, float):
                value_str = f"{value:.2f}"
            else:
                value_str = str(value)
            
            lines.append(f"- **{friendly_name}**: {value_str}")
        
        return "\n".join(lines)
    
    # =========================================================================
    # HISTORY MANAGEMENT
    # =========================================================================
    
    def add_to_history(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            role: Either "user" or "assistant"
            content: The message content
        """
        self.history.append({
            "role": role,
            "content": content
        })
        
        # Trim history if it exceeds the maximum length
        max_length = config.GEMINI_HISTORY_LENGTH
        if len(self.history) > max_length:
            self.history = self.history[-max_length:]
    
    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.history = []
    
    def get_history(self) -> List[Dict[str, str]]:
        """
        Get the current conversation history.
        
        Returns:
            List of message dictionaries
        """
        return self.history.copy()
    
    def set_history(self, history: List[Dict[str, str]]) -> None:
        """
        Set the conversation history (e.g., from session state).
        
        Args:
            history: List of message dictionaries
        """
        self.history = history.copy() if history else []
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    @staticmethod
    def _log_info(message: str) -> None:
        """Log an informational message."""
        print(f"[GeminiAssistant INFO] {message}")
    
    @staticmethod
    def _log_warning(message: str) -> None:
        """Log a warning message."""
        print(f"[GeminiAssistant WARNING] {message}")


# =============================================================================
# CHART CAPTURE UTILITIES
# =============================================================================

def capture_plotly_figure(fig, format: str = "png", scale: int = 2) -> Optional[str]:
    """
    Capture a Plotly figure as a base64-encoded image.
    
    Converts a Plotly figure to a PNG image and encodes it as base64,
    ready to be sent to Gemini for visual analysis.
    
    Args:
        fig: Plotly figure object
        format: Image format (default: "png")
        scale: Image scale factor for higher resolution (default: 2)
    
    Returns:
        Base64-encoded image string, or None if capture fails
    
    Example:
        >>> import plotly.graph_objects as go
        >>> fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))
        >>> image_b64 = capture_plotly_figure(fig)
        >>> if image_b64:
        ...     response = assistant.send_message("Describe this chart", image_base64=image_b64)
    """
    try:
        # Convert figure to bytes
        image_bytes = fig.to_image(format=format, scale=scale)
        
        # Encode as base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        
        return image_base64
        
    except Exception as e:
        print(f"[GeminiAssistant WARNING] Failed to capture figure: {str(e)}")
        print("Make sure 'kaleido' is installed: pip install kaleido")
        return None


def is_gemini_available() -> bool:
    """
    Check if Gemini is properly configured and available.
    
    Returns:
        True if API key is set and library is available, False otherwise
    """
    if not GENAI_AVAILABLE:
        return False
    
    api_key = os.environ.get(config.GEMINI_API_KEY_ENV)
    return bool(api_key)


def get_gemini_status() -> Dict[str, Any]:
    """
    Get detailed status information about Gemini configuration.
    
    Useful for debugging and displaying status in the UI.
    
    Returns:
        Dictionary with status information:
        - library_installed: Whether google-generativeai is installed
        - api_key_set: Whether the API key environment variable is set
        - model: The configured model name
        - is_available: Whether Gemini is fully available
    """
    api_key = os.environ.get(config.GEMINI_API_KEY_ENV, "")
    
    return {
        "library_installed": GENAI_AVAILABLE,
        "api_key_set": bool(api_key),
        "api_key_preview": f"{api_key[:8]}..." if len(api_key) > 8 else "(not set)",
        "model": config.GEMINI_MODEL,
        "max_tokens": config.GEMINI_MAX_TOKENS,
        "temperature": config.GEMINI_TEMPERATURE,
        "history_length": config.GEMINI_HISTORY_LENGTH,
        "is_available": GENAI_AVAILABLE and bool(api_key)
    }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global assistant instance (lazy initialization)
_assistant_instance: Optional[GeminiAssistant] = None


def get_assistant() -> GeminiAssistant:
    """
    Get the global GeminiAssistant instance.
    
    Uses lazy initialization to create the assistant only when first needed.
    Subsequent calls return the same instance.
    
    Returns:
        GeminiAssistant instance
    
    Example:
        >>> assistant = get_assistant()
        >>> response = assistant.send_message("Hello!")
    """
    global _assistant_instance
    
    if _assistant_instance is None:
        _assistant_instance = GeminiAssistant()
    
    return _assistant_instance


def reset_assistant() -> None:
    """
    Reset the global assistant instance.
    
    Forces re-initialization on next get_assistant() call.
    Useful if API key is changed during runtime.
    """
    global _assistant_instance
    _assistant_instance = None


# =============================================================================
# PAGE CONTEXT BUILDERS
# =============================================================================
# These functions help build page-specific context dictionaries

def build_single_asset_context(
    asset: str,
    asset_display: str,
    granularity: str,
    start_date: str,
    end_date: str,
    zscore_threshold: float,
    anomalies_price: int,
    anomalies_volume: int,
    anomalies_volatility: int
) -> Dict[str, Any]:
    """
    Build context dictionary for the Single Asset Analysis page.
    
    Args:
        asset: Asset key (e.g., "sp500")
        asset_display: Display name (e.g., "S&P 500")
        granularity: Data granularity ("minute", "hourly", "daily")
        start_date: Start of date range
        end_date: End of date range
        zscore_threshold: Current Z-score threshold
        anomalies_price: Count of price anomalies
        anomalies_volume: Count of volume anomalies
        anomalies_volatility: Count of volatility anomalies
    
    Returns:
        Context dictionary
    """
    return {
        "page": "Single Asset Analysis",
        "asset": asset,
        "asset_display": asset_display,
        "granularity": granularity,
        "date_range": f"{start_date} → {end_date}",
        "zscore_threshold": zscore_threshold,
        "anomalies_price": anomalies_price,
        "anomalies_volume": anomalies_volume,
        "anomalies_volatility": anomalies_volatility,
        "anomalies_count": anomalies_price + anomalies_volume + anomalies_volatility
    }


def build_realtime_context(
    asset: str,
    asset_display: str,
    selected_day: str,
    window_size: int,
    zscore_threshold: float,
    simulation_progress: float,
    anomalies_found: int
) -> Dict[str, Any]:
    """
    Build context dictionary for the Real-time IoT Simulation page.
    
    Args:
        asset: Asset key
        asset_display: Display name
        selected_day: The day being simulated
        window_size: Sliding window size
        zscore_threshold: Current Z-score threshold
        simulation_progress: Progress percentage (0-100)
        anomalies_found: Number of anomalies detected so far
    
    Returns:
        Context dictionary
    """
    return {
        "page": "Real-time IoT Simulation",
        "asset": asset,
        "asset_display": asset_display,
        "selected_day": selected_day,
        "window_size": window_size,
        "zscore_threshold": zscore_threshold,
        "simulation_progress": f"{simulation_progress:.1f}%",
        "anomalies_found": anomalies_found
    }


def build_cross_asset_context(
    start_date: str,
    end_date: str,
    correlation_window: int,
    systemic_threshold: int,
    systemic_events: int,
    selected_pair: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build context dictionary for the Cross-Asset Analysis page.
    
    Args:
        start_date: Start of date range
        end_date: End of date range
        correlation_window: Rolling correlation window size
        systemic_threshold: Threshold for systemic events
        systemic_events: Number of systemic events detected
        selected_pair: Currently selected asset pair (if any)
    
    Returns:
        Context dictionary
    """
    context = {
        "page": "Cross-Asset Analysis",
        "date_range": f"{start_date} → {end_date}",
        "correlation_window": f"{correlation_window} giorni",
        "systemic_threshold": f"{systemic_threshold} asset",
        "systemic_events": systemic_events
    }
    
    if selected_pair:
        context["selected_pair"] = selected_pair
    
    return context


def build_pattern_context(
    asset: str,
    asset_display: str,
    start_date: str,
    end_date: str,
    tolerance: float,
    prominence: float,
    patterns_doji: int,
    patterns_hammer: int,
    patterns_engulfing_bullish: int,
    patterns_engulfing_bearish: int,
    chart_patterns_count: int
) -> Dict[str, Any]:
    """
    Build context dictionary for the Pattern Recognition page.
    
    Args:
        asset: Asset key
        asset_display: Display name
        start_date: Start of date range
        end_date: End of date range
        tolerance: Price tolerance percentage
        prominence: Peak prominence percentage
        patterns_doji: Count of Doji patterns
        patterns_hammer: Count of Hammer patterns
        patterns_engulfing_bullish: Count of Bullish Engulfing patterns
        patterns_engulfing_bearish: Count of Bearish Engulfing patterns
        chart_patterns_count: Total count of chart patterns
    
    Returns:
        Context dictionary
    """
    return {
        "page": "Pattern Recognition",
        "asset": asset,
        "asset_display": asset_display,
        "date_range": f"{start_date} → {end_date}",
        "tolerance": f"{tolerance}%",
        "prominence": f"{prominence}%",
        "patterns_doji": patterns_doji,
        "patterns_hammer": patterns_hammer,
        "patterns_engulfing_bullish": patterns_engulfing_bullish,
        "patterns_engulfing_bearish": patterns_engulfing_bearish,
        "chart_patterns_count": chart_patterns_count
    }
