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
from io import BytesIO
import base64

# Import logger
from utils.logger import logger
from utils.serialization import to_json_string

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
# CHART CONVERSION UTILITIES
# =============================================================================

def fig_to_png(fig:Any) -> Image.Image:
    img_bytes = fig.to_image(
        format="png",
        width = 1200,
        height= 800,
        scale= 2.0
    )
    img = Image.open(BytesIO(img_bytes))
    return img

def fig_to_base64_image(
    fig: Any,
    format: str = 'png',
    width: int = 1200,
    height: int = 800,
    scale: float = 2.0
) -> Optional[Dict[str, Any]]:
    """
    Converti una figura Plotly in immagine base64 per Gemini Vision API.
    
    Args:
        fig: Plotly Figure object
        format: 'png' o 'jpeg'
        width: Larghezza in pixel
        height: Altezza in pixel  
        scale: Fattore di scala per qualità (2 = retina)
    
    Returns:
        Dict con mime_type e data, None se errore
    """
    if not KALEIDO_AVAILABLE:
        logger.warning("Kaleido not available. Cannot export chart.")
        return None
    
    try:
        # Export figura a bytes usando kaleido
        img_bytes = fig.to_image(
            format=format,
            width=width,
            height=height,
            scale=scale
        )
        
        # Converti in base64
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # Determina mime type
        mime_type = f"image/{format}"
        
        # Formato compatibile con genai
        return {
            "mime_type": mime_type,
            "data": img_b64
        }
        
    except Exception as e:
        logger.error(f"Error converting figure to image: {e}")
        return None


def prepare_multimodal_content(
    text_prompt: str,
    chart_images: List[Dict[str, Any]]
) -> List[Any]:
    """
    Prepara contenuto per Gemini multimodal (testo + immagini).
    
    Args:
        text_prompt: Il prompt testuale
        chart_images: Lista di dict da fig_to_base64_image()
    
    Returns:
        Lista di parti per generate_content()
    """
    if not chart_images or len(chart_images) == 0:
        return [text_prompt]
    
    # Inizia con il testo
    parts = [text_prompt]
    
    # Aggiungi ogni immagine
    for i, img_data in enumerate(chart_images):
        if img_data and 'mime_type' in img_data and 'data' in img_data:
            # Gemini accetta dict con mime_type e data
            parts.append({ # type: ignore
                "mime_type": img_data["mime_type"],
                "data": img_data["data"]
            })
            
            # Aggiungi separator tra immagini multiple
            if i < len(chart_images) - 1:
                parts.append(f"\n--- Chart {i+2} ---\n")
    
    return parts

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
            prompt_text = self._build_prompt(question, page_context, history, chart_figures)
            content: List[str|Image.Image] = [fig_to_png(fig) for fig in (chart_figures or [])]
            content+= [prompt_text] 

            # Send to Gemini with new API
            config = types.GenerateContentConfig( # type: ignore
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                system_instruction=self.system_prompt
            )

            logger.info(f"{content}")
            
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
        chart_figures: Optional[List] = None
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
        # Prepare content (text-only or multimodal)
        # if chart_figures and len(chart_figures) > 0:
        #     # Multimodal: convert charts to images
        #     chart_images = []
        #     for fig in chart_figures[:5]:  # Limit to 5 charts max
        #         img_data = fig_to_base64_image(fig)
        #         if img_data:
        #             chart_images.append(img_data)
            
        #     if len(chart_images) > 0:
        #         # Use multimodal content
        #         parts.append("Moreover, these base64 encoded charts are provided to help you analyze the data:\n")
        #         parts.extend(to_json_string(x) for x in prepare_multimodal_content("", chart_images))

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


# =============================================================================
# CONTEXT BUILDERS
# =============================================================================

def build_single_asset_context(
    asset: str,
    asset_display: str,
    granularity: str,
    start_date: str,
    end_date: str,
    total_records: int,
    price_stats: Dict[str, Any],
    anomaly_counts: Dict[str, int],
    anomaly_details: List[Dict[str, Any]],
    zscore_current: Optional[Dict[str, str]] = None,
    volume_stats: Optional[Dict[str, Any]] = None,
    volatility_stats: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Build comprehensive context for the Single Asset Analysis page.
    
    Args:
        asset: Internal asset key
        asset_display: Display name of the asset
        granularity: Data granularity (minute/hourly/daily)
        start_date: Start date of the analysis period
        end_date: End date of the analysis period
        total_records: Total number of data points
        price_stats: Price statistics dictionary
        anomaly_counts: Dictionary with anomaly counts by type
        anomaly_details: List of anomaly detail dictionaries
        zscore_current: Current Z-scores (optional)
        volume_stats: Volume statistics (optional)
        volatility_stats: Volatility statistics (optional)
    
    Returns:
        Complete context dictionary
    """
    context = {
        "page": "Single Asset Analysis",
        "asset": asset,
        "asset_display": asset_display,
        "granularity": granularity,
        "period": {
            "start": start_date,
            "end": end_date,
            "total_records": total_records
        },
        "price_statistics": price_stats,
        "anomalies": {
            "counts": anomaly_counts,
            "details": anomaly_details
        }
    }
    
    if zscore_current:
        context["zscore_details"] = zscore_current
    
    if volume_stats:
        context["volume_statistics"] = volume_stats
    
    if volatility_stats:
        context["volatility_statistics"] = volatility_stats
    
    return context


def build_realtime_context(
    asset: str,
    asset_display: str,
    simulation_day: str,
    window_size: int,
    zscore_threshold: float,
    progress_pct: float,
    points_streamed: int,
    total_points: int,
    anomalies_found: int,
    anomaly_list: List[Dict[str, Any]],
    current_price: Optional[float] = None,
    current_zscore: Optional[float] = None,
    window_stats: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Build context for the Real-time IoT Simulation page.
    
    Args:
        asset: Internal asset key
        asset_display: Display name
        simulation_day: The day being simulated
        window_size: Sliding window size
        zscore_threshold: Anomaly detection threshold
        progress_pct: Simulation progress percentage
        points_streamed: Number of points processed
        total_points: Total points in the day
        anomalies_found: Count of anomalies detected
        anomaly_list: List of anomaly details
        current_price: Current price (optional)
        current_zscore: Current Z-score (optional)
        window_stats: Window statistics (optional)
    
    Returns:
        Complete context dictionary
    """
    context = {
        "page": "Real-time IoT Simulation",
        "asset": asset,
        "asset_display": asset_display,
        "granularity": "minute",
        "period": {
            "day": simulation_day,
        },
        "simulation": {
            "window_size": window_size,
            "zscore_threshold": zscore_threshold,
            "progress_pct": f"{progress_pct:.1f}%",
            "points_streamed": points_streamed,
            "total_points": total_points,
            "current_price": f"${current_price:.2f}" if current_price else "N/A",
            "current_zscore": f"{current_zscore:.2f}σ" if current_zscore else "N/A"
        },
        "realtime_anomalies": anomaly_list,
        "anomalies": {
            "counts": {"total": anomalies_found},
            "details": anomaly_list[:10]
        }
    }
    
    if window_stats:
        context["window_statistics"] = window_stats
    
    return context


def build_cross_asset_context(
    start_date: str,
    end_date: str,
    correlation_matrix: Dict[str, float],
    systemic_events: Dict[str, Any],
    pair_name: Optional[str] = None,
    pair_analysis: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Build context for the Cross-Asset Analysis page.
    
    Args:
        start_date: Start date
        end_date: End date
        correlation_matrix: Dictionary of asset pair correlations
        systemic_events: Systemic event statistics
        pair_name: Selected pair name (optional)
        pair_analysis: Detailed pair analysis (optional)
    
    Returns:
        Complete context dictionary
    """
    context = {
        "page": "Cross-Asset Analysis",
        "asset": "Multiple",
        "asset_display": "All Assets",
        "granularity": "daily",
        "period": {
            "start": start_date,
            "end": end_date
        },
        "correlations": {
            "matrix": correlation_matrix
        },
        "systemic_events": systemic_events
    }
    
    if pair_name and pair_analysis:
        context["pair_analysis"] = {
            "pair": pair_name,
            **pair_analysis
        }
    
    return context


def build_pattern_context(
    asset: str,
    asset_display: str,
    start_date: str,
    end_date: str,
    candlestick_counts: Dict[str, int],
    chart_patterns: List[Dict[str, Any]],
    pattern_distribution: Optional[Dict[str, int]] = None
) -> Dict[str, Any]:
    """
    Build context for the Pattern Recognition page.
    
    Args:
        asset: Internal asset key
        asset_display: Display name
        start_date: Start date
        end_date: End date
        candlestick_counts: Counts of each candlestick pattern
        chart_patterns: List of detected chart patterns
        pattern_distribution: Pattern frequency distribution (optional)
    
    Returns:
        Complete context dictionary
    """
    context = {
        "page": "Pattern Recognition",
        "asset": asset,
        "asset_display": asset_display,
        "granularity": "daily",
        "period": {
            "start": start_date,
            "end": end_date
        },
        "candlestick_patterns": candlestick_counts,
        "chart_patterns": chart_patterns
    }
    
    if pattern_distribution:
        context["pattern_distribution"] = pattern_distribution
    
    return context

def filter_context(
    full_context: Dict[str, Any], 
    selected_keys: List[str]
) -> Dict[str, Any]:
    """
    Filter the full context to include only selected data sections.
    
    Always includes: page, asset, period, basic info
    Conditionally includes: detailed statistics based on selection
    
    Args:
        full_context: Complete page context dictionary
        selected_keys: List of selected data option keys
    
    Returns:
        Filtered context dictionary
    """
    # Base context always included
    filtered = {
        "page": full_context.get("page", "Unknown"),
        "asset": full_context.get("asset", "Unknown"),
        "asset_display": full_context.get("asset_display", "Unknown"),
        "granularity": full_context.get("granularity", "daily"),
        "period": full_context.get("period", {}),
    }
    
    # Map of keys to context fields
    key_mapping = {
        "price_stats": "price_statistics",
        "anomalies": "anomalies",
        "zscore_details": "zscore_details",
        "volume_stats": "volume_statistics",
        "volatility_stats": "volatility_statistics",
        "simulation_progress": "simulation",
        "realtime_anomalies": "realtime_anomalies",
        "window_stats": "window_statistics",
        "correlation_matrix": "correlations",
        "systemic_events": "systemic_events",
        "pair_analysis": "pair_analysis",
        "candlestick_patterns": "candlestick_patterns",
        "chart_patterns": "chart_patterns",
        "pattern_distribution": "pattern_distribution",
    }
    
    for key in selected_keys:
        if key in key_mapping:
            field = key_mapping[key]
            if field in full_context:
                filtered[field] = full_context[field]
    
    return filtered