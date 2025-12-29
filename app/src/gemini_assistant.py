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
import json
from typing import Any, Dict, List, Optional

# Try to import python-dotenv for .env file support
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    load_dotenv = None

# Try to import config
try:
    import config
except ImportError:
    config = None

# Try to import Gemini library
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None


# =============================================================================
# CONFIGURATION DEFAULTS
# =============================================================================

DEFAULT_MODEL = "gemini-2.0-flash"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.7
DEFAULT_HISTORY_LENGTH = 14

# System prompt in Italian for financial analysis assistant
DEFAULT_SYSTEM_PROMPT = """Sei un assistente esperto di analisi finanziaria e IoT integrato in una dashboard di analytics.

CONTESTO APPLICAZIONE:
Questa è una dashboard universitaria che analizza dati finanziari (S&P 500, Gold, Oil, USD Index, Bitcoin) 
usando tecniche IoT per il rilevamento di anomalie e pattern recognition.

LE TUE COMPETENZE:
1. **Z-Score e Anomalie**: Sai spiegare come il Z-score misura le deviazioni standard dalla media.
   - Z > 3 o Z < -3 indica anomalie significative
   - Puoi interpretare cosa significa un'anomalia di prezzo vs volume vs volatilità

2. **Sliding Window**: Comprendi l'elaborazione real-time con finestre mobili tipica dei sistemi IoT.

3. **Correlazioni**: Sai interpretare matrici di correlazione e identificare relazioni tra asset.
   - Correlazione positiva: asset si muovono insieme
   - Correlazione negativa: asset si muovono in direzioni opposte
   - Gold e USD tipicamente negativamente correlati

4. **Pattern Recognition**: Conosci i pattern candlestick (Doji, Hammer, Engulfing) e chart patterns 
   (Double Top/Bottom, Head & Shoulders, Cup & Handle).

LINEE GUIDA RISPOSTE:
- Rispondi SEMPRE in italiano
- Sii conciso ma informativo (max 200 parole per risposta normale)
- Usa emoji per rendere le risposte più leggibili
- Quando ricevi dati contestuali, analizzali specificamente
- Se non hai dati sufficienti, chiedi all'utente di selezionare più opzioni nel menu "Dati da allegare"

COSA NON FARE:
- NON dare consigli di investimento specifici ("compra", "vendi")
- NON fare previsioni sul prezzo futuro
- NON inventare dati che non ti sono stati forniti
- Se non sai qualcosa, ammettilo onestamente

Sei pronto ad aiutare con l'analisi dei dati finanziari!"""

# Mock response for testing without API key
MOCK_RESPONSE = """**Modalità Demo**

Gemini non è configurato. Per attivare l'assistente:

1. Ottieni una API key gratuita da [Google AI Studio](https://aistudio.google.com/)
2. Imposta la variabile d'ambiente:
   ```
   export GEMINI_API_KEY="la-tua-chiave"
   ```
3. Riavvia l'applicazione

La tua domanda era: "{question}"

In modalità demo, posso comunque spiegarti i concetti base:
- **Z-Score**: Misura quante deviazioni standard un valore è dalla media
- **Anomalia**: Un valore con |Z| > 3 (molto raro, ~0.3% probabilità)
- **Correlazione**: Misura da -1 a +1 quanto due asset si muovono insieme"""


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
        model: The Gemini model instance (or None in mock mode)
        api_key_set: Whether a valid API key is configured
        history: List of conversation messages
    """
    
    def __init__(self):
        """Initialize the Gemini assistant with configuration from config.py or defaults."""
        self.model = None
        self.api_key_set = False
        self.history: List[Dict[str, str]] = []
        
        # Load configuration
        self.model_name = getattr(config, 'GEMINI_MODEL', DEFAULT_MODEL) if config else DEFAULT_MODEL
        self.max_tokens = getattr(config, 'GEMINI_MAX_TOKENS', DEFAULT_MAX_TOKENS) if config else DEFAULT_MAX_TOKENS
        self.temperature = getattr(config, 'GEMINI_TEMPERATURE', DEFAULT_TEMPERATURE) if config else DEFAULT_TEMPERATURE
        self.max_history = getattr(config, 'GEMINI_HISTORY_LENGTH', DEFAULT_HISTORY_LENGTH) if config else DEFAULT_HISTORY_LENGTH
        self.system_prompt = getattr(config, 'GEMINI_SYSTEM_PROMPT', DEFAULT_SYSTEM_PROMPT) if config else DEFAULT_SYSTEM_PROMPT
        
        # Initialize API if available
        self._initialize_api()
    
    def _initialize_api(self) -> None:
        """Initialize the Gemini API with the configured API key."""
        if not GENAI_AVAILABLE:
            return
        
        # Load .env file if available
        if DOTENV_AVAILABLE:
            load_dotenv()
        
        api_key = os.environ.get('GEMINI_API_KEY', '')
        
        if not api_key:
            return
        
        try:
            genai.configure(api_key=api_key)
            
            # Create model with generation config
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config,
            )
            
            self.api_key_set = True
            
        except Exception as e:
            print(f"[GeminiAssistant] API initialization error: {e}")
            self.model = None
            self.api_key_set = False
    
    def send_message(
        self,
        question: str,
        page_context: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Send a message to Gemini and get a response.
        
        Args:
            question: The user's question
            page_context: Dictionary with current page data and statistics
            history: Optional conversation history to include
        
        Returns:
            The assistant's response text
        """
        # Use mock mode if API not available
        if not self.api_key_set or self.model is None:
            return self._get_mock_response(question)
        
        try:
            # Build the full prompt
            prompt = self._build_prompt(question, page_context, history)
            
            # Send to Gemini
            response = self.model.generate_content(prompt)
            
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
        history: Optional[List[Dict[str, str]]]
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
        
        # System prompt
        parts.append(f"=== ISTRUZIONI SISTEMA ===\n{self.system_prompt}\n")
        
        # Page context (if provided)
        if page_context:
            context_str = self._format_context(page_context)
            parts.append(f"=== CONTESTO DATI CORRENTI ===\n{context_str}\n")
        
        # Conversation history (limited)
        if history:
            recent_history = history[-self.max_history:]
            if recent_history:
                history_str = self._format_history(recent_history)
                parts.append(f"=== CONVERSAZIONE PRECEDENTE ===\n{history_str}\n")
        
        # Current question
        parts.append(f"=== DOMANDA UTENTE ===\n{question}\n")
        parts.append("=== TUA RISPOSTA ===")
        
        return "\n".join(parts)
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """
        Format the page context as readable text for the LLM.
        
        Args:
            context: Dictionary with page data
        
        Returns:
            Formatted context string
        """
        lines = []
        
        # Basic info
        lines.append(f"Pagina: {context.get('page', 'N/A')}")
        lines.append(f"Asset: {context.get('asset_display', context.get('asset', 'N/A'))}")
        lines.append(f"Granularità: {context.get('granularity', 'N/A')}")
        
        # Period
        period = context.get('period', {})
        if period:
            lines.append(f"Periodo: {period.get('start', 'N/A')} → {period.get('end', 'N/A')}")
            if 'total_records' in period:
                lines.append(f"Record totali: {period['total_records']}")
        
        # Price statistics
        price_stats = context.get('price_statistics', {})
        if price_stats:
            lines.append("\nSTATISTICHE PREZZO:")
            for key, value in price_stats.items():
                lines.append(f"  - {key}: {value}")
        
        # Anomalies
        anomalies = context.get('anomalies', {})
        if anomalies:
            lines.append("\n⚠️ ANOMALIE RILEVATE:")
            if 'counts' in anomalies:
                counts = anomalies['counts']
                lines.append(f"  - Prezzo: {counts.get('price', 0)}")
                lines.append(f"  - Volume: {counts.get('volume', 0)}")
                lines.append(f"  - Volatilità: {counts.get('volatility', 0)}")
            if 'details' in anomalies and anomalies['details']:
                lines.append("  Dettagli (ultimi 10):")
                for a in anomalies['details'][:10]:
                    lines.append(f"    • {a.get('date', 'N/A')}: {a.get('type', 'N/A')} (Z={a.get('zscore', 'N/A')})")
        
        # Z-Score details
        zscore = context.get('zscore_details', {})
        if zscore:
            lines.append("\nZ-SCORE ATTUALI:")
            for key, value in zscore.items():
                lines.append(f"  - {key}: {value}")
        
        # Volume statistics
        volume_stats = context.get('volume_statistics', {})
        if volume_stats:
            lines.append("\nSTATISTICHE VOLUME:")
            for key, value in volume_stats.items():
                lines.append(f"  - {key}: {value}")
        
        # Volatility statistics
        volatility_stats = context.get('volatility_statistics', {})
        if volatility_stats:
            lines.append("\nSTATISTICHE VOLATILITÀ:")
            for key, value in volatility_stats.items():
                lines.append(f"  - {key}: {value}")
        
        # Simulation data (realtime page)
        simulation = context.get('simulation', {})
        if simulation:
            lines.append("\nSIMULAZIONE:")
            for key, value in simulation.items():
                lines.append(f"  - {key}: {value}")
        
        # Correlations (cross-asset page)
        correlations = context.get('correlations', {})
        if correlations:
            lines.append("\n🔗 CORRELAZIONI:")
            if 'matrix' in correlations:
                lines.append("  Matrice:")
                for pair, corr in correlations['matrix'].items():
                    lines.append(f"    • {pair}: {corr}")
        
        # Systemic events
        systemic = context.get('systemic_events', {})
        if systemic:
            lines.append("\n🌐 EVENTI SISTEMICI:")
            lines.append(f"  - Totale giorni: {systemic.get('total_days', 0)}")
            lines.append(f"  - Soglia: {systemic.get('threshold', 3)} asset")
            if 'events' in systemic:
                lines.append("  Eventi recenti:")
                for e in systemic['events'][:5]:
                    lines.append(f"    • {e.get('date', 'N/A')}: {e.get('assets', 'N/A')}")
        
        # Candlestick patterns
        candle_patterns = context.get('candlestick_patterns', {})
        if candle_patterns:
            lines.append("\n🕯️ PATTERN CANDLESTICK:")
            for pattern, count in candle_patterns.items():
                lines.append(f"  - {pattern}: {count}")
        
        # Chart patterns
        chart_patterns = context.get('chart_patterns', [])
        if chart_patterns:
            lines.append("\nPATTERN GRAFICI:")
            for p in chart_patterns[:5]:
                lines.append(f"  - {p.get('type', 'N/A')}: {p.get('start_date', '')} → {p.get('end_date', '')} ({p.get('signal', '')})")
        
        return "\n".join(lines)
    
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
            role = "Utente" if msg.get("role") == "user" else "Assistente"
            content = msg.get("content", "")
            # Truncate long messages
            if len(content) > 300:
                content = content[:300] + "..."
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
    
    def _get_mock_response(self, question: str) -> str:
        """
        Get a mock response when API is not available.
        
        Args:
            question: The user's question
        
        Returns:
            Mock response string
        """
        mock_template = getattr(config, 'GEMINI_MOCK_RESPONSE', MOCK_RESPONSE) if config else MOCK_RESPONSE
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
    return assistant.api_key_set and assistant.model is not None


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
    zscore_current: Optional[Dict[str, float]] = None,
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
