from .ui import PageType


# =============================================================================
# DATA ATTACHMENT CONFIGURATION
# =============================================================================

# Available data options for each page type
DATA_OPTIONS = {
    PageType.SINGLE_ASSET: {
        "price_stats": {
            "label": "Statistiche prezzo",
            "description": "Min, max, current, % change",
            "default": True
        },
        "anomalies": {
            "label": "⚠️ Lista anomalie",
            "description": "All detected anomalies with details",
            "default": True
        },
        "zscore_details": {
            "label": "Dettagli Z-Score",
            "description": "Current Z-scores for price, volume, volatility",
            "default": False
        },
        "volume_stats": {
            "label": "Statistiche volume",
            "description": "Volume statistics and trends",
            "default": False
        },
        "volatility_stats": {
            "label": "Statistiche volatilità",
            "description": "Volatility range and patterns",
            "default": False
        }
    },
    PageType.REALTIME: {
        "simulation_progress": {
            "label": "Progresso simulazione",
            "description": "Current progress and points streamed",
            "default": True
        },
        "realtime_anomalies": {
            "label": "⚠️ Anomalie rilevate",
            "description": "Anomalies found during simulation",
            "default": True
        },
        "window_stats": {
            "label": "Statistiche finestra",
            "description": "Rolling window statistics",
            "default": False
        }
    },
    PageType.CROSS_ASSET: {
        "correlation_matrix": {
            "label": "🔗 Matrice correlazioni",
            "description": "Full correlation matrix between assets",
            "default": True
        },
        "systemic_events": {
            "label": "🌐 Eventi sistemici",
            "description": "Days with multiple asset anomalies",
            "default": True
        },
        "pair_analysis": {
            "label": "Analisi coppia",
            "description": "Selected pair detailed statistics",
            "default": False
        }
    },
    PageType.PATTERNS: {
        "candlestick_patterns": {
            "label": "🕯️ Pattern candlestick",
            "description": "Doji, Hammer, Engulfing patterns",
            "default": True
        },
        "chart_patterns": {
            "label": "Pattern grafici",
            "description": "Double Top/Bottom, H&S, Cup & Handle",
            "default": True
        },
        "pattern_distribution": {
            "label": "Distribuzione",
            "description": "Pattern frequency and timeline",
            "default": False
        }
    }
}

# Page-specific suggested questions for welcome message
PAGE_SUGGESTIONS = {
    PageType.SINGLE_ASSET: [
        "Cosa significa Z-score?",
        "Spiega le anomalie rilevate",
        "Analizza il trend del prezzo"
    ],
    PageType.REALTIME: [
        "Come funziona la sliding window?",
        "Spiega le anomalie in tempo reale",
        "Cosa indica la simulazione?"
    ],
    PageType.CROSS_ASSET: [
        "Spiega la matrice di correlazione",
        "Cosa sono gli eventi sistemici?",
        "Analizza le relazioni tra asset"
    ],
    PageType.PATTERNS: [
        "Cosa indica un pattern Doji?",
        "Spiega i pattern rilevati",
        "Qual è il segnale più importante?"
    ]
}
