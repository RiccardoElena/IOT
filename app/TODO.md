# Refactoring Roadmap

## FASE 1: Cleanup e Standardizzazione (Quick Wins) ✅

- [x] **Rimuovere codice morto**
  - [x] Eliminare commenti debug da `config.py:13-18`
  - [x] Cercare altri blocchi commentati
- [x] **Centralizzare costanti**
  - [x] Creare `src/constants.py` con:
    - [x] `PAGE_TYPES = ["single_asset", "realtime", "cross_asset", "patterns"]`
    - [x] `DEFAULT_CHART_CONFIG = {...}`
    - [x] Magic numbers (heights, distances, etc.)
  - [x] Aggiornare tutti i riferimenti
- [x] **Standardizzare imports**
  - [x] Creare `__init__.py` che configura path
  - [x] Rimuovere `sys.path.insert` ripetuto da tutti i file
  - [x] Usare imports relativi dove possibile
- [x] **Unificare error handling**
  - [x] Creare `src/utils/logger.py` con logger configurato
  - [x] Sostituire `print()` con `logger.warning/error/info`
  - [x] Aggiungere try-catch consistenti
- [x] **Aggiungere type hints mancanti**
  - [x] Completare hints in tutte le pagine
  - [x] Aggiungere `from __future__ import annotations` dove serve
  - [x] Configurare mypy per validazione

## FASE 2: Modularizzazione Components

- [x] **Dividere `components.py` in moduli**
  - [x] `src/ui/page_components.py` → title, footer
  - [x] `src/ui/gemini/chat_ui.py` → render_chat_messages, render_chat_input
  - [x] `src/ui/gemini/sidebar.py` → render_gemini_sidebar
  - [x] `src/ui/gemini/session_manager.py` → session state management
  - [x] `src/ui/gemini/chart_manager.py` → chart selection logic
  - [x] `src/ui/gemini/data_config.py` → DATA_OPTIONS, PAGE_SUGGESTIONS
- [ ] **Estrarre logica session state**
  - [ ] Creare `src/state/session_manager.py`:
    - [ ] `class PageState` con metodi `init()`, `reset()`, `update()`
    - [ ] Factory per ogni tipo di pagina
    - [ ] Rimuovere duplicazione dalle pagine
- [ ] **Creare componenti riusabili**
  - [ ] `src/ui/controls.py`:
    - [ ] `render_asset_selector()`
    - [ ] `render_date_range_selector()`
    - [ ] `render_granularity_selector()`
    - [ ] `render_threshold_slider()`
  - [ ] Usare nelle pagine

## FASE 3: Data Layer Refactoring

- [ ] **Unificare data loading**
  - [ ] Creare `src/data/cache_manager.py`:
    - [ ] Centralizza tutti i decoratori `@st.cache_data`
    - [ ] Funzioni riusabili: `load_cached_data()`, `process_cached_data()`
    - [ ] Rimuovere duplicati dalle pagine
- [ ] **Creare Data Transfer Objects (DTOs)**
  - [ ] `src/models/asset_data.py`:
    - [ ] `@dataclass AssetData`
    - [ ] `@dataclass AnomalyData`
    - [ ] `@dataclass PatternData`
  - [ ] Sostituire dict con DTOs per type safety
- [ ] **Migliorare gestione NaN**
  - [ ] Creare `src/utils/nan_handler.py`:
    - [ ] `safe_fillna()` con strategia configurabile
    - [ ] `validate_dataframe()` che verifica NaN
  - [ ] Standardizzare uso in tutti i moduli

## FASE 4: Business Logic Separation

- [ ] **Creare Service Layer**
  - [ ] `src/services/anomaly_service.py`:
    - [ ] Sposta logica da pagine + anomaly_detection.py
    - [ ] Metodi: `detect_all_anomalies()`, `get_anomaly_summary()`
  - [ ] `src/services/pattern_service.py`: Logica pattern recognition
  - [ ] `src/services/cross_asset_service.py`: Logica correlazioni
- [ ] **Dependency Injection per config**
  - [ ] Creare `src/config/config_provider.py`:
    - [ ] `class ConfigProvider` con metodi getter
    - [ ] Singleton pattern o context manager
    - [ ] Passare config ai servizi invece di import globale
- [ ] **Validazione input centralizzata**
  - [ ] `src/utils/validators.py`:
    - [ ] `validate_date_range()`
    - [ ] `validate_asset_key()`
    - [ ] `validate_threshold()`
  - [ ] Usare in tutti i punti di ingresso

## FASE 5: Gemini Refactoring

- [ ] **Semplificare context builders**
  - [ ] Creare classe base `BaseContextBuilder` con template method
  - [ ] Sottoclassi per ogni page type
  - [ ] Rimuovere duplicazione nel formatting
- [ ] **Separare chart conversion**
  - [ ] Spostare `fig_to_base64_image` in `src/utils/chart_utils.py`
  - [ ] Aggiungere caching per conversioni ripetute
  - [ ] Gestire fallback in modo robusto
- [ ] **Migliorare mock mode**
  - [ ] Creare `src/gemini/mock_assistant.py` dedicato
  - [ ] Risposte mock più intelligenti basate su context
  - [ ] Modalità demo più user-friendly

## FASE 6: Testing & Documentation

- [ ] **Aggiungere tests**
  - [ ] `tests/unit/` per ogni modulo src
  - [ ] `tests/integration/` per flussi completi
  - [ ] Mock Streamlit e Gemini API
  - [ ] Target: >70% coverage
- [ ] **Documentazione tecnica**
  - [ ] `docs/ARCHITECTURE.md` con diagrammi
  - [ ] `docs/DATA_FLOW.md` con spiegazione flussi
  - [ ] `docs/API_REFERENCE.md` per servizi
  - [ ] Aggiornare docstring con esempi
- [ ] **Performance profiling**
  - [ ] Identificare colli di bottiglia con cProfile
  - [ ] Ottimizzare caching strategy
  - [ ] Lazy loading dove possibile
  - [ ] Benchmark prima/dopo
