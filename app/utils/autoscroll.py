import streamlit.components.v1 as components

# =============================================================================
# AUTO-SCROLL JAVASCRIPT
# =============================================================================

def inject_auto_scroll_js(anchor_id: str) -> None:
    """
    Execute surgical scroll using components.html (iframe)
    to manipulate parent DOM (window.parent).
    Uses getBoundingClientRect for precise positioning.
    Waits for Streamlit's scroll to stop before applying our scroll.
    """
    
    js_code = f"""
    <script>
        (function() {{
            // Find scrollable parent by walking up the DOM tree
            function getScrollParent(node) {{
                if (!node) return null;
                
                let current = node.parentElement;
                while (current) {{
                    const style = window.parent.getComputedStyle(current);
                    if (style.overflowY === 'auto' || style.overflowY === 'scroll') {{
                        return current;
                    }}
                    current = current.parentElement;
                }}
                return null;
            }}

            function performScroll() {{
                const anchor = window.parent.document.getElementById('{anchor_id}');
                if (!anchor) return;

                const container = getScrollParent(anchor);
                if (!container) return;

                const anchorRect = anchor.getBoundingClientRect();
                const containerRect = container.getBoundingClientRect();
                const relativeTop = anchorRect.top - containerRect.top;
                
                // Scroll to position the anchor 5px from top of container
                container.scrollTop += (relativeTop - 5);
            }}

            // Wait for container to exist, then observe when scrolling stops
            function waitForContainer() {{
                const anchor = window.parent.document.getElementById('{anchor_id}');
                if (!anchor) {{
                    setTimeout(waitForContainer, 50);
                    return;
                }}

                const container = getScrollParent(anchor);
                if (!container) {{
                    setTimeout(waitForContainer, 50);
                    return;
                }}

                // Listen to scroll events and wait for them to stop (debounce)
                let scrollTimeout;
                const scrollHandler = function() {{
                    clearTimeout(scrollTimeout);
                    scrollTimeout = setTimeout(() => {{
                        // Scroll has stopped for 150ms - now apply our scroll
                        container.removeEventListener('scroll', scrollHandler);
                        performScroll();
                    }}, 150);
                }};

                container.addEventListener('scroll', scrollHandler);
                
                // Trigger initial check in case Streamlit hasn't scrolled yet
                setTimeout(() => {{
                    if (scrollTimeout === undefined) {{
                        // No scroll detected after 200ms, scroll immediately
                        container.removeEventListener('scroll', scrollHandler);
                        performScroll();
                    }}
                }}, 200);
            }}

            waitForContainer();
        }})();
    </script>
    """
    
    # components.html creates iframe that ALWAYS executes JS
    components.html(js_code, height=0, width=0)