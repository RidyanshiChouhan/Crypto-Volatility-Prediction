"""Premium fintech dark theme CSS for Streamlit."""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a0e17 0%, #111827 50%, #0f172a 100%);
    color: #e2e8f0;
}

.block-container {
    padding-top: 1.5rem;
    max-width: 1400px;
}

/* Glassmorphism cards */
.glass-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

.kpi-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #00d4aa;
    margin: 0;
}

.kpi-label {
    font-size: 0.85rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.risk-badge-low {
    background: rgba(0, 212, 170, 0.2);
    color: #00d4aa;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    font-weight: 600;
    border: 1px solid #00d4aa;
}

.risk-badge-medium {
    background: rgba(240, 180, 41, 0.2);
    color: #f0b429;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    font-weight: 600;
    border: 1px solid #f0b429;
}

.risk-badge-high {
    background: rgba(255, 71, 87, 0.2);
    color: #ff4757;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    font-weight: 600;
    border: 1px solid #ff4757;
}

.hero-title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00d4aa, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

section[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.95);
    border-right: 1px solid rgba(255,255,255,0.08);
}

div[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.05);
    padding: 1rem;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.08);
}

.stButton > button {
    background: linear-gradient(90deg, #00d4aa, #3b82f6);
    color: #0a0e17;
    font-weight: 600;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.5rem;
}

.stButton > button:hover {
    opacity: 0.9;
    border: none;
}
</style>
"""
