# 🦋 Hashimoto Tracker & Coach

A personal Hashimoto's thyroiditis tracker powered by Groq AI.

## Files
- `hashimoto_tracker.py` — main app
- `requirements.txt` — dependencies

## Deploy to Streamlit Cloud (Free)

1. Create a free account at github.com
2. Create a new repository called `hashimoto-tracker`
3. Upload both files to the repository
4. Go to share.streamlit.io
5. Sign in with GitHub
6. Click New App → select your repository
7. Set main file as `hashimoto_tracker.py`
8. Click Advanced Settings → Secrets and add:
   GROQ_API_KEY = "your_groq_key_here"
9. Click Deploy

Get your free Groq API key at console.groq.com

## Run Locally

```
pip install -r requirements.txt
streamlit run hashimoto_tracker.py
```
