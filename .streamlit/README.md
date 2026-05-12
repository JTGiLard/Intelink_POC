# Streamlit configuration

- **`secrets.toml`** is listed in `.gitignore`. Do not commit API keys or credentials.
- **Local:** create `.streamlit/secrets.toml` with at least `OPENAI_API_KEY = "..."` or use a `.env` file in the project root (see `.env.example`).
- **Streamlit Community Cloud:** add `OPENAI_API_KEY` (and any other keys) in **App settings → Secrets**.

Optional: add `config.toml` here for theme/UI (safe to commit if it contains no secrets).
