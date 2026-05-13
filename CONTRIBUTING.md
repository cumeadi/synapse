# Contributing to Synapse

First off, thank you for considering contributing to Synapse! We welcome contributions from everyone.

## Local Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/synapse.git
   cd synapse
   ```

2. **Start the Database:**
   Synapse requires Postgres with the `pgvector` extension. You can spin this up easily using docker-compose:
   ```bash
   docker-compose up -d
   ```

3. **Install Dependencies:**
   Create a virtual environment and install the requirements:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

4. **Set up Environment Variables:**
   Copy `.env.example` to `.env` and fill in your API keys (e.g., `LLM_MODEL`, `OPENAI_API_KEY`, etc.)

5. **Run the API Server Locally:**
   ```bash
   uvicorn app.main:app --reload
   ```

## Pull Request Process
1. Fork the repo and create your branch from `main`.
2. Write tests for your new feature or bug fix.
3. Ensure the test suite passes locally.
4. Update documentation if necessary.
5. Submit a pull request!

## Code of Conduct
Please be respectful and constructive when interacting with the community. We are building this together!
