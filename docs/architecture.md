# BERTokenScope Architecture

BERTokenScope separates the product into four layers:

1. **Model adapters** load Hugging Face models lazily and expose stable service contracts.
2. **Analysis engines** convert model outputs into attention links, masked-token predictions, and financial signals.
3. **Interfaces** expose the system through Streamlit and FastAPI.
4. **Tests and deterministic fallbacks** keep the project reliable when model weights are unavailable.

## CS50AI Extension

The CS50AI Attention project focuses on three core functions:

- locating the `[MASK]` token
- converting attention scores to grayscale cells
- generating diagrams for attention heads

BERTokenScope preserves those ideas in `attention.engine`, then expands them into reusable modules:

- `AttentionEngine` for masked-token prediction
- `strongest_attention_links` for token relationship inspection
- Plotly heatmaps for interactive attention visualization
- dashboard and API interfaces

## Production Direction

The current implementation includes the Phase 2 interpretability foundation:

- selectable attention layers and heads
- attention rollout across layers
- attention entropy and focus diagnostics
- strongest token-to-token attention links
- CS50AI-style PNG attention diagram export
- API access through `/attention/explore`
- financial transcript chunking and period-over-period tone drift
- optional FinBERT adapter with deterministic financial baseline fallback
- embedding intelligence with transformer/fallback encoders, PCA projection, clustering, and similarity pairs
- model comparison across masked language modeling, financial sentiment, and embeddings
- explainability reports with attention attribution and counterfactual token impacts
- local artifact tracking and run history for API workflows

The next high-value additions are:

- hosted deployment with authentication and centralized observability
- external object storage for large model artifacts
- live transformer model cache management
