"""
Integrations for external services and platforms.

Available integrations:
- serve: Serve components as HTTP APIs with FastAPI
- ray: Deploy with Ray Serve for distributed scaling
- cloudflare: Deploy to Cloudflare Workers
- data_designer: Generate synthetic datasets with NVIDIA Data Designer
- docker: Docker container management
- transformers: HuggingFace Transformers training callbacks
"""

from dreadnode.core.integrations import cloudflare, ray, serve

__all__ = ["cloudflare", "ray", "serve"]
