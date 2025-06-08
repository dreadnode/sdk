# Environment variable names

ENV_SERVER_URL = "DREADNODE_SERVER_URL"
ENV_SERVER = "DREADNODE_SERVER"  # alternative to SERVER_URL
ENV_API_TOKEN = "DREADNODE_API_TOKEN"  # noqa: S105 # nosec
ENV_API_KEY = "DREADNODE_API_KEY"  # pragma: allowlist secret (alternative to API_TOKEN)
ENV_LOCAL_DIR = "DREADNODE_LOCAL_DIR"
ENV_PROJECT = "DREADNODE_PROJECT"

# Default values

DEFAULT_SERVER_URL = "https://platform.dreadnode.io"
DEFAULT_LOCAL_OBJECT_DIR = ".dreadnode/objects"

# Default values for the S3 storage
MAX_INLINE_OBJECT_BYTES = 10 * 1024  # 10KB
