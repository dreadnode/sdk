import os
import pathlib

from dreadnode.version import VERSION

#
# Defaults
#

# name of the default local storage path
DEFAULT_LOCAL_STORAGE_DIR = pathlib.Path.home() / ".dreadnode"
# name of the default server profile
DEFAULT_PROFILE_NAME = "main"
# default poll interval for the authentication flow
DEFAULT_POLL_INTERVAL = 5
# default maximum poll time for the authentication flow
DEFAULT_MAX_POLL_TIME = 300
# default maximum token TTL in seconds
DEFAULT_TOKEN_MAX_TTL = 60
# Default values for the S3 storage
DEFAULT_MAX_INLINE_OBJECT_BYTES = 10 * 1024  # 10KB
# default platform domain
DEFAULT_PLATFORM_BASE_DOMAIN = "dreadnode.io"
# default server URL
DEFAULT_SERVER_URL = f"https://platform.{DEFAULT_PLATFORM_BASE_DOMAIN}"
# default local directory for dreadnode objects
DEFAULT_LOCAL_OBJECT_DIR = f"{DEFAULT_LOCAL_STORAGE_DIR}/objects"
# default docker registry subdomain
DEFAULT_DOCKER_REGISTRY_SUBDOMAIN = "registry"
# default docker registry local port
DEFAULT_DOCKER_REGISTRY_LOCAL_PORT = 5005
# default docker registry image tag
DEFAULT_DOCKER_REGISTRY_IMAGE_TAG = "registry"
# default workspace name
DEFAULT_WORKSPACE_NAME = "Personal Workspace"
# default project name
DEFAULT_PROJECT_NAME = "Default"
# default project key
DEFAULT_PROJECT_KEY = "default"

#
# Environment Variable Names
#

ENV_SERVER_URL = "DREADNODE_SERVER_URL"
ENV_SERVER = "DREADNODE_SERVER"  # alternative to SERVER_URL
ENV_API_TOKEN = "DREADNODE_API_TOKEN"  # noqa: S105 # nosec
ENV_API_KEY = "DREADNODE_API_KEY"  # pragma: allowlist secret (alternative to API_TOKEN)
ENV_LOCAL_DIR = "DREADNODE_LOCAL_DIR"
ENV_ORGANIZATION = "DREADNODE_ORGANIZATION"
ENV_WORKSPACE = "DREADNODE_WORKSPACE"
ENV_PROJECT = "DREADNODE_PROJECT"
ENV_PROFILE = "DREADNODE_PROFILE"
ENV_CONSOLE = "DREADNODE_CONSOLE"

#
# Environment
#

# enable debugging
DEBUG = bool(os.getenv("DREADNODE_DEBUG")) or False

# server url
PLATFORM_BASE_URL = os.getenv(ENV_SERVER, os.getenv(ENV_SERVER_URL, DEFAULT_SERVER_URL))

# path to the user configuration file
USER_CONFIG_PATH = pathlib.Path(
    # allow overriding the user config file via env variable
    os.getenv("DREADNODE_USER_CONFIG_FILE") or DEFAULT_LOCAL_STORAGE_DIR / "config"
)

# Default values for the file system credential management
FS_CREDENTIAL_REFRESH_BUFFER = 900  # 15 minutes in seconds

# Default User-Agent
DEFAULT_USER_AGENT = f"dreadnode/{VERSION}"


# Crescendo attack variant paths
CRESCENDO_VARIANTS_PATH = pathlib.Path(__file__).parent / "data" / "templates" / "crescendo"
CRESCENDO_VARIANT_1 = CRESCENDO_VARIANTS_PATH / "variant_1.yaml"
CRESCENDO_VARIANT_2 = CRESCENDO_VARIANTS_PATH / "variant_2.yaml"
CRESCENDO_VARIANT_3 = CRESCENDO_VARIANTS_PATH / "variant_3.yaml"
CRESCENDO_VARIANT_4 = CRESCENDO_VARIANTS_PATH / "variant_4.yaml"
CRESCENDO_VARIANT_5 = CRESCENDO_VARIANTS_PATH / "variant_5.yaml"

# Adversarial benchmark dataset path
ADVERSARIAL_BENCHMARK_DATASET_PATH = (
    pathlib.Path(__file__).parent / "data" / "assets" / "adversarial_benchmark_subset.csv"
)

ASSETS_PATH = pathlib.Path(__file__).parent / "data" / "assets"
