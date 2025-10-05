import typing as t

from dreadnode.constants import DEFAULT_LOCAL_STORAGE_DIR

PlatformService = t.Literal["dreadnode-api", "dreadnode-ui"]
PLATFORM_SERVICES = t.cast("list[PlatformService]", t.get_args(PlatformService))
API_SERVICE: PlatformService = "dreadnode-api"
UI_SERVICE: PlatformService = "dreadnode-ui"

SupportedArchitecture = t.Literal["amd64", "arm64"]
SUPPORTED_ARCHITECTURES = t.cast("list[SupportedArchitecture]", t.get_args(SupportedArchitecture))

DEFAULT_DOCKER_PROJECT_NAME = "dreadnode-platform"

PLATFORM_STORAGE_DIR = DEFAULT_LOCAL_STORAGE_DIR / "platform"
VERSION_CONFIG_PATH = PLATFORM_STORAGE_DIR / "versions.json"
