import typing as t

PlatformService = t.Literal["dreadnode-api", "dreadnode-ui"]
API_SERVICE: PlatformService = "dreadnode-api"
UI_SERVICE: PlatformService = "dreadnode-ui"
SERVICES: list[PlatformService] = [API_SERVICE, UI_SERVICE]
VERSIONS_MANIFEST = "versions.json"

SupportedArchitecture = t.Literal["amd64", "arm64"]
SUPPORTED_ARCHITECTURES: list[SupportedArchitecture] = ["amd64", "arm64"]

DEFAULT_DOCKER_PROJECT_NAME = "dreadnode-platform"
