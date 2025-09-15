import typing as t

PlatformService = t.Literal["api", "ui"]
API_SERVICE: PlatformService = "api"
UI_SERVICE: PlatformService = "ui"
SERVICES: list[PlatformService] = [API_SERVICE, UI_SERVICE]
VERSIONS_MANIFEST = "versions.json"

SupportedArchitecture = t.Literal["amd64", "arm64"]
SUPPORTED_ARCHITECTURES: list[SupportedArchitecture] = ["amd64", "arm64"]

DEFAULT_DOCKER_PROJECT_NAME = "dreadnode-platform"
