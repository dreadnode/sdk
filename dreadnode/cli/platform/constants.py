import typing as t

API_SERVICE = "api"
UI_SERVICE = "ui"
SERVICES = [API_SERVICE, UI_SERVICE]
VERSIONS_MANIFEST = "versions.json"

SupportedArchitecture = t.Literal["amd64", "arm64"]
SUPPORTED_ARCHITECTURES: list[SupportedArchitecture] = ["amd64", "arm64"]
