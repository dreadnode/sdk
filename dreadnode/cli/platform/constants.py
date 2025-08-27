from pathlib import Path

API_SERVICE = "api"
UI_SERVICE = "ui"
SERVICES = [API_SERVICE, UI_SERVICE]

TEMPLATE_DIR = Path(__file__).parent / "templates"
DOCKER_COMPOSE_TEMPLATE = TEMPLATE_DIR / "docker-compose.yaml.j2"
API_ENV_TEMPLATE = TEMPLATE_DIR / ".api.env.j2"
UI_ENV_TEMPLATE = TEMPLATE_DIR / ".ui.env.j2"
