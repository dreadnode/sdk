from pydantic import BaseModel


class Runnable(BaseModel):
    name: str
    description: str
