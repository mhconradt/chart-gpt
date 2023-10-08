from pydantic import BaseModel


class ChartGptModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True
