from pydantic import BaseModel


class ChartGptModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    def llm_content(self) -> str:
        return self.model_dump_json(indent=4)
