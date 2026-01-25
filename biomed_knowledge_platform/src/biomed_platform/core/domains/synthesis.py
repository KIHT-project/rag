from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from biomed_platform.api.models.generated import schemas


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    answer: schemas.AnswerPayload
    citations: list[schemas.Citation]


class SynthesisOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: schemas.AnswerPayload
    citations: list[schemas.Citation]
