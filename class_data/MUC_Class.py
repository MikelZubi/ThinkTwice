from typing import Literal
from pydantic import BaseModel, Field

class Template(BaseModel):
    incident_type: Literal['kidnapping','attack','bombing','robbery','arson','forced work stoppage'] = Field(description="The type of incident, the values can be: 'kidnapping','attack','bombing','robbery','arson', or 'forced work stoppage'")
    PerpInd: list[list[str]] = Field(description="A person responsible for the incident.")
    PerpOrg: list[list[str]] = Field(description="An organization responsible for the incident.")
    Target: list[list[str]] = Field(description="An inanimate object that was attacked.")
    Victim: list[list[str]] = Field(description="The name of a person who was the obvious or apparent target of the attack or who became a victim of the attack.")
    Weapon: list[list[str]] = Field(description="A device used by the perpetrator(s) in carrying out the terrorist act.")


class Base(BaseModel):
    templates: list[Template]


class Entities(BaseModel):
    entities: list[list[str]]


class Incident_Types(BaseModel):
    incident_types: list[Literal['kidnapping','attack','bombing','robbery','arson','forced work stoppage']]