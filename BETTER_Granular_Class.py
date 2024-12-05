from typing import Optional
from pydantic import BaseModel, conlist

#TODO: Optional bezala sartu 
class span(BaseModel):
    string: str
    synclass: str|None

class spans(BaseModel):
    spans: list[span]

class event(BaseModel):
    anchors: spans
    agents: list[spans]
    patients: list[spans]
    event_type : str

class event_set(BaseModel):
    event: event
    irrealis: str|None
    time_attachments: list[spans]|None

class span_set(BaseModel):
    span: spans
    irrealis: str|None


class Protestplate(BaseModel):
    arrested: list[span_set]|None
    imprisioned: list[span_set]|None
    killed: list[span_set]|None
    occupy: list[span_set]|None
    over_time: bool|None
    organizer: list[span_set]|None
    outcome_averted: list[span_set|event_set]|None
    outcome_ocurred: list[span_set]|None
    outcome_hypothetical: list[span_set|event_set]|None
    protest_against: list[span_set|event_set]|None
    protest_event: list[span_set|event_set]|None #Unscored
    protest_for: list[span_set|event_set]|None
    when: list[span_set]|None
    where: list[span_set]|None
    who: list[span_set]|None
    wounded: list[span_set]|None
    template_type: str = "Protestplate"
    template_anchor: spans



class Corruplate(BaseModel):
    charged_with: list[span_set|event_set]|None
    corrupt_event: list[span_set|event_set]|None #Unscored
    judicial_actions: list[span_set|event_set]|None
    fine: list[span_set]|None
    over_time: bool|None
    outcome_averted: list[span_set]|None
    outcome_ocurred: list[span_set]|None
    outcome_hypothetical: list[span_set|event_set]|None
    prison_term : list[span_set]|None
    where: list[span_set]|None
    who: list[span_set]|None
    template_type: str = "Corruplate"
    template_anchor: spans


class Terrorplate(BaseModel):
    blamed_by: list[span_set]|None
    claimed_by: list[span_set]|None
    completion: str|None
    coordinated: bool|None
    killed: list[span_set]|None
    kidnapped: list[span_set]|None
    named_perp: list[span_set]|None
    named_perp_org: list[span_set]|None
    named_organizer: list[span_set]|None
    over_time: bool|None
    outcome_averted: list[span_set]|None
    outcome_ocurred: list[span_set]|None
    outcome_hypothetical: list[span_set|event_set]|None
    perp_captured: list[span_set]|None
    perp_objective: list[span_set|event_set]|None
    perp_wounded: list[span_set]|None
    target_human: list[span_set]|None
    target_physical: list[span_set]|None
    terror_event: list[span_set|event_set]|None #Unscored
    type: str|None
    weapon: list[span_set]|None
    when: list[span_set]|None
    where: list[span_set]|None
    wounded: list[span_set]|None
    template_type: str = "Terrorplate"
    template_anchor: spans


class Epidemiplate(BaseModel):
    hospitalized_count: list[span_set]|None
    hospitalized_cumulative: list[span_set]|None
    hospitalized_individuals: list[span_set]|None
    infected_count: list[span_set]|None
    infected_cumulative: list[span_set]|None
    infected_individuals: list[span_set]|None
    killed_count: list[span_set]|None
    killed_cumulative: list[span_set]|None
    killed_individuals: list[span_set]|None
    NPI_events: list[span_set]|None
    outbreak_event: list[span_set|event_set]|None #Unscored
    tested_count: list[span_set]|None
    tested_cumulative: list[span_set]|None
    tested_individuals: list[span_set]|None
    recoverd_count: list[span_set]|None
    recoverd_cumulative: list[span_set]|None
    recoverd_individuals: list[span_set]|None
    vaccinated_count: list[span_set]|None
    vaccinated_cumulative: list[span_set]|None
    vaccinated_individuals: list[span_set]|None
    when: list[span_set]|None
    where: list[span_set]|None
    template_type: str = "Epidemiplate"
    template_anchor: spans



class Disasterplate(BaseModel):
    major_disaster_event: list[span_set|event_set]|None #Unscored
    over_time: bool|None
    where: list[span_set]|None
    when: list[span_set]|None
    injured_count: list[span_set]|None
    killed_count: list[span_set]|None
    missing_count: list[span_set]|None
    outcome: list[event_set]|None
    responders: list[span_set]|None
    damage: list[span_set]|None
    affected_cumulative_count: list[span_set]|None
    individuals_affected: list[span_set]|None
    rescued_count: list[span_set]|None
    rescue_events: list[event_set]|None
    assitance_provided: list[event_set]|None|None
    assistance_needed: list[event_set]|None
    realted_natural_phenomena: list[event_set]|None
    announce_disaster_warnings: list[event_set]|None
    declare_emergency: list[event_set]|None
    disase_outbreak_events: list[event_set]|None
    repair: list[event_set]|None
    human_displacement_events: list[event_set]|None
    template_type: str = "Disasterplate"
    template_anchor: spans



class Displacementplate(BaseModel):
    human_dispacement_event: list[event_set]|None #Unscored
    over_time: bool|None
    origin: list[span_set]|None
    current_location: list[span_set]|None
    transitiong_location: list[span_set]|None
    destination: list[span_set]|None
    when: list[span_set]|None
    total_displaced_count: list[span_set]|None
    event_or_soa_at_origin: list[event_set]|None
    settlement_status_event_or_soa: list[event_set]|None
    outcome: list[event_set]|None
    group_identity: list[span_set]|None
    injured_count: list[span_set]|None
    killed_count: list[span_set]|None
    missing_count: list[span_set]|None
    detained_count: list[span_set]|None
    blocked_migration_count: list[span_set]|None
    transitory_events: list[span_set]|None
    assistance_provided: list[span_set]|None
    assistance_needed: list[span_set]|None
    template_type: str = "Displacementplate"
    template_anchor: spans


class Template(BaseModel):
    templates: conlist(item_type=Protestplate|Corruplate|Terrorplate|Epidemiplate|Disasterplate|Displacementplate, min_length=0, max_length=9)