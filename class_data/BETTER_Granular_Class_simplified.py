from typing import Literal, Optional, Union
from pydantic import BaseModel, Field, conlist



class span(BaseModel):
    string: str
    synclass: Optional[Literal['name', 'nominal', 'pronoun', 'event-anchor', 'template-anchor', 'time-mention', 'duration-mention']] = None

class event(BaseModel):
    anchors: span
    event_type : Literal['Aid-Needs', 'Apply-NPI', 'Award-Contract', 'Bribery', 'Business-Event-or-SoA', 'Change-of-Govt', 'Change-Repayment', 'Close-Schools', 'Communicate-Event', 'Conduct-Diplomatic-Talks', 'Conduct-Medical-Research', 'Conduct-Meeting', 'Conduct-Protest', 'Conduct-Violent-Protest', 'Conspiracy', 'Construct-Project', 'Coordinated-Comm', 'Corruption', 'Coup', 'Cull-Livestock', 'Cyber-Crime-Attack', 'Cyber-Crime-Other', 'Cybersecurity-Measure', 'Cybersecurity-Response', 'Death-from-Crisis-Event', 'Declare-Emergency', 'Disease-Exposes', 'Disease-Infects', 'Disease-Kills', 'Disease-Outbreak', 'Disease-Recovery', 'Dismiss-Workers', 'Distribute-PPE', 'Economic-Event-or-SoA', 'Employ-Workers', 'Environmental-Event-or-SoA', 'Establish-Project', 'Evacuate', 'Expel', 'Extortion', 'Famine-Event-or-SoA', 'Financial-Crime', 'Financial-Loss', 'Fiscal-or-Monetary-Action', 'Fund-Project', 'Hospitalize', 'Identify-Vulnerability', 'Illegal-Entry', 'Impose-Quarantine', 'Information-Release', 'Information-Theft', 'Infrastructure-Operation', 'Interrupt-Construction', 'Interrupt-Operations', 'Judicial-Acquit', 'Judicial-Convict', 'Judicial-Indict', 'Judicial-Other', 'Judicial-Plead', 'Judicial-Prosecute', 'Judicial-Seize', 'Judicial-Sentence', 'Kidnapping', 'Law-Enforcement-Arrest', 'Law-Enforcement-Extradite', 'Law-Enforcement-Investigate', 'Law-Enforcement-Other', 'Leave-Job', 'Legislative-Action', 'Lift-PPE-Requirements', 'Lift-Quarantine', 'Loosen-Business-Restrictions', 'Loosen-Travel-Restrictions', 'Make-Repayment', 'Migrant-Detain', 'Migrant-Relocation', 'Migrant-Smuggling', 'Migration-Blocked', 'Migration-Impeded-Failed', 'Military-Attack', 'Military-Declare-War', 'Military-Other', 'Missing-from-Crisis-Event', 'Monitor-Disease', 'Natural-Disaster-Event-or-SoA', 'Natural-Phenomenon-Event-or-SoA', 'Open-Schools', 'Organize-Protest', 'Other-Crime', 'Other-Government-Action', 'Pay-Ransom', 'Persecution', 'Political-Election-Event', 'Political-Event-or-SoA', 'Political-Other', 'Provide-Aid', 'Propose-Project', 'Refugee-Movement', 'Repair', 'Require-PPE', 'Rescue', 'Restrict-Business', 'Restrict-Travel', 'Sign-Agreement', 'Suppress-Communication', 'Suppression-of-Free-Speech', 'Suppress-Meeting', 'Suppress-or-Breakup-Protest', 'Test-Patient', 'Treat-Patient', 'Vaccinate', 'Violence', 'Violence-Attack', 'Violence-Bombing', 'Violence-Damage', 'Violence-Kill', 'Violence-Other', 'Violence-Set-Fire', 'Violence-Wound', 'War-Event-or-SoA', 'Weather-Event-or-SoA', 'Weather-or-Environmental-Damage', 'Wounding-from-Crisis-Event']

class Protestplate(BaseModel):
    template_type: Literal["Protestplate"] = "Protestplate"
    arrested: Optional[list[span]] = Field(
        description='Description or count of those arrested.',
        default=None)
    imprisoned: Optional[list[span|event]] = Field(
        description='Description or count of those jailed.',
        default=None)
    killed: Optional[list[span]] = Field(
        description='Description or count of those killed, e.g., "two people," "two"',
        default=None)
    occupy: Optional[list[span]] = Field(
        description='Any space or building taken over, e.g., "the local government offices".',
        default=None)
    over_time: Optional[bool] = Field(
        description='A flag indicating whether this is an individual protest, or a period of frequent civil unrest, with regular protests arising in different places and times, e.g., the Arab Spring.',
        default=None)
    organizer: Optional[list[span]] = Field(
        description='The group/individuals leading the protest, e.g., "the Workers Party".',
        default=None)
    outcome_averted: Optional[list[span|event]] = Field(
        description='Events that were either averted or are noted as not having occurred, e.g., "no injuries" (Basic events do not code negation or other realis factors).',
        default=None)
    outcome_occurred: Optional[list[span|event]] = Field(
        description='Events that occurred because of the corruption.',
        default=None)
    outcome_hypothetical: Optional[list[span|event]] = Field(
        description='Events that are only noted as potentially occurring because of the corruption.',
        default=None)
    protest_against: Optional[list[span|event]] = Field(
        description='Any events presented as what the protest is meant to end, e.g. “corruption,” “unemployment” (an state-of-affairs Basic event), “a [ban] on [the washing lines],” etc.',
        default=None)
    protest_event: Optional[list[span|event]]  = Field(
        description='Triggers of the template',
        default=None) #Unscored
    protest_for: Optional[list[span|event]] = Field(
        description='Any event presented as the aim of the protest, e.g., “the corrupt must face justice,” coded as {agt ø, head “face justice,” ptt “the corrupt”}.',
        default=None)
    when: Optional[list[span]] = Field(
        description='Date of the protest, as best identifiable: “Thursday,” “last month,” etc.',
        default=None)
    where: Optional[list[span]] = Field(
        description='Location(s) of the protest.',
        default=None)
    who: Optional[list[span]] = Field(
        description='References to protest participants, e.g., “hundreds of young men.”.',
        default=None)
    wounded: Optional[list[span]] = Field(
        description='Description of any injured participants, or a count if that is all that is available, e.g., “a woman,” “40,” “several police officers”.',
        default=None)
    template_anchor: span = Field(
        description='The anchor of the template',
        default=None)



class Corruplate(BaseModel):
    template_type: Literal["Corruplate"] = "Corruplate"
    charged_with: Optional[list[span|event]] = Field(
        description='The crimes that the individual has been charged with, coded as Basic events “[patient Uyukaeve] had been caught accepting the [event bribe]”',
        default=None)
    corrupt_event: Optional[list[span|event]]  = Field(
        description='Triggers of the template',
        default=None) #Unscored
    judicial_actions: Optional[list[span|event]] = Field(
        description='Investigations, trials, sentences, and so forth noted in the narrative as having taken place',
        default=None) 
    fine: Optional[list[span]] = Field(
        description='Any monetary damages or seizures levelled as punishment.',
        default=None) 
    over_time: Optional[bool] = Field(
        description='A flag indicating whether this is an individual case of corruption, or a systematic state of corruption affecting many corrupt individuals or institutions.',
        default=None) 
    outcome_averted: Optional[list[span]] = Field(
        description='Events that were either averted or are noted as not having occurred, e.g., “no injuries” (Basic events do not code negation or other realis factors).',
        default=None) 
    outcome_occurred: Optional[list[span|event]] = Field(
        description='Events that occurred because of the corruption.',
        default=None) 
    outcome_hypothetical: Optional[list[span|event]] = Field(
        description='Events that are only noted as potentially occurring because of the corruption.',
        default=None) 
    prison_term : Optional[list[span]] = Field(
        description='The duration(s) of any prison sentence mentioned as applicable (with appropriate irrealis status indicated as appropriate).',
        default=None) 
    where: Optional[list[span]] = Field(
        description='Location(s) of the corruption.',
        default=None) 
    who: Optional[list[span]] = Field(
        description='The individual(s) being accused of corruption.',
        default=None) 
    template_anchor: span = Field(
        description='The anchor of the template',
        default=None)


class Terrorplate(BaseModel):
    template_type: Literal["Terrorplate"] = "Terrorplate"
    blamed_by: Optional[list[span]] = Field(
        description='Those who are asserting the identity of the perpetrators.',
        default=None) 
    claimed_by: Optional[list[span]] = Field(
        description='Those who have claimed responsibility for the terrorist event(s).',
        default=None) 
    completion: Optional[Literal["planned","thwarted","failed","successful"]] = Field(
        description='Whether the terrorism event(s) are considered to be completed or not. One of "planned", "thwarted", "failed", or "successful".',
        default=None) 
    coordinated: Optional[bool] = Field(
        description='Whether the terrorism event(s) are considered to be coordinated or not.',
        default=None) 
    killed: Optional[list[span]] = Field(
        description='Mentions of those people who were killed.',
        default=None) 
    kidnapped: Optional[list[span]] = Field(
        description='Mentions of those people who were kidnapped.',
        default=None) 
    named_perp: Optional[list[span]] = Field(
        description='Those to whom the terrorist event(s) are attributed.',
        default=None) 
    named_perp_org: Optional[list[span]] = Field(
        description='Mentions of the organization(s) the perpetrators belong to.',
        default=None) 
    named_organizer: Optional[list[span]] = Field(
        description='Those to whom the planning of the terrorist event(s) are attributed.',
        default=None) 
    over_time: Optional[bool] = Field(
        description='A flag indicating whether this is an individual case of terrorism, or a systematic state of terrorism',
        default=None) 
    outcome_averted: Optional[list[span]] = Field(
        description='Events that were prevented by virtue of the events described in this template.',
        default=None) 
    outcome_occurred: Optional[list[span|event]] = Field(
        description='Events or states-of-affairs that have actually taken place by virtue of the terrorist events.',
        default=None) 
    outcome_hypothetical: Optional[list[span|event]] = Field(
        description='Events or states-of-affairs that could have taken place by due to the terrorist events.',
        default=None) 
    perp_captured: Optional[list[span]] = Field(
        description='Mentions of perpetrators of the terrorist events who were captured.',
        default=None) 
    perp_killed: Optional[list[span]] = Field(
        description='Mentions of perpetrators who were killed in the course of the terrorist events.',
        default=None) 
    perp_objective: Optional[list[span|event]] = Field(
        description='Mentions of events which are identified as being desired to have taken place (or states-of-affairs to have come about) by virtue of the terrorist events.',
        default=None) 
    perp_wounded: Optional[list[span]] = Field(
        description='Mentions of perpetrators who were wounded in the course of the terrorist events.',
        default=None) 
    target_human: Optional[list[span]] = Field(
        description='One or more people, named or unnamed, who are said to be the targets of the terrorist events.',
        default=None) 
    target_physical: Optional[list[span]] = Field(
        description='The facility or geo-political location that was being targeted by the terrorist events.',
        default=None) 
    terror_event: Optional[list[span|event]] = Field(
        description='Triggers of the template',
        default=None)  #Unscored
    type: Optional[Literal["arson", "assault", "bombing", "kidnapping", "murder", "unspecified"]] = Field(
        description='One of "arson", "assault", "bombing", "kidnapping", "murder", or "unspecified".',
        default=None)
    weapon: Optional[list[span]] = Field(
        description='Mentions of the weapons or other instruments used to carry out the terrorist events.',
        default=None)
    when: Optional[list[span]] = Field(
        description='Mentions of times or durations associated with the events involved.',
        default=None)
    where: Optional[list[span]] = Field(
        description='Mentions of the location(s) at which the terrorist events have taken place.',
        default=None)
    wounded: Optional[list[span]] = Field(
        description='Mentions of those people who were wounded.',
        default=None)
    template_anchor: span = Field(
        description='The anchor of the template',
        default=None)



class Epidemiplate(BaseModel):
    template_type: Literal["Epidemiplate"] = "Epidemiplate"
    disease: Optional[list[span]] = Field(
        description='Mentions of the disease that is at the heart of the outbreak',
        default=None)
    exposed_count: Optional[list[span]] = Field(
        description='Mentions of counts of people who have become exposed.',
        default=None)
    exposed_cumulative: Optional[list[span]] = Field(
        description='Mentions of cumulative counts of people who have become exposed.',
        default=None)
    exposed_individuals: Optional[list[span]] = Field(
        description='Mentions of people (or groups of people) who have become exposed to the disease.',
        default=None)
    hospitalized_count: Optional[list[span]] = Field(
        description='Mentions of counts of people who have become hospitalized due to the disease.',
        default=None)
    hospitalized_cumulative: Optional[list[span]] = Field(
        description='Mentions of cumulative counts of people who have become hospitalized due to the disease.',
        default=None)
    hospitalized_individuals: Optional[list[span]] = Field(
        description='Mentions of people (or groups of people) who have become hospitalized due to the disease.',
        default=None)
    infected_count: Optional[list[span]] = Field(
        description='Mentions of counts of people who have become infected.',
        default=None)
    infected_cumulative: Optional[list[span]] = Field(
        description='Mentions of cumulative counts of people who have become infected.',
        default=None)
    infected_individuals: Optional[list[span]] = Field(
        description='Mentions of people (or groups of people) who have become infected with the disease.',
        default=None)
    killed_count: Optional[list[span]] = Field(
        description='Mentions of counts of people who have been killed by the disease.',
        default=None)
    killed_cumulative: Optional[list[span]] = Field(
        description='Mentions of cumulative counts of people who have been killed by the disease.',
        default=None)
    killed_individuals: Optional[list[span]] = Field(
        description='Mentions of people (or groups of people) who have been killed by the disease.',
        default=None)
    NPI_Events: Optional[list[span|event]] = Field(
        description='Non-Pharmacologic-Intervention-Events. Interventions taken by authorities to prevent the further disease, such as closing schools, limiting travel, etc.',
        default=None)
    outbreak_event: Optional[list[span|event]] = Field(
        description='Triggers of the template',
        default=None) #Unscored
    tested_count: Optional[list[span]] = Field(
        description='Mentions of counts of people who have been tested for the disease.',
        default=None)
    tested_cumulative: Optional[list[span]] = Field(
        description='Mentions of cumulative counts of people who have been tested for the disease.',
        default=None)
    tested_individuals: Optional[list[span]] = Field(
        description='Mentions of people (or groups of people) who have been tested for the disease.',
        default=None)
    recovered_count: Optional[list[span]] = Field(
        description='Mentions of counts of people who have recovered from the disease.',
        default=None)
    recovered_cumulative: Optional[list[span]] = Field(
        description='Mentions of cumulative counts of people who have recovered from the disease.',
        default=None)
    recovered_individuals: Optional[list[span]] = Field(
        description='Mentions of people (or groups of people) who have recovered from the disease.',
        default=None)
    vaccinated_count: Optional[list[span]] = Field(
        description='Mentions of counts of people who have been vaccinated against the disease.',
        default=None)
    vaccinated_cumulative: Optional[list[span]] = Field(
        description='Mentions of cumulative counts of people who have been vaccinated against the disease.',
        default=None)
    vaccinated_individuals: Optional[list[span]] = Field(
        description='Mentions of people (or groups of people) who have been vaccinated against the disease.',
        default=None)
    when: Optional[list[span]] = Field(
        description='Mentions of times or durations associated with the events involved.',
        default=None)
    where: Optional[list[span]] = Field(
        description='Mentions of one or more instances of where the disease has occurred.',
        default=None)
    template_anchor: span = Field(
        description='The anchor of the template',
        default=None)




class Disasterplate(BaseModel):
    template_type: Literal["Disasterplate"] = "Disasterplate"
    major_disaster_event: Optional[list[span|event]] = Field(
        description='Holds instances of Environmental, Natural Phenomenon, or Famine events (or SoAs), or also Weather/Environmental damage events. Can be considered to serve as triggers.',
        default=None) #Unscored
    over_time: Optional[bool] = Field(
        description='A flag indicating whether this is an individual event, or an extended state of affairs.',
        default=None)
    where: Optional[list[span]] = Field(
        description='Mentions of one or more locations associated with the disaster.',
        default=None)
    when: Optional[list[span]] = Field(
        description='Mentions of times or durations associated with the events involved.',
        default=None)
    injured_count: Optional[list[span]] = Field(
        description='Mentions that enumerate the individuals injured in a disaster',
        default=None)
    killed_count: Optional[list[span]] = Field(
        description='Mentions that enumerate the individuals who died in a disaster.',
        default=None)
    missing_count: Optional[list[span]] = Field(
        description='Mentions that enumerate individuals who have gone missing in a disaster.',
        default=None)
    outcome: Optional[list[event]] = Field(
        description='Events that occurred as a result of a disaster but are not already covered in specific slots of the Disasterplate.',
        default=None)
    responders: Optional[list[span]] = Field(
        description='Entities designating the responders to the natural disaster.',
        default=None)
    damage: Optional[list[span]] = Field(
        description='Entities that have been damaged in the disaster, typically the patients of a Weather-or-environmental-Damage Basic event',
        default=None)
    affected_cumulative_count: Optional[list[span]] = Field(
        description='For those cases where the toll of a disaster is identified over time: this slot holds entities that enumerate the cumulative count of affected individuals.',
        default=None)
    individuals_affected: Optional[list[span]] = Field(
        description='Entities that provide descriptions of the affected individuals.',
        default=None)
    rescued_count: Optional[list[span]] = Field(
        description='Entities enumerating the individuals rescued in the response to a disaster.',
        default=None)
    rescue_events: Optional[list[event]] = Field(
        description='This slot holds Basic events of the Rescue or Evacuate types (as opposed to the patients of these events, as in the fillers of the rescued-count slot).',
        default=None)
    assistance_provided: Optional[list[event]] = Field(
        description='This slot captures Provide-Aid events that were in response to the disaster.',
        default=None)
    assistance_needed: Optional[list[event]] = Field(
        description='Similarly to the previous slot, this slot captures Aid-Needs events that originated in the disaster.',
        default=None)
    related_natural_phenomena: Optional[list[event]] = Field(
        description='Weather, Environmental, or similar events that occurred as a result of the predicating disaster. For example, floods might be related to a hurricane disaster. Because it can sometimes be difficult to determine which event should be treated as “primary,” in this example, both “floods” and “hurricane” would be included in this slot.',
        default=None)
    announce_disaster_warnings: Optional[list[event]] = Field(
        description='Any communication events that warn of the impending disaster before it strikes.',
        default=None)
    declare_emergency: Optional[list[event]] = Field(
        description='Any communication events declaring the emergency (typically a Declare-Emergency Basic event).',
        default=None)
    disease_outbreak_events: Optional[list[event]] = Field(
        description='Disease outbreak events that follow upon the disaster, e.g., the cholera outbreak after the Haiti earthquake. Not just Disease-Outbreak; this can also include other epidemic event types.',
        default=None)
    repair: Optional[list[event]] = Field(
        description='Repair events that are part of the response to the disaster.',
        default=None)
    human_displacement_events: Optional[list[event]] = Field(
        description='Any refugee movement events resulting from the disaster (including those related to detaining refugees).',
        default=None)
    template_anchor: span = Field(
        description='The anchor of the template',
        default=None)




class Displacementplate(BaseModel):
    template_type: Literal["Displacementplate"] = "Displacementplate"
    human_displacement_event: Optional[list[event]] = Field(
        description='Holds event anchors typically associated with human displacement -primarily refugee- movement events. Can be considered to serve as triggers.',
        default=None) #Unscored
    over_time: Optional[bool] = Field(
        description='A flag indicating whether this is an individual event, or an extended state of affairs.',
        default=None)
    origin: Optional[list[span]] = Field(
        description='Mentions of one or more locations from which the displaced humans are fleeing.',
        default=None)
    current_location: Optional[list[span]] = Field(
        description='Mentions of any locations reported to be where the displaced humans currently are.',
        default=None)
    transiting_location: Optional[list[span]] = Field(
        description='Mentions of any locations through which the displaced humans transited on their way to their current location.',
        default=None)
    destination: Optional[list[span]] = Field(
        description='Mentions of any locations that the displaced humans intend to continue on towards.',
        default=None)
    when: Optional[list[span]] = Field(
        description='Mentions of times or durations associated with the events involved',
        default=None)
    total_displaced_count: Optional[list[span]] = Field(
        description='As with Epidemiplates, these are phrases that enumerate the total population of displaced humans.',
        default=None)
    event_or_SoA_at_origin: Optional[list[event]] = Field(
        description='Events that precipitated the migration: these can be natural disasters, war, famine, political or economic crises, etc.',
        default=None)
    settlement_status_event_or_SoA: Optional[list[event]] = Field(
        description='Events (of any type) that pertain to the status of the displaced humans. Events that indicate the final status of the individuals (e.g., asylum is granted, refugees are resettled, etc.). Note that applying for asylum is not a settlement status event, as an application for asylum is not an indication of where the individual(s) ultimately settled.',
        default=None)
    outcome: Optional[list[event]] = Field(
        description='Captures events that occurred as a result of the human displacement event but are not already covered in specific slots of the Displacementplate. Also used for hypothetical and averted events, as in Phase 1 templates: for these cases an irrealis marker is used.',
        default=None)
    group_identity: Optional[list[span]] = Field(
        description='Mentions pertaining to the group identity of the displaced humans, be that ethnic, national, religious, or otherwise.',
        default=None)
    injured_count: Optional[list[span]] = Field(
        description='Mentions enumerating those displaced humans who have been injured during their migration.',
        default=None)
    killed_count: Optional[list[span]] = Field(
        description='Mentions enumerating those displaced humans who have been killed during their migration.',
        default=None)
    missing_count: Optional[list[span]] = Field(
        description='Mentions enumerating those displaced humans who have gone missing during their migration.',
        default=None)
    detained_count: Optional[list[span]] = Field(
        description='Mentions enumerating those displaced humans who have been detained during their migration.',
        default=None)
    blocked_migration_count: Optional[list[span]] = Field(
        description='Mentions enumerating those displaced humans who have been blocked from further migration.',
        default=None)
    Transitory_events: Optional[list[span|event]] = Field(
        description='Events that occurred during the displaced humans’ travels.',
        default=None)
    Assistance_provided: Optional[list[event]] = Field(
        description='Assistance events provided to the displaced humans, typically Provide-Aid events. Note that the type field on these events may provide useful information.',
        default=None)
    Assistance_needed: Optional[list[event]] = Field(
        description='Assistance request events provided to the displaced humans, typically Aid-Needs events. Note that the type field on these events may provide useful information.',
        default=None)
    template_anchor: span = Field(
        description='The anchor of the template',
        default=None)



class Template(BaseModel):
    #templates: conlist(item_type=Protestplate|Corruplate|Terrorplate|Epidemiplate|Disasterplate|Displacementplate, max_length=9)
    templates: list[Union[Protestplate,Corruplate,Terrorplate,Epidemiplate,Disasterplate,Displacementplate], Field(discriminator='template_type')]

class Indv_Protestplate(BaseModel):
    templates: list[Protestplate]

class Indv_Corruplate(BaseModel):
    templates: list[Corruplate]

class Indv_Terrorplate(BaseModel):
    templates: list[Terrorplate]

class Indv_Epidemiplate(BaseModel):
    templates: list[Epidemiplate]

class Indv_Disasterplate(BaseModel):
    templates: list[Disasterplate]

class Indv_Displacementplate(BaseModel):
    templates: list[Displacementplate]
    