# MUC Database
The Message Understanding Conferences (MUC) focused on extracting information about terrorist events. The objtective of this database it to do template filling, in which systems must identify incidents, represented by predefined schemas or templates, in a document, and populate fields in those templates with relevant information extracted or inferred from the text.

## Relevant Incidents
Relevant incidents are, in general, violent acts perpetrated with political aims and a motive of intimidation. These are acts of terrorism. Terrorist acts may be perpetrated by an "illegal, subnational, clandestine group," which includes known guerrilla and drug-trafficking organizations. Their targets may be just about anything/anybody except (a) another such group or member of such a group or (b) a military/police intallation or force (with the same nationality as the perpetrator), or a member of such a force, in which case the incident is presumed to be a purely guerrilla act and is not relevant to the MUC terrorist incident database.

However, if a guerrilla warfare incident happens to affect civilian personnel or property, whether intentionally or accidentally, the incident becomes relevant and should be included in the database. In these cases, the database should contain information on both the military and the nonmilitary targets.

Similarly, the database will include incidents of sabotage on a nation's infrastructure. It will also include incidents perpetrated against individuals who are former members of the military, e.g., a murder perpetrated against a retired general.

Also, under certain circumstances, the perpetrator may be a member of the government, including the military. This is the case when the target is civilian.

If an article discusses multiple relevant types of terrorist incidents, each should be recorded in a separate template; however, some articles may not contain any relevant incidents. If an article discusses more than one instance of the same relevant type of terrorist incident, each such incident should be captured in a separate template . A "separate instance" is one which has different information about the location, date or perpetrator. 

Incidents fall into the following categories: 'kidnapping,' 'attack,' 'bombing,' 'robbery,' 'arson,' or 'forced work stoppage.' The 'attack' category should only be used when a terrorist incident does not clearly fit into any of the other categories."

## Fields
The slots for each template are the next ones: 

### `incident_type`
The type of incident, the values can be: 'kidnapping','attack','bombing','robbery','arson', or 'forced work stoppage'
### `PerpInd`
A person responsible for the incident. 
### `PerpOrg`
An organization responsible for the incident.
### `Target`
An inanimate object that was attacked.
### `Victim`
The name of a person who was the obvious or apparent target of the attack or who became a victim of the attack.
### `Weapon`
A device used by the perpetrator(s) in carrying out the terrorist act.

## Format
The format of the output needs to be as follows:
templates:[*List of identified `templates`; empty if there are not any*]
Where each `template` is as follows:
{{incident_type: *Identified incident*,
PerpInd: [*List of identified individuals; empty if no information is found in the text.*],
PerpOrg: [*List of identified organizations,; empty if no information is found in the text.*],
Target: [*List of targets; empty if no information is found in the text.*],
Victim: [*List of victims; empty if no information is found in the text.*],
Weapon: [*List of weapons; empty if no information is found in the text.*]}}