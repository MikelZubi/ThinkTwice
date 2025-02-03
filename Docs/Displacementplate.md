# Displacementplate
## Fields

### `human_displacement_event`

**Type:** `list[event (optional)`

Holds event anchors typically associated with human displacement -primarily refugee- movement events. Can be considered to serve as triggers.

### `over_time`

**Type:** `bool (optional)`

A flag indicating whether this is an individual event, or an extended state of affairs.

### `origin`

**Type:** `list[span (optional)`

Mentions of one or more locations from which the displaced humans are fleeing.

### `current_location`

**Type:** `list[span (optional)`

Mentions of any locations reported to be where the displaced humans currently are.

### `transiting_location`

**Type:** `list[span (optional)`

Mentions of any locations through which the displaced humans transited on their way to their current location.

### `destination`

**Type:** `list[span (optional)`

Mentions of any locations that the displaced humans intend to continue on towards.

### `when`

**Type:** `list[span (optional)`

Mentions of times or durations associated with the events involved

### `total_displaced_count`

**Type:** `list[span (optional)`

As with Epidemiplates, these are phrases that enumerate the total population of displaced humans.

### `event_or_SoA_at_origin`

**Type:** `list[event (optional)`

Events that precipitated the migration: these can be natural disasters, war, famine, political or economic crises, etc.

### `settlement_status_event_or_SoA`

**Type:** `list[event (optional)`

Events (of any type) that pertain to the status of the displaced humans. Events that indicate the final status of the individuals (e.g., asylum is granted, refugees are resettled, etc.). Note that applying for asylum is not a settlement status event, as an application for asylum is not an indication of where the individual(s) ultimately settled.

### `outcome`

**Type:** `list[event (optional)`

Captures events that occurred as a result of the human displacement event but are not already covered in specific slots of the Displacementplate. Also used for hypothetical and averted events, as in Phase 1 templates: for these cases an irrealis marker is used.

### `group_identity`

**Type:** `list[span (optional)`

Mentions pertaining to the group identity of the displaced humans, be that ethnic, national, religious, or otherwise.

### `injured_count`

**Type:** `list[span (optional)`

Mentions enumerating those displaced humans who have been injured during their migration.

### `killed_count`

**Type:** `list[span (optional)`

Mentions enumerating those displaced humans who have been killed during their migration.

### `missing_count`

**Type:** `list[span (optional)`

Mentions enumerating those displaced humans who have gone missing during their migration.

### `detained_count`

**Type:** `list[span (optional)`

Mentions enumerating those displaced humans who have been detained during their migration.

### `blocked_migration_count`

**Type:** `list[span (optional)`

Mentions enumerating those displaced humans who have been blocked from further migration.

### `Transitory_events`

**Type:** `list[span | event (optional)`

Events that occurred during the displaced humans’ travels.

### `Assistance_provided`

**Type:** `list[event (optional)`

Assistance events provided to the displaced humans, typically Provide-Aid events. Note that the type field on these events may provide useful information.

### `Assistance_needed`

**Type:** `list[event (optional)`

Assistance request events provided to the displaced humans, typically Aid-Needs events. Note that the type field on these events may provide useful information.

### `template_anchor`

**Type:** `<class 'span'>`

The anchor of the template
