# Disasterplate
## Fields

### `major_disaster_event`

**Type:** `list[span | event (optional)`

Holds instances of Environmental, Natural Phenomenon, or Famine events (or SoAs), or also Weather/Environmental damage events. Can be considered to serve as triggers.

### `over_time`

**Type:** `bool (optional)`

A flag indicating whether this is an individual event, or an extended state of affairs.

### `where`

**Type:** `list[span (optional)`

Mentions of one or more locations associated with the disaster.

### `when`

**Type:** `list[span (optional)`

Mentions of times or durations associated with the events involved.

### `injured_count`

**Type:** `list[span (optional)`

Mentions that enumerate the individuals injured in a disaster

### `killed_count`

**Type:** `list[span (optional)`

Mentions that enumerate the individuals who died in a disaster.

### `missing_count`

**Type:** `list[span (optional)`

Mentions that enumerate individuals who have gone missing in a disaster.

### `outcome`

**Type:** `list[event (optional)`

Events that occurred as a result of a disaster but are not already covered in specific slots of the Disasterplate.

### `responders`

**Type:** `list[span (optional)`

Entities designating the responders to the natural disaster.

### `damage`

**Type:** `list[span (optional)`

Entities that have been damaged in the disaster, typically the patients of a Weather-or-environmental-Damage Basic event

### `affected_cumulative_count`

**Type:** `list[span (optional)`

For those cases where the toll of a disaster is identified over time: this slot holds entities that enumerate the cumulative count of affected individuals.

### `individuals_affected`

**Type:** `list[span (optional)`

Entities that provide descriptions of the affected individuals.

### `rescued_count`

**Type:** `list[span (optional)`

Entities enumerating the individuals rescued in the response to a disaster.

### `rescue_events`

**Type:** `list[event (optional)`

This slot holds Basic events of the Rescue or Evacuate types (as opposed to the patients of these events, as in the fillers of the rescued-count slot).

### `assistance_provided`

**Type:** `list[event (optional)`

This slot captures Provide-Aid events that were in response to the disaster.

### `assistance_needed`

**Type:** `list[event (optional)`

Similarly to the previous slot, this slot captures Aid-Needs events that originated in the disaster.

### `related_natural_phenomena`

**Type:** `list[event (optional)`

Weather, Environmental, or similar events that occurred as a result of the predicating disaster. For example, floods might be related to a hurricane disaster. Because it can sometimes be difficult to determine which event should be treated as “primary,” in this example, both “floods” and “hurricane” would be included in this slot.

### `announce_disaster_warnings`

**Type:** `list[event (optional)`

Any communication events that warn of the impending disaster before it strikes.

### `declare_emergency`

**Type:** `list[event (optional)`

Any communication events declaring the emergency (typically a Declare-Emergency Basic event).

### `disease_outbreak_events`

**Type:** `list[event (optional)`

Disease outbreak events that follow upon the disaster, e.g., the cholera outbreak after the Haiti earthquake. Not just Disease-Outbreak; this can also include other epidemic event types.

### `repair`

**Type:** `list[event (optional)`

Repair events that are part of the response to the disaster.

### `human_displacement_events`

**Type:** `list[event (optional)`

Any refugee movement events resulting from the disaster (including those related to detaining refugees).

### `template_anchor`

**Type:** `<class 'span'>`

The anchor of the template
