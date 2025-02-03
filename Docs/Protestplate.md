# Protestplate
## Fields

### `arrested`

**Type:** `list[span (optional)`

Description or count of those arrested.

### `imprisoned`

**Type:** `list[span | event (optional)`

Description or count of those jailed.

### `killed`

**Type:** `list[span (optional)`

Description or count of those killed, e.g., "two people," "two"

### `occupy`

**Type:** `list[span (optional)`

Any space or building taken over, e.g., "the local government offices".

### `over_time`

**Type:** `bool (optional)`

A flag indicating whether this is an individual protest, or a period of frequent civil unrest, with regular protests arising in different places and times, e.g., the Arab Spring.

### `organizer`

**Type:** `list[span (optional)`

The group/individuals leading the protest, e.g., "the Workers Party".

### `outcome_averted`

**Type:** `list[span | event (optional)`

Events that were either averted or are noted as not having occurred, e.g., "no injuries" (Basic events do not code negation or other realis factors).

### `outcome_occurred`

**Type:** `list[span | event (optional)`

Events that occurred because of the corruption.

### `outcome_hypothetical`

**Type:** `list[span | event (optional)`

Events that are only noted as potentially occurring because of the corruption.

### `protest_against`

**Type:** `list[span | event (optional)`

Any events presented as what the protest is meant to end, e.g. “corruption,” “unemployment” (an state-of-affairs Basic event), “a [ban] on [the washing lines],” etc.

### `protest_event`

**Type:** `list[span | event (optional)`

Triggers of the template

### `protest_for`

**Type:** `list[span | event (optional)`

Any event presented as the aim of the protest, e.g., “the corrupt must face justice,” coded as {agt ø, head “face justice,” ptt “the corrupt”}.

### `when`

**Type:** `list[span (optional)`

Date of the protest, as best identifiable: “Thursday,” “last month,” etc.

### `where`

**Type:** `list[span (optional)`

Location(s) of the protest.

### `who`

**Type:** `list[span (optional)`

References to protest participants, e.g., “hundreds of young men.”.

### `wounded`

**Type:** `list[span (optional)`

Description of any injured participants, or a count if that is all that is available, e.g., “a woman,” “40,” “several police officers”.

### `template_anchor`

**Type:** `<class 'span'>`

The anchor of the template
