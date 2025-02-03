# Corruplate
## Fields

### `charged_with`

**Type:** `list[span | event (optional)`

The crimes that the individual has been charged with, coded as Basic events “[patient Uyukaeve] had been caught accepting the [event bribe]”

### `corrupt_event`

**Type:** `list[span | event (optional)`

Triggers of the template

### `judicial_actions`

**Type:** `list[span | event (optional)`

Investigations, trials, sentences, and so forth noted in the narrative as having taken place

### `fine`

**Type:** `list[span (optional)`

Any monetary damages or seizures levelled as punishment.

### `over_time`

**Type:** `bool (optional)`

A flag indicating whether this is an individual case of corruption, or a systematic state of corruption affecting many corrupt individuals or institutions.

### `outcome_averted`

**Type:** `list[span (optional)`

Events that were either averted or are noted as not having occurred, e.g., “no injuries” (Basic events do not code negation or other realis factors).

### `outcome_occurred`

**Type:** `list[span | event (optional)`

Events that occurred because of the corruption.

### `outcome_hypothetical`

**Type:** `list[span | event (optional)`

Events that are only noted as potentially occurring because of the corruption.

### `prison_term`

**Type:** `list[span (optional)`

The duration(s) of any prison sentence mentioned as applicable (with appropriate irrealis status indicated as appropriate).

### `where`

**Type:** `list[span (optional)`

Location(s) of the corruption.

### `who`

**Type:** `list[span (optional)`

The individual(s) being accused of corruption.

### `template_anchor`

**Type:** `<class 'span'>`

The anchor of the template
