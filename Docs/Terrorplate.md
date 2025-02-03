# Terrorplate
## Fields

### `blamed_by`

**Type:** `list[span (optional)`

Those who are asserting the identity of the perpetrators.

### `claimed_by`

**Type:** `list[span (optional)`

Those who have claimed responsibility for the terrorist event(s).

### `completion`

**Type:** `Literal['planned', 'thwarted', 'failed', 'successful' (optional)`

Whether the terrorism event(s) are considered to be completed or not. One of "planned", "thwarted", "failed", or "successful".

### `coordinated`

**Type:** `bool (optional)`

Whether the terrorism event(s) are considered to be coordinated or not.

### `killed`

**Type:** `list[span (optional)`

Mentions of those people who were killed.

### `kidnapped`

**Type:** `list[span (optional)`

Mentions of those people who were kidnapped.

### `named_perp`

**Type:** `list[span (optional)`

Those to whom the terrorist event(s) are attributed.

### `named_perp_org`

**Type:** `list[span (optional)`

Mentions of the organization(s) the perpetrators belong to.

### `named_organizer`

**Type:** `list[span (optional)`

Those to whom the planning of the terrorist event(s) are attributed.

### `over_time`

**Type:** `bool (optional)`

A flag indicating whether this is an individual case of terrorism, or a systematic state of terrorism

### `outcome_averted`

**Type:** `list[span (optional)`

Events that were prevented by virtue of the events described in this template.

### `outcome_occurred`

**Type:** `list[span | event (optional)`

Events or states-of-affairs that have actually taken place by virtue of the terrorist events.

### `outcome_hypothetical`

**Type:** `list[span | event (optional)`

Events or states-of-affairs that could have taken place by due to the terrorist events.

### `perp_captured`

**Type:** `list[span (optional)`

Mentions of perpetrators of the terrorist events who were captured.

### `perp_killed`

**Type:** `list[span (optional)`

Mentions of perpetrators who were killed in the course of the terrorist events.

### `perp_objective`

**Type:** `list[span | event (optional)`

Mentions of events which are identified as being desired to have taken place (or states-of-affairs to have come about) by virtue of the terrorist events.

### `perp_wounded`

**Type:** `list[span (optional)`

Mentions of perpetrators who were wounded in the course of the terrorist events.

### `target_human`

**Type:** `list[span (optional)`

One or more people, named or unnamed, who are said to be the targets of the terrorist events.

### `target_physical`

**Type:** `list[span (optional)`

The facility or geo-political location that was being targeted by the terrorist events.

### `terror_event`

**Type:** `list[span | event (optional)`

Triggers of the template

### `type`

**Type:** `Literal['arson', 'assault', 'bombing', 'kidnapping', 'murder', 'unspecified' (optional)`

One of "arson", "assault", "bombing", "kidnapping", "murder", or "unspecified".

### `weapon`

**Type:** `list[span (optional)`

Mentions of the weapons or other instruments used to carry out the terrorist events.

### `when`

**Type:** `list[span (optional)`

Mentions of times or durations associated with the events involved.

### `where`

**Type:** `list[span (optional)`

Mentions of the location(s) at which the terrorist events have taken place.

### `wounded`

**Type:** `list[span (optional)`

Mentions of those people who were wounded.

### `template_anchor`

**Type:** `<class 'span'>`

The anchor of the template
