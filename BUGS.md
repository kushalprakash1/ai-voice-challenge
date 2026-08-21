# Gold-call findings

These findings describe remote-agent behavior observed in Calls 1–6. They do
not include failures attributed to VoiceProbe.

## Transfer reached a test-line dead end

**Observed in:** Call 2

**Expected:** Accepting an offered staff escalation should connect the caller
to a useful support destination or clearly report that transfer is unavailable.

**Observed:** The remote agent announced a transfer, but the handoff reached a
test-line goodbye instead of clinic staff.

**Why it matters:** A successful transfer announcement can mask an unusable
escalation path.

## Medication demo workflow cannot add refill data

**Observed in:** Call 3

**Expected:** The demo workflow should provide a coherent way to add or collect
the medication information required for its refill flow.

**Observed:** The agent reported that the chart contained no refillable
medications, then said adding lisinopril to the demo profile was unsupported.

**Why it matters:** A caller can enter the advertised refill workflow but
cannot supply the state needed to complete it.

## Location inventory and address changed during one call

**Observed in:** Call 4

**Expected:** Offered office locations and their addresses should remain stable
through selection and follow-up questions.

**Observed:** The agent first reported one main office in Nashville, then
identified the selected main clinic as an Austin address, and later treated the
Nashville address as a second location.

**Why it matters:** Inconsistent location identity makes directions and
location-specific scheduling unreliable.

## Office-hours question was not recognized

**Observed in:** Call 4

**Expected:** “What are the hours for that location?” should return the active
office's hours or request a day if necessary.

**Observed:** The agent interpreted “hours” as “towers” and returned building
information instead of office hours.

**Why it matters:** A common spoken request failed under ordinary recognition
variation.

## Registered name was corrupted when repeated

**Observed in:** Call 5

**Expected:** A registered caller name should be repeated and spelled
consistently with the value supplied during profile creation.

**Observed:** After registering “Gyeong-hyeon Gwak,” the agent repeated a
materially different name and spelling.

**Why it matters:** Name corruption can attach later workflow state to the
wrong identity.

## Doctor-specific hours collapsed to clinic hours

**Observed in:** Call 5

**Expected:** A question about one doctor's location and working hours should
return doctor-specific information or state that it is unavailable.

**Observed:** The agent supplied a location for the selected doctor but then
answered with generic clinic hours.

**Why it matters:** Clinic opening hours do not establish that a particular
doctor is available then.

## Farthest-date objective was not followed initially

**Observed in:** Call 6

**Expected:** A request for the furthest currently bookable date should search
the scheduling horizon before proposing a slot.

**Observed:** The agent first returned the next available date and only exposed
the furthest date after the caller rejected the earlier option and restated the
objective.

**Why it matters:** “Next” and “furthest” are different scheduling goals and can
produce materially different appointments.

## Accepted appointment ended without confirmation

**Observed in:** Call 1

**Expected:** After the caller accepts a concrete slot, the agent should confirm
whether the appointment was booked before ending the interaction.

**Observed:** The caller accepted the offered appointment, but the interaction
terminated before an explicit booking confirmation was received.

**Why it matters:** Slot acceptance alone does not prove that a scheduling
transaction completed.
