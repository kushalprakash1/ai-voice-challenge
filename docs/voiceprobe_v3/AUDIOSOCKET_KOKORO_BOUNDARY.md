# AudioSocket / Kokoro v3 media boundary

This phase adapts the existing working media contracts without changing the
legacy Asterisk adapter or `autonomous_phone.py`.

The extracted production contracts are:

- Kokoro output is converted from 24 kHz float audio to 8 kHz PCM16.
- AudioSocket speech is split into 20 ms frames and sent at playback cadence.
- Patient speech and continuous idle silence share one send lock.
- Continuous idle silence starts after Asterisk supplies the AudioSocket UUID.
- All inbound PCM is recorded.
- Inbound STT forwarding is muted only during actual patient playback plus the
  existing 350 ms echo guard; reasoning/TTS preparation does not stop listening.

`KokoroTelephonyRenderer` lazily reuses the existing synthesis and telephony
conversion functions.

`AudioSocketKokoroSpeechTask` satisfies the `queue_frame()` contract required
by `PipecatRuntimeBridge`. It schedules synthesis and AudioSocket playback in
background threads so the asyncio loop remains available to Deepgram Flux.
Runtime busy state is released only through the playback-finished callback.

`AudioSocketV3MediaBoundary` owns the shared AudioSocket send lock contract,
starts the existing idle-silence sender, records every inbound PCM payload, and
applies the existing playback/echo mute boundary before handing PCM to the
future Pipecat input feeder.

This phase intentionally does not change dialing, AMI Originate ordering,
AudioSocket socket ownership, objective classification, or live-call behavior.
