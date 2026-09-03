# StemForge

Stem separation, MIDI transcription, and key/BPM detection for macOS.

Point it at a track and it writes a folder containing separated stems, a MIDI
file per stem, and the key, tempo and downbeat position — then hands the whole
thing to Logic Pro or GarageBand.

Everything runs locally. No audio leaves the machine, and no account is needed.

## Install

```bash
make install
```

Python 3.10–3.12 and `ffmpeg` (for MP3/M4A/AAC input) are the prerequisites:

```bash
brew install python@3.11 ffmpeg
```

Model weights download on first use, roughly 300 MB for the default separation
model, and are cached in `~/.cache` afterwards.

## The app

```bash
make app
```

That builds `dist/StemForge.app`. Drag it to `/Applications` and double-click
it. It starts the local engine and opens the interface in your browser; closing
the window leaves nothing running that you have to clean up, and reopening the
app reuses the running engine.

The bundle launches the virtualenv it was built against rather than embedding
Python, so keep this checkout in place — or rebuild after moving it.

## Command line

```bash
stemforge analyze track.wav
```

```
Key    F# minor  (Camelot 11A, 74% confident)
Tempo  174.02 BPM  (96% confident)
First downbeat  0.312 s
Duration        212.4 s
```

The full pipeline, with sixteenth-note quantisation on the transcribed MIDI:

```bash
stemforge process track.wav --quantize 4 -o ~/Music/StemForge
```

Six stems instead of four, three separation passes for a cleaner split, and
straight into Logic afterwards:

```bash
stemforge process track.wav --model htdemucs_6s --shifts 3 --open-in logic
```

`stemforge models` lists the separation models and reports which compute device
is in use. `stemforge process --help` covers the rest.

## What comes out

```
Track/
├── analysis.json          key, tempo, beat grid, note counts, settings used
├── stems/
│   ├── drums.wav          24-bit WAV, source sample rate, aligned to 00:00:00
│   ├── bass.wav
│   ├── other.wav
│   └── vocals.wav
├── midi/
│   ├── bass.mid           tempo- and key-stamped, one instrument per file
│   ├── drums.mid          General MIDI percussion
│   └── all_stems.mid      every transcription in one multi-track file
└── daw/
    ├── tempo_map.mid      tempo and key signature only
    └── README.txt         the import steps for this specific track
```

## Logic Pro and GarageBand

StemForge is a standalone app, not an Audio Units plugin. Separation and
transcription take seconds to minutes per track and need the whole file up
front, so neither fits inside a real-time plugin slot.

What it does instead is produce a session both DAWs import cleanly, and drive
the hand-off for you. **Stems → Logic Pro** and **Stems → GarageBand** in the
app open the stems directly; the `daw/` folder covers the manual route:

1. New empty project.
2. Drag `daw/tempo_map.mid` in and accept the tempo import. The project is now
   at the detected tempo and key.
3. Drag the `stems` folder in as multiple tracks, all at bar 1.
4. Drag files from `midi/` onto software-instrument tracks.

Because every stem starts at 00:00:00 and is the same length as the source,
they stay phase-aligned with each other and sum back to the original mix.

`daw/README.txt` reports the first downbeat position. When a track does not
start exactly on the beat, that number is how far to nudge the regions so bar 1
lands on the downbeat.

## Precision controls

**Separation passes** (`--shifts`, the Precision slider) re-run the model on
randomly offset copies of the audio and average the results, which suppresses
the artefacts that a single pass leaves behind. Two or three passes is a
noticeable improvement on dense mixes; processing time scales with the count.

**Model** picks the trade-off. `htdemucs` is the fast default; `htdemucs_ft` is
the cleanest four-stem split and around four times slower; `htdemucs_6s` adds
guitar and piano stems, with piano the least reliable of the six.

**Isolation** — the “everything else” mix (`--residual`) writes the sum of the
stems you did not select, so you get a soloed part and its exact complement
rather than two independent guesses.

**Quantisation** snaps transcribed notes to a grid derived from the detected
tempo and downbeat. Strength below 100% moves notes part of the way, which
keeps some of the original feel. Leave it off to preserve the performance as
played.

## How the analysis works

**Key** correlates the track's chroma against Albrecht & Shanahan key profiles
across all 24 major and minor rotations. Two extra pieces of evidence break the
tie that catches simple implementations — tracks tend to open and close on the
tonic, so the first and last sections are weighted separately, and the bass
line is analysed on its own, since root movement is what distinguishes a key
from its relative. Output includes the Camelot code and the runners-up.

**Tempo** beat-tracks the onset envelope, then refines the result from the
median inter-beat interval with outliers discarded, giving a fractional BPM
rather than a rounded one. Octave errors are folded into 70–180 BPM.

**Downbeats** come from the beat phase whose chroma flux and low-frequency
energy are highest — chord changes land on bar lines. The grid is then
extrapolated back to the start of the file, since beat trackers routinely miss
the opening beat.

**MIDI** uses Basic Pitch per stem, with the frequency range and onset
thresholds set per instrument: transcribing a bass line with vocal settings
turns its harmonics into notes that were never played. Drums take a different
path — onsets are detected per frequency band, pooled, and each one classified
by which bands actually spike, so a snare's low-frequency body is not also
counted as a kick while a simultaneous kick and hi-hat still both get written.

## Limitations

- Transcription of dense polyphonic material (the `other` stem, layered synths)
  is approximate. Bass, single-voice melodies and piano are the reliable cases.
- The drum transcriber writes kick, snare and hi-hat. Toms, cymbals and
  percussion are not classified, and hi-hats struck at the same moment as a
  snare are sometimes absorbed by it.
- Key detection assumes one key for the whole track. Modulations are not
  reported; a low confidence score is the signal that something is off.
- Tempo detection assumes a roughly constant tempo. Live and rubato material
  gives a low confidence score and an unreliable beat grid.

## Development

```bash
make test        # fast unit tests
make test-all    # adds the end-to-end run through both models
```

The end-to-end tests render a synthetic eight-bar track in A minor at 128 BPM
with a known chord progression and drum pattern, then assert that the pipeline
recovers the key, the tempo, the chord roots in the bass MIDI, and the grid the
notes were quantised to.

## Windows and Linux

The engine is portable — `stemforge process` and `stemforge serve` work
anywhere PyTorch does. What is macOS-only is the `.app` bundle, the native file
pickers, and the Logic/GarageBand hand-off. On other platforms, run
`stemforge serve` from a terminal and type paths into the file field.
