# UI string catalogues

BabelBrain's user-visible labels go through Qt's translation machinery. This
directory holds the catalogues; `../Localization.py` holds the loading logic and
the rules for marking a string up in the source.

## What is here

| File | Role |
|---|---|
| `babelbrain_tvus_en.ts` | Editable source catalogue: the **TVUS overlay**. Open it in Qt Linguist. |
| `babelbrain_tvus_en.qm` | Compiled form, loaded at run time. Committed so a build needs no Qt tools. |
| `update_translations.sh` | Rescans the sources (`lupdate`) and recompiles (`lrelease`). |
| `normalize_context.py` | Called by that script; keeps the context of `TR()` strings consistent across an `lupdate` run. |

Naming is `babelbrain_<mode>_<language>.qm` for an overlay and
`babelbrain_<language>.qm` for a full language catalogue.

## How the stacking works

`install_translators()` installs the base language first, then the mode overlay
on top. Qt searches installed translators in **reverse** order of installation
and falls through when one has no entry for a string, so:

```
"Max. Temp. Skull"
   -> babelbrain_tvus_en.qm   has an entry  -> "Max. Temp. Vertebra"
"CSF"
   -> babelbrain_tvus_en.qm   no entry
   -> (base language)         no entry
   -> English source text     -> "CSF"
```

That fall-through is the whole point: the overlay only carries the 22 anatomy
terms that actually differ between transcranial and transvertebral use, and the
GUI code contains no mode test at all. `lrelease` drops untranslated entries
from the `.qm`, so an entry left empty in the `.ts` costs nothing.

A `.ts` therefore lists **every** marked-up string in the app (246 today), most
of them empty. That is expected — it is the translator's worklist, and it is
what a future full catalogue such as `babelbrain_fr.ts` will fill in completely.

## Adding a new language

Nothing translates anything automatically. A `.ts` file is a table of rows, one
per user-visible string, and a person who speaks the language types the right
hand side. The steps below are just how that table gets created and shipped.

### 1. Generate the empty catalogue (maintainer, once per language)

A new language has no home in the repo until you make one. Add it to
`CATALOGUES` in `update_translations.sh` — the full catalogue, and optionally a
TVUS overlay for the same language:

```bash
CATALOGUES=(
    babelbrain_tvus_en
    babelbrain_fr          # full French
    babelbrain_tvus_fr     # French vertebral terms only (optional)
)
```

Then run it:

```bash
export QTBIN=$CONDA_PREFIX/lib/qt6/bin
./update_translations.sh
```

That creates `babelbrain_fr.ts` holding every marked-up string in the app, each
one with an empty translation:

```xml
<message>
    <source>Hide marks</source>
    <translation type="unfinished"/>
</message>
```

Commit the generated `.ts`. No code change is needed: `Localization.py` finds a
catalogue by filename, so `babelbrain_fr.qm` and `babelbrain_tvus_fr.qm` are
picked up as soon as they exist.

Nobody ever writes a `.ts` by hand, and nobody edits the `<source>` side — that
half is regenerated from the code on every run.

### 2. Fill it in (translator)

Send the translator that **one file**. They need Qt Linguist and nothing else —
not the repo, not Python, not a conda environment. Linguist is a standalone
download, or `$CONDA_PREFIX/lib/qt6/bin/Linguist.app` on macOS.

They open `babelbrain_fr.ts`, see the source strings on the left and an empty box
on the right, type, and `Ctrl+Enter` to confirm each one and move on. Then they
save and send the file back.

Worth telling them explicitly:

* **Leave anything you are unsure of empty.** An empty row falls back to English,
  which is always safe; a wrong translation is worse than none. A 30 %-translated
  build ships perfectly well — 30 % translated, 70 % English, nothing broken.
* **Keep the placeholders.** `%s`, `%3.2f`, `%1` and `\n` must survive into the
  translation. Linguist warns when they do not match.
* The `<comment>` shown on some entries is a disambiguation note explaining where
  the string is used. It is never displayed to the user.

For a TVUS overlay (`babelbrain_tvus_fr.ts`) ask for **only** the anatomy terms —
about twenty rows — and to leave the rest empty; those fall through to the full
language catalogue. That is a much smaller job, and it is usually a different
person: it needs the clinical vocabulary rather than general fluency.

### 3. Integrate and ship (maintainer)

```bash
cp ~/Downloads/babelbrain_fr.ts .
./update_translations.sh          # merges the new text, recompiles the .qm
```

Check the report — `Generated N translation(s)` should match what they filled in.
Commit both the `.ts` and the `.qm`. The `.qm` is committed deliberately so that
a PyInstaller build needs no Qt tools, and `BabelBrain.spec` globs `i18n/*.qm`,
so a new language is bundled with no change to the spec.

### 4. Selecting the language at run time — NOT IMPLEMENTED YET

`install_translators()` accepts any language code and the stacking works, but
`main()` currently hardcodes `language='en'`:

```python
_installed = Localization.install_translators(
    app, language='en',
    mode=Localization.mode_for_config(args.bTVUSOperation))
```

So a translated catalogue can be built and bundled today, but a user has no way
to ask for it. Before a localized build is useful, add one of: a CLI flag
(mirroring `-bTVUSOperation`), a dropdown in Advanced Options persisted to the
config, or auto-detection from `QLocale.system()`. The choice takes effect at the
next launch either way — the forms set their text once, at construction.

## What must NOT be translatable

Combo-box entries in the two Designer forms are device names, algorithm names,
tool names and `NO`/`YES` flags. The code reads them back with `currentText()` /
`findText()`, persists them to the `.ini` and YAML, and the scripting, server and
pytest interfaces address them by those exact strings. `uic` used to wrap them in
`QCoreApplication.translate`, which meant a translator could have broken all of
that from a `.ts` file. They are now marked in the `.ui` as

```xml
<string notr="true">real CT</string>
```

so `uic` emits a plain literal and they never reach a catalogue. Keep it that way
for any new combo entry, and apply the same rule in code: never mark up a string
that is written to a file, compared against, or used as a key. The CSV /
Brainsight export headers in `Babel_Thermal.py` are the other example - their
wording deliberately duplicates translated on-screen labels and must stay
English.

## Contexts

A catalogue entry is keyed by *(context, source text)*, not by source text alone.

| Context | Where it comes from |
|---|---|
| `BabelBrain` | Hand-written markup: `TR("...")` (see below) |
| `SelFilesDialog` | `SelFiles/form.ui`, wrapped automatically by `pyside6-uic` |
| `OptionsDialog` | `Options/form.ui`, likewise |

The two dialog contexts used to both be called `Dialog`, which silently merged
their entries; they were renamed so that identical wording in the two dialogs
can be translated independently. Do not rename them again once translations
exist — every entry would be orphaned.

## `TR()` and the context dance

Source code marks a string with the short helper from `Localization.py`:

```python
from Localization import TR
...
self.LocMTB = make_button("LocMTB", TR("Max. Temp. Brain"))
```

`TR(text)` is `QCoreApplication.translate("BabelBrain", text)` — one context for
all hand-written markup, so the overlay stays a single small block. The argument
must be a plain string **literal**; `lupdate` reads the source statically, so an
f-string or a concatenation is invisible to it. Interpolate afterwards:

```python
TR("Y pos = %3.2f mm") % ycoord          # good
TR(f"Y pos = {ycoord:3.2f} mm")          # NOT extracted — silently untranslatable
```

When the same English word needs different translations in different places,
pass a disambiguation tag as a second argument. It is a note to the translator,
never shown to the user, and it gives each occurrence its own catalogue row:

```python
TR("Range", "ZTE normalized")            # its own row
TR("Range", "HU threshold")              # a separate row
```

Strings in different contexts (`BabelBrain` vs `SelFilesDialog` vs
`OptionsDialog`) are already separate rows and need no tag.

`lupdate` learns about `TR` from `-tr-function-alias tr+=TR`. Because `TR` takes
no context argument, lupdate files those messages under an **empty** context,
while the catalogue stores them under `BabelBrain`. `normalize_context.py`
converts between the two representations around the lupdate call:

```
normalize_context.py --denormalize   "BabelBrain" -> ""     (before lupdate)
lupdate ...                          contexts now agree, translations survive
normalize_context.py                 ""  -> "BabelBrain"    (after lupdate)
```

Both directions are required. Without the `--denormalize` step, lupdate compares
freshly extracted empty-context messages against stored `BabelBrain` ones,
concludes every string vanished and every string is new, and drops the
translations. `update_translations.sh` already does this; just use the script.

The normalizer refuses to run if lupdate emits any context other than the empty
one, `BabelBrain`, `SelFilesDialog` or `OptionsDialog` — that would mean the
catalogue and the run-time lookup had started to disagree, which otherwise fails
silently by falling back to the English source text.

## Rebuilding

```bash
export QTBIN=$CONDA_PREFIX/lib/qt6/bin     # lupdate/lrelease live here
./update_translations.sh                   # rescan sources, then compile
./update_translations.sh --release         # compile only
```

Use the Qt tools matching the PySide6 the app runs on (6.9.3 for the `-314`
environments). `lupdate` **merges**: existing translations survive, new strings
arrive as unfinished, and strings that vanished from the source are marked
`vanished` rather than deleted.

When the markup sweep widens, add the newly marked files to `SOURCES` in
`update_translations.sh` — anything not listed there is invisible to `lupdate`.

## Editing translations

See **Adding a new language** above for the full round trip. For a quick edit to
an existing catalogue: open the `.ts` in Qt Linguist
(`$CONDA_PREFIX/lib/qt6/bin/Linguist.app` on macOS, `linguist` elsewhere) or edit
the `<translation>` elements by hand, leave an entry empty to inherit the English
wording, then run `./update_translations.sh --release` to regenerate the `.qm`
and commit both files.

`lrelease` ships a translation that has text even when it is still marked
`unfinished`; only genuinely empty entries are dropped. So a translator who
forgets to confirm an entry still gets their text into the build — the finished
flag tracks progress, it does not gate the output.

## Gotcha to watch for

An overlay entry is keyed by the **English source text**. Rewording a base label
in the source silently orphans its override — `lupdate` will mark the old entry
`vanished` and add the new wording as unfinished, and the app quietly falls back
to the transcranial term. After any label rewording, re-run
`update_translations.sh` and check the report for vanished entries.
