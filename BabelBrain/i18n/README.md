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

That fall-through is the whole point: the overlay only carries the ~19 anatomy
terms that actually differ between transcranial and transvertebral use, and the
GUI code contains no mode test at all. `lrelease` drops untranslated entries
from the `.qm`, so an entry left empty in the `.ts` costs nothing.

A `.ts` therefore lists **every** marked-up string in the app (145 today), most
of them empty. That is expected — it is the translator's worklist, and it is
what a future full catalogue such as `babelbrain_fr.ts` will fill in completely.

## Contexts

A catalogue entry is keyed by *(context, source text)*, not by source text alone.

| Context | Where it comes from |
|---|---|
| `BabelBrain` | Hand-written markup: `QCoreApplication.translate("BabelBrain", "...")` |
| `SelFilesDialog` | `SelFiles/form.ui`, wrapped automatically by `pyside6-uic` |
| `OptionsDialog` | `Options/form.ui`, likewise |

The two dialog contexts used to both be called `Dialog`, which silently merged
their entries; they were renamed so that identical wording in the two dialogs
can be translated independently. Do not rename them again once translations
exist — every entry would be orphaned.

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

Use Qt Linguist (`$CONDA_PREFIX/lib/qt6/bin/Linguist.app` on macOS, `linguist`
elsewhere), or edit the `<translation>` elements by hand. Leave an entry empty to
inherit the English wording. After editing, run `./update_translations.sh
--release` to regenerate the `.qm` and commit both files.

## Gotcha to watch for

An overlay entry is keyed by the **English source text**. Rewording a base label
in the source silently orphans its override — `lupdate` will mark the old entry
`vanished` and add the new wording as unfinished, and the app quietly falls back
to the transcranial term. After any label rewording, re-run
`update_translations.sh` and check the report for vanished entries.
