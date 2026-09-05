#!/bin/bash
# Rescan the sources for marked-up strings and recompile the catalogues.
#
#   ./update_translations.sh            # update .ts, then compile .qm
#   ./update_translations.sh --release  # compile .qm only
#
# lupdate MERGES: existing translations are kept, new strings are added as
# unfinished, and strings that disappeared from the source are marked vanished.
# Never hand-edit the .ts files apart from the <translation> elements - use
# Qt Linguist ($CONDA_PREFIX/lib/qt6/bin/Linguist.app on macOS).
#
# lrelease drops untranslated (empty/unfinished) entries from the .qm. That is
# exactly what makes the overlay catalogues work: an entry left empty in
# babelbrain_tvus_en.ts falls through to the base English source text, so an
# overlay only needs to fill in the terms it actually changes.
#
# Requires the Qt tools that ship with the conda env, e.g.
#   $CONDA_PREFIX/lib/qt6/bin/{lupdate,lrelease}
# Use the ones matching the PySide6 the app runs on (6.9.3 for the -314 envs).

set -e
cd "$(dirname "$0")"

QTBIN="${QTBIN:-$CONDA_PREFIX/lib/qt6/bin}"
LUPDATE="$QTBIN/lupdate"
LRELEASE="$QTBIN/lrelease"

# Sources carrying marked-up strings. Add files here as the markup sweep
# widens (Phase 2) - anything not listed is invisible to lupdate.
SOURCES=(
    ../BabelBrain.py
    ../Babel_Thermal/Babel_Thermal.py
    ../Babel_Thermal/ThermalForm.py
    ../GUIComponents/TxPanelBase.py
)

# Qt Designer forms. uic already wraps their strings; lupdate reads the .ui
# directly, under the contexts SelFilesDialog / OptionsDialog.
FORMS=(
    ../SelFiles/form.ui
    ../Options/form.ui
)

# One catalogue per (language, mode). "tvus_en" is an overlay: it is installed
# on top of the base language and only overrides the anatomy terms.
CATALOGUES=(
    babelbrain_tvus_en
)

if [ "$1" != "--release" ]; then
    for cat in "${CATALOGUES[@]}"; do
        echo "== lupdate -> $cat.ts"
        "$LUPDATE" -locations none "${SOURCES[@]}" "${FORMS[@]}" -ts "$cat.ts"
    done
fi

for cat in "${CATALOGUES[@]}"; do
    echo "== lrelease -> $cat.qm"
    "$LRELEASE" "$cat.ts" -qm "$cat.qm"
done
