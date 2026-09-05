"""
Centralised handling of user-visible strings for BabelBrain.

BabelBrain uses Qt's native translation machinery. Every user-visible string is
marked up in the source with its *English* wording, and a compiled catalogue
(``.qm``) supplies a replacement at run time. Nothing else in the code changes:
variable names, dict keys, HDF5 fields, CSV headers and the ``.ini``/YAML
persisted values all stay in English.

Two independent axes are supported, and they stack:

  * **language**  - a full catalogue, e.g. ``babelbrain_fr.qm``.
  * **mode**      - a small *overlay* catalogue that only overrides the handful
                    of anatomy terms that differ between the transcranial (TU)
                    and transvertebral (TVUS) applications, e.g.
                    ``babelbrain_tvus_en.qm``.

Qt searches installed translators in *reverse* order of installation and falls
through to the next one when a translator has no entry for a string. So the
overlay is installed last and only needs to carry the ~20 strings it actually
changes; everything else falls through to the language catalogue, and then to
the English source text. This is why TVUS labelling needs no ``if/else`` at all
in the GUI code.

--------------------------------------------------------------------------
MARKING UP A STRING - read this before adding one
--------------------------------------------------------------------------
Wrap the literal English text in ``TR()``::

    from Localization import TR
    ...
    self.LocMTB = make_button("LocMTB", TR("Max. Temp. Brain"))

``TR`` takes the text only; the context is always ``TRANSLATION_CONTEXT``
("BabelBrain"), so hand-written markup all lands in one small catalogue block.

The argument must be a plain string **literal**. ``lupdate`` reads the source
statically, so a name, an f-string or a concatenation is invisible to it. For a
message that needs values interpolated, mark up the *format* and apply the
values afterwards::

    TR("Y pos = %3.2f mm") % ycoord                        # good
    TR(f"Y pos = {ycoord:3.2f} mm")                        # NOT extracted

``update_translations.sh`` teaches ``lupdate`` about ``TR`` with
``-tr-function-alias tr+=TR`` and then pins the extracted context; see that
script and ``i18n/README.md``. ``QCoreApplication.translate("BabelBrain", ...)``
still works and is still extracted - ``TR`` is simply the short spelling of it.

Strings coming from the two Qt Designer forms (``SelFiles``, ``Options``) are
already wrapped by ``pyside6-uic`` under the contexts ``SelFilesDialog`` and
``OptionsDialog``; they need no source change.

Only mark strings that reach a widget or a plot. Do not mark strings that are
written to a file or compared against - notably the CSV/Brainsight export
headers in ``Babel_Thermal.py`` and any combo-box entry read back with
``currentText()``. Combo entries defined in the Designer forms carry
``notr="true"`` so that ``uic`` emits a plain literal for them; keep that on any
entry you add.

--------------------------------------------------------------------------
REBUILDING THE CATALOGUES
--------------------------------------------------------------------------
``lupdate`` rescans the sources and merges into the ``.ts`` files (keeping
existing translations); ``lrelease`` compiles them to the ``.qm`` files that
ship with the app. Both live in the Qt bin directory of the conda environment,
e.g. ``$CONDA_PREFIX/lib/qt6/bin``. See ``i18n/README.md``.
"""

import os
import platform
import sys
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QLocale, QTranslator

# Context used by every hand-written markup call. Designer-generated forms use
# their own contexts ("SelFilesDialog", "OptionsDialog").
TRANSLATION_CONTEXT = "BabelBrain"

# Operating modes that have an overlay catalogue. "tu" (transcranial) is the
# source language itself, so it has no overlay.
MODE_TU = None
MODE_TVUS = 'tvus'

_IS_MAC = platform.system() == 'Darwin'

# Keeps the installed QTranslator objects alive: Qt does not take ownership,
# and a garbage-collected translator silently stops translating.
_INSTALLED = []


def resource_path():  # needed for bundling
    """Absolute path to the resource root; works for dev and for PyInstaller.

    Mirrors the helper of the same name in BabelBrain.py and CTZTEProcessing.py.
    Duplicated rather than imported so that this module stays free of any
    BabelBrain import (it is loaded before the main window exists).
    """
    if not _IS_MAC:
        return os.path.split(Path(__file__))[0]

    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        bundle_dir = Path(sys._MEIPASS)
    else:
        bundle_dir = Path(__file__).parent

    return bundle_dir


def i18n_dir():
    """Directory holding the compiled .qm catalogues."""
    return os.path.join(resource_path(), 'i18n')


def _load(basename):
    """Load <i18n_dir>/<basename>.qm, or return None when it is absent."""
    path = os.path.join(i18n_dir(), basename + '.qm')
    if not os.path.isfile(path):
        return None
    tr = QTranslator()
    if not tr.load(path):
        print('Localization: failed to load ' + path)
        return None
    return tr


def install_translators(app, language='en', mode=MODE_TU):
    """Install the language catalogue and, on top of it, the mode overlay.

    Call once, right after the QApplication is created and *before* any widget
    is built - Qt resolves a translation when the string is used, and the forms
    set their text only at construction time.

    ``language`` is an ISO code ('en', 'fr', ...); ``mode`` is MODE_TU or
    MODE_TVUS. Missing catalogues are skipped silently, so an untranslated build
    simply shows the English source text. Returns the list of names installed,
    which is useful for the log line in main().
    """
    installed = []

    # Base language. 'en' is the source language, so it normally has no
    # catalogue - the source text is already English.
    if language and language != 'en':
        tr = _load('babelbrain_' + language)
        if tr is not None:
            app.installTranslator(tr)
            _INSTALLED.append(tr)
            installed.append('babelbrain_' + language)

    # Mode overlay, installed last so that it wins over the base language.
    if mode:
        name = 'babelbrain_' + mode + '_' + (language or 'en')
        tr = _load(name)
        if tr is not None:
            app.installTranslator(tr)
            _INSTALLED.append(tr)
            installed.append(name)
        else:
            print('Localization: no overlay catalogue "' + name +
                  '" - labels stay in transcranial wording')

    # The UI language must not change how numbers are parsed or written: a
    # French locale would switch QDoubleSpinBox to a decimal comma, which would
    # corrupt the values persisted to the .ini/YAML files and break the
    # scripting and server interfaces. Numeric formatting stays C/English.
    QLocale.setDefault(QLocale(QLocale.English, QLocale.UnitedStates))

    return installed


def mode_for_config(bTVUS_OPERATION):
    """Map the BabelBrain operating-mode flag onto an overlay name."""
    return MODE_TVUS if bTVUS_OPERATION else MODE_TU


def TR(text):
    """Translate *text* in the shared "BabelBrain" context.

    The standard way to mark a user-visible string. Pass a plain string literal
    so that lupdate can extract it - see the module docstring for the rules and
    for how the build script is told about this function.
    """
    return QCoreApplication.translate(TRANSLATION_CONTEXT, text)
