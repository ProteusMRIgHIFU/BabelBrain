"""Import plumbing for user-created (custom) transducers.

A custom transducer is generated into ``~/.config/BabelBrain/Transducers`` and
imported from there at runtime, so it never ships inside the frozen bundle --
only the templates it builds on do. This module keeps those templates reachable
for transducers that were generated before the templates used package-qualified
imports.
"""

import importlib
import importlib.abc
import importlib.util
import logging
import sys

# Packages holding the template modules an older generated file may import by
# bare name, tried in order.
_TEMPLATE_PACKAGES = (
    "babel_transducers.transducer_templates",
    "TranscranialModeling.babel_integration.integration_templates",
)


class _AliasLoader(importlib.abc.Loader):
    """Re-exports an already-imported module under a second, bare name.

    The alias gets its own module object so the packaged module's ``__name__``
    and ``__spec__`` are left alone, but the classes themselves are shared --
    subclassing the alias and the package path gives the same base class.
    """

    def __init__(self, target):
        self._target = target

    def create_module(self, spec):
        return None  # default module object

    def exec_module(self, module):
        for key, value in vars(self._target).items():
            if not key.startswith("__"):
                setattr(module, key, value)


class _TemplateAliasFinder(importlib.abc.MetaPathFinder):
    """Last-resort finder mapping a bare template name onto its package path."""

    def find_spec(self, fullname, path=None, target=None):
        # Only bare top-level names, and only ones that look like a template.
        if path is not None or "." in fullname or not fullname.startswith("babel_"):
            return None

        for package_name in _TEMPLATE_PACKAGES:
            try:
                module = importlib.import_module(f"{package_name}.{fullname}")
            except ImportError:
                continue
            logging.info(f"Resolved legacy transducer import '{fullname}' "
                         f"to {package_name}.{fullname}")
            return importlib.util.spec_from_loader(fullname, _AliasLoader(module))

        return None


def register_template_aliases() -> None:
    """Make the transducer templates importable by their bare module names.

    Transducers generated before this change do

        module_directory = Path.cwd() / 'BabelBrain' / 'babel_transducers' / ...
        sys.path.insert(0, str(module_directory))
        from babel_focused_array_tx import FocusedArrayTx

    which only resolves when the app happens to be started from the repository
    root, and never resolves in a frozen app because the templates live in the
    PYZ archive instead of on disk -- BabelBrain then reports the transducer as
    unsupported. The finder installed here is consulted only after the normal
    import machinery has failed, so it rescues those files without shadowing
    anything. Newly generated transducers import the packages directly and never
    reach it.
    """

    if any(isinstance(finder, _TemplateAliasFinder) for finder in sys.meta_path):
        return

    sys.meta_path.append(_TemplateAliasFinder())
