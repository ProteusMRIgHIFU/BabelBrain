"""
Centralized BabelBrain artifact naming (Phase 1).

A single place that enumerates the files the pipeline produces, so callers
(the job server today; the GUI result-loading later) get a *deterministic*
manifest instead of globbing the output folder.

Phase 1 intentionally does NOT touch the writers. It reproduces the exact names
by reading the already-resolved paths off the live BabelBrain widget and its
Step-2/Step-3 sub-objects, plus GetThermalOutName for the per-timing thermal
files. Because it reads the same attributes the app itself used, it can't drift
from what was actually written.

Phase 2 will invert this: make the builders here the single source of truth and
have BabelBrain.py / OutputFileNames / GetThermalOutName delegate to them (pure
function of Config), at which point build_manifest can run without a live widget.

Naming convention (for reference): every artifact derives from the prefix
    <trajectoryID>_<TxSystem>_<freqkHz>kHz_<PPW>PPW_
(merged results join IDs with '+'). See BabelBrain.py:UpdateMaskParameters,
TranscranialModeling OutputFileNames(), and ThermalModeling GetThermalOutName().
"""
import os
from collections import namedtuple


# key    : stable logical name (e.g. 'acoustic_field', 'thermal_field')
# path   : absolute file path
# step   : 1 (domain) | 2 (acoustic) | 3 (thermal)
# fmt    : nifti | h5 | npz | mat | yaml | csv | stl
# role   : primary (a result a client wants) | intermediate | export
Artifact = namedtuple('Artifact', ['key', 'path', 'step', 'fmt', 'role'])


class Manifest:
    """An immutable, filterable list of Artifacts with JSON-friendly output."""

    def __init__(self, artifacts):
        # de-duplicate on path while preserving order
        seen, uniq = set(), []
        for a in artifacts:
            if a.path and a.path not in seen:
                seen.add(a.path)
                uniq.append(a)
        self._arts = uniq

    def __iter__(self):
        return iter(self._arts)

    def __len__(self):
        return len(self._arts)

    def primary(self):
        return Manifest([a for a in self._arts if a.role == 'primary'])

    def intermediates(self):
        return Manifest([a for a in self._arts if a.role == 'intermediate'])

    def by_step(self, step):
        return Manifest([a for a in self._arts if a.step == step])

    def existing(self):
        return Manifest([a for a in self._arts if os.path.isfile(a.path)])

    def paths(self):
        return [a.path for a in self._arts]

    def to_list(self):
        return [a._asdict() for a in self._arts]


# ── helpers ──────────────────────────────────────────────────────────────────
def _as_list(v):
    """Sol names are a str for single-element Tx, a list for phased arrays."""
    if v is None:
        return []
    return list(v) if isinstance(v, (list, tuple)) else [v]


def _first(v):
    lst = _as_list(v)
    return lst[0] if lst else None


_STEP2_SUFFIX_CACHE = None


def _step2_suffix_map():
    """{logical key -> filename suffix after CPREFIX}, extracted live from
    TranscranialModeling.OutputFileNames so the Step-2 output set (the
    FullElasticSolution*/RayleighFreeWater* NIfTIs from Step10_GetResults) is
    reproduced from its authoritative definition and can't drift."""
    global _STEP2_SUFFIX_CACHE
    if _STEP2_SUFFIX_CACHE is None:
        _STEP2_SUFFIX_CACHE = {}
        try:
            from TranscranialModeling.BabelIntegrationBASE import OutputFileNames
            sent = os.sep + '__BABEL_SENT__' + os.sep + 'mask.nii.gz'
            of = OutputFileNames(sent, 'TGT', 250e3, 6, '', False)
            cprefix = of['DataForSim'][:-len('DataForSim.h5')]
            for k, v in of.items():
                if k == 'DataForSim' or not isinstance(v, str) or not v.startswith(cprefix):
                    continue
                suf = v[len(cprefix):]
                # The writers transform the name: SaveNiftiEnforcedISO strips the
                # '__' marker (X__.nii.gz -> X.nii.gz); ResaveNormalized writes a
                # _Sub_NORM sibling for each _Sub NIfTI. Reproduce both so the
                # manifest matches what is actually on disk. (Phase 2 will move
                # these transforms into the writers' single source of truth.)
                if suf.endswith('.nii.gz'):
                    real = suf.split('__.nii.gz')[0] + '.nii.gz' if suf.endswith('__.nii.gz') else suf
                    _STEP2_SUFFIX_CACHE[real] = real
                    if real.endswith('_Sub.nii.gz'):
                        norm = real.replace('_Sub.nii.gz', '_Sub_NORM.nii.gz')
                        _STEP2_SUFFIX_CACHE[norm] = norm
                else:
                    _STEP2_SUFFIX_CACHE[suf] = suf
        except Exception:
            _STEP2_SUFFIX_CACHE = {}
    return _STEP2_SUFFIX_CACHE


def _step2_extra_files(sol_name, role, keyprefix='acoustic_'):
    """Every Step-2 output that shares sol_name's CPREFIX (field/phase/refocus
    NIfTIs, etc.). sol_name is the DataForSim.h5, whose CPREFIX = name[:-len].
    Reuses OutputFileNames' suffixes, so no name is hardcoded here."""
    if not sol_name or not sol_name.endswith('DataForSim.h5'):
        return []
    cprefix = sol_name[:-len('DataForSim.h5')]
    out = []
    for suf in _step2_suffix_map():           # keys are the real suffixes
        fmt = 'nifti' if suf.endswith('.nii.gz') else ('h5' if suf.endswith('.h5') else 'other')
        label = suf.rsplit('.', 1)[0].replace('.nii', '')   # drop extension
        out.append(Artifact(keyprefix + label, cprefix + suf, 2, fmt, role))
    return out


def _thermal_basenames(base_field, combos, isppa):
    """The thermal output base names (no extension) for each timing combination,
    reproducing ThermalModeling.GetThermalOutName exactly."""
    if not base_field or isppa is None:
        return []
    from ThermalModeling.CalculateTemperatureEffects import GetThermalOutName
    out = []
    for c in combos or []:
        out.append(GetThermalOutName(base_field,
                                     c['Duration'], c['DurationOff'], c['DC'],
                                     isppa, c['PRF'], c['Repetitions']))
    return out


# ── manifest builder (Phase 1: reads the live widget) ────────────────────────
def build_manifest(bb):
    """Return the Manifest of artifacts for the pipeline state of BabelBrain
    widget *bb*. Paths are the *expected* names (whether or not the file exists);
    call .existing() to filter to what was actually written."""
    C = bb.Config
    ids = C.get('ID', []) or []
    n = len(ids)
    arts = []

    # ── Step 1: domain generation (per trajectory) ──
    outmask = getattr(bb, '_outnameMask', [])
    t1wres = getattr(bb, '_T1W_resampled_fname', [])
    prefpath = getattr(bb, '_prefix_path', [])
    for i in range(n):
        if i < len(outmask):
            arts.append(Artifact('mask', outmask[i], 1, 'nifti', 'primary'))
        if i < len(t1wres):
            arts.append(Artifact('t1w_resampled', t1wres[i], 1, 'nifti', 'intermediate'))
        if i < len(prefpath):
            p = prefpath[i]
            if C.get('bUseCT'):
                arts.append(Artifact('ct', p + 'CT.nii.gz', 1, 'nifti', 'intermediate'))
                arts.append(Artifact('ct_calibration', p + 'CT-cal.npz', 1, 'npz', 'intermediate'))
            if C.get('bExtractAirRegions'):
                arts.append(Artifact('air_regions', p + 'AirRegions.nii.gz', 1, 'nifti', 'intermediate'))
    tracking = getattr(bb, '_trackingtimefile', None)
    if tracking:
        arts.append(Artifact('execution_times', tracking, 1, 'yaml', 'intermediate'))

    # ── Step 2: acoustic field (per trajectory + merged) ──
    ac_panels = getattr(bb.AcSim, '_acPanels', []) if hasattr(bb, 'AcSim') else []
    for i in range(n):
        panel = ac_panels[i] if i < len(ac_panels) else None
        if not panel:
            continue
        for p in _as_list(panel.get('FullSolName')):
            arts.append(Artifact('acoustic_field', p, 2, 'h5', 'primary'))
            arts.extend(_step2_extra_files(p, 'primary'))
        for p in _as_list(panel.get('WaterSolName')):
            arts.append(Artifact('acoustic_field_water', p, 2, 'h5', 'intermediate'))
            arts.extend(_step2_extra_files(p, 'intermediate', keyprefix='acoustic_water_'))
    merged_ac = getattr(getattr(bb, 'AcSim', None), '_MergedResultsFullSolName', None)
    for p in _as_list(merged_ac):
        arts.append(Artifact('acoustic_field_merged', p, 2, 'h5', 'primary'))
        arts.extend(_step2_extra_files(p, 'primary', keyprefix='acoustic_merged_'))

    # ── Step 3: thermal field (per trajectory × timing combo + merged) ──
    # Timing combinations live on ThermalSim.Config (parsed from the thermal
    # profile YAML), a separate object from the main Config.
    tconf = getattr(getattr(bb, 'ThermalSim', None), 'Config', None)

    def _tget(key, default):
        for src in (tconf, C):
            if src is None:
                continue
            try:
                return src[key]
            except Exception:
                continue
        return default

    combos = _tget('AllDC_PRF_Duration', [])
    isppa = _tget('BaseIsppa', None)
    for i in range(n):
        panel = ac_panels[i] if i < len(ac_panels) else None
        base = _first(panel.get('FullSolName')) if panel else None
        for th in _thermal_basenames(base, combos, isppa):
            arts.append(Artifact('thermal_field', th + '.h5', 3, 'h5', 'primary'))
            arts.append(Artifact('thermal_field_mat', th + '.mat', 3, 'mat', 'intermediate'))
    for th in _thermal_basenames(_first(merged_ac), combos, isppa):
        arts.append(Artifact('thermal_field_merged', th + '.h5', 3, 'h5', 'primary'))

    return Manifest(arts)
