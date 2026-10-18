'''
JSON-based synchronization with Brainsight (Brainsight 2.6 and newer).

Brainsight 2.6 replaces the legacy text files in ~/.BabelBrainSync
(Input_Target.txt, Input_Anatomical.txt, Output.txt, ...) with two JSON files:

    BabelBrainInput.json   written by Brainsight before launching BabelBrain
                           with -bInUseWithBrainsight (anatomical, segmentation
                           and targets with their UUIDs)
    BabelBrainOutput.json  written by BabelBrain to report acoustic/thermal
                           results and modified trajectories, keyed by the
                           target UUIDs received in the input

The *only* trigger for the JSON scheme is the presence of BabelBrainInput.json.
As soon as it is found, it is renamed to BabelBrainInput-read.json (before
parsing), so a later session started by an older Brainsight (that only writes
the legacy text files) never picks up a stale JSON input. When the JSON scheme
is in use, the legacy output files are not written.

Both files carry a ``schemaVersion`` field to allow BabelBrain and Brainsight to
evolve independently. BabelBrain answers in the same schema version it received.

This module has no Qt dependency so it can be unit-tested in isolation.
'''
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SYNC_PATH = str(Path.home()) + os.sep + '.BabelBrainSync'

INPUT_JSON = 'BabelBrainInput.json'
INPUT_JSON_READ = 'BabelBrainInput-read.json'
OUTPUT_JSON = 'BabelBrainOutput.json'

# Legacy (pre Brainsight 2.6) output files; never written in JSON mode
LEGACY_OUTPUT_FILES = ['Output.txt',
                       'Output.json',
                       'Output_TargetModified.txt',
                       'Output_Thermal.txt',
                       'Output_Thermal.json']

# Number of entries used for positions in the output: a 3x4 [R|t] matrix in
# row-major order, the same layout Brainsight uses for "position" in the input
POSITION_LENGTH = 12

# Mechanical corrections below this norm (mm) are considered as no correction
_MIN_CORRECTION = 1e-6

# Characters in target names that would break the Brainsight trajectory text
# file (tab-separated) or the output file names built from them
_INVALID_NAME_CHARS = re.compile(r'[\t\r\n/\\]')


class BrainsightSyncError(Exception):
    '''Error with the content of the Brainsight JSON input.'''


@dataclass
class BrainsightTarget:
    name: str
    uuid: str
    mat4: np.ndarray


@dataclass
class BrainsightInput:
    schema_version: int
    created_by: str
    segmentation_path: str
    anatomical_path: str
    coordinate_system: str
    targets: list
    output_path: str = None
    source_file: str = ''

    @property
    def names(self):
        return [t.name for t in self.targets]

    @property
    def uuids(self):
        return [t.uuid for t in self.targets]


###############################################################################
# Detection of the JSON scheme
###############################################################################

def IsJSONSyncRequested(sync_path=SYNC_PATH):
    '''True if Brainsight left a (not yet consumed) BabelBrainInput.json.'''
    return os.path.isfile(os.path.join(sync_path, INPUT_JSON))


def ClaimInputJSON(sync_path=SYNC_PATH):
    '''
    Rename BabelBrainInput.json to BabelBrainInput-read.json and return the new
    path. This is done before parsing so the input is consumed exactly once,
    even if its content turns to be invalid.
    '''
    src = os.path.join(sync_path, INPUT_JSON)
    dst = os.path.join(sync_path, INPUT_JSON_READ)
    os.replace(src, dst)
    return dst


def RemoveStaleOutputs(sync_path=SYNC_PATH):
    '''Remove outputs (JSON and legacy) left by a previous session.'''
    for fname in [OUTPUT_JSON] + LEGACY_OUTPUT_FILES:
        fpath = os.path.join(sync_path, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)


###############################################################################
# Parsing
###############################################################################

_TRAILING_COMMA = re.compile(r'("(?:\\.|[^"\\])*")|,(\s*[\]}])')


def _StripTrailingCommas(text):
    return _TRAILING_COMMA.sub(lambda m: m.group(1) if m.group(1) is not None else m.group(2), text)


def LoadJSONText(text):
    '''
    Parse JSON text. Brainsight must produce strict JSON, but as a safety net
    we tolerate trailing commas (seen in early examples), with a warning.
    '''
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        relaxed = _StripTrailingCommas(text)
        if relaxed == text:
            raise
        data = json.loads(relaxed)
        print('WARNING: Brainsight JSON input is not strictly compliant (%s); '
              'accepted after removing trailing commas' % str(e))
        return data


def PositionToMat4(position):
    '''
    Convert a Brainsight JSON position (12 values, 3x4 [R|t] row-major; 16
    values for a full 4x4 are also accepted) to a 4x4 matrix with the same
    convention as ConvMatTransform.ReadTrajectoryBrainsight (translation in
    Mat4[:3,3], orientation axes in the columns Mat4[:3,0..2]).
    '''
    arr = np.asarray(position, dtype=float).ravel()
    if arr.size not in (12, 16):
        raise BrainsightSyncError('position must have 12 or 16 values, it has %i' % arr.size)
    Mat4 = np.eye(4)
    Mat4[:3, :] = arr[:12].reshape(3, 4)
    if arr.size == 16 and not np.allclose(arr[12:], [0, 0, 0, 1]):
        raise BrainsightSyncError('last row of a 16-value position must be [0, 0, 0, 1]')
    return Mat4


def Mat4ToPosition(Mat4, length=POSITION_LENGTH):
    '''Inverse of PositionToMat4.'''
    Mat4 = np.asarray(Mat4, dtype=float)
    values = [float(v) for v in Mat4[:3, :].ravel()]
    if length == 16:
        values += [0.0, 0.0, 0.0, 1.0]
    return values


def _Require(data, key, kind, context='input'):
    if key not in data:
        raise BrainsightSyncError('Missing required field "%s" in Brainsight %s' % (key, context))
    value = data[key]
    if not isinstance(value, kind):
        raise BrainsightSyncError('Field "%s" in Brainsight %s has the wrong type' % (key, context))
    return value


def _ParseInputV1(data):
    targets = []
    for n, entry in enumerate(_Require(data, 'targets', list)):
        context = 'target #%i' % (n + 1)
        if not isinstance(entry, dict):
            raise BrainsightSyncError('Brainsight %s is not an object' % context)
        name = _Require(entry, 'name', str, context).strip()
        uuid = _Require(entry, 'uuid', str, context).strip()
        position = _Require(entry, 'position', list, context)
        if len(name) == 0 or len(uuid) == 0:
            raise BrainsightSyncError('Brainsight %s has an empty name or uuid' % context)
        if _INVALID_NAME_CHARS.search(name):
            raise BrainsightSyncError('Brainsight target name "%s" contains invalid characters (tab, new line, / or \\)' % name)
        targets.append(BrainsightTarget(name=name, uuid=uuid, mat4=PositionToMat4(position)))

    if len(targets) == 0:
        raise BrainsightSyncError('Brainsight input does not contain any target')
    names = [t.name for t in targets]
    if len(set(names)) != len(names):
        raise BrainsightSyncError('Brainsight target names must be unique (they are used to name output files): ' + ', '.join(names))
    uuids = [t.uuid for t in targets]
    if len(set(uuids)) != len(uuids):
        raise BrainsightSyncError('Brainsight target UUIDs must be unique')

    output_path = data.get('outputPath', None)
    if output_path is not None:
        if not isinstance(output_path, str):
            raise BrainsightSyncError('Field "outputPath" in Brainsight input has the wrong type')
        output_path = output_path.strip()
        if len(output_path) == 0:
            output_path = None

    return BrainsightInput(schema_version=1,
                           created_by=data.get('createdBy', 'Brainsight'),
                           segmentation_path=_Require(data, 'segmentationPath', str).strip(),
                           anatomical_path=_Require(data, 'anatomicalPath', str).strip(),
                           coordinate_system=_Require(data, 'coordinateSystem', str).strip(),
                           targets=targets,
                           output_path=output_path)


# One parser per schema version; add new versions here
_INPUT_PARSERS = {1: _ParseInputV1}
SUPPORTED_SCHEMA_VERSIONS = tuple(sorted(_INPUT_PARSERS.keys()))


def ParseInput(data):
    '''Validate and convert the decoded JSON input to a BrainsightInput.'''
    if not isinstance(data, dict):
        raise BrainsightSyncError('Brainsight input must be a JSON object')
    version = data.get('schemaVersion', None)
    if not isinstance(version, int) or isinstance(version, bool):
        raise BrainsightSyncError('Missing or invalid "schemaVersion" in Brainsight input')
    if version not in _INPUT_PARSERS:
        if version > max(SUPPORTED_SCHEMA_VERSIONS):
            raise BrainsightSyncError('Brainsight input uses schema version %i, but this version of BabelBrain '
                                      'only supports up to version %i.\nPlease update BabelBrain.'
                                      % (version, max(SUPPORTED_SCHEMA_VERSIONS)))
        raise BrainsightSyncError('Unsupported Brainsight input schema version %i' % version)
    return _INPUT_PARSERS[version](data)


def ReadInputFile(fname):
    with open(fname, 'r', encoding='utf-8') as f:
        text = f.read()
    try:
        data = LoadJSONText(text)
    except json.JSONDecodeError as e:
        raise BrainsightSyncError('Brainsight input %s is not valid JSON:\n%s' % (fname, str(e)))
    res = ParseInput(data)
    res.source_file = fname
    return res


###############################################################################
# Output
###############################################################################

def WriteJSONAtomic(fname, data):
    '''Write via a temporary file + rename so readers never see a partial file.'''
    dirname = os.path.dirname(fname)
    fd, tmpname = tempfile.mkstemp(dir=dirname, prefix='.tmp_', suffix='.json')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        os.replace(tmpname, fname)
    except BaseException:
        if os.path.isfile(tmpname):
            os.remove(tmpname)
        raise


class BrainsightJSONSync:
    '''
    Session state reported to Brainsight through BabelBrainOutput.json.

    Targets are always matched by index (order of the targets in the input,
    which is also the order of the trajectories in BabelBrain), never by
    name, as names can be modified (e.g. "_mod_" prefix) during the session.

    For each target we keep:
        base pose       pose used to generate the domain in Step 1 (input
                        pose, or the one selected in PlanTUS)
        correction      mechanical correction (RAS, mm) found in Step 2; the
                        reported pose is the base pose with the translation
                        reduced by this correction. It is always applied to the
                        base pose, so repeated exports never accumulate.

    The whole output file is rewritten on every change.
    '''

    def __init__(self, BrainsightIn, output_fname, created_by):
        self._input = BrainsightIn
        self.output_fname = output_fname
        self.created_by = created_by
        n = len(BrainsightIn.targets)
        self._base = [t.mat4.copy() for t in BrainsightIn.targets]
        self._base_sources = [[] for _ in range(n)]
        self._correction = [np.zeros(3) for _ in range(n)]
        self._acoustic = [''] * n
        self._thermal = [''] * n
        self._acoustic_combined = ''
        self._thermal_combined = ''

    @property
    def NumberTargets(self):
        return len(self._input.targets)

    def CurrentMat4(self, idx):
        Mat4 = self._base[idx].copy()
        Mat4[:3, 3] -= self._correction[idx]
        return Mat4

    def ModifiedBy(self, idx):
        sources = list(self._base_sources[idx])
        if np.linalg.norm(self._correction[idx]) > _MIN_CORRECTION:
            sources.append('MechanicalCorrection')
        return sources

    def _SplitResults(self, paths):
        '''
        BabelBrain keeps result paths in a list with one entry per target and,
        when there is more than one target, an extra last entry for the merged
        result. Empty strings mean "not available".
        '''
        n = self.NumberTargets
        paths = list(paths)
        individual = (paths[:n] + [''] * n)[:n]
        combined = paths[n] if (n > 1 and len(paths) > n) else ''
        return individual, combined

    def UpdateAcousticResults(self, paths):
        self._acoustic, self._acoustic_combined = self._SplitResults(paths)
        self.Write()

    def UpdateThermalResults(self, paths):
        self._thermal, self._thermal_combined = self._SplitResults(paths)
        self.Write()

    def SetMechanicalCorrection(self, idx, correction_ras):
        self._correction[idx] = np.asarray(correction_ras, dtype=float).ravel()[:3].copy()
        self.Write()

    def ReplaceTrajectory(self, idx, Mat4, source):
        '''
        A new base pose for target idx (e.g. selected in PlanTUS). Results
        for that target and the merged results are discarded, as they no longer
        correspond to the trajectory; Step 1 and Step 2 need to be run again.
        '''
        self._base[idx] = np.asarray(Mat4, dtype=float).copy()
        self._base_sources[idx] = [source]
        self._correction[idx] = np.zeros(3)
        self._acoustic[idx] = ''
        self._thermal[idx] = ''
        self._acoustic_combined = ''
        self._thermal_combined = ''
        self.Write()

    def _TargetEntry(self, idx):
        t = self._input.targets[idx]
        return {'targetName': t.name,
                'targetUUID': t.uuid,
                'modifiedPosition': Mat4ToPosition(self.CurrentMat4(idx))}

    def _ResultEntries(self, paths):
        res = []
        for idx, p in enumerate(paths):
            if p:
                entry = self._TargetEntry(idx)
                res.append({'targetName': entry['targetName'],
                            'targetUUID': entry['targetUUID'],
                            'imagePath': p,
                            'modifiedPosition': entry['modifiedPosition']})
        return res

    def _CombinedEntries(self, path):
        if not path:
            return []
        return [{'targetUUIDList': self._input.uuids, 'imagePath': path}]

    def BuildOutput(self):
        targets = []
        for idx in range(self.NumberTargets):
            entry = self._TargetEntry(idx)
            entry['modifiedBy'] = self.ModifiedBy(idx)
            targets.append(entry)
        return {'schemaVersion': self._input.schema_version,
                'createdBy': self.created_by,
                'coordinateSystem': self._input.coordinate_system,
                'targets': targets,
                'acousticResults': self._ResultEntries(self._acoustic),
                'acousticCombinedResults': self._CombinedEntries(self._acoustic_combined),
                'thermalResults': self._ResultEntries(self._thermal),
                'thermalCombinedResults': self._CombinedEntries(self._thermal_combined)}

    def Write(self):
        WriteJSONAtomic(self.output_fname, self.BuildOutput())
