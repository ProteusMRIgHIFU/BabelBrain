'''
Reading, modifying and writing Brainsight trajectory (target) text files.

A Brainsight export looks like::

    # Version: 15
    # Coordinate system: NIfTI:S:Aligned
    # Created by: Brainsight 2.6
    # Units: millimetres, degrees, milliseconds, and microvolts
    # Encoding: UTF-8
    # Notes: Each column is delimited by a tab. ...
    # Target Name<TAB>Loc. X<TAB>Loc. Y<TAB>Loc. Z<TAB>m0n0<TAB>...<TAB>m2n2
    T1<TAB>-9.826350809<TAB>8.357484884<TAB>...

Lines starting with '#' are metadata only *before* the column header line
("# Target Name<TAB>Loc. X<TAB>..."). Brainsight allows target names starting
with '#', so after the column header a line is identified as a trajectory by
its content (tab-separated, numeric location and orientation columns), not by
its first character.

Lines that are not trajectories (metadata, blank lines, other sections after
the targets) are kept verbatim when writing the file back, and trajectories
that are not modified are written back exactly as read.

Orientation convention (same as BabelBrain has always used): translation in
Mat4[:3,3] and the three orientation vectors (m0n*, m1n*, m2n*) in the columns
Mat4[:3,0], Mat4[:3,1] and Mat4[:3,2].
'''
import re

import numpy as np

COLUMN_NAME = 'Target Name'
LOCATION_COLUMNS = ('Loc. X', 'Loc. Y', 'Loc. Z')
MATRIX_COLUMNS = ('m0n0', 'm0n1', 'm0n2',
                  'm1n0', 'm1n1', 'm1n2',
                  'm2n0', 'm2n1', 'm2n2')
DEFAULT_COLUMNS = (COLUMN_NAME,) + LOCATION_COLUMNS + MATRIX_COLUMNS

DEFAULT_METADATA = (('Version', '15'),
                    ('Coordinate system', 'NIfTI:S:Aligned'),
                    ('Created by', 'BabelBrain'),
                    ('Units', 'millimetres, degrees, milliseconds, and microvolts'),
                    ('Encoding', 'UTF-8'),
                    ('Notes', 'Each column is delimited by a tab. Each value within a column is delimited by a semicolon.'))

_METADATA_LINE = re.compile(r'^#\s*([^:\t]+?)\s*:\s?(.*)$')


def _FormatValue(v):
    return '{0:.9f}'.format(float(v))


def _ValidateName(name):
    name = str(name)
    if len(name) == 0 or re.search(r'[\t\r\n]', name):
        raise ValueError('Invalid Brainsight target name %r (empty, or contains tab or new line)' % name)
    return name


def _ParseColumnHeader(line):
    '''Return the list of column names if line is the target column header, else None.'''
    if not line.startswith('#'):
        return None
    fields = [f.strip() for f in line[1:].split('\t')]
    if len(fields) < len(DEFAULT_COLUMNS) or fields[0] != COLUMN_NAME:
        return None
    if not all(c in fields for c in LOCATION_COLUMNS + MATRIX_COLUMNS):
        return None
    return fields


class BrainsightTrajectory:
    '''
    A Brainsight trajectory file with one or more trajectories (targets).

    Typical use::

        traj = BrainsightTrajectory.from_file(fname)
        Mat4 = traj.get_mat4(0)
        traj.set_location(0, Mat4[:3,3] - correction)   # mechanical correction
        traj.replace(1, NewMat4, name='T2')              # e.g. PlanTUS result
        traj.set_metadata('Created by', 'BabelBrain 0.8.8')
        traj.to_file(outfname)
    '''

    def __init__(self, lines, columns, row_lines, newline='\n'):
        # lines: all lines of the file (without line terminators)
        # columns: column names of the target section
        # row_lines: index in `lines` of each trajectory row
        self._lines = list(lines)
        self._columns = list(columns)
        self._row_lines = list(row_lines)
        self._newline = newline
        self._index = {c: self._columns.index(c) for c in DEFAULT_COLUMNS}
        self._rows = [self._lines[n].split('\t') for n in self._row_lines]
        self._names = [r[self._index[COLUMN_NAME]] for r in self._rows]
        self._mats = [self._RowToMat4(r, n) for n, r in zip(self._row_lines, self._rows)]
        self._modified = [False] * len(self._rows)

    ###########################################################################
    # Construction
    ###########################################################################

    @classmethod
    def from_file(cls, fname):
        with open(fname, 'r', newline='') as f:
            text = f.read()
        try:
            return cls.from_text(text)
        except ValueError as e:
            raise ValueError('%s: %s' % (fname, str(e)))

    @classmethod
    def from_text(cls, text):
        newline = '\r\n' if '\r\n' in text else '\n'
        lines = text.splitlines()

        header_line = None
        for n, line in enumerate(lines):
            columns = _ParseColumnHeader(line)
            if columns is not None:
                header_line = n
                break

        if header_line is None:
            # No column header: old-style file; we can only rely on '#' for comments
            columns = list(DEFAULT_COLUMNS)
            row_lines = [n for n, line in enumerate(lines)
                         if len(line.strip()) > 0 and not line.startswith('#')]
            for n in row_lines:
                if cls._TryParseRow(lines[n], columns) is None:
                    raise ValueError('line %i is not a valid Brainsight trajectory: %r' % (n + 1, lines[n]))
        else:
            row_lines = []
            for n in range(header_line + 1, len(lines)):
                line = lines[n]
                if len(line.strip()) == 0:
                    continue
                if cls._TryParseRow(line, columns) is not None:
                    # this can be a target whose name starts with '#'
                    row_lines.append(n)
                elif line.startswith('#'):
                    # comment or start of another section: end of the targets
                    break
                else:
                    raise ValueError('line %i is not a valid Brainsight trajectory: %r' % (n + 1, line))

        if len(row_lines) == 0:
            raise ValueError('no trajectories found in Brainsight file')
        return cls(lines, columns, row_lines, newline)

    @classmethod
    def new(cls, names, mats, coordinate_system, created_by='BabelBrain', version='15', notes=None):
        '''Create a new trajectory file content from names and 4x4 matrices.'''
        if len(names) != len(mats) or len(names) == 0:
            raise ValueError('names and mats must have the same, non-zero, length')
        metadata = dict(DEFAULT_METADATA)
        metadata['Version'] = str(version)
        metadata['Coordinate system'] = coordinate_system
        metadata['Created by'] = created_by
        if notes is not None:
            metadata['Notes'] = notes
        lines = ['# %s: %s' % (k, v) for k, v in metadata.items()]
        lines.append('# ' + '\t'.join(DEFAULT_COLUMNS))
        row_lines = []
        for name, Mat4 in zip(names, mats):
            row_lines.append(len(lines))
            lines.append('\t'.join([_ValidateName(name)] + [_FormatValue(v) for v in cls._Mat4ToValues(Mat4)]))
        return cls(lines, DEFAULT_COLUMNS, row_lines)

    ###########################################################################
    # Row conversion
    ###########################################################################

    @staticmethod
    def _TryParseRow(line, columns):
        fields = line.split('\t')
        if len(fields) < len(columns):
            return None
        try:
            return [float(fields[columns.index(c)]) for c in LOCATION_COLUMNS + MATRIX_COLUMNS]
        except ValueError:
            return None

    def _RowToMat4(self, fields, nline):
        try:
            loc = [float(fields[self._index[c]]) for c in LOCATION_COLUMNS]
            mat = [float(fields[self._index[c]]) for c in MATRIX_COLUMNS]
        except (ValueError, IndexError):
            raise ValueError('line %i is not a valid Brainsight trajectory' % (nline + 1))
        Mat4 = np.eye(4)
        Mat4[:3, 3] = loc
        Mat4[:3, 0] = mat[0:3]
        Mat4[:3, 1] = mat[3:6]
        Mat4[:3, 2] = mat[6:9]
        return Mat4

    @staticmethod
    def _Mat4ToValues(Mat4):
        Mat4 = np.asarray(Mat4, dtype=float)
        if Mat4.shape != (4, 4):
            raise ValueError('Mat4 must be a 4x4 matrix')
        return np.concatenate((Mat4[:3, 3], Mat4[:3, 0], Mat4[:3, 1], Mat4[:3, 2]))

    ###########################################################################
    # Access
    ###########################################################################

    def __len__(self):
        return len(self._rows)

    @property
    def names(self):
        return list(self._names)

    @property
    def metadata(self):
        '''"# Key: value" lines found before the column header, as a dict.'''
        res = {}
        end = self._HeaderLineIndex()
        for line in self._lines[:end]:
            m = _METADATA_LINE.match(line)
            if m is not None:
                res[m.group(1)] = m.group(2).strip()
        return res

    @property
    def version(self):
        return self.metadata.get('Version', None)

    @property
    def coordinate_system(self):
        return self.metadata.get('Coordinate system', None)

    def get_mat4(self, idx):
        return self._mats[idx].copy()

    def get_name(self, idx):
        return self._names[idx]

    def ReturnBabelBrainTrajectories(self, bGetID=False):
        '''
        Same return values BabelBrain has always used for Brainsight files:
        a 4x4 matrix and a name for a single trajectory, or a 4x4xN array and
        a list of names for several.
        '''
        if len(self) == 1:
            Mat4 = self.get_mat4(0)
            ID = self._names[0]
        else:
            Mat4 = np.zeros((4, 4, len(self)))
            for n in range(len(self)):
                Mat4[:, :, n] = self._mats[n]
            ID = self.names
        if bGetID:
            return Mat4, ID
        return Mat4

    ###########################################################################
    # Modification
    ###########################################################################

    def _CheckIndex(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError('trajectory %i is out of range, file has %i trajectories' % (idx, len(self)))

    def set_location(self, idx, xyz):
        '''Change the location (Loc. X/Y/Z) of trajectory idx, keeping its orientation.'''
        self._CheckIndex(idx)
        self._mats[idx][:3, 3] = np.asarray(xyz, dtype=float).ravel()[:3]
        self._modified[idx] = True

    def replace(self, idx, Mat4, name=None):
        '''Replace location and orientation of trajectory idx (e.g. with a PlanTUS result).'''
        self._CheckIndex(idx)
        self._Mat4ToValues(Mat4)  # validates shape
        self._mats[idx] = np.asarray(Mat4, dtype=float).copy()
        if name is not None:
            self._names[idx] = _ValidateName(name)
        self._modified[idx] = True

    def rename(self, idx, name):
        self._CheckIndex(idx)
        self._names[idx] = _ValidateName(name)
        self._modified[idx] = True

    def _HeaderLineIndex(self):
        for n, line in enumerate(self._lines):
            if _ParseColumnHeader(line) is not None:
                return n
        return self._row_lines[0]

    def set_metadata(self, key, value):
        '''Set a "# Key: value" line, adding it before the column header if missing.'''
        value = str(value).strip()
        if re.search(r'[\r\n]', value):
            raise ValueError('metadata value cannot contain new lines')
        end = self._HeaderLineIndex()
        newline = '# %s: %s' % (key, value)
        for n in range(end):
            m = _METADATA_LINE.match(self._lines[n])
            if m is not None and m.group(1) == key:
                self._lines[n] = newline
                return
        self._lines.insert(end, newline)
        self._row_lines = [n + 1 if n >= end else n for n in self._row_lines]

    ###########################################################################
    # Output
    ###########################################################################

    def to_text(self):
        lines = list(self._lines)
        for i, n in enumerate(self._row_lines):
            if not self._modified[i]:
                continue  # untouched trajectories are written exactly as read
            fields = list(self._rows[i])
            fields[self._index[COLUMN_NAME]] = self._names[i]
            for c, v in zip(LOCATION_COLUMNS + MATRIX_COLUMNS, self._Mat4ToValues(self._mats[i])):
                fields[self._index[c]] = _FormatValue(v)
            lines[n] = '\t'.join(fields)
        return self._newline.join(lines) + self._newline

    def to_file(self, fname):
        with open(fname, 'w', newline='') as f:
            f.write(self.to_text())
