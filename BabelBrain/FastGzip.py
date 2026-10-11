"""
Faster .nii.gz writing via python-isal (Intel ISA-L).

nibabel and SimpleITK compress with single-threaded zlib, which dominates the
time to save large float volumes (that barely compress anyway). ISA-L produces
standard gzip streams readable by any tool, ~3x faster single-threaded and
~9x faster multi-threaded (300^3 float32: 2.4 s -> 0.27 s).

Two mechanisms:
  * save_nifti / write_sitk: explicit helpers for the pipeline's save call
    sites; compress with a multi-threaded ISA-L writer.
  * importing this module patches nibabel so any other `to_filename` /
    `nibabel.save` of a .gz file uses single-threaded ISA-L. (nibabel's writer
    calls tell()/seek(), which the threaded writer does not support.)

Both write with mtime=0, like nibabel, so identical data gives identical bytes
(FileManager hashes file contents to decide on reuse).

If isal is not installed, everything falls back to plain nibabel/SimpleITK.
"""
import os
import shutil
import tempfile

try:
    from isal import igzip, igzip_threaded
    HAVE_ISAL = True
except ImportError:
    HAVE_ISAL = False

COMPRESSLEVEL = 1  # nibabel's default; higher levels gain little on float fields


def _is_gz(path):
    return str(path).lower().endswith('.gz')


def save_nifti(img, path):
    """Save a nibabel image; .gz paths are compressed with multi-threaded ISA-L."""
    path = str(path)
    if HAVE_ISAL and _is_gz(path):
        with igzip_threaded.open(path, 'wb', compresslevel=COMPRESSLEVEL, threads=-1) as f:
            f.write(img.to_bytes())
    else:
        img.to_filename(path)
    return path


def gzip_file(src, dst, remove_src=True):
    """Compress an existing file `src` into gzip file `dst`."""
    with open(src, 'rb') as fin, \
         igzip_threaded.open(dst, 'wb', compresslevel=COMPRESSLEVEL, threads=-1) as fout:
        shutil.copyfileobj(fin, fout, 4 * 1024 * 1024)
    if remove_src:
        os.remove(src)
    return dst


def write_sitk(img, path):
    """sitk.WriteImage replacement; .nii.gz paths are written uncompressed to a
    temporary file and then compressed with multi-threaded ISA-L."""
    import SimpleITK as sitk
    path = str(path)
    if not (HAVE_ISAL and path.lower().endswith('.nii.gz')):
        sitk.WriteImage(img, path)
        return path
    fd, tmp = tempfile.mkstemp(suffix='.nii', dir=os.path.dirname(os.path.abspath(path)))
    os.close(fd)
    try:
        sitk.WriteImage(img, tmp)
        gzip_file(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def _install_nibabel_patch():
    from nibabel.openers import Opener, ImageOpener
    gz_read = Opener.gz_def[0]

    def _gz_open(filename, mode='rb', compresslevel=COMPRESSLEVEL, mtime=0, keep_open=False):
        if 'w' in mode:
            return igzip.IGzipFile(filename, mode, compresslevel=compresslevel, mtime=mtime)
        return gz_read(filename, mode, compresslevel=compresslevel, mtime=mtime, keep_open=keep_open)

    # ImageOpener (used by to_filename) keeps its own copy of the map
    for cls in (Opener, ImageOpener):
        cls.compress_ext_map['.gz'] = (_gz_open, Opener.gz_def[1])


if HAVE_ISAL:
    try:
        _install_nibabel_patch()
    except Exception as e:  # never let an optimization break saving
        print('FastGzip: nibabel patch not installed:', e)
