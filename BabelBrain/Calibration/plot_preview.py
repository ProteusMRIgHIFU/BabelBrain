"""PDF plots with a PNG twin for the in-app viewer.

The frozen app does not bundle a PDF renderer (PyMuPDF alone is ~45 MB), so
Calibration.ViewResults displays the PNG saved next to each PDF.
"""
import os


def png_preview_path(pdf_path):
    return os.path.splitext(pdf_path)[0] + '.png'


def save_figure_with_preview(fig, pdf_path, dpi=200, **kwargs):
    """Save ``fig`` (a Figure, or the pyplot module) as ``pdf_path`` and its PNG twin."""
    fig.savefig(pdf_path, **kwargs)
    fig.savefig(png_preview_path(pdf_path), dpi=dpi, **kwargs)
