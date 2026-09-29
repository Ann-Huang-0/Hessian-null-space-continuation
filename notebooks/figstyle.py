"""Plot style and colours of the paper figures."""
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TEXT_W, HALF_W = 5.5, 2.65          # ICLR text width and half width, in inches

MUTED = dict(blue='#3660AF', teal='#08886D', mauve='#A46BBD', sage='#709C3C', wine='#912F52',
             ochre='#C28412', slate='#0393C2', rust='#BC5840', ink='#3B3B38')
STATE8 = [MUTED[k] for k in ('blue', 'sage', 'wine', 'ochre', 'slate', 'rust', 'mauve', 'teal')]
PAL = {'anchor': MUTED['teal'], 'undirected': MUTED['mauve'], 'cka': '#1B3F82', 'dsa': MUTED['ochre'],
       'seed': '0.5', 'target': MUTED['ink'], 'ref': MUTED['ink']}
LS = {'target': ':', 'ref': ':'}


def setup():
    plt.style.use(Path(__file__).with_name('paper.mplstyle'))


def row(n, h=1.7, width=TEXT_W, sharey=True, **kw):
    """A row of n small panels spanning the text width."""
    fig, ax = plt.subplots(1, n, figsize=(width, h), sharey=sharey, **kw)
    return fig, np.atleast_1d(ax)


def panel(w=HALF_W, h=1.9, **kw):
    return plt.subplots(figsize=(w, h), **kw)


def legend_above(fig, ax, ncol=3, **kw):
    """One shared legend above the panels."""
    for old in list(fig.legends):
        old.remove()
    fig._legend_above = True
    return fig.legend(*ax.get_legend_handles_labels(), loc='lower center', ncol=ncol, bbox_to_anchor=(0.5, 1.0), **kw)


def legend_outside(ax, ncol=1, **kw):
    """Legend to the right of the axes."""
    ax.get_figure()._legend_right = True
    return ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=ncol, **kw)


def finish(fig, rect=None):
    """tight_layout, leaving room for a legend above or to the right."""
    if rect is None:
        rect = ([0, 0, 0.84, 1] if getattr(fig, '_legend_right', False) else
                [0, 0, 1, 0.92] if getattr(fig, '_legend_above', False) else [0, 0, 1, 1])
    with warnings.catch_warnings():             # 3-D axes do not take part in tight_layout
        warnings.simplefilter('ignore')
        fig.tight_layout(rect=rect)
    return fig


def ref_line(ax, y, label=None):
    """A horizontal reference level such as a loss ceiling."""
    return ax.axhline(y, color=PAL['ref'], ls=LS['ref'], lw=1.0, label=label)
