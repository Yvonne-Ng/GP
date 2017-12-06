"""
Wrappers for basic matplotlib figures.
"""

import os

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

# some magic here
from matplotlib.pyplot import setp

class Canvas:
    default_name = 'test.pdf'
    def __init__(self, title="", ratioTitle1="", ratioTitle2="", ratioTitle3="", ratioNum=2, out_path=None, figsize=(8,10), ext=None):
        self.fig = Figure(figsize)
        self.fig.subplots_adjust(top=0.85)
        self.canvas = FigureCanvas(self.fig)
        #grid = GridSpec(2,1, height_ratios=[3,1])
        #self.ax = self.fig.add_subplot(grid[0])
        #self.ratio = self.fig.add_subplot(grid[1], sharex=self.ax)
        grid = GridSpec(3,1, height_ratios=[5,1,1])
        if ratioNum==3:
            grid = GridSpec(4,1, height_ratios=[5,1,1,1])
        self.ax = self.fig.add_subplot(grid[0])
        self.ax.set_title(title)
        #self.ax.set_xlabel('mjj (GeV)')
        self.ax.set_ylabel('Event count')
        
        self.ratio = self.fig.add_subplot(grid[1], sharex=self.ax)
        self.ratio.set_title(ratioTitle1)
        self.ratio2 = self.fig.add_subplot(grid[2], sharex=self.ax)
        self.ratio2.set_title(ratioTitle2)
        self.ratio2.set_xlabel(r'$m_{jj}$(GeV)', ha='right', x=0.98)
        if ratioNum==3:
            self.ratio3 = self.fig.add_subplot(grid[3], sharex=self.ax)
            self.ratio3.set_title(ratioTitle3)
            self.ratio3.set_xlabel(r'$m_{jj}(GeV)$', ha='right', x=0.98)
        self.out_path = out_path
        self.ext = ext
        #self.fig.show()

    def save(self, out_path=None, ext=None):
        output = out_path or self.out_path
        assert output, "an output file name is required"
        out_dir, out_file = os.path.split(output)
        if ext:
            out_file = '{}.{}'.format(out_file, ext.lstrip('.'))
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        self.fig.tight_layout(pad=0.3, h_pad=0.3, w_pad=0.3)
        self.canvas.print_figure(output)

    def __enter__(self):
        if not self.out_path:
            self.out_path = self.default_name
        return self
    def __exit__(self, extype, exval, extb):
        if extype:
            return None
        self.save(self.out_path, ext=self.ext)
        return True
