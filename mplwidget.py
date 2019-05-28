# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 13:59:22 2019

@author:
Understanding optics with Python - [Multidisciplinary & applied optics] Ammar, Ahmed_ Lakshminarayanan, Vasudevan_ Varadharajan, L. Srinivasa - CRC 2017.pdf
"""
import os
#os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"

'''MATPLOTLIB WIDGET'''
# Python Qt5 bindings for GUI objects
from PyQt5.QtWidgets import QSizePolicy, QWidget, QVBoxLayout
# import the Qt5Agg FigureCanvas object, that binds Figure to
# Qt5Agg backend. It also inherits from QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# Matplotlib Toolbar
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
# Matplotlib Figure object
from matplotlib.figure import Figure
from matplotlib import rcParams, gridspec
rcParams['font.size'] = 9
#rcParams['toolbar'] = 'None'
rcParams["figure.edgecolor"] = 'darkblue'

import matplotlib.pyplot as plt
#plt.tight_layout(pad = 1.25)

class MplCanvas(FigureCanvas):
    """
    Class to represent the FigureCanvas widget
    """
    def __init__(self):
        # setup Matplotlib Figure and Axis
        self.fig = plt.figure(figsize=(6, 6))
        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.98, top=0.95, wspace=0.5, hspace=0.5)
        #gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[2,1]) 
        gs = plt.GridSpec(3, 8, wspace=0.5, hspace=0.3)
        #plt.axis('tight')
        ax1 = self.fig.add_subplot(gs[:2, 2:6])  # xC + yP
        ax2 = self.fig.add_subplot(gs[:1, :2], sharex=ax1)  # 1P
        ax3 = self.fig.add_subplot(gs[:1, 6:], sharex=ax1) # 1C
        ax4 = self.fig.add_subplot(gs[2:, :3]) # S_T
        ax5 = self.fig.add_subplot(gs[2:, 3:6], sharex=ax4)  # P&L
        axes = [ax1, ax2, ax3, ax4, ax5]
        self.axes = axes
        FigureCanvas.__init__(self, self.fig)
        # we define the widget as expandable
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        # notify the system of updated policy
        FigureCanvas.updateGeometry(self)

class MPL_WIDGET(QWidget):
    """Widget defined in Qt Designer"""
    def  __init__(self, parent=None):
        # initialization of Qt MainWindow widget
        QWidget.__init__(self, parent)
        # set the canvas to the Matplotlib widget
        self.canvas = MplCanvas()
        # create a navigation toolbar for our plot canvas
        #self.navi_toolbar = NavigationToolbar(self.canvas, self)
        # create a vertical box layout
        self.vbl = QVBoxLayout()
        # add mpl widget to vertical box
        self.vbl.addWidget(self.canvas)
        # add the navigation toolbar to vertical box
        #self.vbl.addWidget(self.navi_toolbar)
        # set the layout to vertical box
        self.setLayout(self.vbl)
        #
        self.canvas.fig.set_visible(False)


