# -*- coding: utf-8 -*-

import time, datetime, os
#os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

import sys, copy
#import design
#from PyQt5.QtCore import Qt
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout, QHBoxLayout, QSlider, QLineEdit, QLabel, QTableWidgetItem, QSizePolicy, QTableWidgetSelectionRange
from matplotlib.widgets import Slider, Button, RadioButtons

# Libraries for drawing figures
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.ticker import FuncFormatter
# Ensure using PyQt5 backend
import matplotlib as mpl
mpl.use('QT5Agg')
import matplotlib.pyplot as plt
#plt.style.use('ggplot')
#plt.style.use('seaborn')
#plt.tight_layout(pad = 1.25)

# Libraries for stochastic simulation
import numpy as np
from numpy import log, exp, sqrt
from scipy.stats import skew, kurtosis
from math import ceil, floor
from scipy.special import binom
from scipy.stats import norm
import bsm_base as bsm

#from Simulation import merton_jump_diffusion_simulate as mjd
from Simulation import geometric_brownian_motion_simulate as gbm

class Window(QtWidgets.QDialog):
    def __init__(self):
        QtWidgets.QDialog.__init__(self)
        # load the .ui file created in Qt Creator directly
        print(f"screen width = {screen_width}")
        if screen_width >= 1920:
            Interface = uic.loadUi('mod5.ui', self)
        else:
            Interface = uic.loadUi('mod5.ui', self)

        # other useful into
        # default values for all parameters
        self.S0 = 100
        self.K_C = self.slider_K_C.value()
        self.K_P = self.K_C
        self.r   = self.slider_r.value()/100
        self.q   = self.slider_q.value()/100
        self.σ   = self.slider_sigma.value()/100
        self.T_C = self.slider_T_C.value()*.25
        self.T_P = self.T_C
        self.μ   = self.slider_mu.value()/100
        self.Δt  = 0.001
        self.seed = 12345
        self.simSteps = self.spin_numSteps.value() # no. of steps in simulated share price changes; default 5
        self.simCurrentStep = 0
        self.simY = None
        self.Ydiscrete = None
        self.circleCoords = []
        self.PL_TVOM = [[None for col in range(4)] for row in range(6)] # initialize matrix (6x4) 
        self.PL      = [[None for col in range(4)] for row in range(6)] # same as above, but no TVOM 
        #params = {'S0': log(self.S0), 'K': log(self.K_C), 'r': self.r, 'q': self.q, 'σ': self.σ, 'T': self.T_C, 'seed': self.seed}
        #self.params = params
 
        self.slider_K_C.valueChanged.connect(self.on_change_K_C)
        self.slider_K_P.valueChanged.connect(self.on_change_K_P)
        self.slider_T_C.valueChanged.connect(self.on_change_T_C)

        self.slider_r.valueChanged.connect(self.on_change_r)
        self.slider_q.valueChanged.connect(self.on_change_q)
        self.slider_sigma.valueChanged.connect(self.on_change_sigma)
        self.slider_mu.valueChanged.connect(self.on_change_mu)

        #self.slider_K_P.valueChanged.connect(self.slider_K_C)
        #self.slider_r_P.valueChanged.connect(self.slider_r_C)
        #self.slider_q_P.valueChanged.connect(self.slider_q_C)
        #self.slider_s_P.valueChanged.connect(self.slider_s_C)
        #self.slider_T_P.valueChanged.connect(self.slider_T_C)

        self.button_Simulate.clicked.connect(self.on_click_ShowSimulation)

        self.button_NextStep.clicked.connect(self.on_click_NextStep)

        self.button_NewSim.clicked.connect(self.on_click_NewSim)

        self.spin_numSteps.valueChanged.connect(self.on_valueChange_numSteps)
        self.spin_numSteps.visible = False

        self.radio_ignore_TVOM.clicked.connect(self.on_click_ignore_TVOM)

        # initialize default parameter values

        # Run some initialization stuff
        # simulate a batch of prices
        self.on_click_NewSim()

        # Initialize all parameter labels 
        self.label_K_C.setText(f"K = {self.K_C}") 
        self.label_K_P.setText(f"K = {self.K_P}")   
        self.label_T_C.setText(f"T = {self.T_C}")   
        self.label_T_P.setText(f"T = {self.T_P}")   
        self.label_q.setText(f"q = {self.q}")     
        self.label_r.setText(f"r = {self.r}")      
        self.label_sigma.setText(f"σ = {self.σ}") 
        self.label_mu.setText(f"μ = {self.μ}") 

    def on_click_ignore_TVOM(self):
        if self.radio_ignore_TVOM.isChecked():
            #print(f"ignore checked!")
            self.PL_TVOM = copy.deepcopy(self.PL)
        else:
            #print(f"ignore not chekced!")
            pass
        self.UpdatePL_TVOM()
        #self.DisplayPL_Table()
        self.showGraph()

    def DisplayPL_Table(self):

        # Set up the Table Widget (QTableView) environment - headers, widths, etc.
        self.tablePL.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.tablePL.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.tablePL.setColumnCount(4)
        self.tablePL.setRowCount(6)
        #hori_labels = ['1P Value', '1C Value', 'xC+yP Value', 'P/L']
        hori_labels = ['1P\nValue', '1C\nValue', 'xC+yP\nValue', 'P/L']
        self.tablePL.setHorizontalHeaderLabels(hori_labels)
        font = QFont('Segoe UI', 9)
        font.setBold(True)
        self.tablePL.horizontalHeader().setFont(font)
        vert_labels = [f"t{str(j)}" for j in range(6)]
        self.tablePL.setVerticalHeaderLabels(vert_labels)
        font = QFont('Segoe UI', 9)
        font.setBold(False)
        self.tablePL.verticalHeader().setFont(font)
        # Headers
        self.tablePL.resizeColumnsToContents()
        header = self.tablePL.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        for col in range(4):    
            header.setSectionResizeMode(col, QtWidgets.QHeaderView.Stretch)

        self.tablePL.clearSelection()

        # copy from self.PL_TVOM to self.tablePL
        k = self.simCurrentStep
        for row in range(k+1):
            for col in range(4):
                item = QTableWidgetItem(f"{self.PL_TVOM[row][col]:.2f}")
                item.setTextAlignment(Qt.AlignRight)
                self.tablePL.setItem(row, col, item)
        

    def on_valueChange_numSteps(self, value):
        self.simSteps = value

    def on_click_NextStep(self, value):
        self.tablePL.clear()
        #print(f"length of circleCoords = {len(self.circleCoords)}")
        self.simCurrentStep += 1
        if self.simCurrentStep > self.simSteps:
            self.simCurrentStep = 0
            #self.circleCoords = []
        #print(f"Current step = {self.simCurrentStep}")
        self.showGraph()


    def on_change_K_C(self, value):
        self.K_C = value
        self.label_K_C.setText(f"K = {str(self.K_C)}")
        self.on_click_NewSim()
        self.showGraph()
    def on_change_K_P(self, value):
        self.K_P = value
        self.label_K_P.setText(f"K = {str(self.K_P)}")
        self.on_click_NewSim()
        self.showGraph()
    def on_change_T_C(self, value): # T is the same both P and C, for now
        self.T_C = value*.25
        self.label_T_C.setText(f"T = {str(self.T_C)}")
        self.slider_T_P.setValue(self.T_C)
        self.label_T_P.setText(f"T = {str(self.T_C)}")
        self.on_click_NewSim()
        self.showGraph()

    def on_change_r(self, value):
        self.r = value/100
        self.label_r.setText(f"r = {str(self.r)}")
        self.on_click_NewSim()
        self.showGraph()
    def on_change_q(self, value):
        self.q = value/100
        self.label_q.setText(f"q = {str(self.q)}")
        self.on_click_NewSim()
        self.showGraph()
    def on_change_sigma(self, value):
        self.σ = value/100
        self.label_sigma.setText(f"σ = {str(self.σ)}")
        self.on_click_NewSim()
        #self.recomputeCircle()
        self.showGraph()
    def on_change_mu(self, value):
        self.μ = value/100
        self.label_mu.setText(f"σ = {str(self.μ)}")
        self.showGraph()

    def on_click_ShowSimulation(self):
        #print("Simulation button clicked")
        self.showGraph()

    def on_click_NewSim(self):
        params = {'S0': self.S0, 'μ': self.μ, 'σ': self.σ, 'Δt': self.Δt, 'T': self.T_C, 'N': 1, 'P': 1, 'seed': np.random.randint(32768)}
        self.simY = gbm(params)
        self.simCurrentStep = 0
        self.circleCoords = []
        self.X = np.linspace(0.0, self.T_C, int(1/self.Δt)+1)

        self.Xdiscrete = np.linspace(0.0, self.T_C, self.simSteps+1)
        self.Ydiscrete = [self.simY[int((len(self.X)-1)/self.simSteps * i)] for i in range(self.simSteps+1) ] # sample of simY at discrete time points
        S0   = self.S0
        K_C  = self.K_C
        K_P  = self.K_P
        r    = self.r
        q    = self.q
        σ    = self.σ
        T_C  = self.T_C
        T_P  = self.T_P
        #seed = np.random.randint(0,65535)
        α = self.spin_C.value()
        β = self.spin_P.value()
        for i in range(0, self.simSteps+1):
            if i == 0:
                current_spot = S0
            else:
                current_spot = self.Ydiscrete[i][0]
            spot_optval = α * bsm.bs_call_price(current_spot, K_C, r, q, σ, T_C*(1-(i)/(self.simSteps))) + \
                          β * bsm.bs_put_price(current_spot, K_P, r, q, σ, T_P*(1-(i)/(self.simSteps)))
            self.circleCoords.append((current_spot, spot_optval))
        
        # clear P/L table widget
        self.tablePL.clear()
        self.showGraph()

    def recomputeCircle(self):
        '''
        Assuming self.Ydiscrete
        '''
        newCircles = []
        S0   = self.S0
        K_C  = self.K_C
        K_P  = self.K_P
        r    = self.r
        q    = self.q
        σ    = self.σ
        T_C  = self.T_C
        T_P  = self.T_P
        α = self.spin_C.value()
        β = self.spin_P.value()
        for point in self.circleCoords:
            y_call = bsm.bs_call_price(point[0], K_C, r, q, σ, T_C*(1-(self.simCurrentStep)/(self.simSteps)))
            y_put  = bsm.bs_put_price(point[0], K_P, r, q, σ, T_P*(1-(self.simCurrentStep)/(self.simSteps)))
            y      = α * y_call + β * y_put
            newCircles.append((point[0], y))
        self.circleCoords = newCircles


    def UpdatePL_TVOM(self):
        '''
        This method recomputes the P/L table based on the current time step
        Results (no explicit output, only self.PL_TVOM is updated):
            An updated PL_TVOM matrix up to the current step as a side effect
        '''
        #print(f"entering UpdatePL_TVOM...")
        k = self.simCurrentStep
        r = self.r # current interest rate
        deltaT = self.T_C / self.simSteps
        self.PL_TVOM = copy.deepcopy(self.PL)
        if not self.radio_ignore_TVOM.isChecked(): # TVOM
            for row in range(k+1):
                for col in range(3):
                    self.PL_TVOM[row][col] = self.PL[row][col] * np.exp(r*(k - row)*deltaT)
            self.PL_TVOM[0][3] = 0.0   
            for row in range(1, k+1):
                # P/L column
                self.PL_TVOM[row][3] = self.PL_TVOM[row][2] - self.PL_TVOM[0][2]

    def showGraph(self):

        mpl = self.oMplCanvas.canvas
        [s1, s2, s3, s4, s5] = mpl.axes

        #print(f"showGraph::very:beginning:PL={self.PL}")

        S0   = self.S0
        K_C  = self.K_C
        K_P  = self.K_P
        r    = self.r
        q    = self.q
        σ    = self.σ
        T_C  = self.T_C
        T_P  = self.T_P

        k = self.simCurrentStep  # this is used a lot below 

        # Get the x, y units of Call and Put options, respectively
        α = self.spin_C.value()  # x
        β = self.spin_P.value()  # y
        #print(f"α={α}, β={β}")

        # Set minimum and maximum x-axis values for s1 (main plot)
        # get the y-coordinates of self.circleCoords (options values) for t0, t1, ..., t5
        circle_y_list = [self.circleCoords[i][0] for i in range(len(self.circleCoords))]
        s1_x_max = max(140, max(circle_y_list)+10)
        s1_x_min = min(60, min(circle_y_list)-10)

        # x-axis points for s1 (main) plot
        x      = np.linspace(s1_x_min, s1_x_max, 51)
        # y-axis points for s3 (1-unit call) plot
        y_call = bsm.bs_call_price(x, K_C, r, q, σ, T_C*(1-k/(self.simSteps)))
        # y-axis points for s2 (1-unit put) plot
        y_put  = bsm.bs_put_price(x, K_P, r, q, σ, T_P*(1-k/(self.simSteps)))
        # y-axis points for s1 (combined options)
        y      = α * y_call + β * y_put
        # payoff for 1-unit call
        z_call = bsm.payoff_call(x, K_C)
        # payoff for 1-unit put
        z_put  = bsm.payoff_put(x, K_P)
        # payoff for combined
        z      = α * z_call + β * z_put

        # s2: 1-unit Put options
        s2.clear()
        s2.set_title(f'$P$', fontsize=12, color='brown')
        s2.grid(True)
        s2.plot(x, y_put, 'r', lw=0.6, label='value')
        s2.plot(x, z_put, 'b-.', lw=1, label='payoff')
        # plot circle on s2
        put_circle_x = self.circleCoords[k][0]
        put_circle_y = bsm.bs_put_price(put_circle_x, K_P, r, q, σ, T_P*(1-k/(self.simSteps)))
        s2.plot(put_circle_x, put_circle_y, 'black', marker="o")
        self.text_PutValue1Unit.setPlainText(f"{put_circle_y:.2f}")

        # s3: 1-unit Call options
        s3.clear()
        s3.set_title(f'$C$', fontsize=12, color='brown')
        s3.grid(True)
        s3.plot(x, y_call, 'r', lw=0.6, label='value')
        s3.plot(x, z_call, 'b-.', lw=1, label='payoff')
        # plot circle on s3
        call_circle_x = self.circleCoords[k][0]
        call_circle_y = bsm.bs_call_price(call_circle_x, K_C, r, q, σ, T_C*(1-k/(self.simSteps)))
        s3.plot(call_circle_x, call_circle_y, 'black', marker="o")
        self.text_CallValue1Unit.setPlainText(f"{call_circle_y:.2f}")

        # Plot all (discretized continuous) simulated share prices
        s4.clear()
        s4.set_title(f'Simulated share prices', fontsize=12, color='brown')
        s4.yaxis.set_tick_params(labelright=False, labelleft=True)
        s4.grid(True)
        #print(f"size of X = {len(self.X)}")
        # set the upper limit of index for which the simulated share prices will be shown on s4
        upperLimit = int((len(self.X)-1) * k / self.simSteps)
        #print(f"upperLimit = {upperLimit}")
        s4.plot(self.X[:upperLimit], self.simY[:upperLimit], 'g', lw=0.75, label='$S_T$')
        s4.set_xlim(right=self.T_C, left=0.0)
        s4.set_ylim(top=self.simY.max()+10, bottom=self.simY.min()-10)
        # plot simulated prices at discrete time points (t0, t1, ..., t5)
        s4.plot(self.Xdiscrete[:k+1], self.Ydiscrete[:k+1], 'black', lw=1.25, label='price')
        #legend = s4.legend(loc='upper left', shadow=False, fontsize='medium')
        #print(f"Ydiscrete = {self.Ydiscrete}")

        # Plot main graph (s1: top row, middle)
        s1.clear()
        #if int(α) - float(α) == 0.0: # α is an integer
        s1.set_title(f'${α}C + {β}P$', fontsize=12, color='brown')
        s1.yaxis.set_tick_params(labelright=False, labelleft=True)
        s1.grid(True)
        s1.plot(x, z, 'b-.', lw=1.5) # payoff
        s1.set_xlim(right=s1_x_max, left=s1_x_min)

        # plot a number of option value curves (from 0 to 5)
        for i in range(0, k+1):
            y_call = bsm.bs_call_price(x, K_C, r, q, σ, T_C*(1-(i)/(self.simSteps)))
            y_put  = bsm.bs_put_price(x, K_P, r, q, σ, T_P*(1-(i)/(self.simSteps)))
            y      = α * y_call + β * y_put
            # plot option value curve          
            s1.plot(x, y, 'red', alpha=0.6, lw=0.4+0.05*i, label=f'value = {put_circle_y + call_circle_y}')
            #s1.legend(loc='upper left', shadow=False, fontsize='medium')
            
        # Now plot the circles
        #self.recomputeCircle()
        for i, point in enumerate(self.circleCoords):
            if i == 0:
                s1.plot(point[0], point[1], 'black', marker="o")
            elif i > 0 and i <= k: # skip step 0, i.e. when there's only one curve
                #print(f"i={i}, {self.circleCoords}")
                p_x, p_y = self.circleCoords[i-1] # previous circle's coordinates
                c_x, c_y = self.circleCoords[i]   # current circle's coordinates
                #print(f"p_x = {p_x:.2f}, p_y = {p_y:.2f}, c_x = {c_x:.2f}, c_y = {c_y:.2f}")
                s1.plot(point[0], point[1], 'black', marker="o")
                s1.arrow(p_x, p_y, c_x - p_x, c_y - p_y, color='purple', width=0.00025, head_width=1.0, length_includes_head=True)


        # Plot P&L graph
        s5.clear()
        s5.set_title(f'Profit/Loss', fontsize=12, color='brown')
        s5.yaxis.set_tick_params(labelright=False, labelleft=True)
        s5.grid(True)

        # 1P value
        self.PL[k][0] = put_circle_y
        # 1C vlaue
        self.PL[k][1] = call_circle_y
        # xC + yP
        self.PL[k][2] = α*call_circle_y + β*put_circle_y
        # P/L
        #print(f"k={k}")
        self.PL[k][3] = self.PL[k][2] - self.PL[0][2]
        self.UpdatePL_TVOM()
        # copy from self.PL_TVOM to self.tablePL
        self.DisplayPL_Table()
                
        #item.setTextAlignment(Qt.AlignVCenter)
        # Select a row
        self.tablePL.setRangeSelected(QTableWidgetSelectionRange(k, 0, k, 3), True)
        for row in range(6):
            self.tablePL.setRowHeight(row, 20)

        # Finally, plot the P/L line
        # find min and max of option values    
        all_options_values = [self.circleCoords[i][1] for i in range(len(self.circleCoords))]
        s5_x_max = max(all_options_values)+10
        s5_x_min = min(all_options_values)-10

        xPL = self.Xdiscrete
        option_val_t0 = self.PL_TVOM[0][2]  # option value at t0
        yPL = [self.PL_TVOM[row][2] for row in range(k+1)]
        s5.set_ylim(top=s5_x_max, bottom=s5_x_min)
        s5.plot(xPL[:k+1], yPL, 'r-.', lw=1.5, label='P&L')
        # plot horizontal line indicating the option value at t0
        s5.axhline(y=option_val_t0, color="black", alpha=1, linewidth=1)
        # annotate plot with P/L values 
        for row in range(0, k+1):
            s5.annotate(f"{self.PL_TVOM[row][3]:+.2f}", xy=[self.Xdiscrete[row]-0.1, self.PL_TVOM[row][3]+option_val_t0+2])
        for row in range(k+1):
            s5.plot(self.Xdiscrete[row], self.PL_TVOM[row][3]+option_val_t0, marker='o', color='black')

        #self.tablePL.setStyleSheet("QTableView::item:selected { color:red; background:yellow; font-weight: bold; } " + "QTableView::vertical:header {font-size: 9}")


        # Display entire canvas!
        mpl.fig.set_visible(True)
        mpl.draw()

 




# This creates the GUI window:
if __name__ == '__main__':

    import sys
    app = QtWidgets.QApplication(sys.argv)
    ### load appropriately sized UI based on screen resolution detected
    screen = app.primaryScreen()
    screen_width=screen.size().width()
    window = Window()
    w = window
    t = w.tablePL
    #window.showGraph()
    window.show()
    sys.exit(app.exec_())

