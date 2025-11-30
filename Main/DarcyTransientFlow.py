import numpy as np
from BaseClasses import BasicData
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class DarcyTransientFlow(BasicData):
    def __init__(self, L:float, PLeft:float, PRight:float, k:float, mu:float, phi:float, ct:float, startTime:float=0.001, endTime:float=1.0):
        super().__init__()
        self.L = L                  #* Length of the domain (m)
        self.PLeft = PLeft          #* Pressure at the left boundary (Pa)
        self.PRight = PRight        #* Pressure at the right boundary (Pa)
        self.k = k                  #* Permeability (m^2)
        self.mu = mu                #* Dynamic viscosity (Pa.s)
        self.phi = phi              #* Porosity (-)
        self.ct = ct                #* Total compressibility (1/Pa)

        self.startTime = startTime
        self.endTime = endTime

        self.DeactivateAttr()

    def InfiniteSum(self, x:float, t:float, NInf:int=100) -> float:
        """
        Calculate the infinite series of the transient Darcy flow solution in a 1D domain.

        Parameters:
        x: Spatial coordinate (m).
        t: Time (s).

        Returns:
        Value of the infinite series at position x and time t.
        """
        D = self.k / (self.mu * self.phi * self.ct) #* Diffusivity
        k = lambda n: (n * np.pi) / self.L #* Spatial frequency
        eq = lambda n: (1 / n) * np.exp(-k(n)**2 * D * t) * np.sin(k(n) * x) #* Series term

        return sum([eq(n) for n in range(1, NInf + 1)])

    def AnalyticalSolution(self, x:float, t:float, NInf:int=100) -> float:
        """
        Calculate the analytical solution for transient Darcy flow in a 1D domain.

        Parameters:
        x: Spatial coordinate (m).
        t: Time (s).

        Returns:
        Pressure at position x and time t (Pa).
        """

        return self.PLeft + (self.PRight - self.PLeft) * (x / self.L + 2 / np.pi * self.InfiniteSum(x, t, NInf))
    
    def Plot(self):
        x = np.linspace(0, self.L, 100)

        fig, ax = plt.subplots()
        line, = ax.plot(x, self.AnalyticalSolution(x, self.startTime), lw=2)
        ax.set_xlabel('Position (m)')

        fig.subplots_adjust(bottom=0.25)

        ax_time = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        time_slider = Slider(
            ax = ax_time,
            label = 'Time (s)',
            valmin = self.startTime,
            valmax = self.endTime,
            valinit = self.startTime,
        )

        def update(val):
            line.set_ydata(self.AnalyticalSolution(x, time_slider.val))
            fig.canvas.draw_idle()

        time_slider.on_changed(update)

        fig.suptitle('Transient Darcy Flow in 1D Domain', y=0.95)
        plt.show()