import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sym
import os
import glob
from pathlib import *
from mpl_toolkits import mplot3d
from scipy import signal
from matplotlib.widgets import Slider, Button
import math
import matplotlib.patches as patches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QTabWidget, QVBoxLayout
import re
import sys
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import interp1d
import inspect
import functools
from scipy.optimize import root_scalar

# Functions for making the print of the method docstrings look better
# Removes the 2 tab indents made by the code block alignment
def clean_docstring(method):
    if method.__doc__:
        lines = method.__doc__.splitlines()
        # cleaned_lines = [line.lstrip('\t') for line in lines]
        cleaned_lines = [line.replace('\t\t', '', 1) for line in lines]
        method.__doc__ = '\n'.join(cleaned_lines)
    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        return method(*args, **kwargs)
    return wrapper

def apply_clean_docstrings(cls):
    for attr_name, attr_value in cls.__dict__.items():
        if callable(attr_value) and not attr_name.startswith('__'):
            setattr(cls, attr_name, clean_docstring(attr_value))
    return cls

@apply_clean_docstrings
class QuasiPhaseMatching():
	def __init__(self, lambda_p, temperature, material='MgO:PPLN_Gayer', crystal_period=16e-6):
		if material:
			self.material = material
		self.lambda_p = lambda_p
		self.temperature = temperature
		self.crystal_period = self.temperature_dependent_grating_period(crystal_period=crystal_period, material=material)
		if material in ('PPLN_Zelmon', 'MgO:PPLN_Zelmon', 'MgO:PPLN_Gayer', 'MgO:PPSLT_Dolev'):
			self.ne_p, self.no_p = self.sellmeier_equations(wavelength=lambda_p, lvl=1)
		else:
			self.ne_p = self.sellmeier_equations(wavelength=lambda_p, lvl=1)

		# Defining constants
		self.epsilon_0 = 8.8541878128e-12 # vacuum permittivity [F/m]
		self.e = 1.602176634e-19 # elementary charge [C]
		self.m_0 = 9.10938356e-31 # free electron mass [kg]
		self.c = 2.99792458e8 # speed of light in vacuum [m/s]
		self.hbar = 1.054571817e-34 # Reduced Plank's constant [Js]
		self.axes = {}

	def update_parameters(self, lambda_p, temperature, material=None, crystal_period=16e-6):
		self.lambda_p = lambda_p
		self.temperature = temperature
		self.crystal_period = crystal_period
		if material:
			self.material = material

	def define_material_parameters(self, alpha=None, beta=None, a_coef:list=None, b_coef:list=None):
		self._a_coef = a_coef
		self._b_coef = b_coef
		self._alpha = alpha
		self._beta = beta 

	def material_setup(self, material=None, lvl=2):
		"""
		Function which defines which material to initialize.

		Parameters:
			material (string):	Defines the material to use. A list of defined materials can be found below

		Predefined materials available:
			From Manjooran et al. 2012 (DOI: 10.1134/S1054660X12080142):

				- "MgO:PPSLT_Manjooran"
				- "PPKTP"
				- "MgO:PPLN_Manjooran"

			From Jundt 1997 (DOI: 10.1364/OL.22.001553):

				- "PPLN"

			From Zelmon et al. 1997 (DOI: 10.1364/JOSAB.14.003319):
				**This contains both ne and no refractive indicies as two returns**

				- "MgO:PPLN_Zelmon" **Note: Columns have been swapped**
				- "PPLN_Zelmon"

			From Gayer et al. 2008 (DOI: 10.1007/s00340-008-2998-2):
				**This contains both ne and no refractive indicies as two returns**

				- "MgO:PPLN"

			From Dolev et al. 2009 (DOI: 10.1007/s00340-009-3502-3)
				**This contains both ne and no refractive indicies as two returns**

				- "MgO:PPSLT"
				
		If material is on predefined, use method "define_material_parameters" to initialize custom material.
		"""

		if material is None:
			material = self.material

		if material == 'MgO:PPSLT_Manjooran': # Following Manjooran et al. 2012 (DOI: 10.1134/S1054660X12080142)
			self._a_coef = [4.5615, 0.08488, 0.1927, 5.5832, 8.3067, 0.021696]
			self._b_coef = [4.782e-7, 3.0913e-8, 2.7326e-8, 1.4837e-5, 1.3647e-7]
			self._alpha = 1.6e-5
			self._beta = 7e-9
			material_text = f'0.5% MgO doped PPSLT'
			reference_text = f'Manjooran et al. 2012 (DOI: 10.1134/S1054660X12080142)'
			refractive_index_text = f'n_e'
			temperature_text = f'Yes'
		elif material == 'PPKTP_Manjooran': # Following Manjooran et al. 2012 (DOI: 10.1134/S1054660X12080142)
			self._a_coef = [0]
			self._b_coef = [0]
			self._alpha = 6.7e-6
			self._beta = 11e-9
			material_text = f'PPKTP'
			reference_text = f'Manjooran et al. 2012 (DOI: 10.1134/S1054660X12080142)'
			refractive_index_text = f'n_e'
			temperature_text = f'Yes'
		elif material == 'MgO:PPLN_Manjooran': # Following Manjooran et al. 2012 (DOI: 10.1134/S1054660X12080142)
			self._a_coef = [5.756, 0.0983, 0.2020, 189.32, 12.52, 1.32e-2]
			self._b_coef = [2.860e-6, 4.700e-8, 6.113e-8, 1.516e-4]
			self._alpha = 1.54e-5
			self._beta = 5.3e-9
			material_text = f'5% MgO doped congruent PPLN'
			reference_text = f'Manjooran et al. 2012 (DOI: 10.1134/S1054660X12080142)'
			refractive_index_text = f'n_e'
			temperature_text = f'Yes'
		elif material == 'PPLN_Jundt': # Following Jundt 1997 (DOI: 10.1364/OL.22.001553)
			self._a_coef = [5.35583, 0.100473, 0.20692, 100, 11.34927, 1.5334e-2]
			self._b_coef = [4.629e-7, 3.862e-8, -0.89e-8, 2.657e-5]
			self._alpha = 1.54e-5 # [1/K]
			self._beta = 5.3e-9 # [1/K^2]
			material_text = f'Congruent PPLN'
			reference_text = f'Jundt 1997 (DOI: 10.1364/OL.22.001553)'
			refractive_index_text = f'n_e'
			temperature_text = f'Yes'
		elif material == 'MgO:PPLN_Zelmon': # Following Zelmon et al. 1997 (DOI: 10.1364/JOSAB.14.003319)
			self._coef_ne = [2.2454, 0.01242, 1.3005, 0.05313, 6.8972, 331.33]
			self._coef_no = [2.4272, 0.01478, 1.4617, 0.05612, 9.6536, 371.216]
			self._alpha = 1.54e-5 # Copied from Jundt
			self._beta = 5.3e-9 # Copied from Jundt
			material_text = f'5% MgO doped congruent PPLN'
			reference_text = f'Zelmon et al. 1997 (DOI: 10.1364/JOSAB.14.003319)'
			refractive_index_text = f'n_e, n_o'
			temperature_text = f'No'
		elif material == 'PPLN_Zelmon': # Following Zelmon et al. 1997 (DOI: 10.1364/JOSAB.14.003319)
			self._coef_ne = [2.9804, 0.02047, 0.5981, 0.0666, 8.9543, 416.08]
			self._coef_no = [2.6734, 0.01764, 1.2290, 0.05914, 12.614, 474.6]
			self._alpha = 1.54e-5 # Copied from Jundt
			self._beta = 5.3e-9 # Copied from Jundt
			material_text = f'Congruent PPLN'
			reference_text = f'Zelmon et al. 1997 (DOI: 10.1364/JOSAB.14.003319)'
			refractive_index_text = f'n_e, n_o'
			temperature_text = f'No'
		elif material in ('MgO:PPLN_Gayer', 'MgO:PPLN'): # Following Gayer et al. 2008 (DOI: 10.1007/s00340-008-2998-2)
			self._a_coef_ne = [5.756, 0.0983, 0.2020, 189.32, 12.52, 1.32e-2]
			self._a_coef_no = [5.653, 0.1185, 0.2091, 89.61, 10.85, 1.97e-2]
			self._b_coef_ne = [2.860e-6, 4.700e-8, 6.113e-8, 1.516e-4]
			self._b_coef_no = [7.941e-7, 3.134e-8, -4.641e-8, -2.188e-6]
			self._alpha = 1.54e-5 # Copied from Jundt
			self._beta = 5.3e-9 # Copied from Jundt
			material_text = f'5% MgO doped congruent PPLN'
			reference_text = f'Gayer et al. 2008 (DOI: 10.1007/s00340-008-2998-2)'
			refractive_index_text = f'n_e, n_o'
			temperature_text = f'Yes'
		elif material in ('MgO:PPSLT_Dolev', 'MgO:PPSLT'): # Following Dolev et al. 2009 (DOI: 10.1007/s00340-009-3502-3)
			self._a_coef_ne = [4.5615, 0.08488, 0.1927, 5.5832, 8.3067, 0.021696]
			self._b_coef_ne = [4.782e-7, 3.0913e-8, 2.7326e-8, 1.4837e-5, 1.3647e-7]
			self._a_coef_no = [4.5082, 0.084888, 0.19552, 1.1570, 8.2517, 0.0237]
			self._b_coef_no = [2.0704e-8, 1.4449e-8, 1.597e-8, 4.768e-6, 1.1127e-5]
			self._alpha = 1.6e-5
			self._beta = 7e-9
			material_text = f'0.5% MgO doped PPLST'
			reference_text = f'Dolev et al. 2009 (DOI: 10.1007/s00340-009-3502-3)'
			refractive_index_text = f'n_e, n_o'
			temperature_text = f'Yes'
		else:
			return print('Please choose an indexed material.')

		print_text = (f'Material data initialised:\n'+
			f'\tMaterial: {material_text}\n'+
			f'\tReference: {reference_text}\n'+
			f'\tRefractive index: {refractive_index_text}\n'+
			f'\tTemperature dependent: {temperature_text}')

		if lvl <= 1:
			print(print_text)

	def generate_signal_idler_wavelengths(self, lambda_s_min, lambda_s_max, num_points=2**10, lambda_p=None):
		'''
		Function that generates the signal and idler arrays based on a specific pump wavelength, which
		satisfies energy conservation,

							        1/λ_p = 1/λ_s + 1/λ_i.

		From this relation and pre-determined bounds on the signal wavelength, the functions calculates
		the corresponding idler wavelengths as,
	
									λ_i = λ_s * λ_p / (λ_s - λ_p).

		Parameters:
		-----------
			lambda_s_min (float):				Lower bound on the signal wavelength.
			lambda_s_max (float):				Upper bound on the singal wavelength.
			num_points (float):					Number of datapoints in the wavelength arrays.
			lambda_p (float):					Wavelength of the pump.

		Returns:
		--------
			lambda_s (ndarray)					Numpy array containing all signal wavelengths
			lambda_i (ndarray)					Numpy array containing all corresponding idler wavelengths

		'''

		if lambda_p is None:
			lambda_p = self.lambda_p

		# Generate an array of signal wavelengths within the specified range
		lambda_s = np.linspace(lambda_s_min, lambda_s_max, num_points)

		# Calculate the corresponding idler wavelengths using the phase matching condition
		lambda_i = (lambda_s * lambda_p) / (lambda_s - lambda_p)

		# Filter out invalid (negative or zero) idler wavelengths
		valid_indices = lambda_i > 0
		lambda_s = lambda_s[valid_indices]
		lambda_i = lambda_i[valid_indices]
    
		return lambda_s, lambda_i

	def generate_DFG(self, lambda_1, lambda_2_min, lambda_2_max, num_points=2**10):
		# Generate an array of signal wavelengths within the specified range
		lambda_2 = np.linspace(lambda_2_min, lambda_2_max, num_points)

		if lambda_2_max > lambda_1:
			# Calculate the corresponding idler wavelengths using the phase matching condition
			lambda_3 = (lambda_1 * lambda_2) / (lambda_2 - lambda_1)
		else:
			lambda_3 = (lambda_1 * lambda_2) / (lambda_1 - lambda_2)
    
		return lambda_2, lambda_3

	def sellmeier_equations(self, wavelength, temperature=None, material=None, lvl=2):
		'''
		Function for calculating the Sellmeier equations for a given material.

		Parameters:
		-----------
			wavelength (float):					Wavelength at which to calculate the refractive index.
			temperature (float):				Temperature of the material in degrees Celcius.
			material (string):					Material in which to calculate the refractive index.
			lvl (int):							Parameter for determining how much to print into the consol.
													A lvl=1 is equal to printing all statements.

		Returns:
		--------
			n_e (float):						Extraordinary refractive index at the specified wavelength
			(n_o (float)):						Ordinary refractive index at the specified wavelength.
													Some materials only contain an extraordinary index!
		'''

		if temperature is None:
			temperature = self.temperature

		if material is None:
			material = self.material

		self.material_setup(material=material, lvl=lvl)

		# Convert wavelength in SI units to emperical units used in the Sellmeier equations
		# Input into the equations are in [um]
		wavelength = wavelength*1e6

		if material == 'PPKTP_Manjooran':
			n_e = np.sqrt(2.12725 + (1.18431/(1 - (5.14852e-2/wavelength**2))) + (0.6603/(1 - (100.00507/wavelength**2))) - 9.68956e-3*wavelength**2) 	
			return n_e

		elif material in ('PPLN_Zelmon', 'MgO:PPLN_Zelmon'):
			n_e = np.sqrt((1 + self._coef_ne[0]*wavelength**2 / (wavelength**2 - self._coef_ne[1])) + 
							self._coef_ne[2]*wavelength**2 / (wavelength**2 - self._coef_ne[3]) + 
								self._coef_ne[4]*wavelength**2 / (wavelength**2 - self._coef_ne[5]))

			n_o = np.sqrt((1 + self._coef_no[0]*wavelength**2 / (wavelength**2 - self._coef_no[1])) + 
							self._coef_no[2]*wavelength**2 / (wavelength**2 - self._coef_no[3]) + 
								self._coef_no[4]*wavelength**2 / (wavelength**2 - self._coef_no[5]))

			return n_e, n_o

		elif material in ('MgO:PPLN_Gayer', 'MgO:PPLN'):
			self._f = (temperature - 24.5) * (temperature + 570.82) # In Celcius

			n_e = np.sqrt(self._a_coef_ne[0] + 
				self._b_coef_ne[0]*self._f + 
					((self._a_coef_ne[1] + self._b_coef_ne[1]*self._f) / (wavelength**2 - (self._a_coef_ne[2] + self._b_coef_ne[2]*self._f)**2)) + 
						((self._a_coef_ne[3] + self._b_coef_ne[3]*self._f) / (wavelength**2 - (self._a_coef_ne[4])**2)) - 
							self._a_coef_ne[5]*(wavelength**2))

			n_o = np.sqrt(self._a_coef_no[0] + 
				self._b_coef_no[0]*self._f + 
					((self._a_coef_no[1] + self._b_coef_no[1]*self._f) / (wavelength**2 - (self._a_coef_no[2] + self._b_coef_no[2]*self._f)**2)) + 
						((self._a_coef_no[3] + self._b_coef_no[3]*self._f) / (wavelength**2 - (self._a_coef_no[4])**2)) - 
							self._a_coef_no[5]*(wavelength**2))
			return n_e, n_o

		elif material in ('MgO:PPLN_Manjooran', 'PPLN_Jundt'):
			self._f = (temperature - 24.5) * (temperature + 570.82) # In Celcius

			n_e = np.sqrt(self._a_coef[0] + 
				self._b_coef[0]*self._f + 
					((self._a_coef[1] + self._b_coef[1]*self._f) / (wavelength**2 - (self._a_coef[2] + self._b_coef[2]*self._f)**2)) + 
						((self._a_coef[3] + self._b_coef[3]*self._f) / (wavelength**2 - (self._a_coef[4])**2)) - 
							self._a_coef[5]*(wavelength**2))
			return n_e

		elif material in ('MgO:PPSLT_Dolev', 'MgO:PPSLT'):
			self._f = (temperature - 24.5)*(temperature + 24.5 + 2*273.16)

			n_e = np.sqrt(self._a_coef_ne[0] + 
				self._b_coef_ne[0]*self._f + 
					((self._a_coef_ne[1] + self._b_coef_ne[1]*self._f) / (wavelength**2 - (self._a_coef_ne[2] + self._b_coef_ne[2]*self._f)**2)) + 
						((self._a_coef_ne[3] + self._b_coef_ne[3]*self._f) / (wavelength**2 - (self._a_coef_ne[4] + self._b_coef_ne[4]*self._f)**2)) - 
							self._a_coef_ne[5]*(wavelength**2))

			n_o = np.sqrt(self._a_coef_no[0] + 
				self._b_coef_no[0]*self._f + 
					((self._a_coef_no[1] + self._b_coef_no[1]*self._f) / (wavelength**2 - (self._a_coef_no[2] + self._b_coef_no[2]*self._f)**2)) + 
						((self._a_coef_no[3] + self._b_coef_no[3]*self._f) / (wavelength**2 - (self._a_coef_no[4] + self._b_coef_no[4]*self._f)**2)) - 
							self._a_coef_no[5]*(wavelength**2))
			return n_e, n_o

		elif material in ('MgO:PPSLT_Manjooran'):
			self._f = (temperature - 24.5)*(temperature + 24.5 + 2*273.16)

			n_e = np.sqrt(self._a_coef[0] + 
				self._b_coef[0]*self._f + 
					((self._a_coef[1] + self._b_coef[1]*self._f) / (wavelength**2 - (self._a_coef[2] + self._b_coef[2]*self._f)**2)) + 
						((self._a_coef[3] + self._b_coef[3]*self._f) / (wavelength**2 - (self._a_coef[4] + self._b_coef[4]*self._f)**2)) - 
							self._a_coef[5]*(wavelength**2))

			return n_e

		else:
			print('Please select suitable material.')

	def refractive_indices(self, material=None):
		'''
		Calculate the ordinary and extraordinary refractive index based on material.
		'''
		pass

	def temperature_dependent_grating_period(self, crystal_period, temperature=None, material=None):
		'''
		Calculates the temperature dependent period of the nonlinear crystal as,

					Λ_eff = Λ * (1 + α(T - 25) + β*(T - 25)**2)

		Parameters:
		-----------
			crystal_period, Λ (float or array): 	Period of the crystal
			temperature, T (float):					Temperature of the crystal in degrees Celsius
			material (string):						Material of the crystal

		Returns:
		--------
			period, Λ_eff (float or array):			Temperature effective crystal period 				

		'''
		if material is None:
			material = self.material
		if temperature is None:
			temperature = self.temperature

		self.material_setup(material=material)

		period = crystal_period*(1 + self._alpha*(temperature-25) + self._beta*(temperature-25)**2)

		return period

	def refractive_index_angle(self, ne, no, theta, phi=0):
		'''
		Function for calculating the effective extraordinary refractive index experienced by being at an angle to
				the optic axis, theta.
		The refractive index is calculated as (Saleh and Teich pp. 887 (Eq. 21.2-21)):

				1 / n(θ, ω) = (cos(θ + ϕ)^2 / no(ω)^2) + (sin(θ + ϕ)^2 / ne(ω)^2)  

		Parameters:
		-----------
			ne (float or list): 	Value of the extraordinary index
			no (float or list):		Value of the ordinary refractive index
			theta (float):			Incidence angle the the normal of the crystal surface in degrees.
			phi (float):			Pre-defined tilt angle of the optic axis with respect to surface normal. Default is 0.

		Returns:
		--------
			n (float or list):		Effective extraordinary refractive index experienced at angle (theta + phi).
		'''
		radians = (theta+phi)*np.pi / 180

		if (theta+phi) in (90.0, 270.0):
			n = ne
		else:
			n = np.sqrt((ne**2 * no**2) / (np.cos(radians)**2 * ne**2 + np.sin(radians)**2 * no**2))

		return n

	def non_collinear_angle_calculation(self, n_p, n_i, n_s, lambda_p, lambda_i, lambda_s, Lambda, temperature=None):
		'''
		Function for calculating the emission angles of the signal and idler at which the phase
		 matching condition is satisfied for a given poling period, Λ.

		Calculates the idler angle, α, as:

			α = arccos(sqrt((kg^2 - 2*kg*kp + ki^2 + kp^2 - ks^2)^2 / ki^2*(-kp + kg)^2) / 2)

		It then finds the signal angle as:

			β = arcsin((ki / ks) * sin(alpha))

		Parameters:
		-----------
			n_p: (float)			Refractive index of the pump.
			n_s: (float or list)		Refractive index of the signal.
			n_i: (float or list)		Refractive index of the idler.
			lambda_p: (float)		Wavelength of the pump.
			lambda_s: (float or list)	Wavelength of the signal.
			lambda_i: (float or list)	Wavelength of the idler.
			Lambda: (float)			Poling period of the nonlinear crystal
			temperature: (float)		Temperature of the crystal in degrees Celcius. 
										Default uses initialised temperature.	

		Returns: 
		--------
			alpha (float or array)		Emission angle between the idler and pump in radians.
			beta (float or array)		Emission angle between the signal and pump in radian.
		'''
		if temperature is None:
			temperature = self.temperature

		# Defining the wavenumbers
		kp = 2*np.pi*n_p / lambda_p
		ks = 2*np.pi*n_s / lambda_s
		ki = 2*np.pi*n_i / lambda_i
		kg = 2*np.pi / Lambda

		numerator_top = (kg**2 - 2*kp*kg + ki**2 + kp**2 - ks**2)**2
		denominator_top = ki**2 * (-kp + kg)**2

		numerator = np.asarray(np.sqrt(numerator_top / denominator_top))
		
		numerator[(numerator > 2) | (numerator < -2)] = np.nan

		alpha = np.arccos((numerator / 2))

		beta = np.arcsin((ki/ks)*np.sin(alpha))

		return alpha, beta

	def non_collinear_angle_calculation2(self, n_p, n_i, n_s, lambda_p, lambda_i, lambda_s, Lambda, temperature=None):
		'''
		Function for calculating the emission angles of the signal and idler at which the phase
		 matching condition is satisfied for a given poling period, Λ.

		Calculates the idler angle, α, as:

			arg1 = sqrt(-(kg+ki-kp-ks)*(kg-ki-kp-ks)*(kg+ki-kp+ks)*(kg-ki-kp+ks)) / (2*ki*(kg-kp))

			arg2 = -(kg^2-2*kg*kp+ki^2+kp^2-ks^2 / (2*ki*(kg-kp)))

			α = arctan2(arg1, arg2)

		It then finds the signal angle as:

			β = arcsin((ki / ks) * sin(alpha))

		Parameters:
		-----------
			n_p: (float)			Refractive index of the pump.
			n_s: (float or list)		Refractive index of the signal.
			n_i: (float or list)		Refractive index of the idler.
			lambda_p: (float)		Wavelength of the pump.
			lambda_s: (float or list)	Wavelength of the signal.
			lambda_i: (float or list)	Wavelength of the idler.
			Lambda: (float)			Poling period of the nonlinear crystal
			temperature: (float)		Temperature of the crystal in degrees Celcius. Default uses initialised temperature.	

		Returns: (tuple)
		--------
			alpha (float or array)		Emission angle between the idler and pump in radians.
			beta (float or array)		Emission angle between the signal and pump in radian.
		'''
		if temperature is None:
			temperature = self.temperature

		# Defining the wavenumbers
		kp = 2*np.pi*n_p / lambda_p
		ks = 2*np.pi*n_s / lambda_s
		ki = 2*np.pi*n_i / lambda_i
		kg = 2*np.pi / Lambda

		num1 = np.sqrt(-(kg+ki-kp-ks)*(kg-ki-kp-ks)*(kg+ki-kp+ks)*(kg-ki-kp+ks))
		dem1 = 2*(kg-kp)*ki

		arg1 = - num1/dem1

		num2 = kg**2-2*kg*kp+ki**2+kp**2-ks**2
		dem2 = 2*(kg-kp)*ki

		arg2 = -num2/dem2

		alpha = np.arctan2(arg1, arg2)
		beta = np.arcsin((ki/ks)*np.sin(alpha))

		return alpha, beta

	def non_collinear_angle_calculation_delta_k(self, n_p, n_i, n_s, lambda_p, lambda_i, lambda_s, Lambda, dk, temperature=None):
		'''
		Function for calculating the emission angles of the signal and idler at which the phase
		 matching condition is satisfied for a given poling period, Λ.

		Calculates the idler angle, α, as:

			arg1 = sqrt(-(kg+ki-kp-ks+dk)*(kg-ki-kp-ks+dk)*(kg+ki-kp+ks+dk)*(kg-ki-kp+ks+dk)) / (2*ki*(kg-kp+dk))

			arg2 = -dk^2+(-2*kg+2*kp)*dk-kg^2+2*kg*kp-ki^2-kp^2+ks^2 / (2*ki*(kg-kp+dk))

			α = arctan2(arg1, arg2)

		It then finds the signal angle as:

			β = arcsin((ki / ks) * sin(alpha))

		Parameters:
		-----------
			n_p: (float)			Refractive index of the pump.
			n_s: (float or list)		Refractive index of the signal.
			n_i: (float or list)		Refractive index of the idler.
			lambda_p: (float)		Wavelength of the pump.
			lambda_s: (float or list)	Wavelength of the signal.
			lambda_i: (float or list)	Wavelength of the idler.
			Lambda: (float)			Poling period of the nonlinear crystal
			temperature: (float)		Temperature of the crystal in degrees Celcius. Default uses initialised temperature.	

		Returns: 
		--------
			alpha (float or array)		Emission angle between the idler and pump in radians.
			beta (float or array)		Emission angle between the signal and pump in radian.
		'''
		if temperature is None:
			temperature = self.temperature

		# Defining the wavenumbers
		kp = 2*np.pi*n_p / lambda_p
		ks = 2*np.pi*n_s / lambda_s
		ki = 2*np.pi*n_i / lambda_i
		kg = 2*np.pi / Lambda

		self.temp = -(kg+ki-kp-ks+dk)*(kg-ki-kp-ks+dk)*(kg+ki-kp+ks+dk)*(kg-ki-kp+ks+dk)
		# print(temp)
		# num1 = np.sqrt(-(kg+ki-kp-ks+dk)*(kg-ki-kp-ks+dk)*(kg+ki-kp+ks+dk)*(kg-ki-kp+ks+dk))
		self.temp[self.temp < 0] = np.nan

		num1 = np.sqrt(self.temp)

		dem1 = 2*(kg-kp+dk)*ki

		arg1 = - num1/dem1

		num2 = -dk**2+(-2*kg+2*kp)*dk-kg**2+2*kg*kp-ki**2-kp**2+ks**2
		dem2 = 2*(kg-kp+dk)*ki

		arg2 = num2/dem2

		alpha = np.arctan2(arg1, arg2)
		beta = np.arcsin((ki/ks)*np.sin(alpha))

		return alpha, beta

	def phase_matching_periods(self, n_p, n_s, n_i, lambda_p, lambda_s, lambda_i, temperature=None):
		'''
		Calculates the crystal periods which \n\t\tsatisfies the phase matching condition:

			k_p - k_s - k_i - k_g = 0

		where k_j (j=p,s,i) is the wave number of the\n\t\tpump, signal, and idler, and k_g is the \n\t\tperiod grating (k_g = 1 / Λ).
		The function then calculates the grating period as:

			Λ = - (λ_p*λ_s*λ_i) / (λ_p*λ_i*n_s + λ_p*λ_s*n_i - λ_s*λ_i*n_p)

		with n_j (j=p,s,i) being the refractive \n\t\tindicies of the pump, signal, and idler.

		Parameters:
		-------------
			n_p (float):						Refractive index of the pump.
			n_s (float or array):				Refractive index of the idler.
			n_i (float or array):				Refractive index of the signal.
			lambda_p (float):					Wavelength of the pump.
			lambda_s (float or array):			Wavelegnth of the idler.
			lambda_i (float or array):			Wavelength of the signal.

		'''
		if temperature is None:
			temperature = self.temperature
		crystal_period = - (lambda_s*lambda_p*lambda_i) / (-lambda_s*lambda_i*n_p + lambda_p*lambda_i*n_s + lambda_p*lambda_s*n_i)

		periods = crystal_period / (1 + self._alpha*(temperature-25) + self._beta*(temperature-25)**2)

		return periods

	def phase_matching_collinear(self, n_p, n_i, n_s, lambda_p, lambda_i, lambda_s, Lambda):

		Delta_k = (n_p / lambda_p) - (n_s / lambda_s) - (n_i / lambda_i) - (1 / Lambda)

		return Delta_k

	def phase_matching_full(self, n_p, n_i, n_s, lambda_p, lambda_i, lambda_s, Lambda, 
								theta_i_rad=None, theta_i=None, theta_s_rad=None, theta_s=None):
		'''
		Calculates the phase matching condition for noncollinear emission as,

						Δk = 2π(k_p - k_i*cos(α) - k_s*cos(arcsin((k_i/k_s)*sin(α))) - k_g),

		where k_j (j=p,s,i) is the wavenumber of pump, signal, and idler given as k_j = n_j / λ_j, and 
		k_g is the grating period k_g = 2π / Λ.

		Parameters:
		-----------
			n_p (float):						Refractive index of the pump.
			n_i (float or array):				Refractive index of the idler.
			n_s (float or array):				Refractive index of the signal.
			lambda_p (float):					Wavelength of the pump.
			lambda_i (float or array):			Wavelegnth of the idler.
			lambda_s (float or array):			Wavelength of the signal.
			theta_rad (float):					Angle of the idler, α in radians
			theta (float):						Angle of the idler, α in degrees

		Returns:
		--------
			Delta_k (float or array):			Phase matching condition
		'''

		angles = [("theta_i", theta_i), ("theta_i_rad", theta_i_rad), ("theta_s", theta_s), ("theta_s_rad", theta_s_rad)]

		# Filter out None values
		non_none_angles = [(name, x) for name, x in angles if x is not None]

		# Check if more than one variable is not None
		if len(non_none_angles) > 1:
			raise ValueError("More than one variable is provided. Only one should be not None.")
		elif len(non_none_angles) == 0:
			raise ValueError("At least one variable must be provided")

		# Assign the single non-None variable
		angle_name, theta = non_none_angles[0]

		if angle_name in ("theta_i", "theta_i_rad"):
			Delta_k = ((n_p/lambda_p) - (n_i/lambda_i)*np.cos(theta) - 
				(n_s/lambda_s)*np.cos(np.arcsin((n_i*lambda_s/n_s*lambda_i)*np.sin(theta))) - (1/Lambda)) *2*np.pi

		elif angle_name in ("theta_s", "theta_s_rad"):
			Delta_k = ((n_p/lambda_p) - (n_s/lambda_s)*np.cos(theta) - 
				(n_i/lambda_i)*np.cos(np.arcsin((n_s*lambda_i/n_i*lambda_s)*np.sin(theta))) - (1/Lambda)) *2*np.pi

		return Delta_k

	def phase_matching_periods_angle_optic_axis_perpendicular(self, ne_p, ne_s, ne_i, 
																lambda_p, lambda_s, lambda_i, theta, phi=90,
																interaction_type='eoe', 
																no_p=None, no_s=None, no_i=None, temperature=None):
		if temperature is None:
			temperature = self.temperature

		if interaction_type in ('eoe'):
			n_p = self.refractive_index_angle(ne=ne_p, no=no_p, theta=theta, phi=phi)
			n_s = self.refractive_index_angle(ne=ne_s, no=no_s, theta=theta, phi=phi)
			n_i = no_i
		else:
			print('Please check input parameters!')
			return

		crystal_period = - (lambda_s*lambda_p*lambda_i) / (-lambda_s*lambda_i*n_p + lambda_p*lambda_i*n_s + lambda_p*lambda_s*n_i)

		crystal_period = crystal_period * np.cos((90-theta)*np.pi/180)

		periods = crystal_period / (1 + self._alpha*(temperature-25) + self._beta*(temperature-25)**2)

		return periods

	def phase_matching_condition(self, ne_p, ne_s, ne_i, lambda_p, lambda_s, lambda_i, crystal_period=None):
		if crystal_period is None:
			crystal_period = self.crystal_period

		for index, (b, c, l_s, l_i) in enumerate(zip(ne_s, ne_i, lambda_s, lambda_i)):
			if np.isclose((self.ne_p / self.lambda_p) - (b / l_s) - (c / l_i) - (1 / crystal_period), 0):
				return index
		return -1

	def calculate_crystal_period(self, n_p, n_s, n_i, lambda_p, lambda_s, lambda_i):

		reciprocal_period = (n_p / lambda_p) - (n_s / lambda_s) - (n_i / lambda_i)

		period = 1 / reciprocal_period * 1e6 # in um

		return period

	def signal_power_output(self, omega_s, omega_i, n_p, n_s, n_i, L, Lambda, P_p, theta_s):
		'''
		docstring
		'''
		d = 15.1e-12 # [m/V]
		theta_rad = np.reshape((np.deg2rad(theta_s)), (len(theta_s),1))

		dk = self.phase_matching_full(n_p=n_p, n_i=n_i, n_s=n_s, lambda_p=lambda_p, 
			lambda_i=lambda_i, lambda_s=lambda_s, Lambda=Lambda, theta_s_rad=theta_rad)

		# print(dk)
		# print(dk.shape)

		domega_s = abs(omega_s[1] - omega_s[0])
		dtheta_s = abs(theta_rad[1] - theta_rad[0])

		dP_s = np.trapz(self.hbar*omega_s**4*omega_i*n_s**2*L**2*d**2*P_p / 
			(2*np.pi**2*self.c**5*self.epsilon_0*n_p*n_i) * 
			(np.sin(theta_rad) / (np.cos(theta_rad)**3)) * (np.sinc(dk*L/2)**2 * domega_s), dx=dtheta_s, axis=0)

		# dP_s = (self.hbar*omega_s**4*omega_i*n_s**2*L**2*P_p / (2*np.pi**2*self.c**5*self.epsilon_0*n_p*n_i) * 
		# 	(np.sin(theta_rad) / np.cos(theta_rad)**3) * (np.sinc(dk*L/2)**2)*domega_s*dtheta_s)

		# total_angle_range = np.max(theta_rad) - np.min(theta_rad)
		# dP_s_normalized = dP_s / total_angle_range

		return dP_s
		# return dP_s_normalized

	def plot_Manjooran_et_al_fig1(self):
		lambda_s, lambda_i = self.generate_signal_idler_wavelengths(lambda_s_min=600e-9, lambda_s_max=2000e-9, lambda_p=520e-9)

		PPcLN_ne_s = self.sellmeier_equations(wavelength=lambda_s, material='MgO:PPLN_Manjooran', temperature=100)
		PPcLN_ne_i = self.sellmeier_equations(wavelength=lambda_i, material='MgO:PPLN_Manjooran', temperature=100)

		PPSLT_ne_s = self.sellmeier_equations(wavelength=lambda_s, material='MgO:PPSLT_Manjooran', temperature=100)
		PPSLT_ne_i = self.sellmeier_equations(wavelength=lambda_i, material='MgO:PPSLT_Manjooran', temperature=100)

		PPKTP_ne_s = self.sellmeier_equations(wavelength=lambda_s, material='PPKTP_Manjooran', temperature=22)
		PPKTP_ne_i = self.sellmeier_equations(wavelength=lambda_i, material='PPKTP_Manjooran', temperature=22)

		# Calculate the group velocities (factored for representation on graph (1e8))
		# v_g = c / (n - lambda * (dn/dlambda))
		PPcLN_vg = 3 / (PPcLN_ne_s - lambda_s*np.gradient(PPcLN_ne_s, lambda_s)) 
		PPSLT_vg = 3 / (PPSLT_ne_s - lambda_s*np.gradient(PPSLT_ne_s, lambda_s))
		PPKTP_vg = 3 / (PPKTP_ne_s - lambda_s*np.gradient(PPKTP_ne_s, lambda_s))


		fig = plt.figure(figsize=(8,6))
		gs = GridSpec(1,1, figure=fig)

		ax = fig.add_subplot(gs[0,0])
		
		ax.plot(lambda_s, PPcLN_ne_s, color='C0', linestyle='solid', label='PPcLN')
		ax.plot(lambda_s, PPSLT_ne_s, color='C1', linestyle='solid', label='PPSLT')
		ax.plot(lambda_s, PPKTP_ne_s, color='C2', linestyle='solid', label='PPKTP')
		ax.set_xlabel('Wavelength [$\\mu$m]')
		ax.set_ylabel('Refractive index, $n_e$')
		ax.set_ylim(1.8, 2.25)
		ax.set_xlim(0.680e-6, 2e-6)

		ax2 = ax.twinx()
		ax2.plot(lambda_s, PPcLN_vg, color='C0', linestyle='dotted', label='PPcLN')
		ax2.plot(lambda_s, PPSLT_vg, color='C1', linestyle='dotted', label='PPSLT')
		ax2.plot(lambda_s, PPKTP_vg, color='C2', linestyle='dotted', label='PPKTP')
		ax2.set_ylim(1.30, 1.7)
		ax2.set_ylabel('Group velocity, $v_g$ [$10^{8}$ m/s]')

		ax.legend(title='$n_e$', loc=6)
		ax2.legend(title='$v_g$', loc=7)

		fig.set_tight_layout(True)
		fig.show()

	def plot_Manjooran_et_al_fig2(self):
		lambda_s, lambda_i = self.generate_signal_idler_wavelengths(lambda_s_min=600e-9, lambda_s_max=2000e-9, lambda_p=520e-9)


		PPcLN_ne_p = self.sellmeier_equations(wavelength=520e-9, material='MgO:PPLN_Manjooran', temperature=100)
		PPcLN_ne_s = self.sellmeier_equations(wavelength=lambda_s, material='MgO:PPLN_Manjooran', temperature=100)
		PPcLN_ne_i = self.sellmeier_equations(wavelength=lambda_i, material='MgO:PPLN_Manjooran', temperature=100)

		PPSLT_ne_p = self.sellmeier_equations(wavelength=520e-9, material='MgO:PPSLT_Manjooran', temperature=100)
		PPSLT_ne_s = self.sellmeier_equations(wavelength=lambda_s, material='MgO:PPSLT_Manjooran', temperature=100)
		PPSLT_ne_i = self.sellmeier_equations(wavelength=lambda_i, material='MgO:PPSLT_Manjooran', temperature=100)

		PPKTP_ne_p = self.sellmeier_equations(wavelength=520e-9, material='PPKTP_Manjooran', temperature=22)
		PPKTP_ne_s = self.sellmeier_equations(wavelength=lambda_s, material='PPKTP_Manjooran', temperature=22)
		PPKTP_ne_i = self.sellmeier_equations(wavelength=lambda_i, material='PPKTP_Manjooran', temperature=22)

		PPcLN_periods = self.phase_matching_periods(n_p=PPcLN_ne_p, n_s=PPcLN_ne_s, n_i=PPcLN_ne_i, 
			lambda_p=520e-9, lambda_s=lambda_s, lambda_i=lambda_i, temperature=100)
		PPSLT_periods = self.phase_matching_periods(n_p=PPSLT_ne_p, n_s=PPSLT_ne_s, n_i=PPSLT_ne_i, 
			lambda_p=520e-9, lambda_s=lambda_s, lambda_i=lambda_i, temperature=100)
		PPKTP_periods = self.phase_matching_periods(n_p=PPKTP_ne_p, n_s=PPKTP_ne_s, n_i=PPKTP_ne_i, 
			lambda_p=520e-9, lambda_s=lambda_s, lambda_i=lambda_i, temperature=22)

		#------------------------------------------------------------------------------------------------------------#
		#                                         Recreation of figure 1                                             #
		#------------------------------------------------------------------------------------------------------------#

		fig = plt.figure(figsize=(8,6))
		gs = GridSpec(1,1, figure=fig)

		half = int(len(lambda_s)/2)
		idx = (np.abs(lambda_s - 1050e-9)).argmin()

		ax = fig.add_subplot(gs[0,0])
		ax.plot(PPcLN_periods[0:idx]*1e6, lambda_s[0:idx]*1e6, color='C0', linestyle='solid', label='PPcLN')
		ax.plot(PPSLT_periods[0:idx]*1e6, lambda_s[0:idx]*1e6, color='C1', linestyle='solid', label='PPSLT')
		ax.plot(PPKTP_periods[0:idx]*1e6, lambda_s[0:idx]*1e6, color='C2', linestyle='solid', label='PPKTP')
		ax.set_ylabel('Signal wavelength [$\\mu$m]')
		ax.set_xlabel('Grating period [$\\mu$m]')
		ax.set_xlim(6,10)
		ax.set_ylim(0.6,2)
		ax.legend(title='Signal')

		axb = ax.twinx()
		axb.plot(PPcLN_periods*1e6, lambda_i*1e6, color='C0', linestyle='dotted', label='PPcLN')
		axb.plot(PPSLT_periods*1e6, lambda_i*1e6, color='C1', linestyle='dotted', label='PPSLT')
		axb.plot(PPKTP_periods*1e6, lambda_i*1e6, color='C2', linestyle='dotted', label='PPKTP')
		axb.set_ylabel('Idler wavelength [$\\mu$m]')
		axb.legend(title='Idler')
		axb.set_ylim(0.6,2)

		fig.set_tight_layout(True)
		fig.show()

		#------------------------------------------------------------------------------------------------------------#
		#                            Full figure showing full range of signal and idler                              #
		#------------------------------------------------------------------------------------------------------------#

		fig2 = plt.figure(figsize=(8,6))
		gs2 = GridSpec(1,1, figure=fig2)

		ax2 = fig2.add_subplot(gs2[0,0])
		ax2.plot(PPcLN_periods*1e6, lambda_s*1e6, color='C0', linestyle='solid', label='PPcLN')
		ax2.plot(PPSLT_periods*1e6, lambda_s*1e6, color='C1', linestyle='solid', label='PPSLT')
		ax2.plot(PPKTP_periods*1e6, lambda_s*1e6, color='C2', linestyle='solid', label='PPKTP')
		ax2.set_ylabel('Signal wavelength [$\\mu$m]')
		ax2.set_xlabel('Grating period [$\\mu$m]')
		ax2.set_xlim(6,10)
		ax2.legend(title='Signal')

		axb2 = ax2.twinx()
		axb2.plot(PPcLN_periods*1e6, lambda_i*1e6, color='C0', linestyle='dotted', label='PPcLN')
		axb2.plot(PPSLT_periods*1e6, lambda_i*1e6, color='C1', linestyle='dotted', label='PPSLT')
		axb2.plot(PPKTP_periods*1e6, lambda_i*1e6, color='C2', linestyle='dotted', label='PPKTP')
		axb2.set_ylabel('Idler wavelength [$\\mu$m]')
		axb2.legend(title='Idler')

		fig2.set_tight_layout(True)
		fig2.show()

	def plot_refractive_indices(self, show_residual=True):
		wavelength = np.linspace(500e-9, 7e-6, 2**14)

		ne_PPLN_Gayer, no_PPLN_Gayer = self.sellmeier_equations(wavelength=wavelength, material='MgO:PPLN_Gayer')
		ne_PPSLT_Dolev, no_PPSLT_Dolev = self.sellmeier_equations(wavelength=wavelength, material='MgO:PPSLT_Dolev')

		fig = plt.figure(figsize=(8,6))
		gs = GridSpec(2,1, figure=fig)

		axa = fig.add_subplot(gs[0,0])
		axa.plot(wavelength*1e6, ne_PPLN_Gayer, color='C0', linestyle='solid', label='$n_e$ (Gayer)')
		axa.plot(wavelength*1e6, no_PPLN_Gayer, color='C1', linestyle='solid', label='$n_o$ (Gayer)')
		axa.set_xlabel('Wavelength [$\\mu$m]')
		axa.set_ylabel('Refractive Index $n_e, n_o$')
		axa.set_title('5% MgO doped congruent PPLN')
		axa.legend()

		axb = fig.add_subplot(gs[1,0])
		axb.plot(wavelength*1e6, ne_PPSLT_Dolev, color='C0', linestyle='solid', label='$n_e$ (Dolev)')
		axb.plot(wavelength*1e6, no_PPSLT_Dolev, color='C1', linestyle='solid', label='$n_o$ (Dolev)')
		axb.set_xlabel('Wavelength [$\\mu$m]')
		axb.set_ylabel('Refractive Index $n_e, n_o$')
		axb.set_title('0.5% MgO doped PPSLT')
		axb.legend()

		if show_residual:
			axa2 = axa.twinx()
			axa2.plot(wavelength*1e6, (no_PPLN_Gayer - ne_PPLN_Gayer), color='C6', linestyle='dashed', 
				label='$\\Delta n$')
			axa2.set_ylabel('$\\Delta n$')
			axa2.legend(loc=3)

			axb2 = axb.twinx()
			axb2.plot(wavelength*1e6, (no_PPSLT_Dolev - ne_PPSLT_Dolev), color='C6', linestyle='dashed', 
				label='$\\Delta n$')
			axb2.set_ylabel('$\\Delta n$')
			axb2.legend(loc=3)

		fig.set_tight_layout(True)
		fig.show()

	def plot_refractive_index_reference_comparison_PPLN(self, show_residual=True):
		wavelength = np.linspace(600e-9, 5e-6, 2**10)

		ne_PPLN_zelmon, no_PPLN_zelmon = self.sellmeier_equations(wavelength=wavelength, material='PPLN_Zelmon')
		ne_PPLN = self.sellmeier_equations(wavelength=wavelength, material='PPLN_Jundt')
		ne_MgPPLN_zelmon, no_MgPPLN_zelmon = self.sellmeier_equations(wavelength=wavelength, material='MgO:PPLN_Zelmon')
		ne_PPcLN = self.sellmeier_equations(wavelength=wavelength, material='MgO:PPLN_Manjooran')

		fig = plt.figure(figsize=(8,6))
		gs = GridSpec(2,1,figure=fig)

		axa = fig.add_subplot(gs[0,0])
		axa.plot(wavelength*1e6, ne_PPLN_zelmon, color='C0', linestyle='solid', label='$n_e$ (Zelmon)')
		axa.plot(wavelength*1e6, ne_PPLN, color='C1', linestyle='solid', label='$n_e$ (Jundt)')
		axa.plot(wavelength*1e6, no_PPLN_zelmon, color='C0', linestyle='dotted', label='$n_o$ (Zelmon)')
		axa.set_xlabel('Wavelength [$\\mu$m]')
		axa.set_ylabel('Refractive Index $n_e, n_o$')
		axa.set_title('PPLN')
		axa.legend()

		axb = fig.add_subplot(gs[1,0])
		axb.plot(wavelength*1e6, ne_MgPPLN_zelmon, color='C0', linestyle='solid', label='$n_e$ (Zelmon)')
		axb.plot(wavelength*1e6, ne_PPcLN, color='C1', linestyle='solid', label='$n_e$ (Manjooran)')
		axb.plot(wavelength*1e6, no_MgPPLN_zelmon, color='C0', linestyle='dotted', label='$n_o$ (Zelmon)')
		axb.set_xlabel('Wavelength [$\\mu$m]')
		axb.set_ylabel('Refractive Index $n_e, n_o$')
		axb.set_title('5% MgO doped PPLN')
		axb.legend()

		if show_residual:
			axa2 = axa.twinx()
			axa2.plot(wavelength*1e6, (ne_PPLN_zelmon - ne_PPLN), color='C6', linestyle='dashed', label='$\\Delta n_e$')
			axa2.set_ylabel('Residual $n_e$')
			axa2.legend(loc=3)

			axb2 = axb.twinx()
			axb2.plot(wavelength*1e6, (ne_MgPPLN_zelmon - ne_PPcLN), color='C6', linestyle='dashed', label='$\\Delta n_e$')
			axb2.set_ylabel('Residual $n_e$')
			axb2.legend(loc=3)

		fig.set_tight_layout(True)
		fig.show()

	def plot_Hojo2021_fig3(self):

		# Generate response from PPLN
		lambda_p_PPLN = 800e-9
		lambda_s_PPLN, lambda_i_PPLN = self.generate_signal_idler_wavelengths(lambda_p=lambda_p_PPLN, 
			lambda_s_min=900e-9, lambda_s_max=1500e-9, num_points=2**20)
		
		ne_p_PPLN, no_p_PPLN = self.sellmeier_equations(wavelength=lambda_p_PPLN, material='MgO:PPLN_Gayer')
		ne_s_PPLN, no_s_PPLN = self.sellmeier_equations(wavelength=lambda_s_PPLN, material='MgO:PPLN_Gayer')
		ne_i_PPLN, no_i_PPLN = self.sellmeier_equations(wavelength=lambda_i_PPLN, material='MgO:PPLN_Gayer')

		periods_PPLN = self.phase_matching_periods(n_p=ne_p_PPLN, n_s=ne_s_PPLN, n_i=ne_i_PPLN,
												lambda_p=lambda_p_PPLN, lambda_s=lambda_s_PPLN, lambda_i=lambda_i_PPLN,
												temperature=25)

		alpha_PPLN, beta_PPLN = self.non_collinear_angle_calculation(n_p=ne_p_PPLN, n_i=ne_i_PPLN, n_s=ne_s_PPLN, 
										lambda_p=lambda_p_PPLN, lambda_i=lambda_i_PPLN, lambda_s=lambda_s_PPLN, 
										Lambda=21e-6, temperature=25)
		alpha_PPLN_deg, beta_PPLN_deg = np.rad2deg(alpha_PPLN), np.rad2deg(beta_PPLN)

		# Generate reponse from PPSLT
		lambda_p_PPSLT = 750e-9
		lambda_s_PPSLT, lambda_i_PPSLT = self.generate_signal_idler_wavelengths(lambda_p=lambda_p_PPSLT, 
			lambda_s_min=850e-9, lambda_s_max=1500e-9, num_points=2**20)

		ne_p_PPSLT, no_p_PPSLT = self.sellmeier_equations(wavelength=lambda_p_PPSLT, 
															material='MgO:PPSLT_Dolev', temperature=24.5)
		ne_s_PPSLT, no_s_PPSLT = self.sellmeier_equations(wavelength=lambda_s_PPSLT, 
															material='MgO:PPSLT_Dolev', temperature=24.5)
		ne_i_PPSLT, no_i_PPSLT = self.sellmeier_equations(wavelength=lambda_i_PPSLT, 
															material='MgO:PPSLT_Dolev', temperature=24.5)

		
		periods_PPSLT = self.phase_matching_periods(n_p=ne_p_PPSLT, n_s=ne_s_PPSLT, n_i=ne_i_PPSLT,
												lambda_p=lambda_p_PPSLT, lambda_s=lambda_s_PPSLT, lambda_i=lambda_i_PPSLT,
												temperature=25)

		alpha_PPSLT, beta_PPSLT = self.non_collinear_angle_calculation(n_p=ne_p_PPSLT, n_i=ne_i_PPSLT, n_s=ne_s_PPSLT, 
										lambda_p=lambda_p_PPSLT, lambda_i=lambda_i_PPSLT, lambda_s=lambda_s_PPSLT, 
										Lambda=20e-6, temperature=25)
		alpha_PPSLT_deg, beta_PPSLT_deg = np.rad2deg(alpha_PPSLT), np.rad2deg(beta_PPSLT)

		# Create figure
		fig = plt.figure(figsize=(12,6))
		gs = GridSpec(2,2,figure=fig)

		# Create the PPSLT figures
		# Poling period figure
		axa = fig.add_subplot(gs[0,0])
		axa.plot(lambda_s_PPSLT*1e9, periods_PPSLT*1e6, color='blue', label='Signal')
		axa.set_ylabel('Poling Period [$\\mu$m]')
		axa.set_xlabel('Signal wavelength [nm]')
		axa.set_ylim(19.5, 22)
		axa.set_xlim(800, 1500)
		# axa.legend()

		axa2 = axa.twiny()
		axa2.plot(lambda_i_PPSLT*1e6, periods_PPSLT*1e6, color='green', label='Idler')
		axa2.set_xlabel('Idler wavelength [$\\mu$m]')
		axa2.invert_xaxis()
		axa2.set_xlim(6, 1.5)
		# axa2.legend()

		handlesa, labelsa = axa.get_legend_handles_labels()
		handlesa2, labelsa2 = axa2.get_legend_handles_labels()
		handlesa.extend(handlesa2)
		labelsa.extend(labelsa2)

		# Create a single legend
		axa.legend(handlesa, labelsa, loc='upper right', title='PPSLT')

		textstra = f'$\\lambda_p$ = {lambda_p_PPSLT*1e9} nm'
		propsa = dict(boxstyle='round', facecolor='none', alpha=0.5)

		axa.text(0.23, 0.95, textstra, transform=axa.transAxes, fontsize=10,
    		verticalalignment='top', horizontalalignment='right', bbox=propsa)

		axa.axhline(y=20, color='grey', linestyle='--')

		# Angle figure
		axb = fig.add_subplot(gs[0,1])
		axb.plot(lambda_s_PPSLT*1e9, beta_PPSLT_deg, color='blue', label='Signal')
		axb.set_ylabel('SPDC angle [$\\degree$]')
		axb.set_xlabel('Signal wavelength [nm]')
		axb.set_xlim(800, 1500)
		# axa.legend()

		axb2 = axb.twiny()
		axb2.plot(lambda_i_PPSLT*1e6, alpha_PPSLT_deg, color='green', label='Idler')
		axb2.set_xlabel('Idler wavelength [$\\mu$m]')
		axb2.invert_xaxis()
		axb2.set_xlim(6, 1.5)
		# axa2.legend()

		handlesb, labelsb = axb.get_legend_handles_labels()
		handlesb2, labelsb2 = axb2.get_legend_handles_labels()
		handlesb.extend(handlesb2)
		labelsb.extend(labelsb2)

		# Create a single legend
		axb.legend(handlesb, labelsb, loc='upper right', title='PPSLT')

		textstrb = f'$\\Lambda$ = 20 $\\mu$m'
		propsb = dict(boxstyle='round', facecolor='none', alpha=0.5)

		axb.text(0.18, 0.95, textstrb, transform=axb.transAxes, fontsize=10,
    		verticalalignment='top', horizontalalignment='right', bbox=propsb)
		axb.axhline(y=0, color='grey', linestyle='--')


		# Create the PPLN figures
		axc = fig.add_subplot(gs[1,0])
		axc.plot(lambda_s_PPLN*1e9, periods_PPLN*1e6, color='blue', label='Signal')
		axc.set_ylabel('Poling Period [$\\mu$m]')
		axc.set_xlabel('Signal wavelength [nm]')
		axc.set_ylim(20.5, 23)
		# axa.legend()

		axc2 = axc.twiny()
		axc2.plot(lambda_i_PPLN*1e6, periods_PPLN*1e6, color='green', label='Idler')
		axc2.set_xlabel('Idler wavelength [$\\mu$m]')
		axc2.invert_xaxis()
		axc2.set_xlim(6, 1.5)
		# axa2.legend()

		handlesc, labelsc = axc.get_legend_handles_labels()
		handlesc2, labelsc2 = axc2.get_legend_handles_labels()
		handlesc.extend(handlesc2)
		labelsc.extend(labelsc2)

		# Create a single legend
		axc.legend(handlesc, labelsc, loc='upper right', title='PPLN')

		textstrc = f'$\\lambda_p$ = {lambda_p_PPLN*1e9} nm'
		propsc = dict(boxstyle='round', facecolor='none', alpha=0.5)

		axc.text(0.23, 0.95, textstrc, transform=axc.transAxes, fontsize=10,
    		verticalalignment='top', horizontalalignment='right', bbox=propsc)

		axc.axhline(y=21, color='grey', linestyle='--')

		# Angle figure
		axd = fig.add_subplot(gs[1,1])
		axd.plot(lambda_s_PPLN*1e9, beta_PPLN_deg, color='blue', label='Signal')
		axd.set_ylabel('SPDC angle [$\\degree$]')
		axd.set_xlabel('Signal wavelength [nm]')
		axd.set_xlim(900, 1500)
		# axa.legend()

		axd2 = axd.twiny()
		axd2.plot(lambda_i_PPLN*1e6, alpha_PPLN_deg, color='green', label='Idler')
		axd2.set_xlabel('Idler wavelength [$\\mu$m]')
		axd2.invert_xaxis()
		axd2.set_xlim(6, 1.5)
		# axa2.legend()

		handlesd, labelsd = axd.get_legend_handles_labels()
		handlesd2, labelsd2 = axd2.get_legend_handles_labels()
		handlesd.extend(handlesd2)
		labelsd.extend(labelsd2)

		# Create a single legend
		axd.legend(handlesd, labelsd, loc='upper right', title='PPLN')

		textstrd = f'$\\Lambda$ = 21 $\\mu$m'
		propsd = dict(boxstyle='round', facecolor='none', alpha=0.5)

		axd.text(0.18, 0.95, textstrd, transform=axd.transAxes, fontsize=10,
    		verticalalignment='top', horizontalalignment='right', bbox=propsd)

		axd.axhline(y=0, color='grey', linestyle='--')

		fig.set_tight_layout(True)
		fig.show()

	def compare_methods_with_parameters(self, temperature=25, material='MgO:PPLN', Lambda=18e-6, figsize=(9,6),
		lambda_s_min=900e-9, lambda_s_max=None, lambda_p=800e-9,
		interaction_type='eee', show_degenerate_wavelength=True,
		num_points=2**18):

		if not Lambda:
			Lambda = self.crystal_period

		if not lambda_s_max:
			lambda_s_max = lambda_p*2

		lambda_s, lambda_i = self.generate_signal_idler_wavelengths(lambda_p=lambda_p, 
			lambda_s_min=lambda_s_min, 
			lambda_s_max=lambda_s_max, 
			num_points=num_points)

		ne_p, no_p = self.sellmeier_equations(wavelength=lambda_p, material=material,
			temperature=temperature)
		ne_s, no_s = self.sellmeier_equations(wavelength=lambda_s, material=material,
			temperature=temperature)
		ne_i, no_i = self.sellmeier_equations(wavelength=lambda_i, material=material,
			temperature=temperature)

		if interaction_type in ('ooo'): # Type 0
			n_p, n_s, n_i = no_p, no_s, no_i 
		elif interaction_type in ('eee'): # Type 0
			n_p, n_s, n_i = ne_p, ne_s, ne_i
		elif interaction_type in ('ooe'): # Type I
			n_p, n_s, n_i = ne_p, no_s, no_i
		elif interaction_type in ('eeo'): # Type I
			n_p, n_s, n_i = no_p, ne_s, ne_i
		elif interaction_type in ('eoe'): # Type II
			n_p, n_s, n_i = ne_p, ne_s, no_i
		elif interaction_type in ('oee'): # Type II
			n_p, n_s, n_i = ne_p, no_s, ne_i
		elif interaction_type in ('eoo'): # Type II
			n_p, n_s, n_i = no_p, ne_s, no_i
		elif interaction_type in ('oeo'): # Type II
			n_p, n_s, n_i = no_p, no_s, ne_i
		else:
			print('Please choose valid interaction type, e.g., \'eee\', \'ooo\', \'eoe\', etc.')


		# Calculate angles using the three methods
		idler_angle, signal_angle = self.non_collinear_angle_calculation2(n_p=n_p, n_i=n_i, n_s=n_s,
			lambda_p=lambda_p, lambda_s=lambda_s, 
			lambda_i=lambda_i, Lambda=Lambda)
		idler_angle2, signal_angle2 = self.non_collinear_angle_calculation_delta_k(n_p=n_p, n_i=n_i, n_s=n_s,
			lambda_p=lambda_p, lambda_s=lambda_s, 
			lambda_i=lambda_i, Lambda=Lambda, dk=0)
		idler_angle3, signal_angle3 = self.non_collinear_angle_calculation2(n_p=n_p, n_i=n_i, n_s=n_s,
			lambda_p=lambda_p, lambda_s=lambda_s, 
			lambda_i=lambda_i, Lambda=Lambda)

		# Convert angles to degrees
		idler_angle_deg, signal_angle_deg = np.rad2deg(idler_angle), np.rad2deg(signal_angle)
		idler_angle_deg2, signal_angle_deg2 = np.rad2deg(idler_angle2), np.rad2deg(signal_angle2)
		idler_angle_deg3, signal_angle_deg3 = np.rad2deg(idler_angle3), np.rad2deg(signal_angle3)

		# Print and compare results
		print("Method 1 (alpha, beta):", idler_angle_deg, signal_angle_deg)
		print("Method 2 with dk=0 (alpha, beta):", idler_angle_deg2, signal_angle_deg2)
		print("Method 3 (alpha, beta):", idler_angle_deg3, signal_angle_deg3)

		assert np.allclose(idler_angle_deg, idler_angle_deg2, equal_nan=True), "Mismatch between Method 1 and Method 2"
		assert np.allclose(signal_angle_deg, signal_angle_deg2, equal_nan=True), "Mismatch between Method 1 and Method 2"
		assert np.allclose(idler_angle_deg, idler_angle_deg3, equal_nan=True), "Mismatch between Method 1 and Method 3"
		assert np.allclose(signal_angle_deg, signal_angle_deg3, equal_nan=True), "Mismatch between Method 1 and Method 3"

	def plot_non_collinear_emission_angle(self, temperature=25, material='MgO:PPLN', Lambda=18e-6, figsize=(9,6),
									lambda_s_min=900e-9, lambda_s_max=None, lambda_p=800e-9,
									interaction_type='eee',
									show_degenerate_wavelength=True,
									num_points=2**18):

		if not Lambda:
			Lambda = self.crystal_period

		if not lambda_s_max:
			lambda_s_max = lambda_p*2
		lambda_s, lambda_i = self.generate_signal_idler_wavelengths(lambda_p=lambda_p, 
			lambda_s_min=lambda_s_min, lambda_s_max=lambda_s_max, num_points=num_points)

		ne_p, no_p = self.sellmeier_equations(wavelength=lambda_p, material=material,
			temperature=temperature)
		ne_s, no_s = self.sellmeier_equations(wavelength=lambda_s, material=material,
			temperature=temperature)
		ne_i, no_i = self.sellmeier_equations(wavelength=lambda_i, material=material,
			temperature=temperature)

		if interaction_type in ('ooo'): # Type 0
			n_p, n_s, n_i = no_p, no_s, no_i 
		elif interaction_type in ('eee'): # Type 0
			n_p, n_s, n_i = ne_p, ne_s, ne_i
		elif interaction_type in ('ooe'): # Type I
			n_p, n_s, n_i = ne_p, no_s, no_i
		elif interaction_type in ('eeo'): # Type I
			n_p, n_s, n_i = no_p, ne_s, ne_i
		elif interaction_type in ('eoe'): # Type II
			n_p, n_s, n_i = ne_p, ne_s, no_i
		elif interaction_type in ('oee'): # Type II
			n_p, n_s, n_i = ne_p, no_s, ne_i
		elif interaction_type in ('eoo'): # Type II
			n_p, n_s, n_i = no_p, ne_s, no_i
		elif interaction_type in ('oeo'): # Type II
			n_p, n_s, n_i = no_p, no_s, ne_i
		else:
			print('Please choose valid interaction type, e.g., \'eee\', \'ooo\', \'eoe\', etc.')

		idler_angle, signal_angle = self.non_collinear_angle_calculation(n_p=n_p, n_i=n_i, n_s=n_s,
			lambda_p=lambda_p, lambda_s=lambda_s, lambda_i=lambda_i, Lambda=Lambda)

		idler_angle_deg, signal_angle_deg = np.rad2deg(idler_angle), np.rad2deg(signal_angle)

		fig = plt.figure(figsize=(figsize))
		gs = GridSpec(1,1, figure=fig)

		axa = fig.add_subplot(gs[0,0])
		axa.plot(lambda_s*1e6, signal_angle_deg, color='C0', label='Signal')
		axa.set_xlabel('Signal wavelength [$\\mu$m]')
		axa.set_ylabel('Emission angle, $\\alpha, \\beta$ [$\\degree$]')
		axa.set_xlim(lambda_s.min()*1e6, lambda_s.max()*1e6)

		axa2 = axa.twiny()
		axa2.plot(lambda_i*1e6, idler_angle_deg, color='C1', label='Idler')
		axa2.set_xlabel('Idler wavelength [$\\mu$m]')
		axa2.set_ylabel('Emission angle, $\\alpha, \\beta$ [$\\degree$]')
		axa2.invert_xaxis()
		axa2.set_xlim(lambda_i.max()*1e6, lambda_i.min()*1e6)

		handles, labels = axa.get_legend_handles_labels()
		handles2, labels2 = axa2.get_legend_handles_labels()
		handles.extend(handles2)
		labels.extend(labels2)

		axa.legend(handles, labels, loc='upper left', title=material)

		textstr = f'$\\Lambda$ = {Lambda*1e6} $\\mu$m'
		props = dict(boxstyle='round', facecolor='none', alpha=0.5)

		axa.text(0.95, 0.95, textstr, transform=axa.transAxes, fontsize=10,
    		verticalalignment='top', horizontalalignment='right', bbox=props)

		axa.set_title(f'Emission angles of \'{interaction_type}\'-type SPDC')

		# Check to find degenerate angle
		idx = None
		for i in range(len(idler_angle)):
			if idler_angle[i] == signal_angle[i]:
				idx = i

		if show_degenerate_wavelength:
			if idx is not None:
				signal_degenerate = lambda_s[idx]
				idler_degenerate = lambda_i[idx]

				axa.axhline(y=idler_angle_deg[idx], color='grey', linestyle='--')
			
			if lambda_s.max() == lambda_i.min():
				axa.axvline(x=lambda_p*2*1e6, color='grey', linestyle='--')

		fig.set_tight_layout(True)
		fig.show()

		self.axes['plot_non_collinear_emission_angle'] = (fig, axa, axa2)

		return fig, (axa, axa2)

	def plot_non_collinear_emission_angle_min_max_delta_k(self, temperature=25, material='MgO:PPLN', 
									Lambda=18e-6, figsize=(9,6),
									lambda_s_min=900e-9, lambda_s_max=None, lambda_p=800e-9,
									interaction_type='eee',
									show_degenerate_wavelength=True,
									num_points=2**18, L=11e-3):

		if not Lambda:
			Lambda = self.crystal_period

		if not lambda_s_max:
			lambda_s_max = lambda_p*2
		lambda_s, lambda_i = self.generate_signal_idler_wavelengths(lambda_p=lambda_p, 
			lambda_s_min=lambda_s_min, lambda_s_max=lambda_s_max, num_points=num_points)

		ne_p, no_p = self.sellmeier_equations(wavelength=lambda_p, material=material,
			temperature=temperature)
		ne_s, no_s = self.sellmeier_equations(wavelength=lambda_s, material=material,
			temperature=temperature)
		ne_i, no_i = self.sellmeier_equations(wavelength=lambda_i, material=material,
			temperature=temperature)

		if interaction_type in ('ooo'): # Type 0
			n_p, n_s, n_i = no_p, no_s, no_i 
		elif interaction_type in ('eee'): # Type 0
			n_p, n_s, n_i = ne_p, ne_s, ne_i
		elif interaction_type in ('ooe'): # Type I
			n_p, n_s, n_i = ne_p, no_s, no_i
		elif interaction_type in ('eeo'): # Type I
			n_p, n_s, n_i = no_p, ne_s, ne_i
		elif interaction_type in ('eoe'): # Type II
			n_p, n_s, n_i = ne_p, ne_s, no_i
		elif interaction_type in ('oee'): # Type II
			n_p, n_s, n_i = ne_p, no_s, ne_i
		elif interaction_type in ('eoo'): # Type II
			n_p, n_s, n_i = no_p, ne_s, no_i
		elif interaction_type in ('oeo'): # Type II
			n_p, n_s, n_i = no_p, no_s, ne_i
		else:
			print('Please choose valid interaction type, e.g., \'eee\', \'ooo\', \'eoe\', etc.')

		idler_angle, signal_angle = self.non_collinear_angle_calculation_delta_k(n_p=n_p, n_i=n_i, n_s=n_s,
			lambda_p=lambda_p, lambda_s=lambda_s, lambda_i=lambda_i, Lambda=Lambda, dk=-2*np.pi/L)
		idler_angle2, signal_angle2 = self.non_collinear_angle_calculation_delta_k(n_p=n_p, n_i=n_i, n_s=n_s,
			lambda_p=lambda_p, lambda_s=lambda_s, lambda_i=lambda_i, Lambda=Lambda, dk=0)
		idler_angle3, signal_angle3 = self.non_collinear_angle_calculation_delta_k(n_p=n_p, n_i=n_i, n_s=n_s,
			lambda_p=lambda_p, lambda_s=lambda_s, lambda_i=lambda_i, Lambda=Lambda, dk=2*np.pi/L)
		

		idler_angle_deg, signal_angle_deg = np.rad2deg(idler_angle), np.rad2deg(signal_angle)
		idler_angle_deg2, signal_angle_deg2 = np.rad2deg(idler_angle2), np.rad2deg(signal_angle2)
		idler_angle_deg3, signal_angle_deg3 = np.rad2deg(idler_angle3), np.rad2deg(signal_angle3)

		fig = plt.figure(figsize=(figsize))
		gs = GridSpec(1,1, figure=fig)

		axa = fig.add_subplot(gs[0,0])
		# axa.plot(lambda_s*1e6, signal_angle_deg, color='C0', label='Signal', alpha=0.6)
		axa.plot(lambda_s*1e6, signal_angle_deg2, color='C0', label='Signal')
		# axa.plot(lambda_s*1e6, signal_angle_deg3, color='C0', label='Signal', alpha=0.6)
		axa.set_xlabel('Signal wavelength [$\\mu$m]')
		axa.set_ylabel('Emission angle, $\\alpha, \\beta$ [$\\degree$]')
		axa.set_xlim(lambda_s.min()*1e6, lambda_s.max()*1e6)

		axa2 = axa.twiny()
		# axa2.plot(lambda_i*1e6, idler_angle_deg, color='C1', label='Idler', alpha=0.6)
		axa2.plot(lambda_i*1e6, idler_angle_deg2, color='C1', label='Idler')
		# axa2.plot(lambda_i*1e6, idler_angle_deg3, color='C1', label='Idler', alpha=0.6)
		axa2.set_xlabel('Idler wavelength [$\\mu$m]')
		axa2.set_ylabel('Emission angle, $\\alpha, \\beta$ [$\\degree$]')
		axa2.invert_xaxis()
		axa2.set_xlim(lambda_i.max()*1e6, lambda_i.min()*1e6)

		handles, labels = axa.get_legend_handles_labels()
		handles2, labels2 = axa2.get_legend_handles_labels()
		handles.extend(handles2)
		labels.extend(labels2)

		axa.legend(handles, labels, loc='upper left', title=material)

		textstr = f'$\\Lambda$ = {Lambda*1e6} $\\mu$m'
		props = dict(boxstyle='round', facecolor='none', alpha=0.5)

		axa.text(0.95, 0.95, textstr, transform=axa.transAxes, fontsize=10,
    		verticalalignment='top', horizontalalignment='right', bbox=props)

		axa.set_title(f'Emission angles of \'{interaction_type}\'-type SPDC')

		axa.set_ylim(0, 10)

		axa.fill_between(lambda_s*1e6, np.nan_to_num(signal_angle_deg, nan=0), np.nan_to_num(signal_angle_deg3, nan=0),
			where=~np.isnan(signal_angle_deg3), color='C0', alpha=0.5)
		axa2.fill_between(lambda_i*1e6, np.nan_to_num(idler_angle_deg, nan=0), np.nan_to_num(idler_angle_deg3, nan=0),
			where=~np.isnan(signal_angle_deg3), color='C1', alpha=0.5)

		# Check to find degenerate angle
		idx = None
		for i in range(len(idler_angle)):
			if idler_angle[i] == signal_angle[i]:
				idx = i

		if show_degenerate_wavelength:
			if idx is not None:
				signal_degenerate = lambda_s[idx]
				idler_degenerate = lambda_i[idx]

				axa.axhline(y=idler_angle_deg[idx], color='grey', linestyle='--')
			
			if lambda_s.max() == lambda_i.min():
				axa.axvline(x=lambda_p*2*1e6, color='grey', linestyle='--')

		fig.set_tight_layout(True)
		fig.show()

		self.axes['plot_non_collinear_emission_angle'] = (fig, axa, axa2)

		return fig, (axa, axa2)

	def plot_find_crystal_period(self, lambda_p=750e-9, lambda_s_min=850e-9, lambda_s_max=1500e-9,
									material='MgO:PPSLT', temperature=None, num_points=2**16,
									interaction_type='ooo', ylim=(19, 22)):

		if temperature is None:
			temperature = self.temperature

		# Generate reponse from PPSLT
		lambda_s, lambda_i = self.generate_signal_idler_wavelengths(lambda_p=lambda_p, 
			lambda_s_min=lambda_s_min, lambda_s_max=lambda_s_max, num_points=num_points)

		ne_p, no_p = self.sellmeier_equations(wavelength=lambda_p, 
															material=material, temperature=temperature)
		ne_s,  no_s = self.sellmeier_equations(wavelength=lambda_s, 
															material=material, temperature=temperature)
		ne_i, no_i = self.sellmeier_equations(wavelength=lambda_i, 
															material=material, temperature=temperature)

		if interaction_type in ('ooo'): # Type 0
			n_p, n_s, n_i = no_p, no_s, no_i 
		elif interaction_type in ('eee'): # Type 0
			n_p, n_s, n_i = ne_p, ne_s, ne_i
		elif interaction_type in ('ooe'): # Type I
			n_p, n_s, n_i = ne_p, no_s, no_i
		elif interaction_type in ('eeo'): # Type I
			n_p, n_s, n_i = no_p, ne_s, ne_i
		elif interaction_type in ('eoe'): # Type II
			n_p, n_s, n_i = ne_p, ne_s, no_i
		elif interaction_type in ('oee'): # Type II
			n_p, n_s, n_i = ne_p, no_s, ne_i
		elif interaction_type in ('eoo'): # Type II
			n_p, n_s, n_i = no_p, ne_s, no_i
		elif interaction_type in ('oeo'): # Type II
			n_p, n_s, n_i = no_p, no_s, ne_i
		else:
			print('Please choose valid interaction type, e.g., \'eee\', \'ooo\', \'eoe\', etc.')
		
		periods = self.phase_matching_periods(n_p=n_p, n_s=n_s, n_i=n_i,
												lambda_p=lambda_p, lambda_s=lambda_s, lambda_i=lambda_i,
												temperature=temperature)

		fig = plt.figure(figsize=(9,6))
		gs = GridSpec(1,1,figure=fig)

		axa = fig.add_subplot(gs[0,0])
		axa.plot(lambda_s*1e9, periods*1e6, color='blue', label='Signal')
		axa.set_ylabel('Poling Period [$\\mu$m]')
		axa.set_xlabel('Signal wavelength [nm]')
		axa.set_ylim(ylim[0], ylim[1])
		axa.set_xlim((lambda_p+50e-9)*1e9, lambda_s_max*1e9)
		# axa.legend()

		axa2 = axa.twiny()
		axa2.plot(lambda_i*1e6, periods*1e6, color='green', label='Idler')
		axa2.set_xlabel('Idler wavelength [$\\mu$m]')
		axa2.invert_xaxis()
		axa2.set_xlim(lambda_i.max()*1e6, lambda_i.min()*1e6)
		# axa2.legend()

		handlesa, labelsa = axa.get_legend_handles_labels()
		handlesa2, labelsa2 = axa2.get_legend_handles_labels()
		handlesa.extend(handlesa2)
		labelsa.extend(labelsa2)

		# Create a single legend
		axa.legend(handlesa, labelsa, loc='upper right', title=material)

		textstra = f'$\\lambda_p$ = {lambda_p*1e9} nm'
		propsa = dict(boxstyle='round', facecolor='none', alpha=0.5)

		axa.text(0.23, 0.95, textstra, transform=axa.transAxes, fontsize=10,
    		verticalalignment='top', horizontalalignment='right', bbox=propsa)

		# axa.axhline(y=20, color='grey', linestyle='--')

		fig.set_tight_layout(True)
		fig.show()

		self.axes['plot_find_crystal_period'] = (fig, axa, axa2)

		return fig, (axa, axa2)

	def plot_Hojo2021_fig5(self, figsize=(12,6), angle_max=1.2, angle_interval=0.2, Lambda_min=16.75e-6,
							Lambda_max=17.26e-6, Lambda_n=15, num_points_angle=2**6, material='MgO:PPSLT', show_additional_plots=True):

		lambda_p = 638e-9
		lambda_s, lambda_i = self.generate_signal_idler_wavelengths(lambda_p=lambda_p, lambda_s_min=700e-9, 
			lambda_s_max=1200e-9, num_points=2**16)

		ne_p, no_p = self.sellmeier_equations(wavelength=lambda_p, material=material)
		ne_s, no_s = self.sellmeier_equations(wavelength=lambda_s, material=material)
		ne_i, no_i = self.sellmeier_equations(wavelength=lambda_i, material=material)

		if show_additional_plots:

			temp, temp2 = self.plot_non_collinear_emission_angle(temperature=25, material='MgO:PPSLT', Lambda=16.88e-6, figsize=(9,6),
									lambda_s_min=700e-9, lambda_s_max=1200e-9, lambda_p=lambda_p,
									interaction_type='eee',
									show_degenerate_wavelength=True,
									num_points=2**18)

			temp5, temp6 = self.plot_non_collinear_emission_angle_min_max_delta_k(temperature=25, material='MgO:PPSLT', Lambda=16.88e-6, figsize=(9,6),
									lambda_s_min=700e-9, lambda_s_max=1200e-9, lambda_p=lambda_p,
									interaction_type='eee',
									show_degenerate_wavelength=True,
									num_points=2**18)

			temp3, temp4 = self.plot_find_crystal_period(lambda_p=lambda_p, lambda_s_min=700e-9, lambda_s_max=1200e-9,
									material='MgO:PPSLT', temperature=None, num_points=2**18,
									interaction_type='eee', ylim=(14, 18))

		omega_s = 2*np.pi*data.c/lambda_s
		omega_i = 2*np.pi*data.c/lambda_i
		theta = np.linspace(0,0.3,num_points_angle)

		L = np.linspace(Lambda_min, Lambda_max, Lambda_n)
		a = []

		for i in range(len(L)):
			temp = self.signal_power_output(omega_s=omega_s, omega_i=omega_i, n_p=ne_p, n_s=ne_s, n_i=ne_i,
				L=11e-3, Lambda=L[i], P_p=50e-3, theta_s=theta)

			a.append(temp)

		fig = plt.figure(figsize=figsize)
		gs = GridSpec(1,2,figure=fig)

		axa = fig.add_subplot(gs[0,0])
		for i in range(len(L)):
			axa.plot(lambda_s*1e9, (a[i]/a[i].max())+(i*1.2))
			axa.annotate(f'{L[i]*1e6:.2f} $\\mu$m', xy=(705, (i+0.2)*1.2), color=f'C{i}')
		axa.set_xlim(700, 800)
		axa.set_xlabel('Signal wavelength [nm]')
		axa.set_ylabel('Intensity [a.u.]')
		axa.set_title('Collinear emission spectra ($\\alpha$=0-0.3$\\degree$)')

		textstr = f'$\\lambda_p$ = 638 nm'
		props = dict(boxstyle='round', facecolor='none', alpha=0.5)

		axa.text(0.95, 0.97, textstr, transform=axa.transAxes, fontsize=10,
			verticalalignment='top', horizontalalignment='right', bbox=props)

		spacing = np.linspace(0, angle_max, int((angle_max/angle_interval)))

		theta_2 = [np.linspace(0, angle_max, num_points_angle)]

		for i in range(int(len(spacing))-1):
			theta_2.append(np.linspace(spacing[i], spacing[i+1], num_points_angle))

		b = []

		for i in range(len(theta_2)):
			temp = self.signal_power_output(omega_s=omega_s, omega_i=omega_i, n_p=no_p, n_s=no_s, n_i=no_i,
				L=11e-3, Lambda=16.88e-6, P_p=50e-3, theta_s=theta_2[i])

			b.append(temp)

		axb = fig.add_subplot(gs[0,1])
		for i in range(len(theta_2)):
			axb.plot(lambda_s*1e9, b[i]/b[i].max()+i*1.2)
			axb.annotate(f'{theta_2[i].min():.1f}-{theta_2[i].max():.1f}$\\degree$', xy=(705, (i+0.05)*1.22), color=f'C{i}')
		axb.set_xlim(700,800)
		axb.set_ylabel('Intensity [a.u.]')
		axb.set_xlabel('Signal wavelength [nm]')
		axb.set_title('Noncollinear emission spectra')

		textstrb = f'$\\Lambda$ = 16.88 $\\mu$m\n$\\lambda_p$ = 638 nm'
		propsb = dict(boxstyle='round', facecolor='none', alpha=0.5)

		axb.text(0.95, 0.95, textstrb, transform=axb.transAxes, fontsize=10,
			verticalalignment='top', horizontalalignment='right', bbox=propsb)

		fig.set_tight_layout(True)
		fig.show()
		# plt.close(fig2)


if __name__ == '__main__':
	data = QuasiPhaseMatching(lambda_p=800e-9, temperature=25, crystal_period=22e-6, material='MgO:PPSLT')
	# data2 = QuasiPhaseMatching(lambda_p=750e-9, temperature=24.5, crystal_period=22e-6, material='MgO:PPSLT_Dolev')

	lambda_p = 380e-9
	lambda_s, lambda_i = data.generate_signal_idler_wavelengths(lambda_p=lambda_p, lambda_s_min=500e-9, 
		lambda_s_max=760e-9, num_points=2**16)

	ne_p, no_p = data.sellmeier_equations(wavelength=lambda_p)
	ne_s, no_s = data.sellmeier_equations(wavelength=760e-9)
	ne_i, no_i = data.sellmeier_equations(wavelength=760e-9)


