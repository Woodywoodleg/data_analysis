import numpy as np
import sys
import os 
import pandas as pd
import matplotlib.pyplot as plt
import httpimport
with httpimport.github_repo('woodywoodleg', 'thzsnom', ref='master'):
	from snom_microscope_reader import ImageLoader
from matplotlib.gridspec import GridSpec
import glob
import re
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from lmfit.models import GaussianModel, LorentzianModel
from lmfit import Model
from lmfit import Parameters
import lmfit

class Raman_spectrum():
	def __init__(self, path_to_data='./'):
		self.path_to_data = path_to_data
		self.cd_script = os.getcwd() # Get directory containing script
		self.load_data()
		self.change_cd_back()

	def change_cd_back(self):
		os.chdir(self.cd_script) # Change directory back to where the script is located

	def load_data(self):
		self.data = pd.read_csv(self.path_to_data, sep='\t', header=None, names=['Wavenumber', 'Counts'], skiprows=3)

		return self.data

	def plot_raman_spectrum(self):

		fig = plt.figure()
		gs = GridSpec(1,1,figure=fig)

		ax = fig.add_subplot(gs[0,0])
		ax.plot(self.data['Wavenumber'], self.data['Counts'])
		ax.set_xlabel('Wavenumber [$\\mathrm{cm}^{-1}$]')
		ax.set_ylabel('Intensity [a.u.]')
		ax.set_title('Raman spectrum @ 532 nm')

		fig.set_tight_layout(True)
		fig.show()

class Photoluminescence_spectrum():
	def __init__(self, path_to_data='./'):
		self.path_to_data = path_to_data
		self.cd_script = os.getcwd() # Get directory containing script
		self.load_data()
		self.change_cd_back()

	def change_cd_back(self):
		os.chdir(self.cd_script) # Change directory back to where the script is located

	def load_data(self):
		self.data = pd.read_csv(self.path_to_data, sep='\t', header=None, names=['Wavelength', 'Counts'], skiprows=3)

		return self.data

	def plot_PL_spectrum(self):

		fig = plt.figure()
		gs = GridSpec(1,1,figure=fig)

		ax = fig.add_subplot(gs[0,0])
		ax.plot(self.data['Wavelength'], self.data['Counts'])
		ax.set_xlabel('Wavelength [$\\mathrm{nm}$]')
		ax.set_ylabel('Intensity [a.u.]')
		ax.set_title('PL spectrum @ 532 nm')

		fig.set_tight_layout(True)
		fig.show()

class SFG_power_dependence():
	def __init__(self, path_to_data, path_to_data_wavelength, scan_type='IR', init_extra=True):
		self.path_to_data = path_to_data
		self.path_to_data_wavelength = path_to_data_wavelength
		self.scan_type = scan_type
		self.cd_script = os.getcwd() # Get directory containing script
		if init_extra:
			self.load_data()
			self.convert_column_to_watts()
			self.load_data_wavelength_axis()
			self.change_cd_back()

	def change_cd_back(self):
		os.chdir(self.cd_script) # Change directory back to where the script is located

	def load_data(self):
		os.chdir(self.path_to_data) # Set current directory to the folder containing the files of interest

		self.all_files = [] # Create empty array to contain all txt files in the directory
		for file in glob.glob("*.dat"): # Searches the current directory for all txt files
			self.all_files.append(file) # Appends files found

		try:
			if self.scan_type == 'IR':
				SFG_files = [s for s in self.all_files if re.search('.?SFG.+', s)]
			elif self.scan_type == 'Visible' or 'PL':
				SFG_files = [s for s in self.all_files if re.search('.?PL.+', s)]

		except IndexError:
			print('Error: File not found!')
			sys.exit()

		self.signal, self.backround, self.signal_raw = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

		if len(SFG_files) >= 2:
			SFG_files.sort()
			SFG_files_grouped = [SFG_files[i:i+2] for i in range(0, len(SFG_files), 2)]

			signal_raw, background_raw = pd.DataFrame(), pd.DataFrame()

			for i in range(len(SFG_files_grouped)):
				signal_file = [s for s in SFG_files_grouped[i] if re.search('.?BinArea2of2.+', s)]
				background_file = [s for s in SFG_files_grouped[i] if re.search('.?BinArea1of2.+', s)]

				power_match = re.search(r'(\d+)(nW|uW|mW|W)', signal_file[0])

				if power_match:
					power = power_match.group(0)
				else:
					continue

				signal = pd.read_csv(signal_file[0], sep='\t', header=None)
				background = pd.read_csv(background_file[0], sep='\t', header=None)
				self._averages = signal.shape[1]

				names = []
				for j in range(self._averages):
					names.append('Trace '+str(j+1))

				signal_raw, background_raw = signal.set_axis(names, axis=1), background.set_axis(names, axis=1)

				signal_avg, background_avg = signal_raw.mean(axis=1), background_raw.mean(axis=1)

				self.signal[power], self.backround[power] = signal_avg - background_avg, background_avg

				self.signal_raw[power] = self.signal[power]

			self.signal_raw[self.signal_raw < - 5] = 0
			self.signal[self.signal < -5] = 0

			self.signal_normalised = self.signal / self.signal.max()

		elif len(SFG_files) < 2:
			print('Error: Appears to be missing either background or signal file.')
			sys.exit()
		else:
			print('Please check input - file not found!')
			sys.exit()

		return self.signal

	def convert_column_to_watts(self):
		def convert_to_watts(col_name):
			# Use regular expressions to separate the numeric part and the unit
			match = re.match(r'(\d+\.?\d*)([num]?W)', col_name)
			if match:
				value = float(match.group(1))
				unit = match.group(2)

				# Convert to watts
				if unit == 'nW':
				    return value * 1e-6
				elif unit == 'uW':
				    return value * 1e-3
				elif unit == 'mW':
				    return value * 1
				elif unit == 'W':
				    return value * 1e3
			return float('inf')  # Fallback in case the format doesn't match

		# self.signal.columns = [convert_to_watts(col) for col in self.signal.columns]
		# self.signal_raw.columns = [convert_to_watts(col) for col in self.signal_raw.columns]

		sorted_columns = sorted(self.signal.columns, key=convert_to_watts)
		sorted_columns_sci = {col: f'{convert_to_watts(col):.3e} mW' for col in sorted_columns}
		self.signal = self.signal[sorted_columns]
		self.signal.rename(columns=sorted_columns_sci, inplace=True)


	def correct_error_peaks(self, height=35, prominence=30, threshold=30, neighbours=5):
		# Code for correcting error peaks
		for i in range(len(self.signal.shape[1])):
			peaks, properties = find_peaks(abs(self.signal.iloc[:,i]), height=height, prominence=prominence)

			for peak in peaks:                                                                                                                                                                                                       
				left_value = self.signal.iloc[:,i].iloc[peak - 2]                                                                                                                                                                 
				right_value = self.signal.iloc[:,i].iloc[peak + 2]                                                                                                                                                                
				if abs(self.signal.iloc[:,i].iloc[peak]) > abs(left_value) + threshold and abs(self.signal.iloc[:,i].iloc[peak]) > abs(right_value) + threshold:                                                               
					for i in range(neighbours):                                                                                                                                                                                               
						self.signal.iloc[:,i].iloc[peak-i] = left_value / 2                                                                                                                                                       
						self.signal.iloc[:,i].iloc[peak+i] = left_value / 2   


	def load_data_wavelength_axis(self):
		os.chdir(self.path_to_data_wavelength) # Set current directory to the folder containing the files of interest

		self.all_files_axis = [] # Create empty array to contain all txt files in the directory
		for file in glob.glob("*.dat"): # Searches the current directory for all txt files
			self.all_files_axis.append(file) # Appends files found

		try:
			Ne_files = [s for s in self.all_files_axis if re.search('.?Ne.+', s)]

		except IndexError:
			print('Error: File not found!')
			sys.exit()

		self.Ne_100um = pd.read_csv([s for s in Ne_files if re.search('.?100um.+', s)][0], sep='\t', header=None, names=['Wavelength', 'Ne'])
		if len(Ne_files) > 1:
			self.Ne_200um = pd.read_csv([s for s in Ne_files if re.search('.?200um.+', s)][0], sep='\t', header=None, names=['Wavelength', 'Ne'])
		else:
			self.Ne_200um = pd.DataFrame({'Wavelength': [1]})

		self.wavelength_100um = self.Ne_100um['Wavelength'].to_numpy()
		self.wavelength_200um = self.Ne_200um['Wavelength'].to_numpy()
		self.energy_100um = 1238.9/self.wavelength_100um
		self.energy_200um = 1238.9/self.wavelength_200um

		return self.wavelength_100um, self.wavelength_200um, self.energy_100um, self.energy_200um

	def fit_to_peak(self, spectrum, xaxis, fit_type='Gaussian', peaks=2, A=None, x_0=None, sigma=None, gamma=None, eV_range=None, nm_range=None):

		# Slice the spectrum if boundaries are given either in eV or nm
		if eV_range:
			filtered_indices = [i for i, x in enumerate(xaxis) if eV_range[0] <= x <= eV_range[1]]
			spectrum = spectrum.iloc[filtered_indices]
			xaxis = xaxis.iloc[filtered_indices]
		if nm_range:
			filtered_indices = [i for i, x in enumerate(xaxis) if nm_range[0] <= x <= nm_range[1]]
			spectrum = spectrum.iloc[filtered_indices]
			xaxis = xaxis.iloc[filtered_indices]

		# Define parameters for the fit
		A, x_0, sigma = [s if isinstance(s, list) else [s] for s in [A, x_0, sigma]] # Ensure that the input is a list
		Model = [] # Empty list to contain the model
		params = Parameters() # Create a parameters parameter
		for i in range(peaks): # Define type for each peak
			if fit_type == 'Gaussian':
				temp = GaussianModel(prefix=f'g{i+1}_')
			elif fit_type == 'Lorentzian':
				temp = LorentzianModel(prefix=f'g{i+1}_')
			Model.append(temp)

			# Add the defining parameters for the model
			params.add(f'g{i+1}_amplitude', value=A[i], min=0)
			params.add(f'g{i+1}_center', value=x_0[i])
			params.add(f'g{i+1}_sigma', value=sigma[i])

		# Combining the models and parameters for each peak into a composite model
		model = Model[0] # Initialize the combined model
		for s in Model[1:]: # For loop for summing the models
			model += s # Add subsequent models

		result = model.fit(spectrum, params, x=xaxis)
		# print(result.fit_report())

		self.signal_fit_best_fit = result.best_fit # Best combined fit to the whole spectrum
		self.signal_fit_peaks = pd.DataFrame(result.eval_components()) # Individual fit to each of the peaks

		return result

	def power_dependence(self, method='max', eV_range=None):
		# Create list contain all powers
		self.signal_powers = self.signal.columns.tolist()
		# Convert the list of strings into a list of numbers
		self.signal_powers = [float(power.replace(' mW', '')) for power in self.signal_powers]

		if eV_range:
			filtered_indices = [i for i, x in enumerate(self.energy_100um) if eV_range[0] <= x <= eV_range[1]]
			spectrum = self.signal.iloc[filtered_indices]
			# xaxis = xaxis.iloc[filtered_indices]
		else:
			spectrum = self.signal

		if method == 'max' or 'Max':
			temp_intensity = [] # Temporary list for containing the maximum values
			temp_energy = [] # Temp. list for containing the energy location of the maximum
			for i in range(len(self.signal.columns)):
				temp_intensity.append(spectrum.iloc[:,i].max())
				temp_energy.append(self.energy_100um[spectrum.iloc[:,i].idxmax()])

			# Create a dataframe for containing the results
			self.signal_power_dependence = pd.DataFrame({'Power [mW]': self.signal_powers, 'Signal [a.u]': temp_intensity, 'Energy_loc [eV]': temp_energy}) 

		return self.signal_power_dependence

	def create_header(self):
		pass

	def plot_spectra(self):
		fig = plt.figure()
		gs = GridSpec(1,1, figure=fig)

		ax = fig.add_subplot(gs[0,0])
		for i in range(self.signal.shape[1]):
			plt.plot(self.signal.iloc[:, i], label=self.signal.columns[i])
		ax.set_xlabel('Pixel')
		ax.set_ylabel('Intensity [a.u.]')
		ax.legend()

		fig.set_tight_layout(True)
		fig.show()
		return fig

	def plot_spectra_wavelength(self):
		fig = plt.figure()
		gs = GridSpec(1,1, figure=fig)

		ax = fig.add_subplot(gs[0,0])
		for i in range(self.signal.shape[1]):
			plt.plot(self.wavelength_100um, self.signal.iloc[:, i], label=self.signal.columns[i])
		ax.set_xlabel('Wavelength [nm]')
		ax.set_ylabel('Intensity [a.u.]')
		ax.legend()

		fig.set_tight_layout(True)
		fig.show()
		return fig

	def plot_spectra_eV(self):
		fig = plt.figure()
		gs = GridSpec(1,1, figure=fig)

		ax = fig.add_subplot(gs[0,0])
		for i in range(self.signal.shape[1]):
			plt.plot(self.energy_100um, self.signal.iloc[:, i], label=self.signal.columns[i])
		ax.set_xlabel('Photon energy [eV]')
		ax.set_ylabel('Intensity [a.u.]')
		ax.legend()

		fig.set_tight_layout(True)
		fig.show()
		return fig

	def plot_power_dependence(self, method='max', eV_range=None):
		# Linear fit functions
		def fit_linear(x, a, b):
			return a*x + b

		def fit_exponential(x, a, k):
			return a*x**k

		# Isolate the peak of interest
		power = self.power_dependence(method=method, eV_range=eV_range)

		# Create log axes
		power_logx = np.log(power['Power [mW]'].to_numpy())
		power_logy = np.log(power['Signal [a.u]'].to_numpy())

		# Initial parameters for the fit
		initial_params = {
			'a': 3,
			'b': 0
		}

		# Create a new x-axis for plotting later on
		x = np.linspace(power['Power [mW]'].min()*0.8, power['Power [mW]'].max()*1.3, 2**10)

		# Perform the fitting routine
		model = lmfit.Model(fit_linear) # Create the model
		params = model.make_params(**initial_params) # Insert the fitting parameters
		result = model.fit(power_logy, params, x=power_logx) # Perform the fitting 

		# Plot the resulting data
		fig = plt.figure()
		gs = GridSpec(1,1, figure=fig)

		ax = fig.add_subplot(gs[0, 0])
		ax.scatter(power['Power [mW]'], power['Signal [a.u]'], label='Exp.')
		plt.plot(x, fit_exponential(x=x, a=np.exp(result.params['b'].value), k=result.params['a'].value),
		    label=f'$I^{{{result.params["a"].value:.2f}\\pm {result.params["a"].stderr:.2f}}}$', color='r', linestyle='dashed')
		ax.legend(fontsize=9, loc=4)
		ax.set_xlabel('Power [mW]')
		ax.set_ylabel('Intensity [a.u.]')
		ax.set_yscale('log')
		ax.set_xscale('log')

		fig.set_tight_layout(True)
		fig.show()

		return fig

class SFG_load_spectrum_single():
	def __init__(self, path_to_data_Bin1, path_to_data_Bin2):
		self.path_to_data_Bin1 = path_to_data_Bin1
		self.path_to_data_Bin2 = path_to_data_Bin2
		self.cd_script = os.getcwd() # Get directory containing script
		self.load_data()
		self.change_cd_back()

	def change_cd_back(self):
		os.chdir(self.cd_script) # Change directory back to where the script is located

	def load_data(self):
		self.signal, self.backround = pd.DataFrame(), pd.DataFrame()

		signal_raw = pd.read_csv(self.path_to_data_Bin2, sep='\t', header=None)
		background_raw = pd.read_csv(self.path_to_data_Bin1, sep='\t', header=None)

		signal_avg = signal_raw.mean(axis=1)
		background_avg = background_raw.mean(axis=1)

		self.signal, self.backround = signal_avg - background_avg, background_avg

		self.signal_raw = self.signal

		# Code for correcting error peaks
		peaks, properties = find_peaks(abs(self.signal[power]), height=height, prominence=prominence)

		for peak in peaks:                                                                                                                                                                                                       
			left_value = self.signal[power].iloc[peak - 2]                                                                                                                                                                 
			right_value = self.signal[power].iloc[peak + 2]                                                                                                                                                                
			if abs(self.signal[power].iloc[peak]) > abs(left_value) + threshold and abs(self.signal[power].iloc[peak]) > abs(right_value) + threshold:                                                               
				for i in range(neighbours):                                                                                                                                                                                               
					self.signal[power].iloc[peak-i] = left_value / 2                                                                                                                                                       
					self.signal[power].iloc[peak+i] = left_value / 2  

	def load_data_wavelength_axis(self):
		os.chdir(self.path_to_data) # Set current directory to the folder containing the files of interest

		self.all_files = [] # Create empty array to contain all txt files in the directory
		for file in glob.glob("*.dat"): # Searches the current directory for all txt files
			self.all_files.append(file) # Appends files found

		try:
			Ne_files = [s for s in self.all_files if re.search('.?Ne.+', s)]

		except IndexError:
			print('Error: File not found!')
			sys.exit()

		self.wavelength_100um = pd.read_csv([s for s in Ne_files if re.search('.?100um.+', s)][0], sep='\t', header=None, names=['Wavelength', 'Ne'])
		self.wavelength_200um = pd.read_csv([s for s in Ne_files if re.search('.?200um.+', s)][0], sep='\t', header=None, names=['Wavelength', 'Ne'])

		return self.wavelength_100um, self.wavelength_200um

class PL_wavelength_sweep(SFG_power_dependence):
	def __init__(self, path_to_data, path_to_data_wavelength, scan_type='IR'):
		self.path_to_data = path_to_data
		self.path_to_data_wavelength = path_to_data_wavelength
		self.cd_script = os.getcwd() # Get directory containing script
		super().__init__(path_to_data, path_to_data_wavelength, scan_type=scan_type, init_extra=False)
		self.load_data_PL()
		self.load_data_wavelength_axis()
		self.convert_column_to_nm()
		self.change_cd_back()

	def load_data_PL(self):
		os.chdir(self.path_to_data) # Set current directory to the folder containing the files of interest

		self.all_files = [] # Create empty array to contain all txt files in the directory
		for file in glob.glob("*.dat"): # Searches the current directory for all txt files
			self.all_files.append(file) # Appends files found

		try:
			SFG_files = [s for s in self.all_files if re.search('.?PL.+', s)]

		except IndexError:
			print('Error: File not found!')
			sys.exit()

		self.signal, self.backround, self.signal_raw = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

		if len(SFG_files) >= 2:
			SFG_files.sort()
			SFG_files_grouped = [SFG_files[i:i+2] for i in range(0, len(SFG_files), 2)]

			signal_raw, background_raw = pd.DataFrame(), pd.DataFrame()

			for i in range(len(SFG_files_grouped)):
				signal_file = [s for s in SFG_files_grouped[i] if re.search('.?BinArea2of2.+', s)]
				background_file = [s for s in SFG_files_grouped[i] if re.search('.?BinArea1of2.+', s)]

				wavelength_match = re.search(r'(\d+)(A)', signal_file[0])

				if wavelength_match:
					wavelength = wavelength_match.group(0)
				else:
					continue

				signal = pd.read_csv(signal_file[0], sep='\t', header=None)
				background = pd.read_csv(background_file[0], sep='\t', header=None)
				self._averages = signal.shape[1]

				names = []
				for j in range(self._averages):
					names.append('Trace '+str(j+1))

				signal_raw, background_raw = signal.set_axis(names, axis=1), background.set_axis(names, axis=1)
				signal_avg, background_avg = signal_raw.mean(axis=1), background_raw.mean(axis=1)

				self.signal[wavelength], self.backround[wavelength] = signal_avg - background_avg, background_avg
				self.signal_raw[wavelength] = self.signal[wavelength]

			self.signal_raw[self.signal_raw < - 5] = 1e-8
			self.signal[self.signal < -5] = 1e-8

		elif len(PL_files) < 2:
			print('Error: Appears to be missing either background or signal file.')
			sys.exit()
		else:
			print('Please check input - file not found!')
			sys.exit()

		return self.signal

	def convert_column_to_nm(self):
		def convert_to_nm(col_name):
			# Use regular expressions to separate the numeric part and the unit
			match = re.match(r'(\d+)(A)', col_name)
			if match:
				value = float(match.group(1))
				unit = match.group(2)

				# # Convert to nm
				# value * 1e-2
				return value * 1e-1

		# self.signal.columns = [convert_to_watts(col) for col in self.signal.columns]
		# self.signal_raw.columns = [convert_to_watts(col) for col in self.signal_raw.columns]

		sorted_columns = sorted(self.signal.columns, key=convert_to_nm)
		sorted_columns_sci = {col: f'{convert_to_nm(col):.1f} nm' for col in sorted_columns}
		self.signal = self.signal[sorted_columns]
		self.signal.rename(columns=sorted_columns_sci, inplace=True)

	def plot_lambda_spectra(self, normalisation=True):
		fig = plt.figure()
		gs = GridSpec(1,1, figure=fig)

		ax = fig.add_subplot(gs[0,0])
		for i in range(self.signal.shape[1]):
			if normalisation:
				ax.plot(self.energy_100um, self.signal.iloc[:,i]/self.signal.iloc[:,i].max(), label=f'$\\lambda_{{exc.}} = {self.signal.columns[i]}$')
			else:
				ax.plot(self.energy_100um, self.signal.iloc[:,i], label=f'$\\lambda_{{exc.}} = {self.signal.columns[i]}$')
		ax.set_xlabel('Photon Energy [eV]')
		ax.set_ylabel('Intensity [a.u.]')
		ax.set_title('PL at different excitation energies')
		ax.legend()

		fig.set_tight_layout(True)
		fig.show()

class SFG_polarisation_dependence(SFG_power_dependence):
	def __init__(self, path_to_data, path_to_data_wavelength, scan_type='IR', WG1_zero_point=9150, WG2_zero_point=14730):
		self.path_to_data = path_to_data
		self.path_to_data_wavelength = path_to_data_wavelength
		self._WG1_zero_point = WG1_zero_point # zero point of the wiregrid 1 in question, i.e., 0 degrees
		self._WG2_zero_point = WG2_zero_point # zero point of the wiregrid 2 in question, i.e., 0 degrees
		self._pulses_to_degrees = 400 # 400 pulses corresponds to a 1 degree change
		self.cd_script = os.getcwd() # Get directory containing script
		super().__init__(path_to_data, path_to_data_wavelength, scan_type=scan_type, init_extra=False)
		self.load_data_polarisation()
		self.load_data_wavelength_axis()
		# self.convert_column_to_nm()
		self.change_cd_back()

	def load_data_polarisation(self):
		os.chdir(self.path_to_data) # Set current directory to the folder containing the files of interest

		self.all_files = [] # Create empty array to contain all txt files in the directory
		for file in glob.glob("*.dat"): # Searches the current directory for all txt files
			self.all_files.append(file) # Appends files found

		try:
			if self.scan_type == 'IR':
				SFG_files = [s for s in self.all_files if re.search('.?SFG.+', s)]
			elif self.scan_type == 'Visible':
				SFG_files = [s for s in self.all_files if re.search('.?PL.+', s)]

		except IndexError:
			print('Error: File not found!')
			sys.exit()

		self.signal, self.backround, self.signal_raw = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

		if len(SFG_files) >= 2:
			SFG_files.sort()
			SFG_files_grouped = [SFG_files[i:i+2] for i in range(0, len(SFG_files), 2)]

			signal_raw, background_raw = pd.DataFrame(), pd.DataFrame()

			for i in range(len(SFG_files_grouped)):
				signal_file = [s for s in SFG_files_grouped[i] if re.search('.?BinArea2of2.+', s)]
				background_file = [s for s in SFG_files_grouped[i] if re.search('.?BinArea1of2.+', s)]

				sigma_match = re.search(r'(Sigma)(\d+)', signal_file[0])

				if sigma_match:
					sigma = sigma_match.group(0)
					degree = (float(re.findall(r'\d+', sigma)[0]) - self._WG2_zero_point) / self._pulses_to_degrees
				else:
					continue

				signal = pd.read_csv(signal_file[0], sep='\t', header=None)
				background = pd.read_csv(background_file[0], sep='\t', header=None)
				self._averages = signal.shape[1]

				names = []
				for j in range(self._averages):
					names.append('Trace '+str(j+1))

				signal_raw, background_raw = signal.set_axis(names, axis=1), background.set_axis(names, axis=1)

				signal_avg, background_avg = signal_raw.mean(axis=1), background_raw.mean(axis=1)

				self.signal[degree], self.backround[degree] = signal_avg - background_avg, background_avg

				self.signal_raw[degree] = self.signal[degree]

			self.signal_raw[self.signal_raw < - 5] = 1e-8
			self.signal[self.signal < -5] = 1e-8

			self.signal_normalised = self.signal / self.signal.max()

		elif len(SFG_files) < 2:
			print('Error: Appears to be missing either background or signal file.')
			sys.exit()
		else:
			print('Please check input - file not found!')
			sys.exit()

		return self.signal

	def polar_max(self, method='max', eV_range=None):
		# Create list contain all angles
		self.signal_angles = self.signal.columns.tolist()

		if eV_range:
			filtered_indices = [i for i, x in enumerate(self.energy_100um) if eV_range[0] <= x <= eV_range[1]]
			spectrum = self.signal.iloc[filtered_indices]
			# xaxis = xaxis.iloc[filtered_indices]
		else:
			spectrum = self.signal

		if method == 'max' or 'Max':
			temp_intensity = [] # Temporary list for containing the maximum values
			temp_energy = [] # Temp. list for containing the energy location of the maximum
			for i in range(len(self.signal.columns)):
				temp_intensity.append(spectrum.iloc[:,i].max())
				temp_energy.append(self.energy_100um[spectrum.iloc[:,i].idxmax()])

			# Create a dataframe for containing the results
			self.signal_polar_max = pd.DataFrame({'Angles [deg]': self.signal_angles, 'Signal [a.u]': temp_intensity, 'Energy_loc [eV]': temp_energy}) 

		return self.signal_polar_max

	def plot_polar(self, method='max', eV_range=None):
		# Create the list for the peak
		polar = self.polar_max(method=method, eV_range=eV_range)

		fig = plt.figure()
		gs = GridSpec(1,1, figure=fig)

		ax = fig.add_subplot(gs[0,0], polar=True)
		ax.scatter(polar['Angles [deg]']*np.pi/180, 
			polar['Signal [a.u]'], color='r', label=f'{polar["Energy_loc [eV]"][0].value:.3f} eV')
		ax.set_thetalim(0, np.pi)
		ax.set_yticklabels([])
		ax.legend(fontsize=7)

		fig.set_tight_layout(True)
		fig.show()

		return fig


class SFG_PLE(SFG_power_dependence):
	def __init__(self, path_to_data, path_to_data_wavelength):
		self.path_to_data = path_to_data
		self.path_to_data_wavelength = path_to_data_wavelength
		self.cd_script = os.getcwd() # Get directory containing script
		super().__init__(path_to_data, path_to_data_wavelength, scan_type='Visible', init_extra=False)
		self.load_data_PLE()
		self.load_data_wavelength_axis()
		self.create_PLE_axis()
		# self.convert_column_to_nm()
		self.change_cd_back()

	def load_data_PLE(self):
		os.chdir(self.path_to_data) # Set current directory to the folder containing the files of interest

		self.all_files = [] # Create empty array to contain all txt files in the directory
		for file in glob.glob("*.dat"): # Searches the current directory for all txt files
			self.all_files.append(file) # Appends files found

		try:
			SFG_files = [s for s in self.all_files if re.search('.?PL.+', s)]

		except IndexError:
			print('Error: File not found!')
			sys.exit()

		self.signal, self.backround, self.signal_raw = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

		if len(SFG_files) >= 2:
			SFG_files.sort()
			SFG_files_grouped = [SFG_files[i:i+2] for i in range(0, len(SFG_files), 2)]

			signal_raw, background_raw = pd.DataFrame(), pd.DataFrame()

			for i in range(len(SFG_files_grouped)):
				signal_file = [s for s in SFG_files_grouped[i] if re.search('.?BinArea2of2.+', s)]
				background_file = [s for s in SFG_files_grouped[i] if re.search('.?BinArea1of2.+', s)]

				WL_match = re.search(r'(SetWL)(\d+)', signal_file[0])

				if WL_match:
					WL = WL_match.group(0)
					wavelength = (float(re.findall(r'\d+', WL)[0])*1e-1)
				else:
					continue

				signal = pd.read_csv(signal_file[0], sep='\t', header=None)
				background = pd.read_csv(background_file[0], sep='\t', header=None)
				self._averages = signal.shape[1]

				names = []
				for j in range(self._averages):
					names.append('Trace '+str(j+1))

				signal_raw, background_raw = signal.set_axis(names, axis=1), background.set_axis(names, axis=1)

				signal_avg, background_avg = signal_raw.mean(axis=1), background_raw.mean(axis=1)

				self.signal[wavelength], self.backround[wavelength] = signal_avg - background_avg, background_avg

				self.signal_raw[wavelength] = self.signal[wavelength]

			self.signal_raw[self.signal_raw < - 5] = 1e-8
			self.signal[self.signal < -5] = 1e-8

			self.signal_normalised = self.signal / self.signal.max()

		elif len(SFG_files) < 2:
			print('Error: Appears to be missing either background or signal file.')
			sys.exit()
		else:
			print('Please check input - file not found!')
			sys.exit()

		return self.signal

	def create_PLE_axis(self):
		self.PLE_wavelength = self.signal.columns.to_numpy()
		self.PLE_energy = 1238.9/self.PLE_wavelength

class SFG_reflection(SFG_power_dependence):
	def __init__(self, path_to_data, path_to_reference, path_to_data_wavelength):
		self.path_to_data = path_to_data
		self.path_to_data_wavelength = path_to_data_wavelength
		self.path_to_reference = path_to_reference
		self.cd_script = os.getcwd() # Get directory containing script
		super().__init__(path_to_data, path_to_data_wavelength, scan_type='Visible', init_extra=False)
		# self.load_data_reflection(path=self.path_to_data)
		self.referenced()
		self.load_data_wavelength_axis()
		self.change_cd_back()

	def load_data_reflection(self, path):
		os.chdir(path) # Set current directory to the folder containing the files of interest

		self.all_files = [] # Create empty array to contain all txt files in the directory
		for file in glob.glob("*.dat"): # Searches the current directory for all txt files
			self.all_files.append(file) # Appends files found

		try:
			SFG_files = [s for s in self.all_files if re.search('.?reflection.+', s)]

		except IndexError:
			print('Error: File not found!')
			sys.exit()

		self.signal_temp, self.backround_temp, self.signal_raw_temp = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

		if len(SFG_files) >= 2:
			SFG_files.sort()
			SFG_files_grouped = [SFG_files[i:i+2] for i in range(0, len(SFG_files), 2)]

			signal_raw, background_raw = pd.DataFrame(), pd.DataFrame()

			for i in range(len(SFG_files_grouped)):
				signal_file = [s for s in SFG_files_grouped[i] if re.search('.?BinArea2of2.+', s)]
				background_file = [s for s in SFG_files_grouped[i] if re.search('.?BinArea1of2.+', s)]

				power_match = re.search(r'(\d+)(nW|uW|mW|W)', signal_file[0])

				if power_match:
					power = power_match.group(0)
				else:
					continue

				signal = pd.read_csv(signal_file[0], sep='\t', header=None)
				background = pd.read_csv(background_file[0], sep='\t', header=None)
				self._averages = signal.shape[1]

				names = []
				for j in range(self._averages):
					names.append('Trace '+str(j+1))

				signal_raw, background_raw = signal.set_axis(names, axis=1), background.set_axis(names, axis=1)

				signal_avg, background_avg = signal_raw.mean(axis=1), background_raw.mean(axis=1)

				self.signal_temp[power], self.backround_temp[power] = signal_avg - background_avg, background_avg

				self.signal_raw_temp[power] = self.signal_temp[power]

			self.signal_raw_temp[self.signal_raw_temp < - 5] = 1e-8
			self.signal_temp[self.signal_temp < -5] = 1e-8

			self.signal_normalised_temp = self.signal_temp / self.signal_temp.max()

		return self.signal_temp


	def referenced(self, reference_number=0, method='standard'):

		if method == 'advanced':
			self.signal_raw = self.load_data_reflection(path=self.path_to_data)
			self.reference_raw = self.load_data_reflection(path=self.path_to_reference)

			signal = self.signal_raw / self.signal_raw.iloc[0:100].max()
			reference = self.reference_raw / self.reference_raw.iloc[0:100].max()

			self.signal = signal.sub(reference.iloc[:,reference_number], axis=0)
			self.signal_normalised = self.signal / self.signal.max()

		elif method == 'standard':
			self.signal_raw = self.load_data_reflection(path=self.path_to_data)
			self.reference_raw = self.load_data_reflection(path=self.path_to_reference)

			self.signal = self.signal_raw.div(self.reference_raw.iloc[:,reference_number], axis=0)

		self.change_cd_back()






if __name__ == "__main__":

	data_path = r"C:\Users\h_las\OneDrive\Kyoto University\Post doc\Data\samples\CsPbBr3\bulk\20241007\20241007\\"
	raman = Raman_spectrum(
		path_to_data=r"C:\Users\h_las\OneDrive\Kyoto University\Post doc\Data\samples\CsPbBr3\bulk\20241007\20241007\No241007-001-CsPbBr3-Raman-1sx10-G2400DetWL541nm-Exc532nm-ND3-Obj100x-pinslit100um.txt")
	PL = Photoluminescence_spectrum(
		path_to_data=data_path+r"No241007-003-CsPbBr3-PL-100ms-G150DetWL700nm-Exc532nm-ND3-Obj100x-pinslit100um.txt")

	# microscope = ImageLoader(r'C:\Users\h_las\OneDrive\Kyoto University\Post doc\Data\samples\CsPbBr3\bulk\20241007')
	# microscope_SFG = ImageLoader(r'C:\Users\h_las\OneDrive\Kyoto University\Post doc\Data\samples\CsPbBr3\bulk\20241007')

	sfg = SFG_power_dependence(path_to_data=r'C:\Users\h_las\OneDrive\Kyoto University\Post doc\Data\samples\CsPbBr3\bulk\20241010-SFG\30K\SFG full IR power sweep',
		path_to_data_wavelength=r'C:\Users\h_las\OneDrive\Kyoto University\Post doc\Data\samples\CsPbBr3\bulk\20241010-SFG',
		scan_type='IR')

	sfg_10K_IR_BPF = SFG_power_dependence(path_to_data=r'C:\Users\h_las\OneDrive\Kyoto University\Post doc\Data\samples\CsPbBr3\bulk\20241010-SFG\10K\SFG BPF 1100 nm',
		path_to_data_wavelength=r'C:\Users\h_las\OneDrive\Kyoto University\Post doc\Data\samples\CsPbBr3\bulk\20241010-SFG',
		scan_type='IR')

	pl_lambda_sweep = PL_wavelength_sweep(path_to_data=r'C:\Users\h_las\OneDrive\Kyoto University\Post doc\Data\samples\CsPbBr3\bulk\20241010-SFG\10K\Visible wavelength sweep',
		path_to_data_wavelength=r'C:\Users\h_las\OneDrive\Kyoto University\Post doc\Data\samples\CsPbBr3\bulk\20241010-SFG')

	data2 = SFG_power_dependence(path_to_data=r'C:\Users\h_las\Documents\20241112-SFG\SFG 1300LPF ND100k x 100avg x 10set\set1',
		path_to_data_wavelength=r'C:\Users\h_las\Documents\20241112-SFG',
		scan_type='IR')

	data = SFG_PLE(path_to_data=r'C:\Users\h_las\Documents\20241115-SFG 100 avg\new bulk 100 avg\PLE 540-525nm 50 avg para-perp\set1',
		path_to_data_wavelength=r'C:\Users\h_las\Documents\20241115-SFG 100 avg')


	test = SFG_reflection(path_to_data=r'C:\Users\h_las\OneDrive\Kyoto University\Post doc\Data\samples\CsPbBr3\bulk\20241126-SFG\Reflection CsPbBr3',
		path_to_reference=r'C:\Users\h_las\OneDrive\Kyoto University\Post doc\Data\samples\CsPbBr3\bulk\20241126-SFG\Reflection SiO2-Si',
		path_to_data_wavelength=r'C:\Users\h_las\OneDrive\Kyoto University\Post doc\Data\samples\CsPbBr3\bulk\20241126-SFG')
