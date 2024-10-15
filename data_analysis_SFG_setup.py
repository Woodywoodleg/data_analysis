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

# Define Gaussian function
def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

# Define Lorentzian function
def lorentzian(x, a, x0, gamma):
    return a / (1 + ((x - x0) / gamma)**2)

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

class SFG_IR_SFG_power_dependence():
	def __init__(self, path_to_data, path_to_data_wavelength):
		self.path_to_data = path_to_data
		self.path_to_data_wavelength = path_to_data_wavelength
		self.cd_script = os.getcwd() # Get directory containing script
		self.load_data()
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
			SFG_files = [s for s in self.all_files if re.search('.?SFG.+', s)]

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

				power_match = re.search(r'(\d+)(nW|µW|mW|W)', signal_file[0])

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

		elif len(PL_files) < 2:
			print('Error: Appears to be missing either background or signal file.')
			sys.exit()
		else:
			print('Please check input - file not found!')
			sys.exit()

		return self.signal

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

		self.all_files = [] # Create empty array to contain all txt files in the directory
		for file in glob.glob("*.dat"): # Searches the current directory for all txt files
			self.all_files.append(file) # Appends files found

		try:
			Ne_files = [s for s in self.all_files if re.search('.?Ne.+', s)]

		except IndexError:
			print('Error: File not found!')
			sys.exit()

		self.Ne_100um = pd.read_csv([s for s in Ne_files if re.search('.?100um.+', s)][0], sep='\t', header=None, names=['Wavelength', 'Ne'])
		self.Ne_200um = pd.read_csv([s for s in Ne_files if re.search('.?200um.+', s)][0], sep='\t', header=None, names=['Wavelength', 'Ne'])

		self.wavelength_100um = self.Ne_100um['Wavelength'].to_numpy()
		self.wavelength_200um = self.Ne_200um['Wavelength'].to_numpy()

		return self.wavelength_100um, self.wavelength_200um

	def fit_to_peak(self, spectrum, type='Gaussian', A=None, x_0=None, sigma=None, gamma=None):
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
			plt.plot(1238.9/self.wavelength_100um, self.signal.iloc[:, i], label=self.signal.columns[i])
		ax.set_xlabel('Photon energy [eV]')
		ax.set_ylabel('Intensity [a.u.]')
		ax.legend()

		fig.set_tight_layout(True)
		fig.show()
		return fig

class SFG_visible_PL_power_dependence():
	def __init__(self, path_to_data):
		self.path_to_data = path_to_data
		self.cd_script = os.getcwd() # Get directory containing script
		self.load_data()
		self.change_cd_back()

	def change_cd_back(self):
		os.chdir(self.cd_script) # Change directory back to where the script is located

	def load_data(self):
		os.chdir(self.path_to_data) # Set current directory to the folder containing the files of interest

		self.all_files = [] # Create empty array to contain all txt files in the directory
		for file in glob.glob("*.dat"): # Searches the current directory for all txt files
			self.all_files.append(file) # Appends files found

		try:
			PL_files = [s for s in self.all_files if re.search('.?PL.+', s)]
		
		except IndexError:
			print('Error: File not found!')
			sys.exit()

		self.signal, self.backround, self.signal_raw = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

		if len(SFG_files) >= 2:
			PL_files.sort()
			PL_files_grouped = [PL_files[i:i+2] for i in range(0, len(PL_files), 2)]

			signal_raw, background_raw = pd.DataFrame(), pd.DataFrame()

			for i in range(len(PL_files_grouped)):
				signal_file = [s for s in PL_files_grouped[i] if re.search('.?BinArea2of2.+', s)]
				background_file = [s for s in PL_files_grouped[i] if re.search('.?BinArea1of2.+', s)]

				power_match = re.search(r'(\d+)(nW|µW|mW|W)', signal_file[0])

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

				# Code for correcting error peaks
				peaks, properties = find_peaks(abs(self.signal[power]), height=height, prominence=prominence)

				for peak in peaks:                                                                                                                                                                                                       
					left_value = self.signal[power].iloc[peak - 2]                                                                                                                                                                 
					right_value = self.signal[power].iloc[peak + 2]                                                                                                                                                                
					if abs(self.signal[power].iloc[peak]) > abs(left_value) + threshold and abs(self.signal[power].iloc[peak]) > abs(right_value) + threshold:                                                               
						for i in range(neighbours):                                                                                                                                                                                               
							self.signal[power].iloc[peak-i] = left_value / 2                                                                                                                                                       
							self.signal[power].iloc[peak+i] = left_value / 2   

		elif len(PL_files) < 2:
			print('Error: Appears to be missing either background or signal file.')
			sys.exit()
		else:
			print('Please check input - file not found!')
			sys.exit()

		return self.signal

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


	def plot_spectra(self):
		fig = plt.figure()
		gs = GridSpec(1,1, figure=fig)

		ax = fig.add_subplot(gs[0,0])
		for i in range(self.signal.shape[1]):
			plt.plot(self.signal.iloc[:, i], label=slf.signal.columns[i])
		plt.set_xlabel('Pixel')
		plt.set_ylabel('Intensity [a.u.]')
		plt.legend()

		fig.set_tight_layout(True)
		fig.show()

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
 

if __name__ == "__main__":

	data_path = r"C:\Users\h_las\OneDrive\Kyoto University\Post doc\Data\samples\CsPbBr3\bulk\20241007-CsPbBr3\20241007\\"

	raman = Raman_spectrum(
		path_to_data=r"C:\Users\h_las\OneDrive\Kyoto University\Post doc\Data\samples\CsPbBr3\bulk\20241007-CsPbBr3\20241007\No241007-001-CsPbBr3-Raman-1sx10-G2400DetWL541nm-Exc532nm-ND3-Obj100x-pinslit100um.txt")
	PL = Photoluminescence_spectrum(
		path_to_data=data_path+r"No241007-003-CsPbBr3-PL-100ms-G150DetWL700nm-Exc532nm-ND3-Obj100x-pinslit100um.txt")

	microscope = ImageLoader(r'C:\Users\h_las\OneDrive\Kyoto University\Post doc\Data\samples\CsPbBr3\bulk\20241007-CsPbBr3')
	microscope_SFG = ImageLoader(r'C:\Users\h_las\OneDrive\Kyoto University\Post doc\Data\samples\CsPbBr3\bulk\20241007-CsPbBr3-SFG')

	sfg = SFG_IR_SFG_power_dependence(path_to_data=r'C:\Users\h_las\OneDrive\Kyoto University\Post doc\Data\samples\CsPbBr3\bulk\20241007-CsPbBr3-SFG')



