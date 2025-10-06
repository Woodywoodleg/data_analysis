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
from pathlib import Path
from io import StringIO
from matplotlib.font_manager import FontProperties

def parse_filename_meta(fname: str):
    """
    Parse filename like 'deltax_2p00_P_21p42mW.dat'
    → {'deltax_mm': 2.00, 'power_mW': 21.42}
    """
    stem = Path(fname).stem
    m = re.search(r"deltax_([+-]?\d+p\d+)_P_([+-]?\d+p\d+)mW", stem)
    if not m:
        return {}
    def p2float(tok): return float(tok.replace("p", "."))
    return {
        "deltax_mm": p2float(m.group(1)),
        "power_mW":  p2float(m.group(2)),
        "filename":  fname
    }

def load_one_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=['Wavelength', 'Counts'], skiprows=1)  # adjust sep/header
    df.attrs.update(parse_filename_meta(path))
    return df

class spectrometer_SHG():
	def __init__(self, path_to_data='./'):
		self.path_to_data = Path(path_to_data)
		self.cd_script = os.getcwd() # Get directory containing script
		self.load_data()
		self.change_cd_back()

	def change_cd_back(self):
		os.chdir(self.cd_script) # Change directory back to where the script is located

	def load_data(self):
		all_files = sorted(glob.glob(str(self.path_to_data / "*.dat")))
		self._dc_files   = [f for f in all_files if f.endswith("_dc.dat")]
		self._data_files = [f for f in all_files if not f.endswith("_dc.dat")]

		# make lists of DataFrames
		self.dataframes = [load_one_file(f) for f in self._data_files]
		self.dataframes_dc = [load_one_file(f) for f in self._dc_files]

		self.data = []
		for df, dc in zip(self.dataframes, self.dataframes_dc):
			# sanity check: same shape
			if df.shape != dc.shape:
				raise ValueError(f"Shape mismatch between {df.attrs['filename']} and {dc.attrs['filename']}")

			df_corr = df.copy()
			df_corr.iloc[:, 1] = df.values[:, 1] - dc.values[:, 1]  # subtract elementwise

			# keep the metadata from the original
			df_corr.attrs.update(df.attrs)
			self.data.append(df_corr)

	def plot_all_spectra(self, figsize=(10,6), xlim=None):

		fig = plt.figure(figsize=figsize)
		gs = GridSpec(1, 1, figure=fig)

		ax = fig.add_subplot(gs[0, 0])
		for i in self.data:
			ax.plot(i['Wavelength'], i['Counts'])
		ax.set_xlabel('Wavelength [$\\mathrm{nm}$]')
		ax.set_ylabel('Intensity [a.u.]')
		ax.set_title('Spectrometer results')
		if xlim:
			ax.set_xlim(xlim[0], xlim[1])

		fig.set_tight_layout(True)
		fig.show()

class OceanOptics_spectrum():
	def __init__(self, path_to_data='./'):
		self.path_to_data = path_to_data
		self.cd_script = os.getcwd()
		self.load_data()


		self.change_cd_back()

	def change_cd_back(self):
		os.chdir(self.cd_script) # Change directory back to where the script is located

	def load_data(self):
		path = Path(self.path_to_data)
		with open(path, "r", encoding="shift_jis") as f:   # use utf-8 for Japanese
				lines = f.readlines()

		# Find the data start and end markers
		start = next(i for i, line in enumerate(lines) if "Data Start" in line)
		end   = next(i for i, line in enumerate(lines) if "END of Data" in line)

		# Make sure these are LISTS OF LINES, not one long string
		header_lines = [ln.rstrip("\r\n") for ln in lines[:start] if ln.strip()]
		data_lines   = [ln for ln in lines[start+1:end] if ln.strip()]

		# Save header (as a dict or raw text)
		header = self.parse_header(header_lines)

		# Read the data block into pandas
		self.signal = pd.read_csv(StringIO("".join(data_lines)), sep="\t", header=None, names=["Wavelength", "Counts"])

		# Attach header as metadata
		self.signal.attrs["_header"] = header
		self._header = header

		return self.signal

	def parse_header(self, header_lines):
		"""
		Turn header lines into a dict.
		Handles tab or (full-width) colon separators: '\t', ':', '：'.
		"""
		hdr = {}
		for line in header_lines:
			# normalize full-width colon to ASCII, keep tabs
			norm = line.replace("：", ":").strip()

			if "\t" in norm:
				key, val = norm.split("\t", 1)
			elif ":" in norm:
				key, val = norm.split(":", 1)
			else:
				# no recognizable separator → skip or store raw
				# choose skip to avoid junk keys
				continue

			key = key.strip()
			val = val.strip()

			# If multiple tabbed values, keep as list
			vals = [v for v in val.split("\t") if v] if "\t" in val else [val]

			# Try to coerce numbers (int/float) where possible
			def coerce(x):
				try:
					xi = int(x)
					return xi
				except ValueError:
					try:
						xf = float(x)
						return xf
					except ValueError:
						return x

			coerced = [coerce(v) for v in vals]
			hdr[key] = coerced if len(coerced) > 1 else coerced[0]
		return hdr

	def plot_spectrum(self, figsize=(10,6), title=None, xlim=None):
		# Pick a Japanese-capable font (adjust path if needed)
		font_path_candidates = [
			r"C:\Windows\Fonts\meiryo.ttc",           # Windows Meiryo
			r"C:\Windows\Fonts\YuGothR.ttc",          # Windows Yu Gothic
			r"C:\Windows\Fonts\msgothic.ttc",         # Windows MS Gothic
			"/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux
			"/usr/share/fonts/opentype/noto/NotoSansCJKJP-Regular.otf",
			"/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",  # mac
		]

		jp_font = None
		for p in font_path_candidates:
			if os.path.isfile(p):
				jp_font = FontProperties(fname=p)
				break

		fig = plt.figure(figsize=figsize)
		gs = GridSpec(1, 1, figure=fig)

		axa = fig.add_subplot(gs[0, 0])
		# use a useful header field instead of dumping the whole dict
		# label = f"積分時間={self._header.get('積分時間[usec]', '?')} µs"
		# label = self._header
		label = "\n".join(f"{k}: {v}" for k, v in self._header.items())
		axa.plot(self.signal['Wavelength'], self.signal['Counts'], label=label)

		axa.set_xlabel('Wavelength [nm]', fontproperties=jp_font)
		axa.set_ylabel('Counts [a.u.]', fontproperties=jp_font)

		if title:
			axa.set_title('OceanOptics Spectrum: ' + title, fontproperties=jp_font)
		else:
			axa.set_title('OceanOptics Spectrum', fontproperties=jp_font)

		if xlim:
			axa.set_xlim(*xlim)

		axa.legend(prop=jp_font, loc="best", handlelength=1, handletextpad=1)

		fig.tight_layout()
		plt.show()

		return fig

if __name__ == '__main__':




