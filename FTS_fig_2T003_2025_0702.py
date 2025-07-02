# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] jp-MarkdownHeadingCollapsed=true
# # SPMpy_introduction
# * Authors : Dr. Jewook Park, CNMS, ORNL
#     * Center for Nanophase Materials Sciences (CNMS), Oak Ridge National Laboratory (ORNL)
#     * email :  parkj1@ornl.gov
# -

# > **SPMpy** is a collection of Python functions designed to analyze scanning probe microscopy (SPM) data, including scanning tunneling microscopy and spectroscopy (STM/S) and atomic force microscopy (AFM) datasets, which are inherently multidimensional. By leveraging recent advances in image processing, often referred to as computer vision, SPMpy makes use of [building blocks](https://scipy-lectures.org/intro/intro.html#the-scientific-python-ecosystem) and powerful [visualization tools](https://scikit-image.org/) that are readily available within the [scientific python ecosystem](https://lectures.scientific-python.org). Inspired by established SPM data analysis software such as [Wsxm](http://www.wsxm.eu/) and [Gwyddion](http://gwyddion.net/), SPMpy also incorporates insights from [Fundamentals in Data Visualization](https://clauswilke.com/dataviz/).
# > SPMpy is being developed as an open-source project, led by Jewook Park ([SPMPY](https://github.com/jewook-park/SPMPY)). Contributions, suggestions, and error reports are highly encouraged. Feel free to contact via the GitHub page or [email](mailto:parkj1@ornl.gov). Comments and feedback are welcome in either Korean or English.
#
# > SPMpy uses [Xarray](https://docs.xarray.dev/en/stable/#) as its data container. Currently, SPM datasets obtained via the Nanonis Controller ([SPECS](https://www.specs-group.com/nanonis/products/mimea/)) have been integrated for conversion into Xarray, using [nanonispy](https://github.com/underchemist/nanonispy). By utilizing Xarray datasets, SPMpy allows the manipulation of data channels obtained simultaneously while preserving metadata (data headers). Other SPM controller formats can also be analyzed in SPMpy after conversion to Xarray, though this requires specific conversion functions. Additionally, 2D image data analyzed in Gwyddion (*.gwy) can be converted into Xarray for additional analysis.
#
# * 2025 0213 update by **Jewook Park**
#     * for internal review in STM group, CNMS, ORNL 

# # Experiments al Conditions 
# * <font color= Black, font size="4" > **Data Acquistion date**  </font> :  : 2024  0518
# *  <font color= Black, font size="4" > **Sample**  </font> : 
# <font color= White, font size="3" > $FeTe_{0.55}Se_{0.45}$ </font> ( Low Temp (83 K)Cleaving) 
# *  <font color= Black, font size="4" > **Tip**  </font> : PtIr normal metal
#     tip (#26)
# *  <font color= Black, font size="4" > **Temperature**  </font>   :   **40mK**
# *  <font color= Black, font size="4" > **Magnetic field**  </font>  **0-6T (Z)**

# # import necessary Python packages
# > <font color= orange > **Environments check**  </font>
# >    * check necessary packages  &  import modules

# +
import importlib
import subprocess
import sys
from warnings import warn

# Check if the package is installed
def is_package_installed(package_name):
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

# Run shell commands
def run_command(cmd):
    try:
        result = subprocess.run(
            cmd, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {cmd}")
        print(f"Error: {e.stderr}")
        return False

# Install package via conda or pip
def install_package(install_name, conda=True):
    cmds = []
    
    # Special handling: tqdm-joblib is not available in the conda channel so use pip only
    if install_name == 'tqdm-joblib':
        cmds = [f'{sys.executable} -m pip install tqdm-joblib']
    elif install_name == 'scikit-image':
        cmds = [
            'conda install -y scikit-image -c conda-forge',
            'python -m pip install -U pip',
            'python -m pip install -U scikit-image'
        ]
    elif install_name == 'pyqt5':
        cmds = [
            'conda install -y pyqt -c conda-forge',
            f'{sys.executable} -m pip install pyqt5'
        ]
    elif install_name == 'seaborn-image':
        cmds = [
            f'{sys.executable} -m pip install seaborn-image',
            f'{sys.executable} -m pip install seaborn-image --no-cache-dir'
        ]
    elif install_name == 'dask':
        cmds = ['conda install -y dask -c conda-forge']
    elif install_name == 'joblib':
        cmds = [f'{sys.executable} -m pip install joblib']
    elif install_name == 'numba':
        cmds = [f'{sys.executable} -m pip install numba']
    elif install_name == 'jupyter_bokeh':
        cmds = [f'{sys.executable} -m pip install jupyter-bokeh']
    elif install_name in ['umap-learn', 'scikit-optimize']:
        cmds = [f'conda install -c conda-forge {install_name}']
    elif install_name == 'lmfit':
        cmds = [f'{sys.executable} -m pip install lmfit']
    # Default installation method using conda or pip
    elif conda:
        cmds = [
            f'conda install -y {install_name} -c conda-forge',
            f'conda install -y {install_name} -c anaconda',
            f'{sys.executable} -m pip install {install_name}'
        ]
    else:
        cmds = [f'{sys.executable} -m pip install {install_name}']
    
    # Try each installation command
    for cmd in cmds:
        if run_command(cmd):
            print(f"Successfully installed {install_name}")
            return True
    
    print(f"All installation attempts for {install_name} failed")
    return False

# Try to import the package, optionally assigning an alias
def import_package(import_name, alias=None):
    try:
        module = importlib.import_module(import_name)
        if alias:
            globals()[alias] = module
        return module
    except ImportError as e:
        warn(f"Failed to import {import_name}. Error: {str(e)}")
        return None

# Install and import a package, if not already installed
def install_and_import(install_name, import_name=None, alias=None):
    if import_name is None:
        import_name = install_name

    # Check if the package is already installed (using the correct import name)
    if is_package_installed(import_name):
        print(f"{import_name} is already installed. Skipping installation.")
        return import_package(import_name, alias)

    print(f"{import_name} is not installed or failed to import. Installing {install_name}...")
    if not install_package(install_name):
        warn(f"Failed to install {install_name}")
        return None

    return import_package(import_name, alias)

# Packages list updated to use the correct import names:
# - For tqdm-joblib: import as 'tqdm_joblib'
# - For seaborn-image: import as 'seaborn_image'
# - For scikit-optimize: import as 'skopt'
packages = [
    ('xarray', 'xr', 'xarray'),
    ('numpy', 'np', 'numpy'),
    ('scipy', 'sp', 'scipy'),
    ('seaborn', 'sns', 'seaborn'),
    ('pandas', 'pd', 'pandas'),
    ('scikit-image', 'skimage', 'skimage'),
    ('xrft', None, 'xrft'),
    ('holoviews', 'hv', 'holoviews'),
    ('hvplot', None, 'hvplot'),
    ('plotly', None, 'plotly'),
    ('gwyfile', None, 'gwyfile'),
    ('tqdm', None, 'tqdm'),
    ('tqdm-joblib', None, 'tqdm_joblib'),
    # For netcdf4, use the correct import name 'netCDF4'
    ('netcdf4', None, 'netCDF4'),
    ('h5netcdf', None, 'h5netcdf'),
    ('python-pptx', 'pptx', 'pptx'),
    # For pyqt5, use 'PyQt5' for both alias and import name
    ('pyqt5', 'PyQt5', 'PyQt5'),
    ('seaborn-image', 'isns', 'seaborn_image'),
    ('joblib', None, 'joblib'),
    ('numba', None, 'numba'),
    ('dask', None, 'dask'),
    ('jupyter_bokeh', None, 'jupyter_bokeh'),
    ('umap-learn', None, 'umap'),
    ('scikit-optimize', None, 'skopt'),
    ('ipywidgets', None, 'ipywidgets'),
    ('lmfit', None, 'lmfit'),
    ('hyperopt',        None, 'hyperopt'),
    ('kaleido',       None,      'kaleido')
]

# Install and import each package
for install_name, alias, import_name in packages:
    install_and_import(install_name, import_name, alias)

# Additional imports and configurations
try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from PyQt5.QtWidgets import QApplication, QFileDialog
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import panel as pn
    from panel.interact import interact
    from joblib import Parallel, delayed
    from numba import jit
    from skimage.transform import resize
    import panel.widgets as pnw

    # Load SPMpy functions
    # from SPMpyDataAnalysisFunctions_2024Aug15 import *
    # from SPMpyDataAnalysisFunctions_2024Sep5 import *
    # from SPMpyDataAnalysisFunctionLibrary_2025Feb16 import *
    #from SPMpyDataAnalysisFunctionLibrary_2025Mar28 import *
    from SPMpyDataAnalysisFunctionLibrary_2025May31 import *
    

    # Set seaborn-image origin
    import seaborn_image as isns
    isns.set_image(origin='lower')

    print("All packages have been successfully imported.")
except ImportError as e:
    print(f"Error during additional imports: {str(e)}")

# Configure Bokeh for inline plotting
from bokeh.io import output_notebook
output_notebook()

# Activate holoviews with bokeh backend
import holoviews as hv
hv.extension('bokeh')

# Ensure ipywidgets version 8 or higher is installed
run_command('pip install ipywidgets>=8')
print("ipywidgets version updated if necessary.")

# Upgrade Panel and Holoviews to the latest versions that include prebuilt labextensions.
def upgrade_prebuilt_labextensions():
    """
    Upgrade Panel and Holoviews to the latest versions that include
    prebuilt JupyterLab extensions. This avoids the need for manual labextension installation.
    """
    try:
        result = subprocess.run(
            f"{sys.executable} -m pip install --upgrade panel holoviews",
            check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        print(result.stdout)
        print("Successfully upgraded Panel and Holoviews with prebuilt labextensions.")
    except subprocess.CalledProcessError as e:
        print("Failed to upgrade Panel and Holoviews:")
        print(e.stderr)

upgrade_prebuilt_labextensions()

# -

# # Xr DataSet loading 
#
# >  <font color= orange > Choose Folder & Raw Data for analysis</font>
# > * use the file chooser & Choose Data folder
# > * Once data folder is defined, use the'folder path'

# #### INSTALL Revised Nanonispy 
# * original nanonispy is not updated from 2021--> forked and updated by Jewook
#     * changing ```np.float``` to ```float```
#     * #!pip install /Users/gkp/Documents/GitHub/nanonispy

# +
import os
import sys
import warnings

def install_and_import_nanonispy():
    try:
        # Try importing the nanonispy module
        import nanonispy as nap
        print("nanonispy module is already installed and imported.")
    except ModuleNotFoundError:
        # Module not found, proceed with installation
        warnings.warn("ModuleNotFoundError: No module named 'nanonispy'. Attempting installation.")
        
        # Define possible installation paths
        paths_to_try = [
            r'C:\Users\gkp\Documents\GitHub\nanonispy',
            r'C:\temp\nanonispy',
            r'/Users/gkp/Documents/GitHub/nanonispy'
        ]
        
        installed = False
        
        for path in paths_to_try:
            if os.path.exists(path):
                print(f"Found nanonispy at {path}. Attempting to install...")
                os.system(f'pip install "{path}"')
                try:
                    import nanonispy as nap
                    print("Successfully installed and imported nanonispy.")
                    installed = True
                    break
                except ModuleNotFoundError:
                    print(f"Installation from {path} failed. Trying next path...")
        
        if not installed:
            raise ModuleNotFoundError("Failed to install nanonispy. Please check the paths and try again.")

# Call the function
install_and_import_nanonispy()

# -


# ##### memory usage checkup 
#

# +
import psutil
import os
import platform  # For detecting the operating system

# Get memory usage of the current Python process
process = psutil.Process(os.getpid())
memory_info = process.memory_info()

# Retrieve the operating system name
os_name = platform.system()

# Display memory usage information
print(f"RSS: {memory_info.rss / 1024 ** 2:.2f} MB")  # Resident Set Size (actual memory usage)
print(f"VMS: {memory_info.vms / 1024 ** 2:.2f} MB")  # Virtual Memory Size (total reserved memory)

# Add OS-specific explanations
if os_name == "Darwin":  # macOS
    print("Note: On macOS, VMS includes large reserved virtual memory that is not actually used.")
elif os_name == "Windows":
    print("Note: On Windows, VMS reflects the total virtual memory requested by the process.")
else:
    print("Note: VMS interpretation may vary depending on the operating system.")

# -
# ## check Pytorch & scikit learn  installation in the system 
#

# +
import platform
import sys
import subprocess

def get_os_type():
    os_name = platform.system()
    if os_name == "Darwin":
        return "Mac"
    elif os_name == "Windows":
        return "Windows"
    else:
        return "Unsupported OS"

def check_package_installed(package_name):
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def check_python_env(os_type):
    print(f"Operating System: {os_type}\n")

    if os_type == "Mac":
        print(f"macOS Version: {platform.mac_ver()[0]}")
    elif os_type == "Windows":
        print(f"Windows Version: {platform.version()}")

    print(f"Python Version: {sys.version}")
    conda_env = 'conda' in sys.version or 'Continuum' in sys.version
    print(f"Conda Environment: {'Yes' if conda_env else 'No'}")

    packages = ["torch", "sklearn", "umap", "hdbscan"]
    for pkg in packages:
        installed = check_package_installed(pkg)
        print(f"{pkg}: {'Installed' if installed else 'Not Installed'}")

    if os_type == "Mac" and check_package_installed("torch"):
        try:
            import torch
            mps_available = torch.backends.mps.is_available()
            print(f"PyTorch MPS (Metal Performance Shaders) Support: {'Available' if mps_available else 'Not Available'}")
        except ImportError:
            print("PyTorch is not installed, so MPS support cannot be checked.")

    if conda_env:
        print("\nConda Installed Channels:")
        try:
            subprocess.run(["conda", "config", "--show-sources"], check=True)
        except FileNotFoundError:
            print("Conda not found in the system path.")

def install_packages(os_type):
    if os_type == "Unsupported OS":
        print("Unsupported OS detected. Exiting installation.")
        return

    def install_conda_package(pkg_list, channel="conda-forge"):
        try:
            subprocess.run(["conda", "install", "-y"] + pkg_list + ["-c", channel], check=True)
        except Exception as e:
            print(f"Failed to install {pkg_list}: {e}")

    # PyTorch
    if not check_package_installed("torch"):
        print("\nInstalling PyTorch...")
        if os_type == "Mac":
            install_conda_package(["pytorch", "torchvision", "torchaudio"], channel="pytorch")
        elif os_type == "Windows":
            install_conda_package(["pytorch", "torchvision", "torchaudio"], channel="pytorch")
        print("PyTorch installation attempted.")
    else:
        print("PyTorch already installed.")

    # scikit-learn
    if not check_package_installed("sklearn"):
        print("\nInstalling scikit-learn...")
        install_conda_package(["scikit-learn"])
    else:
        print("scikit-learn already installed.")

    # umap-learn
    if not check_package_installed("umap"):
        print("\nInstalling umap-learn...")
        install_conda_package(["umap-learn"])
    else:
        print("umap-learn already installed.")

    # hdbscan
    if not check_package_installed("hdbscan"):
        print("\nInstalling hdbscan...")
        install_conda_package(["hdbscan"])
    else:
        print("hdbscan already installed.")

    print("\nRechecking environment after installations...\n")
    check_python_env(os_type)

if __name__ == "__main__":
    os_type = get_os_type()

    if os_type == "Unsupported OS":
        print("This script supports only macOS and Windows. Exiting.")
        sys.exit(1)

    print("Step 1: Checking Python environment...\n")
    check_python_env(os_type)

    print("\nStep 2: Installing required packages (if missing)...\n")
    install_packages(os_type)

    print("\n✅ All checks and installations complete!")


# +

from PyQt5.QtWidgets import QApplication, QFileDialog

app = QApplication([])
file_dialog = QFileDialog()
folder_path = file_dialog.getExistingDirectory(None, "Select Folder")
print(f"Selected folder: {folder_path}")
#file_path = file_dialog.getOpenFileName()[0]
#print(f"Selected file: {file_path}")



# -


files_df = files_in_folder(folder_path)
files_df


# + [markdown] jp-MarkdownHeadingCollapsed=true
# ### loading gwy files in folder  & convert to xr
# * transfer to dataframe ==> xrray
# -

# or choose the file_path from the file_df
file_list = files_df[files_df.type=='gwy'].file_name#.iloc[1]
file_list

file_list.iloc[0]

FTS_1T_240505_R30_007_gwy_xr = gwy_df2xr(gwy_img2df (file_list.iloc[0]))
FTS_1T_240505_R30_007 = FTS_1T_240505_R30_007_gwy_xr['gwy_xr1'][['Z_fwd','LIX_fwd']]

# +
#### add attributes 
# -

FTS_1T_240505_R30_007.attrs['tip'] = 'PtIr'
FTS_1T_240505_R30_007.attrs['sample'] = 'FeTe0.55Se0.45'
FTS_1T_240505_R30_007.attrs['temperature'] = '40mK'
FTS_1T_240505_R30_007.attrs['ref_a0nm'] = 0.38

# + editable=true slideshow={"slide_type": ""}
from matplotlib_scalebar.scalebar import ScaleBar

plt.rcParams.update({'font.size': 12})  

fig, ax = plt.subplots(figsize = (4,4))

isns.imshow(plane_fit_surface_xr(FTS_1T_240505_R30_007).Z_fwd
            -plane_fit_surface_xr(FTS_1T_240505_R30_007).Z_fwd.min(),
            ax =ax, robust=True, perc = (0.2,99.95),cmap ='YlGnBu_r')
# Add a scale bar manually using matplotlib's ScaleBar
scalebar = ScaleBar(40/1024, units="nm", location='lower left', font_properties="Arial", scale_loc='top',width_fraction=0.02, pad = 1.0, length_fraction=0.3)
ax.add_artist(scalebar)

# Save the figure as SVG
svg_filepath = "FTS_1T_240505_R30_007_z.svg"
fig.savefig(svg_filepath, format='svg')
# +
from matplotlib_scalebar.scalebar import ScaleBar

plt.rcParams.update({'font.size': 12})  

fig, ax = plt.subplots(figsize = (4,4))

isns.imshow(plane_fit_surface_xr(FTS_1T_240505_R30_007).LIX_fwd
            -plane_fit_surface_xr(FTS_1T_240505_R30_007).LIX_fwd.min(), 
            ax =ax, robust=True, perc = (0.5,99.5),cmap ='coolwarm')
# Add a scale bar manually using matplotlib's ScaleBar
scalebar = ScaleBar(40/1024, units="nm", location='lower left', font_properties="Arial", scale_loc='top',width_fraction=0.02, pad = 1.0, length_fraction=0.3)
ax.add_artist(scalebar)

# Save the figure as SVG
svg_filepath = "FTS_1T_240505_R30_007_LIX.svg"
fig.savefig(svg_filepath, format='svg')
# +
#FTS_1T_240505_R30_007.Z_fwd.plot()
FTS_1T_240505_R30_007_fft = twoD_FFT_xr(FTS_1T_240505_R30_007, complex_output = False)#.plot()
FTS_1T_240505_R30_007_fft.attrs = FTS_1T_240505_R30_007.attrs


FTS_1T_240505_R30_007_fft

# +
from matplotlib_scalebar.scalebar import ScaleBar

plt.rcParams.update({'font.size': 12})  

fig, ax = plt.subplots(figsize = (4,4))

# Define ref_q01overnm (example: ref_a0nm = 1.0)
ref_q01overnm = 1 / (FTS_1T_240505_R30_007_fft.ref_a0nm*1E-9)
# Set zoom_in value (example: 2)
zoom_in = 1.5
zoom_range = ref_q01overnm * zoom_in

# Get coordinate information (freq_X, freq_Y)
freq_X = FTS_1T_240505_R30_007_fft.coords['freq_X']
freq_Y = FTS_1T_240505_R30_007_fft.coords['freq_Y']

# Filter the central area based on the zoom_range
FTS_1T_240505_R30_007_fft_filtered= FTS_1T_240505_R30_007_fft.where(
    (np.abs(freq_X) <= zoom_range) & (np.abs(freq_Y) <= zoom_range), drop=True)

isns.imshow(FTS_1T_240505_R30_007_fft_filtered.Z_fwd_fft , ax =ax, robust=True, perc = (52,99.90),cmap ='Greys')
# Add a scale bar manually using matplotlib's ScaleBar
scalebar = ScaleBar(40/1024, units="nm", 
                    location='lower left',
                    font_properties="Arial",
                    scale_loc='top',
                    pad = 1.0, color ='Black',
                    width_fraction=0.02,
                    length_fraction=0.3)
ax.add_artist(scalebar)

# Save the figure as SVG
svg_filepath = "FTS_1T_240505_R30_007_Z_fwd_fft.svg"
fig.savefig(svg_filepath, format='svg')
# +
from matplotlib_scalebar.scalebar import ScaleBar

plt.rcParams.update({'font.size': 12})  

fig, ax = plt.subplots(figsize = (4,4))

# Define ref_q01overnm (example: ref_a0nm = 1.0)
ref_q01overnm = 1 / (FTS_1T_240505_R30_007_fft.ref_a0nm*1E-9)
# Set zoom_in value (example: 2)
zoom_in = 1.5
zoom_range = ref_q01overnm * zoom_in

# Get coordinate information (freq_X, freq_Y)
freq_X = FTS_1T_240505_R30_007_fft.coords['freq_X']
freq_Y = FTS_1T_240505_R30_007_fft.coords['freq_Y']

# Filter the central area based on the zoom_range
FTS_1T_240505_R30_007_fft_filtered= FTS_1T_240505_R30_007_fft.where(
    (np.abs(freq_X) <= zoom_range) & (np.abs(freq_Y) <= zoom_range), drop=True)

isns.imshow(FTS_1T_240505_R30_007_fft_filtered.LIX_fwd_fft , ax =ax, robust=True, perc = (50,99.9900),cmap ='Blues')
# Add a scale bar manually using matplotlib's ScaleBar
scalebar = ScaleBar(40/1024, units="nm",
                    location='lower left', 
                    font_properties="Arial", 
                    scale_loc='top',color ='Black',
                    pad = 1.0,
                    width_fraction=0.02,                    
                    length_fraction=0.3)
ax.add_artist(scalebar)

# Save the figure as SVG
svg_filepath = "FTS_1T_240505_R30_007_LIX_fwd_fft.svg"
fig.savefig(svg_filepath, format='svg')
# + [markdown] jp-MarkdownHeadingCollapsed=true
# #  Load sxm file and 3ds file for the merged area line profile 
# * loading 3ds files  in folder  & convert to LDOS.xr
#     * currently gwy file is not good for merge with 3ds file 
#
#
# -

# ### loas sxm file first

# +
# ### reselect working folder 
from PyQt5.QtWidgets import QApplication, QFileDialog
import sys

def select_folder():
    # Check if QApplication instance already exists
    app = QApplication.instance()
    
    # If no instance exists, create one
    if app is None:
        app = QApplication(sys.argv)
    
    # Create the file dialog to select a folder
    file_dialog = QFileDialog()
    folder_path = file_dialog.getExistingDirectory(None, "Select Folder")
    
    return folder_path

# Call the function to select a folder
selected_folder = select_folder()
if selected_folder:
    print(f"Selected folder: {selected_folder}")
else:
    print("No folder selected.")


# +
folder_path = selected_folder

files_df = files_in_folder(folder_path) 

file_list = files_df[files_df.type=='sxm'].file_name#
file_list
# -


file_list.iloc[2]

# +
FTS_1T_0506_R30_00003_xr = img2xr(file_list.iloc[1],center_offset=False)

FTS_1T_0506_R30_00003_xr = FTS_1T_0506_R30_00003_xr[['z_fwd']].rename({'z_fwd':'topography'}).copy()
FTS_1T_0506_R30_00003_xr

# +
FTS_0T_0504_R30_00001_xr = img2xr(file_list.iloc[2],center_offset=False)

FTS_0T_0504_R30_00001_xr = FTS_0T_0504_R30_00001_xr[['z_fwd']].rename({'z_fwd':'topography'}).copy()
FTS_0T_0504_R30_00001_xr

# +
#FTS_0T_0504_R30_00001_xr
#FTS_1T_0506_R30_00003_xr

# + [markdown] jp-MarkdownHeadingCollapsed=true
# # Load 3ds file ( change the current working folder with qt5 )
#
# > In JupyterLab, running GUI applications like QApplication multiple times can cause kernel errors.
# > This occurs because the QApplication object is already running, and attempting to create another instance results in an error.
# > To resolve this, the code needs to check if a QApplication instance is already running, and if not, create a new one. 

# +
# ### reselect working folder 
from PyQt5.QtWidgets import QApplication, QFileDialog
import sys

def select_folder():
    # Check if QApplication instance already exists
    app = QApplication.instance()
    
    # If no instance exists, create one
    if app is None:
        app = QApplication(sys.argv)
    
    # Create the file dialog to select a folder
    file_dialog = QFileDialog()
    folder_path = file_dialog.getExistingDirectory(None, "Select Folder")
    
    return folder_path

# Call the function to select a folder
selected_folder = select_folder()
if selected_folder:
    print(f"Selected folder: {selected_folder}")
else:
    print("No folder selected.")
# -


# ## loading 3ds files  in folder  & convert to LDOS.xr


selected_folder

# +
# Get the list of all files in the folder
folder_path = selected_folder

files_df = files_in_folder(folder_path)
files_list = files_df[files_df.type == '3ds'].file_name
files_list
# -

files_list.to_list()

for idx, file_name in enumerate(files_list):
    print(file_name)
    if file_name in ['Grid Spectroscopy(X0.01)_0T_40mK_001.3ds',
 'Grid Spectroscopy(X0.01)_0T_40mK_002.3ds',
 'Grid Spectroscopy(X0.01)_0T_40mK_003.3ds',
 'Grid Spectroscopy(X0.01)_2T_40mK_003.3ds',
 'Grid Spectroscopy(X0.01)_2T_40mK_004.3ds',
 'Grid Spectroscopy(X0.01)_2T_40mK_005.3ds',
 'Grid Spectroscopy(X0.01)_4T_40mK_006.3ds',
 'Grid Spectroscopy(X0.01)_4T_40mK_007.3ds',
 'Grid Spectroscopy(X0.01)_4T_40mK_008.3ds',
 'Grid Spectroscopy(X0.01)_4T_40mK_009.3ds',
 'Grid Spectroscopy(X0.01)_4T_40mK_010.3ds',
 'Grid Spectroscopy(X0.01)_6T_40mK_001.3ds']:
        # file_path = os.path.join(folder_path, file_name)
        file_path = file_name 
        print(file_path)
    
        # Load grid data
        grid_xr = grid2xr(file_path)
        
        # Add attributes
        grid_xr.attrs['tip'] = 'PtIr'
        grid_xr.attrs['sample'] = 'FeTe0.55Se0.45'
        grid_xr.attrs['ref_a0nm'] = 0.380
        grid_xr.attrs['temperature'] = '40mK'
    
        # Process grid data
        grid_xr['I_fb'] = (grid_xr.I_fwd + grid_xr.I_bwd) / 2
        grid_xr['LIX_fb'] = (grid_xr.LIX_fwd + grid_xr.LIX_bwd) / 2
    
        grid_topo = grid_xr[['topography']].copy().astype('float64')
        grid_3D = grid_xr[['I_fb', 'LIX_fb']].copy()
    
        # Get random location for plateau checking
        x_loc, y_loc = get_random_location(grid_3D)
    
        # Find plateau
        grid_3D_plateau = find_plateau0(grid_3D, I0_tolerance=5E-11, LIX0_tolerance=5E-12, x_idx=x_loc, y_idx=y_loc)
    
        # Check metallic_mask & bias_mV offset
        metallic_mask, nearest_zero_bias_mV = plot_m_mask_N_bias_ofst_grid_3D_plateau(grid_3D_plateau)
        
        # Shift bias_offset (not shift if it is negligible) 
        grid_3D_pretreat = shift_bias_mV(grid_3D, nearest_zero_bias_mV, method='median', threshold=0.05)
        
        # Check plateau after single value bias_mV shift 
        grid_3D_pretreat_plateau = find_plateau0(grid_3D_pretreat, I0_tolerance=2E-11, LIX0_tolerance=5E-12, x_idx=x_loc, y_idx=y_loc)
        plot_m_mask_N_bias_ofst_grid_3D_plateau(grid_3D_pretreat_plateau)
        
        # Unit calc after plateau check grid_3D_pretreat   
        grid_3D_pretreat_plateau_untclc = grid_3D_unit_calc(grid_3D_pretreat_plateau, ch_I='I_fb', ch_LIX='LIX_fb')
        
        # Save grid_LDOS 
        grid_LDOS = grid_3D_pretreat_plateau_untclc[['LIX_fb_unit_calc']].rename({'LIX_fb_unit_calc': 'LDOS'})

        # Extract both magnetic field and number correctly
        def extract_suffix(file_name):
            # Split to remove the '.3ds' extension
            suffix = file_name.rsplit('.3ds', 1)[0]  
            
            # Find the magnetic field part (e.g., 0T, 2T, 4T, 6T) correctly
            field_part = suffix.split('_')[-3]  # Get the part with the magnetic field (before 40mK)
            
            # Find the number part (e.g., 001, 002, etc.)
            number_part = suffix.split('_')[-1]  # Get the part after the last '_'
            
            # Combine the field and number
            return f'{field_part}_{number_part}'
        
        # Apply the extraction function to the file names and print the updated names
        suffix = extract_suffix(file_name)
        
        topo_file_name = f"GS_topo_{suffix}.nc"
        grid_topo.to_netcdf(topo_file_name)
        
        # Apply Gaussian convolution to grid_LDOS
        grid_LDOS = gaussian_convolution_update_dataset(grid_LDOS, sigma=0.5)
    
        # Smoothing and derivative of LDOS
        grid_LDOS_SnD = smoothing_and_deriv_LDOS(grid_LDOS, window_length_ratio=0.05)
        
        # Create a new file name for grid_LDOS_SnD using the filename suffix
        ldos_SnD_file_name = f"GS_LDOS_{suffix}.nc"
        grid_LDOS_SnD.to_netcdf(ldos_SnD_file_name)


# # OR, Using saved data  $\to$ loaded nc files 

# +
# ### reselect working folder 
from PyQt5.QtWidgets import QApplication, QFileDialog
import sys

def select_folder():
    # Check if QApplication instance already exists
    app = QApplication.instance()
    
    # If no instance exists, create one
    if app is None:
        app = QApplication(sys.argv)
    
    # Create the file dialog to select a folder
    file_dialog = QFileDialog()
    folder_path = file_dialog.getExistingDirectory(None, "Select Folder")
    
    return folder_path

# Call the function to select a folder
selected_folder = select_folder()
if selected_folder:
    print(f"Selected folder: {selected_folder}")
else:
    print("No folder selected.")


# +
# Get the listof all files in the folder
folder_path = selected_folder

files_df = files_in_folder(folder_path)

# Select all '.nc' files
nc_files = files_df[files_df['file_name'].str.endswith('.nc')]['file_name']
filtered_nc_files = nc_files[nc_files.str.contains('GS_LDOS|GS_topo|grid')]
filtered_nc_files.to_list()
# -


# ## select dataset 

# +
#filtered_nc_files_topo_select = [ 'GS_topo_0T_002.nc','GS_topo_2T_003.nc']
''' 
filtered_nc_files_topo_select = [ 'GS_topo_0T_001.nc',
 'GS_topo_0T_002.nc',
 'GS_topo_0T_003.nc',
 'GS_topo_2T_003.nc',
 'GS_topo_2T_004.nc',
 'GS_topo_2T_005.nc',
 'GS_topo_4T_006.nc',
 'GS_topo_4T_007.nc',
 'GS_topo_4T_008.nc',
 'GS_topo_4T_009.nc',
 'GS_topo_4T_010.nc',
 'GS_topo_6T_001.nc',]
'''
filtered_nc_files_topo_select = [ 'GS_topo_0T_001.nc',
 'GS_topo_0T_002.nc',
 'GS_topo_0T_003.nc',
 'GS_topo_2T_003.nc',
 'GS_topo_2T_004.nc',
 'GS_topo_2T_005.nc','updated_GS_topo_0T_002.nc']

#filtered_nc_files_topo_select

filtered_nc_files_LDOS_select = [file.replace('_topo_', '_LDOS_') for file in filtered_nc_files_topo_select]
#filtered_nc_files_LDOS_select

filtered_nc_files_select =  filtered_nc_files_topo_select+filtered_nc_files_LDOS_select

filtered_nc_files_select

# +
import os
import xarray as xr

# Define the default folder path
selected_folder 

def load_nc_files(filtered_nc_files_select, folder_path=selected_folder):
    """
    Safely load .nc files into dynamically generated variables without overwriting existing variables.

    Parameters:
        filtered_nc_files_select (list): List of .nc file names to load.
        folder_path (str): Path to the folder containing the .nc files. Default is selected_folder.

    Returns:
        dict: A dictionary containing the loaded datasets with unique keys.
    """
    loaded_datasets = {}  # Dictionary to store datasets with unique keys

    for file_name in filtered_nc_files_select:
        # Create a valid Python variable name
        variable_name = os.path.splitext(file_name)[0].replace(' ', '_').replace('-', '_')
        
        # Ensure the variable name is unique to prevent overwriting
        if variable_name in loaded_datasets:
            counter = 1
            original_name = variable_name
            while variable_name in loaded_datasets:
                variable_name = f"{original_name}_{counter}"
                counter += 1

        # Full file path
        file_path = os.path.join(folder_path, file_name)
        
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"Warning: File '{file_name}' does not exist in '{folder_path}'. Skipping.")
            continue

        try:
            # Load the dataset and store it in the dictionary
            dataset = xr.open_dataset(file_path)
            loaded_datasets[variable_name] = dataset
            print(f"Loaded '{file_name}' as '{variable_name}'")
        except Exception as e:
            # Handle any errors during the file loading
            print(f"Error loading file '{file_name}': {e}")
    
    return loaded_datasets


# Example usage

# Default folder_path = selected_folder
loaded_datasets = load_nc_files(filtered_nc_files_select)

# Access datasets dynamically
for var_name, dataset in loaded_datasets.items():
    print(f"{var_name}: {dataset}")


# +
#loaded_datasets.keys()
# loaded_datasets['GS_topo_0T_001']

class DatasetVariables:
    def __init__(self, data_dict):
        for key in data_dict:
            setattr(self, key, data_dict[key])

# Load the datasets into an object with attribute access
datasets = DatasetVariables(loaded_datasets)

# Get all keys from loaded_datasets
keys_list = list(loaded_datasets.keys())

# Automatically assign variables from datasets object
for key in keys_list:
    globals()[key] = getattr(datasets, key)

# Example usage (Print a few assigned variables to check)
print(GS_LDOS_0T_001)  # Equivalent to datasets.GS_LDOS_0T_001
print(GS_LDOS_2T_003)  # Equivalent to datasets.GS_LDOS_2T_005
# -

GS_topo_2T_003

# +
#GS_LDOS_2T_003[['LDOS']].to_netcdf("GS_LDOS_2T_003.h5")

# +
#GS_LDOS_2T_003_LDOS = xr.open_dataset("GS_LDOS_2T_003.h5")

# +
#GS_LDOS_2T_003_LDOS
# -

# # Figure S1 Comparison 0T_002 & 2T_005

# ## Figure SI _ high resolution topography 
#



# +
# ### reselect working folder 
from PyQt5.QtWidgets import QApplication, QFileDialog
import sys

def select_folder():
    # Check if QApplication instance already exists
    app = QApplication.instance()
    
    # If no instance exists, create one
    if app is None:
        app = QApplication(sys.argv)
    
    # Create the file dialog to select a folder
    file_dialog = QFileDialog()
    folder_path = file_dialog.getExistingDirectory(None, "Select Folder")
    
    return folder_path

# Call the function to select a folder
selected_folder = select_folder()
if selected_folder:
    print(f"Selected folder: {selected_folder}")
else:
    print("No folder selected.")


# +
folder_path = selected_folder

files_df = files_in_folder(folder_path) 

file_list = files_df[files_df.type=='sxm'].file_name#
file_list
# -


file_list.iloc[0]

# +
FTS_0T_2023_0506_0001_xr = img2xr(file_list.iloc[0],center_offset=False)

FTS_0T_2023_0506_0001_xr = FTS_0T_2023_0506_0001_xr[['z_fwd']].rename({'z_fwd':'topography'}).copy()
FTS_0T_2023_0506_0001_xr

# +
import seaborn_image as isns
import matplotlib.pyplot as plt

# 0. Configure global scalebar style before plotting
isns.set_scalebar(
    color="white",          # scalebar color
    length_fraction=0.5,    # scalebar spans 50% of image width
    location="lower right"  # position
)

# 1. Load dataset and compute pixel size in nm
ds = FTS_0T_2023_0506_0001_xr
dx = ds.attrs['X_spacing'] * 1e9  # nm per pixel

# 2. Extract topography after plane fit
data = plane_fit_surface_xr(ds).topography

# 3. Create figure and axes
fig, ax = plt.subplots(figsize=(6, 5))

# 4. Display image with copper colormap and scalebar via dx & units
isns.imshow(
    data,
    ax=ax,
    cmap='copper',
    dx=dx,         # physical size per pixel (nm)
    units='nm'     # units for scalebar
)

# 5. Save figure in both SVG and PNG formats
fig.savefig('high_resolution_topography.svg', format='svg')
fig.savefig('high_resolution_topography.png', format='png', dpi=600)

#plt.close(fig)

# -

FTS_0T_2023_0506_0001_xr





# ## loading gwy files in folder  & convert to xr
# * transfer to dataframe ==> xrray

# +
# ### reselect working folder 
from PyQt5.QtWidgets import QApplication, QFileDialog
import sys

def select_folder():
    # Check if QApplication instance already exists
    app = QApplication.instance()
    
    # If no instance exists, create one
    if app is None:
        app = QApplication(sys.argv)
    
    # Create the file dialog to select a folder
    file_dialog = QFileDialog()
    folder_path = file_dialog.getExistingDirectory(None, "Select Folder")
    
    return folder_path

# Call the function to select a folder
selected_folder = select_folder()
if selected_folder:
    print(f"Selected folder: {selected_folder}")
else:
    print("No folder selected.")


# +
folder_path = selected_folder

files_df = files_in_folder(folder_path) 

file_list = files_df[files_df.type=='sxm'].file_name#
file_list
# -


files_df

# or choose the file_path from the file_df
file_list = files_df[files_df.type=='gwy'].file_name#.iloc[1]
file_list

file_list.iloc[2]

FTS_40mK_202405061Txdvd_100_0001_gwy_xr = gwy_df2xr(gwy_img2df (file_list.iloc[2]))


FTS_40mK_202405061Txdvd_100_0001_gwy_xr['gwy_xr3']

FTS_40mK_202405061Txdvd_100_0001_gwy_xr['gwy_xr3'].data_vars

FTS_40mK_202405061Txdvd_100_0001 = FTS_40mK_202405061Txdvd_100_0001_gwy_xr['gwy_xr3'][['z_fwd Corrected','LIX_fwd Corrected']]
FTS_40mK_202405061Txdvd_100_0001 = FTS_40mK_202405061Txdvd_100_0001.rename_vars({'z_fwd Corrected':'Z_fwd','LIX_fwd Corrected':'LIX_fwd' })
FTS_40mK_202405061Txdvd_100_0001

# +
#### add attributes 

FTS_40mK_202405061Txdvd_100_0001.attrs['tip'] = 'PtIr'
FTS_40mK_202405061Txdvd_100_0001.attrs['sample'] = 'FeTe0.55Se0.45'
FTS_40mK_202405061Txdvd_100_0001.attrs['temperature'] = '40mK'
FTS_40mK_202405061Txdvd_100_0001.attrs['ref_a0nm'] = 0.38
# -

FTS_40mK_202405061Txdvd_100_0001

# +
import numpy as np
import xarray as xr
from skimage.transform import rotate

def rotateXY_ds(xrdata: xr.Dataset, rotation_angle: float) -> xr.Dataset:
    '''
    Rotate an xarray.Dataset on the XY plane around its center.

    This function supports datasets with the following dimensionalities:
      - (X, Y): 2D data variables are rotated in the XY plane.
      - (X, Y, bias_mV): 3D data variables (e.g., stack of images at different bias voltages)
        are treated as a series of 2D slices along the bias_mV dimension and each slice
        is rotated independently in the XY plane.

    Parameters
    ----------
    xrdata : xr.Dataset
        Input dataset containing one or more data variables with dimensions X and Y,
        optionally with a third dimension (e.g., bias_mV).
    rotation_angle : float
        Rotation angle in degrees. Positive values correspond to counterclockwise rotation.

    Returns
    -------
    xr.Dataset
        A new dataset with all data variables rotated in the XY plane. The X and Y
        coordinates are adjusted so that the geometric center of the data remains
        at the same location as in the input dataset.

    Notes
    -----
    1. Only X and Y dimensions are used for rotation. Any additional dimensions
       (e.g., bias_mV) are preserved and treated as independent slices.
    2. Padding is applied before rotation to prevent cropping of the rotated data.
    3. After rotation, the dataset is re-centered to maintain the original center
       position in the XY plane.
    '''
    # -- 0) Compute global minimum across all data variables for padding fill
    global_min = float(xrdata.to_array().min().item())

    # -- 1) Store original center coordinates of the XY plane
    orig_cx = xrdata.X.mean().item()
    orig_cy = xrdata.Y.mean().item()

    # -- 2) Determine padding size using first data variable
    first_var = list(xrdata.data_vars)[0]
    # Rotate with resize=True to get full output shape
    rot_shape = rotate(xrdata[first_var].values.astype(float), rotation_angle, resize=True).shape[:2]
    orig_shape = xrdata[first_var].shape[:2]
    pad_xy = ((np.array(rot_shape) - np.array(orig_shape) + 1) // 2).astype(int)
    pad_x, pad_y = pad_xy

    # -- 3) Apply symmetric padding on X and Y dimensions
    xrdata_pad = xrdata.pad(
        X=(pad_x, pad_x),
        Y=(pad_y, pad_y),
        mode='constant',
        constant_values=global_min
    )

    # -- 4) Recreate X and Y coordinate arrays for the padded dataset
    dx = float(np.diff(xrdata.X).mean())
    dy = float(np.diff(xrdata.Y).mean())
    nX = xrdata_pad.sizes['X']
    nY = xrdata_pad.sizes['Y']
    x0 = float(xrdata.X.min()) - pad_x * dx
    y0 = float(xrdata.Y.min()) - pad_y * dy
    x_coords = x0 + np.arange(nX) * dx
    y_coords = y0 + np.arange(nY) * dy
    xrdata_pad = xrdata_pad.assign_coords(X=x_coords, Y=y_coords)

    # -- 5) Perform rotation on each data variable
    xrdata_rot = xrdata_pad.copy(deep=True)
    for var in xrdata_pad.data_vars:
        arr = xrdata_pad[var].values.astype(float)
        # resize=False: maintain padded grid size
        rotated_arr = rotate(arr, rotation_angle, resize=False, cval=global_min)
        xrdata_rot[var].values = rotated_arr

    # -- 6) Re-center coordinates to maintain original dataset center
    new_cx = xrdata_rot.X.mean().item()
    new_cy = xrdata_rot.Y.mean().item()
    xrdata_rot = xrdata_rot.assign_coords(
        X=xrdata_rot.X + (orig_cx - new_cx),
        Y=xrdata_rot.Y + (orig_cy - new_cy)
    )

    return xrdata_rot



# -

rotated_ds = rotateXY_ds(FTS_40mK_202405061Txdvd_100_0001, rotation_angle=5.5)
rotated_ds.Z_fwd.plot()

# +
rotated_ds.sel(X=slice(0.0E-8,6.4E-8),Y=slice(0.05E-8,6.4E-8)).Z_fwd.plot()


ds_2d = rotated_ds.sel(X=slice(0.05E-8,1.05E-8),Y=slice(0.05E-8,1.05E-8)).copy()
# -

#FTS_40mK_202405061Txdvd_100_0001.Z_fwd.plot()
ds_2d = FTS_40mK_202405061Txdvd_100_0001.sel(X=slice(0.0E-8,6.4E-8),Y=slice(0.00E-8,6.4E-8)).copy()

#FTS_40mK_202405061Txdvd_100_0001
ds_2d

# + editable=true slideshow={"slide_type": ""}
from matplotlib_scalebar.scalebar import ScaleBar

plt.rcParams.update({'font.size': 12})  

fig, ax = plt.subplots(figsize = (4,4))

isns.imshow(plane_fit_surface_xr(ds_2d).Z_fwd
            -plane_fit_surface_xr(ds_2d).Z_fwd.min(),
            ax =ax, robust=True, perc = (0.2,99.95),
            #cmap ='YlGnBu_r',
            cmap ='copper')
# Add a scale bar manually using matplotlib's ScaleBar

scalebar = ScaleBar(40/1024, units="nm", location='lower right', 
                    font_properties="Arial", scale_loc='top', 
                    width_fraction=0.02, pad=1.0, length_fraction=0.3, 
                    color='white')

ax.add_artist(scalebar)



# Save the figure as SVG
# save high resolution Zoom in 
#fig.savefig('high_resolutionTopo.svg', format='svg')
#fig.savefig('high_resolutionTopo.png', format='png')

fig.savefig('high_resolutionTopo_large.svg', format='svg')
fig.savefig('high_resolutionTopo_large.png', format='png')
# +
#FTS_1T_240505_R30_007.Z_fwd.plot()
ds_2d_fft = twoD_FFT_xr(ds_2d, complex_output = False)#.plot()
ds_2d_fft.attrs = ds_2d.attrs


ds_2d_fft

# +
from matplotlib_scalebar.scalebar import ScaleBar

plt.rcParams.update({'font.size': 12})  

fig, ax = plt.subplots(figsize = (4,4))

# Define ref_q01overnm (example: ref_a0nm = 1.0)
ref_q01overnm = 1 / (ds_2d_fft.ref_a0nm*1E-9)
# Set zoom_in value (example: 2)
zoom_in = 1.5
zoom_range = ref_q01overnm * zoom_in

# Get coordinate information (freq_X, freq_Y)
freq_X = ds_2d_fft.coords['freq_X']
freq_Y = ds_2d_fft.coords['freq_Y']

# Filter the central area based on the zoom_range
ds_2d_fft_filtered= ds_2d_fft.where(
    (np.abs(freq_X) <= zoom_range) & (np.abs(freq_Y) <= zoom_range), drop=True)

isns.imshow(ds_2d_fft_filtered.Z_fwd_fft , ax =ax, robust=True, perc = (52,99.90),cmap ='Greys')
# Add a scale bar manually using matplotlib's ScaleBar
scalebar = ScaleBar(40/1024, units="nm", 
                    location='lower left',
                    font_properties="Arial",
                    scale_loc='top',
                    pad = 1.0, color ='Black',
                    width_fraction=0.02,
                    length_fraction=0.3)
# show scale bar 
# ax.add_artist(scalebar)

# not to show scale bar 
# ax.add_artist(scalebar)


# Save the figure as SVG
#fig.savefig('ds_2d_fft_filtered.svg', format='svg')
fig.savefig('ds_2d_fft_filtered.png', format='png')



fig.savefig('ds_2d_fft_filtered_large.svg', format='svg')
fig.savefig('ds_2d_fft_filtered_large.png', format='png')
# +
import numpy as np
import matplotlib.pyplot as plt
import seaborn_image as isns

# Set global font size for all text elements
plt.rcParams.update({'font.size': 12})

# Optional: apply seaborn-image context style
isns.set_context('notebook')

# Create a square figure and axis for the 2D FFT
fig, ax = plt.subplots(figsize=(4, 4))

# -----------------------------------------------------------------------------
# (1) Compute reciprocal lattice reference frequency (ref_q01overnm) in 1/m
# -----------------------------------------------------------------------------
# ds_2d_fft.ref_a0nm holds the lattice constant in nanometers (e.g., 0.38 nm)
# Convert to meters and take reciprocal to get reference frequency
ref_q01overnm = 1 / (ds_2d_fft.ref_a0nm * 1e-9)      # [m^-1]

# Specify how much to zoom in around the center (e.g., 1.5×)
zoom_in = 1.5
zoom_range = ref_q01overnm * zoom_in                # [m^-1]

# -----------------------------------------------------------------------------
# (2) Extract the frequency coordinate arrays
# -----------------------------------------------------------------------------
freq_X = ds_2d_fft.coords['freq_X']                 # x-axis frequency [m^-1]
freq_Y = ds_2d_fft.coords['freq_Y']                 # y-axis frequency [m^-1]

# -----------------------------------------------------------------------------
# (3) Filter dataset to keep only the central zoomed-in region
# -----------------------------------------------------------------------------
ds_2d_fft_filtered = ds_2d_fft.where(
    (np.abs(freq_X) <= zoom_range) &
    (np.abs(freq_Y) <= zoom_range),
    drop=True
)

# -----------------------------------------------------------------------------
# (4) Calculate pixel spacing in reciprocal units (1/m) for scalebar
# -----------------------------------------------------------------------------
pixel_freq = float(freq_X[1] - freq_X[0])           # [m^-1] per pixel

# -----------------------------------------------------------------------------
# (5) Display the filtered 2D FFT and automatically add a scalebar
# -----------------------------------------------------------------------------
im = isns.imshow(
    ds_2d_fft_filtered.Z_fwd_fft,   # 2D FFT magnitude to plot
    ax=ax,
    robust=True,
    perc=(52, 99.90),
    cmap='Greys',
    dx=pixel_freq,                  # pixel size in reciprocal units (1/m)
    units='1/m',                    # must specify units when dx is given
    dimension='si-reciprocal'       # correct reciprocal SI dimension
)

# -----------------------------------------------------------------------------
# (6) Customize scalebar appearance via seaborn-image
# -----------------------------------------------------------------------------
isns.set_scalebar(
    color='black',            # bar color
    location='lower left',    # bar position
    width_fraction=0.02,      # thickness relative to image width
    length_fraction=0.3,      # length relative to image width
    scale_loc='top',          # position of scale label (number & unit)
    box_alpha=0               # no background box behind the bar
)

# -----------------------------------------------------------------------------
# (7) Add title only (axis labels omitted as requested)
# -----------------------------------------------------------------------------
#ax.set_title('2D FFT Magnitude')

# -----------------------------------------------------------------------------
# (8) Save the figure in multiple formats (PNG and SVG, normal & large)
# -----------------------------------------------------------------------------
plt.tight_layout()
fig.savefig('fft_reciprocal_scalebar.png', dpi=300)
fig.savefig('fft_reciprocal_scalebar.svg', format='svg')
fig.savefig('fft_reciprocal_scalebar_large.png', dpi=300)
fig.savefig('fft_reciprocal_scalebar_large.svg', format='svg')

# -





# ### working folder back to Grid data set 
#

# +
# ### reselect working folder 
from PyQt5.QtWidgets import QApplication, QFileDialog
import sys

def select_folder():
    # Check if QApplication instance already exists
    app = QApplication.instance()
    
    # If no instance exists, create one
    if app is None:
        app = QApplication(sys.argv)
    
    # Create the file dialog to select a folder
    file_dialog = QFileDialog()
    folder_path = file_dialog.getExistingDirectory(None, "Select Folder")
    
    return folder_path

# Call the function to select a folder
selected_folder = select_folder()
if selected_folder:
    print(f"Selected folder: {selected_folder}")
else:
    print("No folder selected.")
# -


# ## Figure SI 1
# * compare 0T  002 (200nm) vs 2T 005 (20nm),

#  ### to compensate rotation_angle of original plot 
#  * use rotate_2D_xr or rotate_3D_xr after grid2xr data loading

# ### check HR vs multiple LDOS topo with B field
# > * **HR_topo_ds0 , HR_LDOS_ds0**
# > > * vs 
#

# #### check overlapped area using topography

keys_list

GS_topo_2T_003

GS_LDOS_2T_003

# +
# Extract only 'GS_topo_' keys from loaded_datasets
topo_keys = [key for key in loaded_datasets.keys() if key.startswith("GS_topo_")]

# Automatically assign variables for topo datasets
for key in topo_keys:
    globals()[key] = getattr(datasets, key)

# Merge selected topo datasets
merge_topos_Bfield = merge_multiple_2Ddatasets(
    [globals()[key] for key in topo_keys],  # Pass the extracted datasets
    dataset_names=[f"{key}.nc" for key in topo_keys],  # Generate corresponding dataset names
    interpolation=True,
    attrs_check=False,
)

# Display merged dataset
merge_topos_Bfield

# -

merge_topos_Bfield_overlaps = plot_overlapped_multiple_regions (merge_topos_Bfield, show_colorbar=False,attrs_same=False,)

# #### topo comparison 
# * check offset between0T and 2T topography
# * find 'xy_offset_topo'

merge_topo_0T002_N_2T003 =  merge_multiple_datasets( [GS_topo_0T_002, GS_topo_2T_003],
                                                    dataset_names=['GS_topo_0T_002.nc','GS_topo_2T_003.nc'],
                                                    channel_name='topography',
                                                    interpolation= True,   
                                                    attrs_check=False,)
plot_overlapped_multiple_regions(merge_topo_0T002_N_2T003)

# +
# Test the plane fit topo

#plane_fit_surface_xr(plane_fit_y_xr(GS_topo_0T_002)).topography.plot()
#plane_fit_surface_xr(plane_fit_y_xr(GS_topo_2T_003)).topography.plot()

# -


def resampling_skimage_with_matching_v2(ds1: xr.Dataset, 
                                        ds2: xr.Dataset, 
                                        ch: str,
                                        cmap='copper',
                                        correlation_cmap='cividis', 
                                        resampling_x_over_y_aspect_ratio=1,
                                        ds1_alpha=0.5,
                                        ds2_alpha=0.5):
    import numpy as np
    import matplotlib.pyplot as plt
    import xarray as xr
    from skimage.transform import resize
    from skimage import exposure
    from skimage.feature import match_template
    import re

    def extract_metadata(attrs):
        image_size = attrs.get("image_size", [None, None])
        size_x_nm = image_size[0] * 1e9 if image_size[0] is not None else None
        size_y_nm = image_size[1] * 1e9 if image_size[1] is not None else None

        title = attrs.get("title", "")
        bias_match = re.search(r'Bias\s*=\s*([\-\d.]+)\s*mV', title)
        current_match = re.search(r'I_t\s*=\s*([\-\d.]+)\s*pA', title)

        v_bias = f"{bias_match.group(1)} mV" if bias_match else "N/A"
        i_tunnel = f"{current_match.group(1)} pA" if current_match else "N/A"

        return size_x_nm, size_y_nm, v_bias, i_tunnel

    x_ds1, y_ds1 = ds1.coords['X'].values, ds1.coords['Y'].values
    x_ds2, y_ds2 = ds2.coords['X'].values, ds2.coords['Y'].values

    x_overlap_min = max(x_ds2.min(), x_ds1.min())
    x_overlap_max = min(x_ds2.max(), x_ds1.max())
    y_overlap_min = max(y_ds2.min(), y_ds1.min())
    y_overlap_max = min(y_ds2.max(), y_ds1.max())

    num_pixels_x_ds2 = np.sum((x_ds2 >= x_overlap_min) & (x_ds2 <= x_overlap_max))
    num_pixels_y_ds2 = np.sum((y_ds2 >= y_overlap_min) & (y_ds2 <= y_overlap_max))

    if resampling_x_over_y_aspect_ratio == 1:
        min_pixels = min(num_pixels_x_ds2, num_pixels_y_ds2)
        target_pixels_x = target_pixels_y = min_pixels
    else:
        if num_pixels_x_ds2 > num_pixels_y_ds2:
            target_pixels_y = num_pixels_y_ds2
            target_pixels_x = int(target_pixels_y * resampling_x_over_y_aspect_ratio)
        else:
            target_pixels_x = num_pixels_x_ds2
            target_pixels_y = int(target_pixels_x / resampling_x_over_y_aspect_ratio)

    resampled_ds1 = resize(ds1[ch].values, (target_pixels_y, target_pixels_x), anti_aliasing=True)
    resampled_xr = xr.DataArray(resampled_ds1,
                                dims=["Y", "X"],
                                coords={"X": np.linspace(x_overlap_min, x_overlap_max, target_pixels_x),
                                        "Y": np.linspace(y_overlap_min, y_overlap_max, target_pixels_y)})

    norm_ds2 = exposure.equalize_hist(ds2[ch].values)
    norm_resampled = exposure.equalize_hist(resampled_xr.values)
    result = match_template(norm_ds2, norm_resampled)
    y_match, x_match = np.unravel_index(np.argmax(result), result.shape)

    x_start, x_end = x_ds2[x_match], x_ds2[x_match + resampled_xr.shape[1] - 1]
    y_start, y_end = y_ds2[y_match], y_ds2[y_match + resampled_xr.shape[0] - 1]
    xy_offset = (x_start - x_overlap_min, y_start - y_overlap_min)

    matched_crop = ds2[ch].sel(X=slice(x_start, x_end), Y=slice(y_start, y_end))
    result_with_nan = xr.full_like(ds2[ch], np.nan)
    result_with_nan.loc[dict(X=matched_crop.coords['X'], Y=matched_crop.coords['Y'])] = matched_crop

    size_x_nm, size_y_nm, v_bias, i_tunnel = extract_metadata(ds1.attrs)

    fig, ax = plt.subplots(figsize=(6, 6))
    z1 = ds1[ch].values - np.nanmin(ds1[ch].values)
    z2 = ds2[ch].values - np.nanmin(ds2[ch].values)

    extent_ds2 = [x_ds2.min(), x_ds2.max(), y_ds2.min(), y_ds2.max()]
    extent_ds1 = [x_ds1.min(), x_ds1.max(), y_ds1.min(), y_ds1.max()]

    ax.imshow(z2, cmap=cmap, origin='lower', alpha=ds2_alpha, extent=extent_ds2)
    ax.imshow(z1, cmap=cmap, origin='lower', alpha=ds1_alpha, extent=extent_ds1)

    ax.add_patch(plt.Rectangle((x_ds2.min(), y_ds2.min()), x_ds2.ptp(), y_ds2.ptp(),
                               edgecolor='skyblue', facecolor='none', lw=2))
    ax.add_patch(plt.Rectangle((x_ds1.min(), y_ds1.min()), x_ds1.ptp(), y_ds1.ptp(),
                               edgecolor='orange', facecolor='none', lw=2))
    ax.add_patch(plt.Rectangle((x_start, y_start), x_end - x_start,
                               y_end - y_start, edgecolor='red', facecolor='none', lw=2))

    for dx0, dy0 in [(x_overlap_min, y_overlap_min), (x_overlap_max, y_overlap_min),
                     (x_overlap_min, y_overlap_max), (x_overlap_max, y_overlap_max)]:
        dx1 = dx0 + xy_offset[0]
        dy1 = dy0 + xy_offset[1]
        #ax.annotate('', xy=(dx1, dy1), xytext=(dx0, dy0),
        #            arrowprops=dict(arrowstyle='->', color='red', linestyle='dashed'))

    ax.text(x_ds1.min(), y_ds1.max(), '2T',
        color='orange', fontsize=20, ha='left', va='bottom',
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1'))
    
    ax.text(x_ds2.min(), y_ds2.max(), '0T',
            color='skyblue', fontsize=20, ha='left', va='bottom',
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1'))
    
    ax.text((x_start + x_end) / 2, y_start - 0.02 * y_ds2.ptp(), 'drift corrected position',
            color='red', fontsize=16, ha='center', va='top',
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1'))

    ax.set_xlim(x_ds2.min(), x_ds2.max())
    ax.set_ylim(y_ds2.min(), y_ds2.max())
    ax.axis('off')
    #ax.text(0.5, -0.05, f"{x_ds2.ptp()*1e9:.1f} nm x {y_ds2.ptp()*1e9:.1f} nm, V_bias = {v_bias}, I = {i_tunnel}",
    #        transform=ax.transAxes, fontsize=10, ha='center', va='top', color='black')

    fig.tight_layout()

    return result_with_nan, fig, xy_offset


def resampling_skimage_with_matching_v2(
    ds1: xr.Dataset,
    ds2: xr.Dataset,
    ch: str,
    cmap: str = 'copper',
    correlation_cmap: str = 'cividis',
    resampling_x_over_y_aspect_ratio: float = 1,
    ds1_alpha: float = 0.5,
    ds2_alpha: float = 0.5
):
    import numpy as np
    import matplotlib.pyplot as plt
    import xarray as xr
    from skimage.transform import resize
    from skimage import exposure
    from skimage.feature import match_template
    from matplotlib_scalebar.scalebar import ScaleBar
    import re

    def extract_metadata(attrs: dict):
        image_size = attrs.get("image_size", [None, None])
        size_x_nm = image_size[0] * 1e9 if image_size[0] is not None else None
        size_y_nm = image_size[1] * 1e9 if image_size[1] is not None else None

        title = attrs.get("title", "")
        bias_match = re.search(r'Bias\s*=\s*([\-\d.]+)\s*mV', title)
        current_match = re.search(r'I_t\s*=\s*([\-\d.]+)\s*pA', title)

        v_bias = f"{bias_match.group(1)} mV" if bias_match else "N/A"
        i_tunnel = f"{current_match.group(1)} pA" if current_match else "N/A"

        return size_x_nm, size_y_nm, v_bias, i_tunnel

    # Extract coordinates from datasets
    x1, y1 = ds1.coords['X'].values, ds1.coords['Y'].values
    x2, y2 = ds2.coords['X'].values, ds2.coords['Y'].values

    # Determine overlapping region
    xmin, xmax = max(x1.min(), x2.min()), min(x1.max(), x2.max())
    ymin, ymax = max(y1.min(), y2.min()), min(y1.max(), y2.max())

    # Count pixels in overlap region for ds2
    nx = np.sum((x2 >= xmin) & (x2 <= xmax))
    ny = np.sum((y2 >= ymin) & (y2 <= ymax))

    # Decide target pixels based on aspect ratio
    if resampling_x_over_y_aspect_ratio == 1:
        tx = ty = min(nx, ny)
    else:
        if nx > ny:
            ty = ny
            tx = int(ty * resampling_x_over_y_aspect_ratio)
        else:
            tx = nx
            ty = int(tx / resampling_x_over_y_aspect_ratio)

    # Resample ds1 channel array
    arr1 = resize(ds1[ch].values, (ty, tx), anti_aliasing=True)
    ds1_rs = xr.DataArray(
        arr1,
        dims=["Y", "X"],
        coords={
            "X": np.linspace(xmin, xmax, tx),
            "Y": np.linspace(ymin, ymax, ty)
        }
    )

    # Normalize and perform template matching
    m1 = exposure.equalize_hist(ds2[ch].values)
    m2 = exposure.equalize_hist(ds1_rs.values)
    res = match_template(m1, m2)
    i0, j0 = np.unravel_index(np.argmax(res), res.shape)

    # Crop matched region from ds2
    x0, x1_ = x2[j0], x2[j0 + ds1_rs.shape[1] - 1]
    y0, y1_ = y2[i0], y2[i0 + ds1_rs.shape[0] - 1]
    offset = (x0 - xmin, y0 - ymin)

    crop = ds2[ch].sel(X=slice(x0, x1_), Y=slice(y0, y1_))
    result = xr.full_like(ds2[ch], np.nan)
    result.loc[dict(X=crop.coords['X'], Y=crop.coords['Y'])] = crop

    # Extract metadata for annotations
    sx_nm, sy_nm, vbias, itunnel = extract_metadata(ds1.attrs)

    # Plot overlay of ds2 and ds1
    fig, ax = plt.subplots(figsize=(6, 6))
    z2 = ds2[ch].values - np.nanmin(ds2[ch].values)
    z1 = ds1[ch].values - np.nanmin(ds1[ch].values)
    ax.imshow(
        z2,
        cmap=cmap,
        origin='lower',
        alpha=ds2_alpha,
        extent=[x2.min(), x2.max(), y2.min(), y2.max()]
    )
    ax.imshow(
        z1,
        cmap=cmap,
        origin='lower',
        alpha=ds1_alpha,
        extent=[x1.min(), x1.max(), y1.min(), y1.max()]
    )

    # Draw bounding boxes
    ax.add_patch(plt.Rectangle((x2.min(), y2.min()), x2.ptp(), y2.ptp(),
                               edgecolor='skyblue', facecolor='none', lw=2))
    ax.add_patch(plt.Rectangle((x1.min(), y1.min()), x1.ptp(), y1.ptp(),
                               edgecolor='orange', facecolor='none', lw=2))
    ax.add_patch(plt.Rectangle((x0, y0), x1_ - x0, y1_ - y0,
                               edgecolor='red', facecolor='none', lw=2))

    # Annotate with labels (no bounding box around text)
    ax.text(x1.min(), y1.max(), '2T', color='orange', fontsize=20,
            ha='left', va='bottom')
    ax.text(x2.min(), y2.max(), '0T', color='skyblue', fontsize=20,
            ha='left', va='bottom')
    ax.text((x0 + x1_) / 2, y0 - 0.02 * y2.ptp(), 'drift corrected position',
            color='red', fontsize=16, ha='center', va='top')

    # Remove axes
    ax.set_xlim(x2.min(), x2.max())
    ax.set_ylim(y2.min(), y2.max())
    ax.axis('off')

    # Add scale bar based on ds2 spacing (display in nanometers)
    pixel_size_nm = ds2.attrs['X_spacing'] * 1e9  # nanometers per pixel
    sb = ScaleBar(
        pixel_size_nm,
        units='m',
        length_fraction=0.2,
        location='lower right',
        pad=0.5,
        border_pad=0.5,
        color='white'
    )
    ax.add_artist(sb)

    fig.tight_layout()
    return result, fig, offset



# +

GS_topo_0T_002 = plane_fit_surface_xr(plane_fit_y_xr(GS_topo_0T_002))
GS_topo_2T_003 = plane_fit_surface_xr(plane_fit_y_xr(GS_topo_2T_003))
# ds1 (small area, high resolution), ds2(large area, low resolution)
resampled_result_topo_0T002_N_2T003,fig, xy_offset_topo = resampling_skimage_with_matching_v2(GS_topo_2T_003, GS_topo_0T_002, ch='topography', cmap='copper')#,ds1_alpha=1, ds2_alpha=0.5)

fig.savefig('LDOS_resampling_location_check.svg', format='svg')
fig.savefig('LDOS_resampling_location_check.png', format='png', dpi=600)
# -


GS_topo_2T_003


# ####   LDOS  xy offset searching with respect to bias_mV 
# * find xy offset of bias_mV dependent LDOS maps between 0T & 2T grid data
# * 3D scatter plot (plotlt) of xy_offset_df
# * 2D (XY projection) scatter plot with KDE 

GS_LDOS_2T_003

# +
# GS_LDOS_0T_002
# GS_LDOS_2T_003

# +
# Define the range for bias_mV
bias_range = np.arange(-2.4, 2.4 + 0.1, 0.1)
bias_range

# Create an empty DataFrame to store the results
xy_offset_df = pd.DataFrame(columns=['bias_mV', 'x_offset', 'y_offset'])

# Loop through the bias values and run the function
for bias in bias_range:
    # Select the corresponding LDOS map for each bias value
    GS_LDOS_0T_002_bias = GS_LDOS_0T_002[['LDOS']].copy().rename({'LDOS': 'LDOSmap'}).sel(bias_mV=bias, method='nearest')
    GS_LDOS_2T_003_bias = GS_LDOS_2T_003[['LDOS']].copy().rename({'LDOS': 'LDOSmap'}).sel(bias_mV=bias, method='nearest')
    
    # Run the resampling function and get the xy_offset
    # ds1 (small area, high resolution), ds2(large area, low resolution)
    _, _, xy_offset = resampling_skimage_with_matching_v2(
        GS_LDOS_2T_003_bias,GS_LDOS_0T_002_bias, ch='LDOSmap', cmap='coolwarm', correlation_cmap='cividis')
    
    # Create a DataFrame for the current result
    current_result = pd.DataFrame({
        'bias_mV': [bias],
        'x_offset': [xy_offset[0]],  # Assuming xy_offset is a tuple (x_offset, y_offset)
        'y_offset': [xy_offset[1]]
    })
    
    # Concatenate the current result with the existing DataFrame
    xy_offset_df = pd.concat([xy_offset_df, current_result], ignore_index=True)

# Display the final DataFrame
xy_offset_df.head()
xy_offset_df

# +
import plotly.express as px

# Create a 3D scatter plot using plotly
fig = px.scatter_3d(
    xy_offset_df, 
    x='x_offset', 
    y='y_offset', 
    z='bias_mV', 
    color='bias_mV', 
    color_continuous_scale='viridis',
    opacity=0.5, 
    title='3D Scatter Plot of X, Y Offsets with Bias Variations'
)


# Update the axis labels and figure size
fig.update_layout(
    width=600,  # Set the width of the figure
    height=600,  # Set the height of the figure
    scene=dict(
        xaxis_title='X Offset',
        yaxis_title='Y Offset',
        zaxis_title='Bias (mV)'
    )
)
# Show the interactive plot
fig.show()
# -


xy_offset, closest_bias_mV = get_representative_xy_offset(xy_offset_df)

print ('xy_offset_topo: ', xy_offset_topo)
print ('xy_offset_LDOS_avg: ', xy_offset)
#xy_offset

# ###  apply_drift_offset_correction_after_resampled_cropping
# * Update larger grid file
# * cropp corresponding area ( applying XY offset between two grid data)
# * use 'xy_offset_topo' or 'xy_offset'
#  

resampled_result_topo_0T002_N_2T003#.plot()
#GS_LDOS_0T_002_bias

# +
# GS_LDOS_0T_002
# GS_LDOS_2T_003

updated_GS_LDOS_0T002_N_2T003 = apply_drift_offset_correction_after_resampled_cropping(resampled_result_topo_0T002_N_2T003, 
                                                                                     GS_LDOS_0T_002, 
                                                                                     xy_offset_topo)
# cropped and XY adjusted area in larger FOV
updated_GS_LDOS_0T002_N_2T003.to_netcdf('updated_GS_LDOS_0T002_N_2T003.nc')

## in this case origina & updated version is same 
## sse the GS_LDOS_0505_1T_003 & GS_LDOS_0505_2T_002
# -
resampled_result_topo_0T002_N_2T003.where(resampled_result_topo_0T002_N_2T003.notnull(), drop=True)


hv.extension('bokeh')
#hv_bias_mV_slicing(GS_LDOS_2T_003, ch = 'LDOS' ,frame_width=300)#.opts(clim = (0,0.5E-11))
hv_bias_mV_slicing(updated_GS_LDOS_0T002_N_2T003, ch = 'LDOS' ,frame_width=300)#.opts(clim = (0,0.5E-11))

# +
#hv_XY_slicing(GS_LDOS_2T_005, ch = 'LDOS' ,frame_width=500,slicing='Y' ).opts(clim = (0,1.5E-10))
# -

resampled_result_topo_0T002_N_2T003

# +
updated_GS_topo_0T002_N_2T003 = apply_drift_offset_correction_after_resampled_cropping(resampled_result_topo_0T002_N_2T003, 
                                                                                     GS_topo_0T_002, 
                                                                                     xy_offset_topo)

updated_GS_topo_0T002_N_2T003.to_netcdf('updated_GS_topo_0T002_N_2T003.nc')

# +
#GS_topo_2T_003

# +
# assign the FOV  & comparea topography

cropped_large_FOV = updated_GS_topo_0T002_N_2T003.copy()  
small_FOV =  plane_fit_surface_xr(plane_fit_y_xr(updated_GS_topo_0T002_N_2T003.copy()))


compare_cropped_large_and_small_area(cropped_large_FOV, 
                                     small_FOV,
                                     perc=(2, 98),
                                     bias_mV_ref=0,
                                     alpha_large=0.3,      
                                     alpha_small=1, cmap_large='copper', cmap_small='copper',
                                     channel='topography')

# +
import matplotlib.pyplot as plt
import numpy as np

def compare_cropped_large_and_small_area(cropped_large_area, small_area, perc=(2, 98), 
                                         bias_mV_ref=0, channel='LDOS', alpha_large=0.5, 
                                         alpha_small=1, cmap_large='Blues', cmap_small='viridis'):
    """
    Compare two datasets (cropped_large_area and small_area). Automatically handles 2D data.
    
    Parameters:
        cropped_large_area (xarray.Dataset): The large field of view dataset (cropped).
        small_area (xarray.Dataset): The smaller field of view dataset.
        perc (tuple): Percentile for robust scaling (default=(2, 98)).
        bㅊias_mV_ref (float): Bias voltage for selecting the slice, ignored for 2D data.
        channel (str): Data channel to plot (default='LDOS').
        alpha_large (float): Transparency level for the large FOV (default=0.5).
        alpha_small (float): Transparency level for the small FOV (default=1).
        cmap_large (str): Colormap for large FOV (default='Blues').
        cmap_small (str): Colormap for small FOV (default='viridis').
    """
    
    # Function to calculate vmin and vmax based on perc
    def get_vmin_vmax(data, perc):
        vmin, vmax = np.percentile(data.values.flatten(), perc)
        return vmin, vmax
    
    # Check if the channel data has a 'bias_mV' dimension
    if 'bias_mV' in cropped_large_area[channel].dims:
        # If the data has a bias_mV axis, select data based on bias_mV_ref
        large_data = cropped_large_area[channel].sel(bias_mV=bias_mV_ref)
        small_data = small_area[channel].sel(bias_mV=bias_mV_ref)
    else:
        # If there is no bias_mV axis, assume the data is 2D
        large_data = cropped_large_area[channel]
        small_data = small_area[channel]
    
    # Get vmin and vmax for each dataset
    vmin_large, vmax_large = get_vmin_vmax(large_data, perc)
    vmin_small, vmax_small = get_vmin_vmax(small_data, perc)
    
    # Create figure and axes for three subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot the first dataset (larger FOV)
    large_data.plot(ax=axs[0], vmin=vmin_large, vmax=vmax_large, cmap=cmap_large, robust=True)
    axs[0].set_title(f'(cropped) large field of view \n at channel: {channel} \n perc: {perc}')
    axs[0].set_aspect('equal')

    # Plot the second dataset (smaller FOV)
    small_data.plot(ax=axs[1], vmin=vmin_small, vmax=vmax_small, cmap=cmap_small, robust=True)
    axs[1].set_title(f'small field of view \n at channel: {channel} \n perc: {perc}')
    axs[1].set_aspect('equal')

    # Overlay both datasets on the third plot
    small_data.plot(ax=axs[2], vmin=vmin_small, vmax=vmax_small, cmap=cmap_small, alpha=alpha_small)
    large_data.plot(ax=axs[2], vmin=vmin_large, vmax=vmax_large, cmap=cmap_large, alpha=alpha_large)
    axs[2].set_title(f'Overlap: large alpha={alpha_large}, small alpha={alpha_small}')
    axs[2].set_aspect('equal')

    # Adjust layout
    plt.tight_layout()
    plt.show()

# Example usage:
# compare_cropped_large_and_small_area(cropped_large_FOV, small_FOV, perc=(2, 98), bias_mV_ref=0, channel='LDOS')
# compare_cropped_large_and_small_area(cropped_large_FOV, small_FOV, perc=(2, 98), channel='topography')  # For 2D data



# -

compare_cropped_large_and_small_area(updated_GS_LDOS_0T002_N_2T003, 
                                     GS_LDOS_2T_003,
                                     perc=(2, 98),
                                     bias_mV_ref=0,
                                     alpha_large=0.3,      
                                     alpha_small=1,
                                     channel='LDOS')


# +
import matplotlib.pyplot as plt
import numpy as np
import seaborn_image as isns
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def compare_cropped_large_and_small_area_v2(cropped_large_area, small_area, perc=(2, 98), 
                                             bias_mV_ref=0, channel='LDOS', 
                                             alpha_large=0.5, alpha_small=1,
                                             cmap_large='Blues', cmap_small='viridis',
                                             scalebar_length_nm=20,
                                             scalebar_colors=('lightgray', 'white', 'white')):

    def get_vmin_vmax(data, perc):
        return np.percentile(data.values.flatten(), perc)

    if 'bias_mV' in cropped_large_area[channel].dims:
        large_data = cropped_large_area[channel].sel(bias_mV=bias_mV_ref)
        small_data = small_area[channel].sel(bias_mV=bias_mV_ref)
    else:
        large_data = cropped_large_area[channel]
        small_data = small_area[channel]

    large_data_nS = large_data * 1e9
    small_data_nS = small_data * 1e9

    vmin_large, vmax_large = get_vmin_vmax(large_data_nS, perc)
    vmin_small, vmax_small = get_vmin_vmax(small_data_nS, perc)
    vmin_combined = min(vmin_large, vmin_small)
    vmax_combined = max(vmax_large, vmax_small)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    def add_scalebar(ax, data, length_nm=20, color='white'):
        x_len = float(data.coords['X'].max() - data.coords['X'].min())
        bar_x_start = float(data.coords['X'].min()) + 0.05 * x_len
        bar_y = float(data.coords['Y'].min()) + 0.05 * x_len
        bar_x_end = bar_x_start + length_nm * 1e-9
        ax.plot([bar_x_start, bar_x_end], [bar_y, bar_y], color=color, lw=2, zorder=10)
        ax.text((bar_x_start + bar_x_end)/2, bar_y + 0.005 * x_len,
                f'{length_nm} nm', color=color, ha='center', va='bottom', fontsize=12, zorder=10)

    extent = [
        float(large_data.coords['X'].min()), float(large_data.coords['X'].max()),
        float(large_data.coords['Y'].min()), float(large_data.coords['Y'].max())
    ]

    # === 1. 0T map ===
    im0 = isns.imshow(large_data_nS.values, ax=axs[0], cmap=cmap_large,
                      vmin=vmin_combined, vmax=vmax_combined, extent=extent,
                      orientation='horizontal', cbar_label=" LDOS (nS)")
    axs[0].set_title("Zero bias conductance map (0T)")
    axs[0].set_xticks([]); axs[0].set_yticks([]); axs[0].set_xlabel(""); axs[0].set_ylabel("")
    add_scalebar(axs[0], large_data_nS, color=scalebar_colors[0])

    # === 2. 2T map ===
    im1 = isns.imshow(small_data_nS.values, ax=axs[1], cmap=cmap_small,
                      vmin=vmin_combined, vmax=vmax_combined, extent=extent,
                      orientation='horizontal', cbar_label=" LDOS (nS)")
    axs[1].set_title("Zero bias conductance map (2T)")
    axs[1].set_xticks([]); axs[1].set_yticks([]); axs[1].set_xlabel(""); axs[1].set_ylabel("")
    add_scalebar(axs[1], small_data_nS, color=scalebar_colors[1])

    # === 3. Overlay ===
    axs[2].set_title("Overlay")
    axs[2].set_xticks([]); axs[2].set_yticks([]); axs[2].set_xlabel(""); axs[2].set_ylabel("")

    # ✅ dummy colorbar (using ScalarMappable, no imshow)
    norm = mcolors.Normalize(vmin=vmin_combined, vmax=vmax_combined)
    sm = cm.ScalarMappable(cmap='gray', norm=norm)
    sm.set_array([])
    cbar_dummy = fig.colorbar(sm, ax=axs[2], orientation='horizontal', fraction=0.05, pad=0.1)
    cbar_dummy.ax.set_visible(False)

    # ✅ actual overlay image
    axs[2].imshow(large_data_nS.values, cmap=cmap_large,
                  vmin=vmin_combined, vmax=vmax_combined,
                  origin='lower', extent=extent, alpha=alpha_large, zorder=1)
    axs[2].imshow(small_data_nS.values, cmap=cmap_small,
                  vmin=vmin_combined, vmax=vmax_combined,
                  origin='lower', extent=extent, alpha=alpha_small, zorder=2)
    add_scalebar(axs[2], large_data_nS, color=scalebar_colors[2])

    return fig



# -

fig= compare_cropped_large_and_small_area_v2(
    cropped_large_area=updated_GS_LDOS_0T002_N_2T003,
    small_area=GS_LDOS_2T_003,
    channel='LDOS',
    cmap_large='Blues',
    cmap_small='viridis',
    alpha_large=0.3,
    alpha_small=0.5,
    scalebar_length_nm=20,scalebar_colors=('gray', 'white', 'white')
)

# +
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.cm as cm

def compare_cropped_large_and_small_area_v2(cropped_large_area, small_area, perc=(2, 98), 
                                             bias_mV_ref=0, channel='LDOS', 
                                             alpha_large=0.5, alpha_small=1,
                                             cmap_large='Blues', cmap_small='viridis',
                                             scalebar_length_nm=20,
                                             scalebar_colors=('lightgray', 'white', 'white')):

    def get_vmin_vmax(data, perc):
        return np.percentile(data.values.flatten(), perc)

    # 데이터 선택
    if 'bias_mV' in cropped_large_area[channel].dims:
        large_data = cropped_large_area[channel].sel(bias_mV=bias_mV_ref)
        small_data = small_area[channel].sel(bias_mV=bias_mV_ref)
    else:
        large_data = cropped_large_area[channel]
        small_data = small_area[channel]

    # 단위 변환: V/A → nS
    large_data_nS = large_data * 1e9
    small_data_nS = small_data * 1e9

    # 색상 범위
    vmin_large, vmax_large = get_vmin_vmax(large_data_nS, perc)
    vmin_small, vmax_small = get_vmin_vmax(small_data_nS, perc)
    vmin_combined = min(vmin_large, vmin_small)
    vmax_combined = max(vmax_large, vmax_small)

    # 공통 extent
    extent = [
        float(large_data.coords['X'].min()), float(large_data.coords['X'].max()),
        float(large_data.coords['Y'].min()), float(large_data.coords['Y'].max())
    ]

    # Figure 및 layout 정의
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), 
                            gridspec_kw={'width_ratios': [1, 1, 1]},
                            constrained_layout=True)

    # Scale bar 추가 함수
    def add_scalebar(ax, data, length_nm=20, color='white'):
        x_len = float(data.coords['X'].max() - data.coords['X'].min())
        bar_x_start = float(data.coords['X'].min()) + 0.05 * x_len
        bar_y = float(data.coords['Y'].min()) + 0.05 * x_len
        bar_x_end = bar_x_start + length_nm * 1e-9
        ax.plot([bar_x_start, bar_x_end], [bar_y, bar_y], color=color, lw=2)
        ax.text((bar_x_start + bar_x_end)/2, bar_y + 0.005 * x_len,
                f'{length_nm} nm', color=color, ha='center', va='bottom', fontsize=10)

    # 1. 0T image
    im0 = axs[0].imshow(large_data_nS.values, cmap=cmap_large,
                        vmin=vmin_combined, vmax=vmax_combined,
                        origin='lower', extent=extent, aspect='equal')
    axs[0].set_title("Zero bias conductance map (0T)")
    axs[0].set_xticks([]); axs[0].set_yticks([])
    add_scalebar(axs[0], large_data_nS, color=scalebar_colors[0])
    cbar0 = fig.colorbar(im0, ax=axs[0], orientation='horizontal', fraction=0.05, pad=0.1)
    cbar0.set_label("LDOS (nA/V)")

    # 2. 2T image
    im1 = axs[1].imshow(small_data_nS.values, cmap=cmap_small,
                        vmin=vmin_combined, vmax=vmax_combined,
                        origin='lower', extent=extent, aspect='equal')
    axs[1].set_title("Zero bias conductance map (2T)")
    axs[1].set_xticks([]); axs[1].set_yticks([])
    add_scalebar(axs[1], small_data_nS, color=scalebar_colors[1])
    cbar1 = fig.colorbar(im1, ax=axs[1], orientation='horizontal', fraction=0.05, pad=0.1)
    cbar1.set_label("LDOS (nA/V)")

    # 3. Overlay image
    axs[2].set_title("Overlaid")
    axs[2].imshow(large_data_nS.values, cmap=cmap_large,
                  vmin=vmin_combined, vmax=vmax_combined,
                  origin='lower', extent=extent, aspect='equal', alpha=alpha_large)
    axs[2].imshow(small_data_nS.values, cmap=cmap_small,
                  vmin=vmin_combined, vmax=vmax_combined,
                  origin='lower', extent=extent, aspect='equal', alpha=alpha_small)
    axs[2].set_xticks([]); axs[2].set_yticks([])
    add_scalebar(axs[2], large_data_nS, color=scalebar_colors[2])

    # Dummy colorbar for alignment
    norm = mcolors.Normalize(vmin=vmin_combined, vmax=vmax_combined)
    sm = cm.ScalarMappable(norm=norm, cmap='gray')
    sm.set_array([])
    dummy_cbar = fig.colorbar(sm, ax=axs[2], orientation='horizontal', fraction=0.05, pad=0.1)
    dummy_cbar.ax.set_visible(False)

    return fig



# -

fig= compare_cropped_large_and_small_area_v2(
    cropped_large_area=updated_GS_LDOS_0T002_N_2T003,
    small_area=GS_LDOS_2T_003,
    channel='LDOS',
    cmap_large='Blues',
    cmap_small='viridis',
    alpha_large=0.7,
    alpha_small=0.4,
    scalebar_length_nm=20,scalebar_colors=('gray', 'white', 'white')
)



updated_GS_LDOS_0T002_N_2T003

GS_LDOS_2T_003





compare_cropped_large_and_small_area(updated_GS_LDOS_0T002_N_2T003, 
                                     GS_LDOS_2T_003,
                                     perc=(2, 100),
                                     bias_mV_ref=0,
                                     alpha_large=0.3,      
                                     alpha_small=1,
                                     channel='LDOS')


grid_data_dim_slicing(GS_LDOS_2T_003)

# + [markdown] jp-MarkdownHeadingCollapsed=true
# #### topo comparison again after drift compensation
# -

merge_topo_0T002_N_2T003 =  merge_multiple_2Ddatasets( [updated_GS_topo_0T002_N_2T003, GS_topo_2T_003], 
                                                                    dataset_names= ['updated_GS_topo_0T002_N_2T003','GS_topo_2T_003'], channel_name='topography',interpolation= True,    attrs_check=False,)
plot_overlapped_multiple_regions(merge_topo_0T002_N_2T003)



merged_LDOS_0T_002_N_2T_003 = merge_multiple_3Ddataset([updated_GS_LDOS_0T002_N_2T003, GS_LDOS_2T_003], interpolation=False, attrs_check=False)
#merged_LDOS_0T_002_N_2T_003.to_netcdf('merged_LDOS_0T_002_N_2T_003.nc')

# %matplotlib inline
merged_LDOS_0T_002_N_2T_003.LDOS_a2.sel(bias_mV=0,method ='nearest').plot()
plt.show()

merged_LDOS_0T_002_N_2T_003.LDOS_a1.dropna(dim='X', how='all').dropna(dim='Y', how='all').sel(bias_mV= 0, method ='nearest').plot(cmap ='cividis')
plt.show()

# %matplotlib  inline

# #### bais_zm

merged_LDOS_0T_002_N_2T_003 = merged_LDOS_0T_002_N_2T_003.where(
    (merged_LDOS_0T_002_N_2T_003.bias_mV > -2.0)&(
        merged_LDOS_0T_002_N_2T_003.bias_mV < 2.0), 
    drop = True)

#updated_GS_LDOS_0T_002_N_2T_005
#updated_GS_topo_0T002_N_2T005
updated_GS_LDOS_0T002_N_2T003

# #### use one 2T data and choose the line profile points
# * compare line profile of Topo & grid_at_bias_mV at the same position (offset corrected)

# # Line Profiles 

# ### select  line start & end points

# +
import numpy as np
import xarray as xr
import skimage.draw
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def plot_LDOS_map_and_profile(dataset, selected_points, ch='LDOS_smoothed', bias_mV_ref=0,
                              LDOS_cmap='viridis', line_cmap='gnuplot', perc=(0, 98)):
    """
    Plots three figures:
    1. An LDOS map for the specified channel with the line profile overlay.
    2. A 2D image of the LDOS line profile with bias (mV) on the x-axis and actual distance (nm) on the y-axis.
    3. A plot of the interpolated LDOS line profile at the specified bias (mV) with distance (nm) on the x-axis.

    Parameters:
    - dataset (xr.Dataset): The xarray dataset containing LDOS data.
    - selected_points (list of tuples): Two points given as real X, Y coordinates.
    - ch (str): The channel name in the dataset to use for mapping and profiling.
    - bias_mV_ref (int or float): The bias (mV) value at which to extract the interpolated profile.
    - LDOS_cmap (str): The colormap for the LDOS images.
    - line_cmap (str): The colormap for the gradient line connecting the selected points.
    - perc (tuple): Percentile range for image filtering.

    Returns:
    - fig (matplotlib.figure.Figure): The figure containing the plots.
    - interpolated_ds (xr.Dataset): The xarray dataset containing the interpolated LDOS line profile.
    """

    # If the bias_mV coordinates are in descending order, sort them in ascending order
    # to ensure the x-axis is displayed correctly.
    if dataset['bias_mV'].values[0] > dataset['bias_mV'].values[-1]:
        dataset = dataset.sortby('bias_mV')

    # Select the specified channel from the dataset.
    ldos_array = dataset[ch]

    # Convert the given start and end points into indices (order: Y, X).
    start_point = selected_points[0]
    end_point   = selected_points[1]
    start_point_index = (
        np.abs(ldos_array.Y.values - start_point[1]).argmin(),
        np.abs(ldos_array.X.values - start_point[0]).argmin()
    )
    end_point_index = (
        np.abs(ldos_array.Y.values - end_point[1]).argmin(),
        np.abs(ldos_array.X.values - end_point[0]).argmin()
    )

    # Generate an anti-aliased line between the two points.
    rr, cc, val = skimage.draw.line_aa(
        start_point_index[0], start_point_index[1],
        end_point_index[0],   end_point_index[1]
    )

    # Filter out indices that fall out of bounds.
    valid_indices = (
        (rr >= 0) & (rr < ldos_array.sizes['Y']) &
        (cc >= 0) & (cc < ldos_array.sizes['X'])
    )
    rr = rr[valid_indices]
    cc = cc[valid_indices]
    val = val[valid_indices]

    # Raise an error if no valid data points were generated.
    if len(rr) == 0 or len(cc) == 0:
        raise ValueError(
            "No valid data points generated from the selected points. "
            "Please check the input points."
        )

    # Interpolate the line profile: for each bias_mV slice, compute a weighted sum.
    ldos_interpolated = []
    for i in range(ldos_array.sizes['bias_mV']):
        ldos_slice = ldos_array.isel(bias_mV=i).values
        profile = np.sum(ldos_slice[rr, cc] * val[:, np.newaxis], axis=0)
        ldos_interpolated.append(profile)
    ldos_interpolated = np.array(ldos_interpolated).T  # Transpose to shape (distance, bias_mV)

    # --- 변경된 거리 계산: 시작점↔끝점 전체 길이를 기준으로 비율 분할 ---
    # start and end coordinates (m)
    x0 = dataset['X'].values[start_point_index[1]]
    y0 = dataset['Y'].values[start_point_index[0]]
    x1 = dataset['X'].values[end_point_index[1]]
    y1 = dataset['Y'].values[end_point_index[0]]
    total_dist_m = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)

    num_pts = len(rr)
    # 전체 거리만큼 0→total 거리로 균등 분할하여 각 포인트 위치 결정 (nm 단위)
    actual_distance = np.linspace(0, total_dist_m * 1e9, num_pts)

    # Convert the interpolation result to an xarray Dataset.
    interpolated_ds = xr.Dataset(
        {
            "LDOS_interpolated": (["distance", "bias_mV"], ldos_interpolated)
        },
        coords={
            "bias_mV":  ldos_array.bias_mV,
            "distance": actual_distance
        }
    )

    # Extract the interpolated profile at the specified bias_mV value.
    interpolated_profile = interpolated_ds.sel(
        bias_mV=bias_mV_ref, method='nearest'
    ).LDOS_interpolated.values

    # Create a figure with three subplots.
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # First subplot: LDOS map with the line profile overlay.
    ldos_min, ldos_max = np.nanpercentile(
        dataset.sel(bias_mV=bias_mV_ref, method='nearest')[ch].values,
        perc
    )
    dataset.sel(bias_mV=bias_mV_ref, method='nearest')[ch] \
           .plot(ax=ax1, cmap=LDOS_cmap, robust=True,
                 vmin=ldos_min, vmax=ldos_max)

    cmap_instance = plt.get_cmap(line_cmap)
    colors = cmap_instance(np.linspace(0, 1, num_pts))
    x_vals = np.linspace(start_point[0], end_point[0], num_pts)
    y_vals = np.linspace(start_point[1], end_point[1], num_pts)

    # Draw the gradient line connecting the selected points.
    for i in range(num_pts - 1):
        ax1.plot(x_vals[i:i+2], y_vals[i:i+2],
                 color=colors[i], lw=3, alpha=0.5)

    ax1.set_aspect('equal')
    ax1.set_title(f"LDOS Map with Line Profile\n(bias_mV={bias_mV_ref})")
    ax1.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{int(x*1e9)}")
    )
    ax1.yaxis.set_major_formatter(
        FuncFormatter(lambda y, pos: f"{int(y*1e9)}")
    )
    ax1.set_xlabel("X (nm)")
    ax1.set_ylabel("Y (nm)")

    # Second subplot: LDOS line profile as a 2D image.
    flipped = np.flipud(interpolated_ds.LDOS_interpolated.values)
    bias_min, bias_max = (
        interpolated_ds.bias_mV.min().item(),
        interpolated_ds.bias_mV.max().item()
    )
    im2 = ax2.imshow(
        flipped, aspect='auto', cmap=LDOS_cmap, origin='upper',
        extent=[bias_min, bias_max, 0, actual_distance[-1]],
        vmin=np.nanpercentile(flipped, perc[0]),
        vmax=np.nanpercentile(flipped, perc[1])
    )
    plt.colorbar(im2, ax=ax2, label='LDOS')

    for i in range(num_pts):
        ax2.plot(bias_max + 0.1, actual_distance[i],
                 'o', color=colors[i], markersize=4)

    ax2.set_xlabel("Bias (mV)")
    ax2.set_ylabel("Distance (nm)")
    ax2.yaxis.set_major_formatter(
        FuncFormatter(lambda y, pos: f"{int(y)}")
    )
    ax2.set_title("LDOS Line Profile")

    # Third subplot: Interpolated LDOS line profile.
    ax3.plot(actual_distance, interpolated_profile, color="blue", linestyle="-")
    ax3.set_title(f"Interpolated LDOS Line Profile\n(bias_mV={bias_mV_ref})")
    ax3.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{int(x)}")
    )
    ax3.set_xlabel("Distance (nm)")
    ax3.set_ylabel("LDOS")

    plt.tight_layout()
    plt.show()

    return fig, interpolated_ds



# -

# %matplotlib qt5
selected_points, fig = GUI_input2pts_and_selected_line(
    updated_GS_LDOS_0T002_N_2T003.sel(bias_mV=0, method='nearest'), 
    ch='LDOS')

# %matplotlib qt5
selected_points, fig = GUI_input2pts_and_selected_line(
    GS_LDOS_2T_003.sel(bias_mV=0, method='nearest'), 
    ch='LDOS')

selected_points1=selected_points.copy()

# + editable=true slideshow={"slide_type": ""}
# %matplotlib inline
fig1, interpolated_ds1 = plot_LDOS_map_and_profile(updated_GS_LDOS_0T002_N_2T003, selected_points, ch='LDOS',bias_mV_ref = 0.0, perc=(0,99.5))
fig1.savefig('interpolated_ds1_0T.svg', format='svg')
fig1.savefig('interpolated_ds1_0T.png', format='png', dpi=600)
fig2, interpolated_ds2 = plot_LDOS_map_and_profile(GS_LDOS_2T_003, selected_points, ch='LDOS',bias_mV_ref = 0, perc=(0, 95))
fig1.savefig('interpolated_ds2_2T.svg', format='svg')
fig1.savefig('interpolated_ds2_2T.png', format='png', dpi=600)


# -


# ### line profile layout change 

def plot_LDOS_map_and_profile_v2_1(
    dataset, selected_points, ch='LDOS_smoothed', bias_mV_ref=0,
    LDOS_cmap='viridis', line_cmap='gnuplot',
    ZBCM_perc=(2, 98), LP_perc=(2, 98),
    scalebar_length_nm=None, scalebar_color='white'):

    if dataset['bias_mV'].values[0] > dataset['bias_mV'].values[-1]:
        dataset = dataset.sortby('bias_mV')

    ldos_array = dataset[ch]
    start_idx = (np.abs(ldos_array.Y.values - selected_points[0][1]).argmin(),
                 np.abs(ldos_array.X.values - selected_points[0][0]).argmin())
    end_idx = (np.abs(ldos_array.Y.values - selected_points[1][1]).argmin(),
               np.abs(ldos_array.X.values - selected_points[1][0]).argmin())

    rr, cc, val = skimage.draw.line_aa(start_idx[0], start_idx[1], end_idx[0], end_idx[1])
    valid = (rr >= 0) & (rr < ldos_array.sizes['Y']) & (cc >= 0) & (cc < ldos_array.sizes['X'])
    rr, cc, val = rr[valid], cc[valid], val[valid]

    if len(rr) == 0:
        raise ValueError("No valid points on line.")

    ldos_interpolated = []
    for i in range(ldos_array.sizes['bias_mV']):
        ldos_slice = ldos_array.isel(bias_mV=i).values
        profile = np.sum(ldos_slice[rr, cc] * val[:, np.newaxis], axis=0)
        ldos_interpolated.append(profile)
    ldos_interpolated = np.array(ldos_interpolated).T

    x_coords = dataset['X'].values
    y_coords = dataset['Y'].values
    X_map = x_coords[cc]
    Y_map = y_coords[rr]
    x0, y0 = selected_points[0]
    x1, y1 = selected_points[1]
    dx, dy = x1 - x0, y1 - y0
    line_length = np.sqrt(dx**2 + dy**2)
    proj_dist = ((X_map - x0) * dx + (Y_map - y0) * dy) / line_length
    distance_nm = proj_dist * 1e9

    interpolated_ds = xr.Dataset(
        {"LDOS_interpolated": (["distance", "bias_mV"], ldos_interpolated)},
        coords={"distance": distance_nm, "bias_mV": ldos_array.bias_mV}
    )

    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3, figure=fig)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # 1. LDOS map
    ldos_map = dataset.sel(bias_mV=bias_mV_ref, method='nearest')[ch].values
    vmin_map, vmax_map = np.nanpercentile(ldos_map, ZBCM_perc)
    extent_map = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()]
    im1 = ax1.imshow(ldos_map, cmap=LDOS_cmap, origin='lower', extent=extent_map,
                     vmin=vmin_map, vmax=vmax_map, aspect='equal')

    num_segments = len(distance_nm)
    colors = plt.get_cmap(line_cmap)(np.linspace(0, 1, num_segments))
    x_line = np.linspace(x0, x1, num_segments)
    y_line = np.linspace(y0, y1, num_segments)
    for i in range(num_segments - 1):
        ax1.plot(x_line[i:i+2], y_line[i:i+2], color=colors[i], lw=3, alpha=0.6)

    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_title(f"LDOS Map with Line Profile \n (bias = {bias_mV_ref} mV)")
    plt.colorbar(im1, ax=ax1, orientation='vertical', fraction=0.035, pad=0.02, label='LDOS')

    # Scale bar
    x_len_nm = (x_coords[-1] - x_coords[0]) * 1e9
    bar_len_nm = scalebar_length_nm if scalebar_length_nm else 0.2 * x_len_nm
    bar_x_start = x_coords[0] + 0.05 * (x_coords[-1] - x_coords[0])
    bar_x_end = bar_x_start + (bar_len_nm * 1e-9)
    bar_y = y_coords[0] + 0.05 * (y_coords[-1] - y_coords[0])
    ax1.plot([bar_x_start, bar_x_end], [bar_y, bar_y], color=scalebar_color, lw=2)
    ax1.text((bar_x_start + bar_x_end)/2, bar_y + 0.005 * (y_coords[-1] - y_coords[0]),
             f"{bar_len_nm:.0f} nm", color=scalebar_color, ha='center', va='bottom', fontsize=10)

    # 2. Line profile
    flipped = np.flipud(interpolated_ds.LDOS_interpolated.values)
    bias_min = interpolated_ds.bias_mV.min().item()
    bias_max = interpolated_ds.bias_mV.max().item()
    extent_profile = [bias_min, bias_max, 0, distance_nm.max()]
    aspect_ratio = (bias_max - bias_min) / distance_nm.max()
    vmin_lp = np.nanpercentile(flipped, LP_perc[0])
    vmax_lp = np.nanpercentile(flipped, LP_perc[1])

    im2 = ax2.imshow(flipped, cmap=LDOS_cmap, origin='upper', extent=extent_profile,
                     vmin=vmin_lp, vmax=vmax_lp, aspect=aspect_ratio)
    plt.colorbar(im2, ax=ax2, orientation='vertical', fraction=0.035, pad=0.02, label='LDOS')

    for i in range(num_segments):
        ax2.plot(bias_max + 0.1, distance_nm[i], 'o', color=colors[i], markersize=4)

    ax2.set_xlabel("Bias (mV)")
    ax2.set_ylabel("Distance (nm)")
    ax2.set_title("LDOS Line Profile")

    plt.show()
    return fig, interpolated_ds



# +
fig1, interpolated_ds1 = plot_LDOS_map_and_profile_v2_1(updated_GS_LDOS_0T002_N_2T003,
                                                       selected_points, 
                                                       ch='LDOS',
                                                       bias_mV_ref = 0.0, 
                                                       ZBCM_perc=(0, 98), LP_perc=(0, 35),
                                                       scalebar_length_nm=20, scalebar_color='white',
                                                      )

fig1.savefig('interpolated_ds1_0T_v2_1.svg', format='svg')
fig1.savefig('interpolated_ds1_0T_v2_1.png', format='png', dpi=600)


fig2, interpolated_ds2 = plot_LDOS_map_and_profile_v2_1(GS_LDOS_2T_003,
                                                       selected_points,
                                                       ch='LDOS',
                                                       bias_mV_ref = 0,
                                                       ZBCM_perc=(0, 98), LP_perc=(0, 40),
                                                       scalebar_length_nm=20, scalebar_color='white',
                                                      )
fig2.savefig('interpolated_ds2_2T_v2_1.svg', format='svg')
fig2.savefig('interpolated_ds2_2T_v2_1.png', format='png', dpi=600)

# -

def plot_LDOS_map_and_profile_v2_2(dataset, selected_points, ch='LDOS_smoothed', bias_mV_ref=0,
                                   LDOS_cmap='viridis', line_cmap='gnuplot',
                                   ZBCM_perc=(2, 98), LP_perc=(2, 98),
                                   scalebar_length_nm=20, scalebar_color='white',
                                   num_interp_points=200):
    from scipy.interpolate import interp1d
    if dataset['bias_mV'].values[0] > dataset['bias_mV'].values[-1]:
        dataset = dataset.sortby('bias_mV')
    ldos_array = dataset[ch]

    # AA line
    start_idx = (np.abs(ldos_array.Y.values - selected_points[0][1]).argmin(),
                 np.abs(ldos_array.X.values - selected_points[0][0]).argmin())
    end_idx = (np.abs(ldos_array.Y.values - selected_points[1][1]).argmin(),
               np.abs(ldos_array.X.values - selected_points[1][0]).argmin())
    rr, cc, val = skimage.draw.line_aa(start_idx[0], start_idx[1], end_idx[0], end_idx[1])
    valid = (rr >= 0) & (rr < ldos_array.sizes['Y']) & (cc >= 0) & (cc < ldos_array.sizes['X'])
    rr, cc, val = rr[valid], cc[valid], val[valid]

    # Coordinates
    x_coords = dataset['X'].values
    y_coords = dataset['Y'].values
    X_map = x_coords[cc]
    Y_map = y_coords[rr]
    x0, y0 = selected_points[0]
    x1, y1 = selected_points[1]
    dx, dy = x1 - x0, y1 - y0
    line_length = np.sqrt(dx**2 + dy**2)
    proj_dist = ((X_map - x0) * dx + (Y_map - y0) * dy) / line_length
    proj_dist_nm = proj_dist * 1e9

    # Raw LDOS
    bias_vals = dataset['bias_mV'].values
    raw_ldos = np.empty((len(val), len(bias_vals)))
    for i in range(len(bias_vals)):
        ldos_slice = ldos_array.isel(bias_mV=i).values
        raw_ldos[:, i] = ldos_slice[rr, cc] * val

    # Interpolation
    interp_bins = np.linspace(proj_dist_nm.min(), proj_dist_nm.max(), num_interp_points)
    interp_ldos = np.empty((num_interp_points, len(bias_vals)))
    for i in range(len(bias_vals)):
        f = interp1d(proj_dist_nm, raw_ldos[:, i], kind='linear', bounds_error=False, fill_value=np.nan)
        interp_ldos[:, i] = f(interp_bins)

        
    def compute_uniform_ldos_profile(ldos_array, rr, cc, width):
        half_w = width // 2
        n_bias = ldos_array.sizes['bias_mV']
        result = np.zeros((len(rr), n_bias))
        for b in range(n_bias):
            ldos_slice = ldos_array.isel(bias_mV=b).values
            for i in range(len(rr)):
                values = []
                for dy in range(-half_w, half_w + 1):
                    for dx in range(-half_w, half_w + 1):
                        ry = rr[i] + dy
                        cx = cc[i] + dx
                        if 0 <= ry < ldos_slice.shape[0] and 0 <= cx < ldos_slice.shape[1]:
                            values.append(ldos_slice[ry, cx])
                result[i, b] = np.nanmean(values) if values else np.nan
        return result

        
    # Uniform profiles
    rr_u, cc_u = skimage.draw.line(start_idx[0], start_idx[1], end_idx[0], end_idx[1])
    profile_w1 = compute_uniform_ldos_profile(ldos_array, rr_u, cc_u, width=1)
    profile_w3 = compute_uniform_ldos_profile(ldos_array, rr_u, cc_u, width=3)
    profile_w5 = compute_uniform_ldos_profile(ldos_array, rr_u, cc_u, width=5)

    # Plot
    fig, axs = plt.subplots(3, 2, figsize=(14, 15), constrained_layout=True)

    # 1. LDOS map
    ldos_map = dataset.sel(bias_mV=bias_mV_ref, method='nearest')[ch].values
    vmin_map, vmax_map = np.nanpercentile(ldos_map, ZBCM_perc)
    extent_map = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()]
    im0 = axs[0, 0].imshow(ldos_map, cmap=LDOS_cmap, origin='lower', extent=extent_map,
                           vmin=vmin_map, vmax=vmax_map, aspect='equal')
    axs[0, 0].set_title(f"LDOS Map with Line Profile\n(bias = {bias_mV_ref:.1f} mV)")
    axs[0, 0].set_xticks([]); axs[0, 0].set_yticks([])
    fig.colorbar(im0, ax=axs[0, 0], fraction=0.035, pad=0.02, label='LDOS')

    # Overlay line
    num_segments = len(val)
    colors = plt.get_cmap(line_cmap)(np.linspace(0, 1, num_segments))
    x_line = np.linspace(x0, x1, num_segments)
    y_line = np.linspace(y0, y1, num_segments)
    for i in range(num_segments - 1):
        axs[0, 0].plot(x_line[i:i+2], y_line[i:i+2], color=colors[i], lw=3, alpha=0.6)

    # 2. Raw AA profile
    im1 = axs[0, 1].imshow(raw_ldos, aspect='auto', cmap=LDOS_cmap,
                           extent=[bias_vals.min(), bias_vals.max(), 0, proj_dist_nm.max()],
                           vmin=np.nanpercentile(raw_ldos, LP_perc[0]),
                           vmax=np.nanpercentile(raw_ldos, LP_perc[1]))
    axs[0, 1].set_title("Anti-aliased LDOS Profile (raw)")
    axs[0, 1].set_xlabel("Bias (mV)"); axs[0, 1].set_ylabel("Distance (nm)")
    fig.colorbar(im1, ax=axs[0, 1], fraction=0.035, pad=0.02, label='LDOS')

    # 3. Interpolated AA profile
    im2 = axs[1, 0].imshow(interp_ldos, aspect='auto', cmap=LDOS_cmap,
                           extent=[bias_vals.min(), bias_vals.max(), 0, interp_bins.max()],
                           vmin=np.nanpercentile(interp_ldos, LP_perc[0]),
                           vmax=np.nanpercentile(interp_ldos, LP_perc[1]))
    axs[1, 0].set_title("Anti-aliased LDOS Profile (interpolated)")
    axs[1, 0].set_xlabel("Bias (mV)"); axs[1, 0].set_ylabel("Distance (nm)")
    fig.colorbar(im2, ax=axs[1, 0], fraction=0.035, pad=0.02, label='LDOS')

    # 4. Uniform width = 1
    im3 = axs[1, 1].imshow(profile_w1, aspect='auto', cmap=LDOS_cmap,
                           extent=[bias_vals.min(), bias_vals.max(), 0, profile_w1.shape[0]],
                           vmin=np.nanpercentile(profile_w1, LP_perc[0]),
                           vmax=np.nanpercentile(profile_w1, LP_perc[1]))
    axs[1, 1].set_title("Uniform LDOS Profile (line width = 1)")
    axs[1, 1].set_xlabel("Bias (mV)"); axs[1, 1].set_ylabel("Distance (px)")
    fig.colorbar(im3, ax=axs[1, 1], fraction=0.035, pad=0.02, label='LDOS')

    # 5. Uniform width = 3
    im4 = axs[2, 0].imshow(profile_w3, aspect='auto', cmap=LDOS_cmap,
                           extent=[bias_vals.min(), bias_vals.max(), 0, profile_w3.shape[0]],
                           vmin=np.nanpercentile(profile_w3, LP_perc[0]),
                           vmax=np.nanpercentile(profile_w3, LP_perc[1]))
    axs[2, 0].set_title("Uniform LDOS Profile (line width = 3)")
    axs[2, 0].set_xlabel("Bias (mV)"); axs[2, 0].set_ylabel("Distance (px)")
    fig.colorbar(im4, ax=axs[2, 0], fraction=0.035, pad=0.02, label='LDOS')

    # 6. Uniform width = 5
    im5 = axs[2, 1].imshow(profile_w5, aspect='auto', cmap=LDOS_cmap,
                           extent=[bias_vals.min(), bias_vals.max(), 0, profile_w5.shape[0]],
                           vmin=np.nanpercentile(profile_w5, LP_perc[0]),
                           vmax=np.nanpercentile(profile_w5, LP_perc[1]))
    axs[2, 1].set_title("Uniform LDOS Profile (line width = 5)")
    axs[2, 1].set_xlabel("Bias (mV)"); axs[2, 1].set_ylabel("Distance (px)")
    fig.colorbar(im5, ax=axs[2, 1], fraction=0.035, pad=0.02, label='LDOS')

    interpolated_ds = xr.Dataset(
        {"LDOS_interpolated": (["distance", "bias_mV"], interp_ldos)},
        coords={"distance": interp_bins, "bias_mV": bias_vals}
    )

    return fig, interpolated_ds

fig, interpolated_ds1 = plot_LDOS_map_and_profile_v2_2(updated_GS_LDOS_0T002_N_2T003,
                                                       selected_points, 
                                                       ch='LDOS',
                                                       bias_mV_ref = 0.2, 
                                                       ZBCM_perc=(2, 98), LP_perc=(2, 78),
                                                       scalebar_length_nm=20, scalebar_color='white')
fig, interpolated_ds2 = plot_LDOS_map_and_profile_v2_2(GS_LDOS_2T_003,
                                                       selected_points,
                                                       ch='LDOS',
                                                       bias_mV_ref = 0,
                                                       ZBCM_perc=(2, 98), LP_perc=(2, 78),
                                                       scalebar_length_nm=20, scalebar_color='white')

fig

# +
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import skimage.draw
from skimage.transform import rotate
from scipy.interpolate import interp1d
from matplotlib.patches import Polygon

def plot_LDOS_map_and_profile_v2_3_updated(
    dataset,
    selected_points,
    ch='LDOS_smoothed',
    bias_mV_ref=0,
    LDOS_cmap='viridis',
    line_cmap='gnuplot',
    ZBCM_perc=(2, 98),
    LP_perc=(2, 98),
    scalebar_length_nm=None,
    scalebar_color='white',
    width_nm=10,
    num_interp_points=200
):
    """
    LDOS visualization function (2x2 layout):
      1. Original map + line + selection box
      2. Rotated map + precise crop
      3. LDOS line profile
      4. Cropped region
    """

    def get_rectangle_corners(x0, y0, x1, y1, width_nm):
        dx, dy = x1 - x0, y1 - y0
        nx, ny = -dy, dx
        norm = np.hypot(nx, ny)
        nx /= norm; ny /= norm
        half = (width_nm * 1e-9) / 2
        return np.array([
            [x0 + nx*half, y0 + ny*half],
            [x0 - nx*half, y0 - ny*half],
            [x1 - nx*half, y1 - ny*half],
            [x1 + nx*half, y1 + ny*half],
        ])

    def rotate_and_crop_rectangle(dataset, selected_points, width_nm, ch, bias_mV_ref):
        img = dataset.sel(bias_mV=bias_mV_ref, method='nearest')[ch].values
        xs = dataset['X'].values; ys = dataset['Y'].values
        dx = xs[1] - xs[0]; dy = ys[1] - ys[0]

        (x0, y0), (x1, y1) = selected_points
        theta = np.arctan2(y1 - y0, x1 - x0)
        # rotate so that start→end becomes vertical up with start at bottom
        angle_deg = np.degrees(theta) - 90

        img_rot = rotate(img, angle=angle_deg, resize=True, order=1,
                         mode='constant', cval=np.nan)

        corners_m = get_rectangle_corners(x0, y0, x1, y1, width_nm)
        corners_px = np.column_stack([
            (corners_m[:,1] - ys[0]) / dy,
            (corners_m[:,0] - xs[0]) / dx
        ])

        h0, w0 = img.shape
        center0 = np.array([h0/2, w0/2])
        h1, w1 = img_rot.shape
        center1 = np.array([h1/2, w1/2])

        rad = np.deg2rad(angle_deg)
        R = np.array([[ np.cos(rad), -np.sin(rad)],
                      [ np.sin(rad),  np.cos(rad)]])

        offs = corners_px - center0
        corners_rot = (R @ offs.T).T + center1

        rmin, rmax = int(corners_rot[:,0].min()), int(corners_rot[:,0].max())
        cmin, cmax = int(corners_rot[:,1].min()), int(corners_rot[:,1].max())
        crop = img_rot[rmin:rmax, cmin:cmax]

        return img_rot, crop, corners_rot

    # ensure ascending bias
    if dataset['bias_mV'].values[0] > dataset['bias_mV'].values[-1]:
        dataset = dataset.sortby('bias_mV')

    x0, y0 = selected_points[0]
    x1, y1 = selected_points[1]
    yi0 = np.abs(dataset.Y.values - y0).argmin()
    xi0 = np.abs(dataset.X.values - x0).argmin()
    yi1 = np.abs(dataset.Y.values - y1).argmin()
    xi1 = np.abs(dataset.X.values - x1).argmin()
    rr, cc, val = skimage.draw.line_aa(yi0, xi0, yi1, xi1)
    mask = (rr>=0)&(rr<dataset.dims['Y'])&(cc>=0)&(cc<dataset.dims['X'])
    rr, cc, val = rr[mask], cc[mask], val[mask]

    dxl, dyl = x1-x0, y1-y0
    length = np.hypot(dxl, dyl)
    Xmap = dataset.X.values[cc]; Ymap = dataset.Y.values[rr]
    proj = ((Xmap-x0)*dxl + (Ymap-y0)*dyl) / length
    distance_nm = proj * 1e9

    biases = dataset.bias_mV.values
    raw = np.array([
        dataset[ch].isel(bias_mV=i).values[rr,cc] * val
        for i in range(len(biases))
    ]).T

    bins = np.linspace(distance_nm.min(), distance_nm.max(), num_interp_points)
    interp_ldos = np.array([
        interp1d(distance_nm, raw[:,j], kind='linear',
                 bounds_error=False, fill_value=np.nan)(bins)
        for j in range(len(biases))
    ]).T

    rot_img, crop_img, corners_rot = rotate_and_crop_rectangle(
        dataset, selected_points, width_nm, ch, bias_mV_ref
    )

    fig, ax = plt.subplots(2,2, figsize=(14,12))
    vmin, vmax = np.nanpercentile(dataset.sel(bias_mV=bias_mV_ref)[ch], ZBCM_perc)
    ext = [dataset.X.min()*1e9, dataset.X.max()*1e9,
           dataset.Y.min()*1e9, dataset.Y.max()*1e9]
    ax[0,0].imshow(dataset.sel(bias_mV=bias_mV_ref)[ch],
                   cmap=LDOS_cmap, origin='lower',
                   extent=ext, vmin=vmin, vmax=vmax, aspect='equal')
    ax[0,0].plot([x0*1e9,x1*1e9],[y0*1e9,y1*1e9],
                 color='red',lw=2,marker='^',markersize=8,markevery=[-1])
    corners_nm = get_rectangle_corners(x0,y0,x1,y1,width_nm)*1e9
    ax[0,0].add_patch(Polygon(corners_nm,closed=True,edgecolor='red',facecolor='none',lw=2))
    ax[0,0].set_title("1. Original LDOS Map + Line + Crop")
    ax[0,0].set_xlabel("X (nm)"); ax[0,0].set_ylabel("Y (nm)")

    ax[0,1].imshow(rot_img, cmap=LDOS_cmap, origin='lower')
    rect = [(c[1],c[0]) for c in corners_rot]
    ax[0,1].add_patch(Polygon(rect,closed=True,edgecolor='red',facecolor='none',lw=2))
    ax[0,1].set_title("2. Rotated LDOS Map + Crop")

    fl = np.flipud(interp_ldos)
    vmin2, vmax2 = np.nanpercentile(fl, LP_perc)
    ar = (biases.max()-biases.min())/bins.max()
    ax[1,0].imshow(fl, cmap=LDOS_cmap, origin='upper',
                   extent=[biases.min(),biases.max(),0,bins.max()],
                   vmin=vmin2, vmax=vmax2, aspect=ar)
    ax[1,0].set_title("3. LDOS Line Profile")
    ax[1,0].set_xlabel("Bias (mV)"); ax[1,0].set_ylabel("Distance (nm)")

    ax[1,1].imshow(crop_img, cmap=LDOS_cmap, origin='lower',
                   vmin=vmin, vmax=vmax, aspect='equal')
    ax[1,1].set_title(f"4. Cropped LDOS (width={width_nm} nm)")

    plt.tight_layout()

    interpolated_ds = xr.Dataset(
        {"LDOS_interpolated": (["distance","bias_mV"], interp_ldos)},
        coords={"distance": bins, "bias_mV": biases}
    )
    return fig, interpolated_ds



# +
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import skimage.draw
from skimage.transform import rotate
from scipy.interpolate import interp1d
from matplotlib.patches import Polygon
from matplotlib.ticker import FuncFormatter

def plot_LDOS_map_and_profile_v2_3_updated(
    dataset,
    selected_points,
    ch='LDOS_smoothed',
    bias_mV_ref=0,
    LDOS_cmap='viridis',
    line_cmap='gnuplot',
    ZBCM_perc=(2, 98),
    LP_perc=(2, 98),
    scalebar_length_nm=None,
    scalebar_color='white',
    width_nm=10,
    num_interp_points=200
):
    """
    Visualize LDOS data in a 2x2 layout:

    1. Original LDOS map with a gradient-colored line and selection rectangle.
    2. The map rotated so the sampling line is vertical, overlaid with the exact crop rectangle.
    3. The LDOS line profile (bias vs. distance) computed by sampling along the line and interpolating.
    4. The cropped region from the rotated map with a red dashed overlay showing the sampled line direction.

    Parameters:
    - dataset (xr.Dataset): Xarray Dataset with coords 'X','Y','bias_mV' and data variable `ch`.
    - selected_points ([(float,float),(float,float)]): Two (x,y) points in meters.
    - ch (str): Name of the LDOS data variable within the Dataset.
    - bias_mV_ref (float): Bias slice at which to plot maps and perform cropping.
    - LDOS_cmap, line_cmap (str): Colormaps for LDOS images and gradient line.
    - ZBCM_perc, LP_perc (tuple): Percentile ranges for map and line-profile color scaling.
    - scalebar_length_nm, scalebar_color: (Unused here) optional scalebar parameters.
    - width_nm (float): Physical width of the sampling rectangle in nanometers.
    - num_interp_points (int): Number of distance bins for interpolation.

    Returns:
    - fig (plt.Figure): The generated figure.
    - interpolated_ds (xr.Dataset): Dataset with dims ('distance','bias_mV') for the interpolated profile.
    """

    def get_rectangle_corners(x0, y0, x1, y1, width_nm):
        """
        Compute the four corners (in meters) of a rectangle of width `width_nm` nm centered
        on the line from (x0,y0) to (x1,y1).
        """
        dx, dy = x1 - x0, y1 - y0
        nx, ny = -dy, dx
        norm = np.hypot(nx, ny)
        nx /= norm; ny /= norm
        half = (width_nm * 1e-9) / 2
        return np.array([
            [x0 + nx*half, y0 + ny*half],
            [x0 - nx*half, y0 - ny*half],
            [x1 - nx*half, y1 - ny*half],
            [x1 + nx*half, y1 + ny*half]
        ])

    def rotate_and_crop_rectangle(dataset, selected_points, width_nm, ch, bias_mV_ref):
        """
        Rotate the 2D LDOS map so the sampling line is vertical (start at bottom),
        then crop the exact rectangle defined originally.

        Returns:
          img_rot      : full rotated image
          crop_img     : cropped subregion
          corners_rot  : pixel coords of rotated rectangle corners
          rmin, cmin   : top-left offsets of the crop in the rotated image
        """
        # Extract the 2D map at the specified bias
        img = dataset.sel(bias_mV=bias_mV_ref, method='nearest')[ch].values
        xs = dataset['X'].values; ys = dataset['Y'].values
        dx = xs[1] - xs[0]; dy = ys[1] - ys[0]

        (x0, y0), (x1, y1) = selected_points
        theta = np.arctan2(y1 - y0, x1 - x0)
        # Rotate so the line points straight up
        angle_deg = np.degrees(theta) - 90

        img_rot = rotate(img, angle=angle_deg, resize=True, order=1,
                         mode='constant', cval=np.nan)

        # Compute original rectangle corners in meter coords
        corners_m = get_rectangle_corners(x0, y0, x1, y1, width_nm)
        # Convert meters to original pixel coordinates (row, col)
        corners_px = np.column_stack([
            (corners_m[:,1] - ys[0]) / dy,  # row index
            (corners_m[:,0] - xs[0]) / dx   # col index
        ])

        # Build rotation transform about center
        h0, w0 = img.shape
        center0 = np.array([h0/2, w0/2])
        h1, w1 = img_rot.shape
        center1 = np.array([h1/2, w1/2])
        rad = np.deg2rad(angle_deg)
        R = np.array([[ np.cos(rad), -np.sin(rad)],
                      [ np.sin(rad),  np.cos(rad)]])

        # Rotate corner coordinates
        offsets = corners_px - center0
        corners_rot = (R @ offsets.T).T + center1

        # Determine crop bounds
        rmin = int(np.floor(corners_rot[:,0].min()))
        rmax = int(np.ceil (corners_rot[:,0].max()))
        cmin = int(np.floor(corners_rot[:,1].min()))
        cmax = int(np.ceil (corners_rot[:,1].max()))
        crop_img = img_rot[rmin:rmax, cmin:cmax]

        return img_rot, crop_img, corners_rot, rmin, cmin

    # Ensure bias axis ascending
    if dataset['bias_mV'].values[0] > dataset['bias_mV'].values[-1]:
        dataset = dataset.sortby('bias_mV')

    x0, y0 = selected_points[0]
    x1, y1 = selected_points[1]

    # Generate anti-aliased line indices and weights
    yi0 = np.abs(dataset.Y.values - y0).argmin()
    xi0 = np.abs(dataset.X.values - x0).argmin()
    yi1 = np.abs(dataset.Y.values - y1).argmin()
    xi1 = np.abs(dataset.X.values - x1).argmin()
    rr, cc, val = skimage.draw.line_aa(yi0, xi0, yi1, xi1)
    mask = (rr>=0)&(rr<dataset.dims['Y'])&(cc>=0)&(cc<dataset.dims['X'])
    rr, cc, val = rr[mask], cc[mask], val[mask]

    # Project distances along line (meters → nm)
    dxl, dyl = x1-x0, y1-y0
    length = np.hypot(dxl, dyl)
    Xmap = dataset.X.values[cc]; Ymap = dataset.Y.values[rr]
    proj = ((Xmap - x0)*dxl + (Ymap - y0)*dyl) / length
    distance_nm = proj * 1e9

    # Sample LDOS along the line with anti-aliased weights
    biases = dataset.bias_mV.values
    raw = np.array([
        dataset[ch].isel(bias_mV=i).values[rr,cc] * val
        for i in range(len(biases))
    ]).T

    # Interpolate to uniform distance bins
    bins = np.linspace(distance_nm.min(), distance_nm.max(), num_interp_points)
    interp_ldos = np.array([
        interp1d(distance_nm, raw[:,j],
                 kind='linear', bounds_error=False, fill_value=np.nan)(bins)
        for j in range(len(biases))
    ]).T

    # Rotate and crop the rectangle region
    img_rot, crop_img, corners_rot, rmin, cmin = rotate_and_crop_rectangle(
        dataset, selected_points, width_nm, ch, bias_mV_ref
    )

    # Create figure and axes
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))

    # 1) Original LDOS map
    vmin_map, vmax_map = np.nanpercentile(
        dataset.sel(bias_mV=bias_mV_ref)[ch].values, ZBCM_perc
    )
    extent_map = [
        dataset.X.values.min()*1e9, dataset.X.values.max()*1e9,
        dataset.Y.values.min()*1e9, dataset.Y.values.max()*1e9
    ]
    axs[0,0].imshow(
        dataset.sel(bias_mV=bias_mV_ref)[ch],
        cmap=LDOS_cmap, origin='lower',
        extent=extent_map, vmin=vmin_map, vmax=vmax_map,
        aspect='equal'
    )
    # Plot gradient-colored sampling line
    num_pts = len(rr)
    gradient = plt.get_cmap(line_cmap)(np.linspace(0,1,num_pts))
    xs_line = np.linspace(x0*1e9, x1*1e9, num_pts)
    ys_line = np.linspace(y0*1e9, y1*1e9, num_pts)
    for i in range(num_pts-1):
        axs[0,0].plot(xs_line[i:i+2], ys_line[i:i+2],
                      color=gradient[i], lw=3, alpha=0.6)
    # Draw selection rectangle
    corners_nm = get_rectangle_corners(x0, y0, x1, y1, width_nm) * 1e9
    axs[0,0].add_patch(Polygon(
        corners_nm, closed=True, edgecolor='red', facecolor='none', lw=2
    ))
    axs[0,0].set_title("1. Original LDOS Map + Line + Crop")
    axs[0,0].set_xlabel("X (nm)"); axs[0,0].set_ylabel("Y (nm)")
    axs[0,0].xaxis.set_major_formatter(FuncFormatter(lambda v,p: f"{int(v)}"))
    axs[0,0].yaxis.set_major_formatter(FuncFormatter(lambda v,p: f"{int(v)}"))

    # 2) Rotated LDOS map + crop overlay
    axs[0,1].imshow(img_rot, cmap=LDOS_cmap, origin='lower')
    rect_xy = [(c[1], c[0]) for c in corners_rot]
    axs[0,1].add_patch(Polygon(
        rect_xy, closed=True, edgecolor='red', facecolor='none', lw=2
    ))
    axs[0,1].set_title("2. Rotated LDOS Map + Crop")

    # 3) LDOS line profile
    flipped = np.flipud(interp_ldos)
    vmin_lp, vmax_lp = np.nanpercentile(flipped, LP_perc)
    aspect = (biases.max() - biases.min()) / bins.max()
    axs[1,0].imshow(
        flipped, cmap=LDOS_cmap, origin='upper',
        extent=[biases.min(), biases.max(), 0, bins.max()],
        vmin=vmin_lp, vmax=vmax_lp, aspect=aspect
    )
    axs[1,0].set_title("3. LDOS Line Profile")
    axs[1,0].set_xlabel("Bias (mV)"); axs[1,0].set_ylabel("Distance (nm)")
    axs[1,0].yaxis.set_major_formatter(FuncFormatter(lambda v,p: f"{int(v)}"))

    # 4) Cropped LDOS with red dashed overlay line
    axs[1,1].imshow(
        crop_img, cmap=LDOS_cmap, origin='lower',
        vmin=vmin_map, vmax=vmax_map, aspect='equal'
    )
    axs[1,1].set_title(f"4. Cropped LDOS (width = {width_nm} nm)")
    # Compute rotated line endpoints within crop coordinates
    start_center = (corners_rot[0] + corners_rot[1]) / 2
    end_center   = (corners_rot[2] + corners_rot[3]) / 2
    # Subtract crop offset
    start_rc = start_center - np.array([rmin, cmin])
    end_rc   = end_center   - np.array([rmin, cmin])
    # Plot red dashed line, 50% transparent
    axs[1,1].plot(
        [start_rc[1], end_rc[1]],
        [start_rc[0], end_rc[0]],
        linestyle='--', color='red', linewidth=2, alpha=0.5
    )

    plt.tight_layout()

    # Build xarray Dataset for the interpolated profile
    interpolated_ds = xr.Dataset(
        {"LDOS_interpolated": (["distance", "bias_mV"], interp_ldos)},
        coords={"distance": bins, "bias_mV": biases}
    )

    return fig, interpolated_ds



# -

fig1, interpolated_ds1 = plot_LDOS_map_and_profile_v2_3_updated(updated_GS_LDOS_0T002_N_2T003,
                                                       selected_points, 
                                                       ch='LDOS',
                                                       bias_mV_ref = 0.0, 
                                                       ZBCM_perc=(0, 98), LP_perc=(0, 65),
                                                       scalebar_length_nm=20, scalebar_color='white',
                                                      )
fig1
fig2, interpolated_ds2 = plot_LDOS_map_and_profile_v2_3_updated(GS_LDOS_2T_003,
                                                       selected_points,
                                                       ch='LDOS',
                                                       bias_mV_ref = 0,
                                                       ZBCM_perc=(0, 98), LP_perc=(0, 70),
                                                       scalebar_length_nm=20, scalebar_color='white',
                                                      )
fig2


def plot_LDOS_map_and_profile_v3(dataset, selected_points, ch='LDOS_smoothed', bias_mV_ref=0.0,
                                  LDOS_cmap='viridis', line_cmap='gnuplot',
                                  ZBCM_perc=(2, 98), LP_perc=(2, 98),
                                  scalebar_length_nm=None, scalebar_color='white',
                                  line_width_px=1, width_nm=10):
    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    import skimage.draw
    from matplotlib.patches import Polygon
    from skimage.transform import rotate
    from scipy.interpolate import interp1d

    # 1. 정렬
    if dataset['bias_mV'].values[0] > dataset['bias_mV'].values[-1]:
        dataset = dataset.sortby('bias_mV')

    x_coords = dataset['X'].values
    y_coords = dataset['Y'].values
    bias_vals = dataset['bias_mV'].values
    ch_data = dataset[ch]

    x0, y0 = selected_points[0]
    x1, y1 = selected_points[1]
    dx, dy = x1 - x0, y1 - y0
    line_length = np.hypot(dx, dy)

    start_idx = (np.abs(y_coords - y0).argmin(), np.abs(x_coords - x0).argmin())
    end_idx = (np.abs(y_coords - y1).argmin(), np.abs(x_coords - x1).argmin())

    rr_u, cc_u = skimage.draw.line(start_idx[0], start_idx[1], end_idx[0], end_idx[1])

    def compute_uniform_ldos_profile(ldos_array, rr, cc, width):
        half_w = width // 2
        n_bias = ldos_array.sizes['bias_mV']
        result = np.zeros((len(rr), n_bias))
        
        for b in range(n_bias):
            ldos_slice = ldos_array.isel(bias_mV=b).values  # 2D array
            for i in range(len(rr)):
                values = []
                for dy in range(-half_w, half_w + 1):
                    for dx in range(-half_w, half_w + 1):
                        ry = rr[i] + dy
                        cx = cc[i] + dx
                        if 0 <= ry < ldos_slice.shape[0] and 0 <= cx < ldos_slice.shape[1]:
                            values.append(ldos_slice[ry, cx])
                result[i, b] = np.nanmean(values) if values else np.nan
    
        return result
    # (1) LDOS map
    ldos_map = dataset.sel(bias_mV=bias_mV_ref, method='nearest')[ch].values
    vmin_map, vmax_map = np.nanpercentile(ldos_map, ZBCM_perc)
    extent_map = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()]

    # (2) Line profile
    profile = compute_uniform_ldos_profile(ch_data, rr_u, cc_u, line_width_px)
    proj_dist = np.linspace(0, line_length * 1e9, len(rr_u))

    # (3) Crop LDOS
    def get_rectangle_corners(x0, y0, x1, y1, width_nm):
        dx, dy = x1 - x0, y1 - y0
        nx, ny = -dy, dx
        norm = np.hypot(nx, ny)
        nx /= norm; ny /= norm
        offset_x = nx * width_nm * 1e-9 / 2
        offset_y = ny * width_nm * 1e-9 / 2
        return np.array([
            [x0 + offset_x, y0 + offset_y],
            [x0 - offset_x, y0 - offset_y],
            [x1 - offset_x, y1 - offset_y],
            [x1 + offset_x, y1 + offset_y]
        ])

    def rotate_and_crop_rectangle(dataset, selected_points, width_nm, ch, bias_mV_ref):
        data_2d = dataset.sel(bias_mV=bias_mV_ref, method='nearest')[ch].values
        x_coords = dataset['X'].values
        y_coords = dataset['Y'].values
        dx_nm = (x_coords[1] - x_coords[0]) * 1e9

        (x0, y0), (x1, y1) = selected_points
        theta_rad = np.arctan2(y1 - y0, x1 - x0)
        angle_deg = -(90 - np.degrees(theta_rad))

        rotated_img = rotate(data_2d, angle=angle_deg, resize=True, order=1, mode='constant', cval=np.nan)

        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2
        center_ix = (center_x - x_coords[0]) / (x_coords[1] - x_coords[0])
        center_iy = (center_y - y_coords[0]) / (y_coords[1] - y_coords[0])
        original_center = np.array([data_2d.shape[0] / 2, data_2d.shape[1] / 2])
        center_pix = np.array([center_iy, center_ix])
        shift = center_pix - original_center

        rot_matrix = np.array([
            [np.cos(theta_rad), -np.sin(theta_rad)],
            [np.sin(theta_rad),  np.cos(theta_rad)]
        ])
        center_rotated = rot_matrix @ shift + np.array(rotated_img.shape) / 2

        width_px = int(width_nm / dx_nm)
        height_px = int(np.hypot(x1 - x0, y1 - y0) / (y_coords[1] - y_coords[0]))

        y_min = int(center_rotated[0] - height_px / 2)
        y_max = int(center_rotated[0] + height_px / 2)
        x_min = int(center_rotated[1] - width_px / 2)
        x_max = int(center_rotated[1] + width_px / 2)

        return rotated_img[y_min:y_max, x_min:x_max], width_px

    cropped_rotated, width_px = rotate_and_crop_rectangle(dataset, selected_points, width_nm, ch, bias_mV_ref)

    # ===== 시각화 =====
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # (1) Map
    axs[0].imshow(ldos_map, cmap=LDOS_cmap, origin='lower', extent=extent_map,
                  vmin=vmin_map, vmax=vmax_map, aspect='equal')
    axs[0].plot([x0, x1], [y0, y1], color='red', lw=2, marker='^', markersize=8, markevery=[-1])
    axs[0].add_patch(Polygon(get_rectangle_corners(x0, y0, x1, y1, width_nm),
                             closed=True, edgecolor='red', facecolor='none', lw=2))
    axs[0].set_title("LDOS Map with Line")

    # (2) Line Profile
    flipped = np.flipud(profile)
    vmin_lp = np.nanpercentile(flipped, LP_perc[0])
    vmax_lp = np.nanpercentile(flipped, LP_perc[1])
    aspect_ratio = (bias_vals.max() - bias_vals.min()) / proj_dist.max()
    axs[1].imshow(flipped, cmap=LDOS_cmap, origin='upper',
                  extent=[bias_vals.min(), bias_vals.max(), 0, proj_dist.max()],
                  vmin=vmin_lp, vmax=vmax_lp, aspect=aspect_ratio)
    axs[1].set_title(f"LDOS Profile (line width = {line_width_px})")
    axs[1].set_xlabel("Bias (mV)")
    axs[1].set_ylabel("Distance (nm)")

    # (3) Cropped
    axs[2].imshow(cropped_rotated, cmap=LDOS_cmap, origin='lower',
                  vmin=vmin_map, vmax=vmax_map, aspect='equal')
    axs[2].axvline(x=width_px // 2, color='white', linestyle='--', lw=1)
    axs[2].set_title(f"Cropped Rotated LDOS (width = {width_nm} nm)")

    plt.tight_layout()

    interpolated_ds = xr.Dataset(
        {"LDOS_interpolated": (["distance", "bias_mV"], profile)},
        coords={"distance": proj_dist, "bias_mV": bias_vals}
    )

    return fig, interpolated_ds



def plot_LDOS_map_and_profile_v3(dataset, selected_points, ch='LDOS_smoothed', bias_mV_ref=0.0,
                                  LDOS_cmap='viridis', line_cmap='gnuplot',
                                  ZBCM_perc=(2, 98), LP_perc=(2, 98),
                                  scalebar_length_nm=None, scalebar_color='white',
                                  line_width_px=1, width_nm=10):
    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    import skimage.draw
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Polygon
    from skimage.transform import rotate

    # 1. 정렬
    if dataset['bias_mV'].values[0] > dataset['bias_mV'].values[-1]:
        dataset = dataset.sortby('bias_mV')

    x_coords = dataset['X'].values
    y_coords = dataset['Y'].values
    bias_vals = dataset['bias_mV'].values
    ch_data = dataset[ch]

    x0, y0 = selected_points[0]
    x1, y1 = selected_points[1]
    dx, dy = x1 - x0, y1 - y0
    line_length = np.hypot(dx, dy)

    start_idx = (np.abs(y_coords - y0).argmin(), np.abs(x_coords - x0).argmin())
    end_idx = (np.abs(y_coords - y1).argmin(), np.abs(x_coords - x1).argmin())
    rr_u, cc_u = skimage.draw.line(start_idx[0], start_idx[1], end_idx[0], end_idx[1])

    # 2. Line profile 계산
    def compute_uniform_ldos_profile(ldos_array, rr, cc, width):
        half_w = width // 2
        n_bias = ldos_array.sizes['bias_mV']
        result = np.zeros((len(rr), n_bias))
        for b in range(n_bias):
            ldos_slice = ldos_array.isel(bias_mV=b).values
            for i in range(len(rr)):
                values = []
                for dy in range(-half_w, half_w + 1):
                    for dx in range(-half_w, half_w + 1):
                        ry = rr[i] + dy
                        cx = cc[i] + dx
                        if 0 <= ry < ldos_slice.shape[0] and 0 <= cx < ldos_slice.shape[1]:
                            values.append(ldos_slice[ry, cx])
                result[i, b] = np.nanmean(values) if values else np.nan
        return result

    profile = compute_uniform_ldos_profile(ch_data, rr_u, cc_u, line_width_px)
    proj_dist = np.linspace(0, line_length * 1e9, len(rr_u))

    # 3. 회전 Crop 이미지 계산
    def get_rectangle_corners(x0, y0, x1, y1, width_nm):
        dx, dy = x1 - x0, y1 - y0
        nx, ny = -dy, dx
        norm = np.hypot(nx, ny)
        nx /= norm; ny /= norm
        offset_x = nx * width_nm * 1e-9 / 2
        offset_y = ny * width_nm * 1e-9 / 2
        return np.array([
            [x0 + offset_x, y0 + offset_y],
            [x0 - offset_x, y0 - offset_y],
            [x1 - offset_x, y1 - offset_y],
            [x1 + offset_x, y1 + offset_y]
        ])

    def rotate_and_crop_rectangle(dataset, selected_points, width_nm, ch, bias_mV_ref):
        data_2d = dataset.sel(bias_mV=bias_mV_ref, method='nearest')[ch].values
        x_coords = dataset['X'].values
        y_coords = dataset['Y'].values
        dx_nm = (x_coords[1] - x_coords[0]) * 1e9

        (x0, y0), (x1, y1) = selected_points
        theta_rad = np.arctan2(y1 - y0, x1 - x0)
        angle_deg = -(90 - np.degrees(theta_rad))
        rotated_img = rotate(data_2d, angle=angle_deg, resize=True, order=1, mode='constant', cval=np.nan)

        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2
        center_ix = (center_x - x_coords[0]) / (x_coords[1] - x_coords[0])
        center_iy = (center_y - y_coords[0]) / (y_coords[1] - y_coords[0])
        original_center = np.array([data_2d.shape[0] / 2, data_2d.shape[1] / 2])
        center_pix = np.array([center_iy, center_ix])
        shift = center_pix - original_center
        rot_matrix = np.array([
            [np.cos(theta_rad), -np.sin(theta_rad)],
            [np.sin(theta_rad),  np.cos(theta_rad)]
        ])
        center_rotated = rot_matrix @ shift + np.array(rotated_img.shape) / 2

        width_px = int(width_nm / dx_nm)
        height_px = len(rr_u)  # ← 수정: 실제 line 길이와 일치하게
        y_min = int(center_rotated[0] - height_px / 2)
        y_max = int(center_rotated[0] + height_px / 2)
        x_min = int(center_rotated[1] - width_px / 2)
        x_max = int(center_rotated[1] + width_px / 2)

        return rotated_img[y_min:y_max, x_min:x_max], width_px

    cropped_rotated, width_px = rotate_and_crop_rectangle(dataset, selected_points, width_nm, ch, bias_mV_ref)

    # 4. 시각화
    fig = plt.figure(figsize=(6, 14), constrained_layout=True)
    gs = GridSpec(3, 1, height_ratios=[1.2, 1, 1], figure=fig)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # (1) LDOS map with colored line
    ldos_map = dataset.sel(bias_mV=bias_mV_ref, method='nearest')[ch].values
    vmin_map, vmax_map = np.nanpercentile(ldos_map, ZBCM_perc)
    extent_map = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()]
    im1 = ax1.imshow(ldos_map, cmap=LDOS_cmap, origin='lower', extent=extent_map,
                     vmin=vmin_map, vmax=vmax_map, aspect='equal')

    num_segments = len(rr_u)
    colors = plt.get_cmap(line_cmap)(np.linspace(0, 1, num_segments))
    x_line = np.linspace(x0, x1, num_segments)
    y_line = np.linspace(y0, y1, num_segments)
    for i in range(num_segments - 1):
        ax1.plot(x_line[i:i+2], y_line[i:i+2], color=colors[i], lw=3, alpha=0.6)

    # Scale bar
    if scalebar_length_nm:
        bar_x_start = x_coords[0] + 0.05 * (x_coords[-1] - x_coords[0])
        bar_x_end = bar_x_start + (scalebar_length_nm * 1e-9)
        bar_y = y_coords[0] + 0.05 * (y_coords[-1] - y_coords[0])
        ax1.plot([bar_x_start, bar_x_end], [bar_y, bar_y], color=scalebar_color, lw=2)
        ax1.text((bar_x_start + bar_x_end)/2, bar_y + 0.01 * (y_coords[-1] - y_coords[0]),
                 f"{scalebar_length_nm:.0f} nm", color=scalebar_color, ha='center', va='bottom', fontsize=10)

    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_title(f"LDOS Map with Line Profile\n(at bias = {bias_mV_ref:.1f} mV)")
    fig.colorbar(im1, ax=ax1, orientation='vertical', fraction=0.035, pad=0.01, label='LDOS')

    # (2) Line Profile
    flipped = np.flipud(profile)
    vmin_lp = np.nanpercentile(flipped, LP_perc[0])
    vmax_lp = np.nanpercentile(flipped, LP_perc[1])
    aspect_ratio = (bias_vals.max() - bias_vals.min()) / proj_dist.max()
    im2 = ax2.imshow(flipped, cmap=LDOS_cmap, origin='upper',
                     extent=[bias_vals.min(), bias_vals.max(), 0, proj_dist.max()],
                     vmin=vmin_lp, vmax=vmax_lp, aspect=aspect_ratio)
    ax2.set_xlabel("Bias (mV)")
    ax2.set_ylabel("Distance (nm)")
    ax2.set_title(f"LDOS Profile")# (line width = {line_width_px})")
    fig.colorbar(im2, ax=ax2, orientation='vertical', fraction=0.035, pad=0.01, label='LDOS')

    # (3) Cropped Rotated LDOS — 이제는 aspect='equal'로 정확히 설정됨
    im3 = ax3.imshow(cropped_rotated, cmap=LDOS_cmap, origin='lower',
                     vmin=vmin_map, vmax=vmax_map, aspect='equal')
    ax3.axvline(x=width_px // 2, color='white', linestyle='--', lw=1)
    #ax3.set_title(f"Cropped Rotated LDOS (width = {width_nm} nm)")
    # 5. 결과 반환
    interpolated_ds = xr.Dataset(
        {"LDOS_interpolated": (["distance", "bias_mV"], profile)},
        coords={"distance": proj_dist, "bias_mV": bias_vals}
    )

    return fig, interpolated_ds



fig, interpolated_ds1 = plot_LDOS_map_and_profile_v3(
    updated_GS_LDOS_0T002_N_2T003,
    selected_points,
    ch='LDOS',
    bias_mV_ref=0.0,
    ZBCM_perc=(0, 98),
    LP_perc=(0, 65),
    scalebar_length_nm=20,
    scalebar_color='white',
    line_width_px=1,           # 또는 1
    width_nm=10                # crop용 rectangle 너비
)
fig, interpolated_ds2 = plot_LDOS_map_and_profile_v3(
    GS_LDOS_2T_003,
    selected_points,
    ch='LDOS',
    bias_mV_ref=0.0,
    ZBCM_perc=(0, 98),
    LP_perc=(0, 65),
    scalebar_length_nm=20,
    scalebar_color='white',
    line_width_px=1,           # 또는 1
    width_nm=10                # crop용 rectangle 너비
)

fig


interpolated_ds1_cleaned = interpolated_ds1.dropna(dim='bias_mV', how='any')
interpolated_ds2_cleaned = interpolated_ds2.dropna(dim='bias_mV', how='any')

# +
#interpolated_ds
fig = plot_interpolated_profiles(interpolated_ds1_cleaned, offset=4E-11, skip=1, show_dots=False)
#fig.savefig('LDOS_lineprofile2.svg')

# in case of bias range zoom '''
interpolated_ds1_cleaned_bias_zm = interpolated_ds1_cleaned.where((
    interpolated_ds1_cleaned.bias_mV>-2.4)&(
        interpolated_ds1_cleaned.bias_mV<2.4),drop=True)
'''
#interpolated_ds
fig = plot_interpolated_profiles(interpolated_ds1_cleaned_bias_zm,
                                 offset=2E-9,
                                 skip=4,
                                 show_dots=True)
#fig.savefig('LDOS_lineprofile2.svg')
'''

#interpolated_ds
fig = plot_interpolated_profiles(interpolated_ds2_cleaned, offset=2E-11, skip=2, show_dots=False)
#fig.savefig('LDOS_lineprofile2.svg')

# -

fig = plot_interpolated_line_profiles_image(interpolated_ds1_cleaned, interpolated_ds2_cleaned, perc=(0,75), figsize=(8,4))

# ## after adjust offset by using  the same area crop 
# * updated_GS_topo_0T_002  or  GS_topo_2T_003
# * grid_LDOS = updated_GS_topo_0T_002.copy()

# ## use the updated 0T data for further analysis of grid_LDOS & grid_topo

# +
#updated_GS_topo_0T_002, GS_topo_2T_003
#updated_GS_LDOS_0T_002
# -

GS_topo_2T_005

# +
## select LDOS &topo files 

# +

#grid_topo = updated_GS_topo_0T_002.copy()
###grid_topo = updated_GS_topo_0T002_N_2T005.copy()
#grid_LDOS = updated_GS_LDOS_0T_002[['LDOS']].copy()
###grid_LDOS = updated_GS_LDOS_0T002_N_2T005[['LDOS']].copy()
# -

# ##### for CNN model , call 0T002 original dataset 
#

'''
GS_topo_0T_002 = loaded_datasets['GS_topo_0T_002']
GS_LDOS_0T_002 = loaded_datasets['GS_LDOS_0T_002']
grid_LDOS = GS_LDOS_0T_002.copy()
grid_topo = GS_topo_0T_002.copy()
'''

# +
#GS_topo_2T_005 = loaded_datasets['GS_topo_2T_005']
#GS_LDOS_2T_005 = loaded_datasets['GS_LDOS_2T_005']
#grid_LDOS = GS_LDOS_2T_005.copy()
#grid_topo = GS_topo_2T_005.copy()
# -

plane_fit_x_xr(plane_fit_y_xr(grid_topo)).topography.plot(cmap ='copper', robust = True)
plt.show()

# # **Grid_Analysis** 
#
#
# * [**Slicing (Holoview: X, Y, bias_mV)**](#Slicing-(Holoview:-X,-Y,-bias_mV))
#   
# * [**2D FFT & Quasi Particle Interferance**](#2D-FFT-&-Quasi-Particle-Interferance)
#
# * [**Filtering, Thresholds, and Segmentation**](#Filtering,-Thresholds,-and-Segmentation)
#
# * [**Flattening and Drift Compensation**](Flattening-and-Drift-Compensation)

# ### Slicing (Holoview: X, Y, bias_mV) 
#
# * back to [**Grid Analysis**](**Grid_Analysis** )

# +
# #%matplotlib inline
# -

grid_topo

ds

# %matplotlib inline
hv.extension('bokeh')
hv_bias_mV_slicing(grid_LDOS, ch = 'LDOS' ,frame_width=300)#.opts(clim = (0,0.5E-11))

# +
#hv_XY_slicing(grid_LDOS, ch = 'LDOS' ,frame_width=400,slicing='Y' ).opts(clim = (0,9.5E-10))

hv_XY_slicing(ds, ch = 'LDOS' ,frame_width=400,slicing='Y' ).opts(clim = (0,9.5E-10))
# -

# #### Gaussain convolution?

# +
#grid_LDOS_eq = rescale_intensity_xr(grid_LDOS, percentile = (2,98))
#choose overwrite == T or F 
#grid_LDOS_g = filter_gaussian_xr (grid_LDOS_eq, sigma=  1.0 ,overwrite =True)
#grid_LDOS = grid_LDOS_g

# +
# Check bias range of grid_3D

#hv_XY_slicing(grid_LDOS_g, ch = 'LDOS_fb',slicing= 'X')

#hv_bias_mV_slicing(grid_LDOS_g, ch = 'LDOS_gaussian',frame_width=300)#.opts(clim = (0,0.5E-11))
# -

# #### Difference of Gaussian (DoG) filtering 
#

# grid_LDOS = filter_diffofgaussians_xr (grid_LDOS,low_sigma= 0.5 , high_sigma=None ,overwrite =True)


# #### LDOS bias_mV slicing images & correlation plot 
# * **recall grid_LDOS** from grid_3D_gap
# * select slicing bias
# * slicing image
# * correlatio plot

grid_LDOS

# +
bias_mV_slices = np.linspace(-2.5 ,2.5, 15)

plot_bias_sliced_grid_LDOS_images(grid_LDOS, bias_mV_slices,ch = 'LDOS',  col_wrap=5, perc=(0, 100))

# +
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Convert xarray to pandas dataframe and unstack 'bias_mV'
grid_LDOS_df = grid_LDOS.LDOS.to_dataframe().unstack('bias_mV')

# Adjust the multi-index of the dataframe to a single index by transposing and resetting index
grid_LDOS_df_T = grid_LDOS_df.T.reset_index().drop(['level_0'], axis=1)

# Round the 'bias_mV' values for cleaner indexing
grid_LDOS_df_T['bias_mV'] = grid_LDOS_df_T['bias_mV'].round(3)

# Set 'bias_mV' as the index for easier correlation analysis
grid_LDOS_df_T = grid_LDOS_df_T.set_index('bias_mV')

# Transpose back to get 'bias_mV' as the column index
grid_LDOS_df = grid_LDOS_df_T.T

# Compute the correlation matrix of the dataframe
grid_LDOS_df_corr = grid_LDOS_df.corr()

####################################
# SKIP setting to reduce resolution
######################################
SKIP = 2  # Desired sampling size

# Sample the data by selecting every SKIP-th row and column
grid_LDOS_df_corr_sampled = grid_LDOS_df_corr.iloc[::SKIP, ::SKIP]

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(grid_LDOS_df_corr_sampled, dtype=bool))

'''
# Generate a mask for the upper triangle of the heatmap to avoid redundant information
mask = np.triu(np.ones_like(grid_LDOS_df_corr, dtype=bool))
'''
# Set up the figure for larger images (can adjust size if needed)
f, ax = plt.subplots(figsize=(10, 8))

# Generate a custom diverging colormap for visual clarity
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
ax = sns.heatmap(grid_LDOS_df_corr_sampled, mask=mask, cmap=cmap, vmax=.3, center=0,
                 square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

# Set the title of the heatmap plot
ax.set_title('2D correlation w.r.t. bias_mV')

# Display the plot
plt.show()
# -


# ##### Correlation between topgraphy vs bias dependent LDOS images

grid_topo.topography.plot(robust= True, cmap  ='copper')
plt.show()

# +
# Convert topography and LDOS data into DataFrame
grid_LDOS_df = grid_LDOS.LDOS.to_dataframe().unstack('bias_mV').reset_index()
grid_LDOS_df.columns = grid_LDOS_df.columns.get_level_values(0)
grid_LDOS_df = grid_LDOS_df.drop(columns=['X', 'Y']).set_axis(grid_LDOS.bias_mV.round(3), axis=1)

# Merge topography data with LDOS data
grid_LDOS_df_topo = pd.concat([grid_topo.to_dataframe().reset_index().topography, grid_LDOS_df], axis=1)

# Calculate correlation between topography and other bias_mV values
topo_correlations = grid_LDOS_df_topo.corr()['topography'][1:]

# Create figure level plot
fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(111)
sns.lineplot(x=topo_correlations.index, y=topo_correlations.values, marker='o', ax=ax)
ax.set_title('Correlation between topo vs bias_mV', fontsize='large')
ax.set_xlabel('bias_mV', fontsize='large')
ax.set_ylabel('Correlation coefficient', fontsize='large')
ax.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# +
# Convert topography and LDOS data into DataFrame
grid_LDOS_df = grid_LDOS.LDOS.to_dataframe().unstack('bias_mV').reset_index()
grid_LDOS_df.columns = grid_LDOS_df.columns.get_level_values(0)
grid_LDOS_df = grid_LDOS_df.drop(columns=['X', 'Y']).set_axis(grid_LDOS.bias_mV.round(3), axis=1)

# Extract the map at bias_mV = 0
ldos_bias0 = grid_LDOS.sel(bias_mV=0).to_dataframe().LDOS

# Merge LDOS bias_mV=0 data with the rest of the LDOS data
grid_LDOS_df_bias0 = pd.concat([ldos_bias0.reset_index(drop=True), grid_LDOS_df], axis=1)

# Calculate correlation between LDOS at bias_mV=0 and other bias_mV values
ldos_bias0_correlations = grid_LDOS_df_bias0.corr().iloc[0, 1:]

# Create figure level plot
fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(111)
sns.lineplot(x=ldos_bias0_correlations.index, y=ldos_bias0_correlations.values, marker='o', ax=ax)
ax.set_title('Correlation between LDOS (bias_mV=0) vs bias_mV', fontsize='large')
ax.set_xlabel('bias_mV', fontsize='large')
ax.set_ylabel('Correlation coefficient', fontsize='large')
ax.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# +
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def plot_find_peaks_pdSeries(series, prominence=0.01, width=None):
    # Convert index to float
    series.index = series.index.astype(float)
    
    # Find peaks using find_peaks
    peaks, properties = find_peaks(series.values, prominence=prominence, width=width)
    
    # Create a DataFrame with peak properties
    series_pks_prprts = pd.DataFrame({
        'position': series.index[peaks],
        'height': series.values[peaks],
        'prominence': properties['prominences'],
        'width': properties['widths']
    })
    
    # Plot
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(series.index, series.values)
    ax.plot(series.index[peaks], series.values[peaks], "x")
    
    # Show prominence
    ax.vlines(x=series.index[peaks], ymin=series.values[peaks] - properties['prominences'],
              ymax=series.values[peaks], color="C1")
    
    # Show width
    for i, peak in enumerate(peaks):
        left_ip = np.interp(properties['left_ips'][i], np.arange(len(series)), series.index)
        right_ip = np.interp(properties['right_ips'][i], np.arange(len(series)), series.index)
        ax.hlines(y=properties['width_heights'][i], xmin=left_ip, xmax=right_ip, color="C1")
    
    ax.set_title('Correlation between topo vs bias_mV', fontsize='large')
    ax.set_xlabel('Bias_mV', fontsize='large')
    ax.set_ylabel('Correlation coefficient', fontsize='large')
    ax.grid(True)
    
    return series_pks_prprts, fig

# Run the function
topo_correlations_pks_prprts, fig = plot_find_peaks_pdSeries(topo_correlations, prominence=1E-12, width=1E-7)
#print(topo_correlations_pks_prprts)
plt.show()
# -


# ### plot line STS 
#

# #### STS curve at XY point using slider 
# * slider X&Y setting

# +
import panel as pn
import panel.widgets as pnw
pn.extension()

# Create sliders
sliderX = pnw.IntSlider(name='X', 
                        start=0, 
                        end=grid_LDOS.X.shape[0])

sliderY = pnw.IntSlider(name='Y', 
                        start=0, 
                        end=grid_LDOS.Y.shape[0])

# Display sliders and values in scientific form with 3 significant digits
pn.Column(
    sliderX,
    sliderY,
    pn.bind(lambda x, y: (f"{grid_LDOS.X[x].values:.3e}", f"{grid_LDOS.Y[y].values:.3e}"), sliderX, sliderY)
)


# +

plot_Yslice_Xline_w_LDOS(grid_LDOS, sliderX = sliderX, sliderY = sliderY, ch = 'LDOS',slicing_bias_mV = 0.0)
plt.show()

# +

plot_Xslice_Yline_w_LDOS(grid_LDOS, sliderX = sliderX, sliderY = sliderY, ch = 'LDOS',slicing_bias_mV = 0.3)
#plot_Yslice_w_LDOS(grid_LDOS, sliderY = sliderY, ch = 'LDOS_fb',slicing_bias_mV = 0)
plt.show()
# -

# #####  plot_bias_sliced_grid_LDOS_images

plot_XYsliced_grid_images_with_topo(grid_LDOS, grid_topo, ch = 'LDOS',  number_of_XYslice=11 ,
                                    slicing='Y',  height=4, col_wrap=4, perc=(0, 75),
                                    ldos_aspect_ratio=1, topo_aspect_ratio=0.25,
                                    use_individual_vlimits=True, bias_mV_Vline_guide = [-2,-1,0,1,2])

plot_XYsliced_grid_images_with_LDOS_lines(grid_LDOS, number_of_XYslice=11, slicing='X', height=4, col_wrap=4, 
                                           perc=(0,75), ldos_aspect_ratio=1, topo_aspect_ratio=0.25,
                                           use_individual_vlimits=True, bias_mV_Vline_guide=[-2,-1, 0, 1, 2],
                                           ch='LDOS', bias_mV_ref=0, line_alpha= 0.5)

# ### plot Detection and Derivatives 
#
#

# +
#grid_LDOS = xr.open_dataset(grid_LDOS_file_name)
#grid_LDOS

# +
#####  grid_LDOS data derivative along bias_mV  (EDC-like)
grid_LDOS_EDC = smoothing_and_deriv_LDOS(grid_LDOS, ch_name='LDOS', window_length=5,polyorder=3)
#grid_LDOS_EDC

# bias_mV slicing (holoview) --> no contrast 
# hv_bias_mV_slicing(grid_LDOS_EDC, ch = 'LDOS_2deriv')
# XY_slicng (holoview) --> emphasizing SC gap 

hv_XY_slicing(grid_LDOS_EDC, ch = 'LDOS_2deriv',slicing= 'Y', frame_width= 500)

# useful to estimate peak detection 
# -

grid_LDOS_EDC

# +
#grid_data_dim_slicing(grid_LDOS_EDC,channel ='LDOS_2deriv')
grid_data_dim_slicing(grid_LDOS_EDC,channel ='LDOS_smoothed')

#####  grid_LDOS data derivative along XY  (MDC-like)

# +
#####  grid_LDOS data derivative along bias_mV  (MDC-like)
grid_LDOS_XY_diff =  gaussian_smoothing_and_deriv_XY(grid_LDOS,sigma=1.0,direction = 'Y',ch_name='LDOS')
# XY slicing (holoview) 
# hv_XY_slicing(grid_LDOS_XY_diff, ch = 'LDOSY_2deriv',slicing= 'Y')
# Bias_mV slicing (holoview)
#hv_bias_mV_slicing(grid_LDOS_XY_diff,ch = 'LDOSX_2deriv')
# ==> no clear contrast 

# use sobel_filter instead for edge finding for bias dependent LDOS map 
hv_XY_slicing (filter_sobel_xr(grid_LDOS, overwrite= True),ch = 'LDOS', slicing= 'X',frame_width= 500)
# -

filter_gaussian_xr

grid_data_dim_slicing(filter_gaussian_xr (grid_LDOS, sigma=1, overwrite= True),channel ='LDOS')


# ### Crop grid LDOS
#

# +
#grid_LDOS

# +
#grid_LDOS.where( (grid_LDOS.X<-3.4E-7)&(grid_LDOS.Y<3.5E-7), drop= True).sel(bias_mV=0).LDOS.plot()
#grid_LDOS = grid_LDOS.where( (grid_LDOS.X<-3.9E-7)&(grid_LDOS.Y<2.0E-7), drop= True)

#grid_topo =  grid_topo.where( (grid_LDOS.X<-3.9E-7)&(grid_LDOS.Y<2.8E-7), drop= True)
# -

# ## Zero Bias Conductance map analysis 
#
# * 1. threshold & gaussian check 
# * 2. assign the bianary filter 
# * 3. Delunary Triangulation 

zero_bias_map = grid_LDOS.where( (grid_LDOS.bias_mV <0.051)&( grid_LDOS.bias_mV <0.051),drop = True).mean('bias_mV')
#zero_bias_map.attrs = grid_LDOS.attrs


# ### Delunary triangulations 

# #### Threshold solutions for bianry image 

# +
#zero_bias_map

# +
# Show thresholding results
g_sigma_list = [0,3, 0.5, 0.8,1]
#zero_bais_map = grid_LDOS.sel(bias_mV = 0 , method = 'nearest')
zero_bias_map = grid_LDOS.where( (grid_LDOS.bias_mV <0.010)&( grid_LDOS.bias_mV >-0.010),drop = True).mean('bias_mV')

zero_bias_map.attrs = grid_LDOS.attrs

show_thresholding_results(plane_fit_surface_xr(zero_bias_map).LDOS.values,  g_sigma_list = g_sigma_list)


# +
#plane_fit_surface_xr(zero_bias_map).LDOS.values
# -

#binary_image = create_binary_image(grid_LDOS, ch = 'LDOS', bias_mV= 0, g_sigma=1, threshold_method='Mean', min_size=100, max_size=None)
binary_image = create_binary_image(plane_fit_surface_xr(zero_bias_map).LDOS.values, 
                                   g_sigma=0.5, 
                                   threshold_method='Otsu', 
                                   min_size= 3,
                                   max_size= None, figsize=(8,4))


# +
#binary_image
# -

# ### Delunary triangulations 

# +
#Mequalize_hist_xr(zero_bias_map).LDOS.plot()
#zero_bias_map
#zero_bias_map.LDOS.plot(robust= True)

# +
from skimage.measure import label, regionprops
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

fig, tri, filtered_edges, distances = plot_delaunay_advanced(zero_bias_map, binary_image, ch='LDOS', 
                                                             filter_boundary=False, show_stats=True, 
                                                             margin_percent=1, min_angle=10
                                                            )


'''
fig, tri, filtered_edges, distances = plot_delaunay_advanced(equalize_hist_xr(zero_bias_map), binary_image, ch='LDOS', 
                                                             filter_boundary=True, show_stats=True, 
                                                             margin_percent=3, min_angle=20
                                                            )
'''
# -


np.save('binary_image.npy',binary_image)


# #### show averaged STS curves from labeled area
# * use the binary image, show labels  

grid_LDOS_label_xr = topography_labeling_and_corresponding_LDOS(grid_topo, grid_LDOS, binary_image, alpha_boundary=1)


# +
grid_LDOS_label_df = grid_LDOS_label_xr.to_dataframe().reset_index()

find_peak_ldos_labels(grid_LDOS_label_df,
                      binary_image,col_wrap=8,
                      bias_range=(-0.1, 0.1),
                      peak_prominence=0.5E-10,
                      show_peaks=True)
# -

# ##### memory usage checkup 
#

# +
import psutil
import os

# Get memory usage of the current Python process
process = psutil.Process(os.getpid())
memory_info = process.memory_info()

print(f"RSS: {memory_info.rss / 1024 ** 2:.2f} MB")  # Resident Set Size (actual memory usage)
print(f"VMS: {memory_info.vms / 1024 ** 2:.2f} MB")  # Virtual Memory Size
# -

# ## 2D FFT & Quasi Particle Interferance
#
#
# * back to [**Preparation**](#Preparation)
# * back to [**Grid_Analysis**](**Grid_Analysis** )

# ### 2D FFT 
# * using "twoD_FFT_xr" for grid_LDOS_fff
# * 

grid_LDOS

# +
grid_LDOS_fft = twoD_FFT_xr(grid_LDOS, complex_output= False)
#grid_LDOS_fft = twoD_FFT_xr(filter_gaussian_xr (grid_LDOS, sigma= 0.8,overwrite = True), complex_output= False)


grid_LDOS_fft
# -

grid_topo_fft = twoD_FFT_xr(grid_topo, complex_output= False)

# ### Create 'global_fft_mask' to filter the FFT results



# +
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, IntSlider, Label
from matplotlib.patches import Polygon, Circle

# Define a global variable to store the mask
global_fft_mask = None

def plot_interactive_fft_filtering(ds_fft, ch_fft='topography_fft', cmap='inferno', vmin_percent=0, vmax_percent=100):
    """
    Interactive function to plot FFT data and create a mask based on a polygon and circle parameters.
    The mask is generated interactively by selecting the radius, polygon sides, and rotation, and is 
    saved to the global variable 'global_fft_mask'.

    Parameters:
    ----------
    ds_fft : xarray.Dataset
        The dataset containing the FFT result to plot and mask.
    
    ch_fft : str, optional
        The name of the FFT channel to plot. Default is 'topography_fft'.
    
    vmin_percent : float, optional
        Minimum percentage for the color range. Default is 0 (0%).
    
    vmax_percent : float, optional
        Maximum percentage for the color range. Default is 100 (100%).
    
    cmap : str, optional
        Colormap to use for plotting the FFT result. Default is 'inferno'.

    Functionality:
    --------------
    - The function generates an interactive plot of the FFT data.
    - Users can control the color range, the polygon size (n_sides), the polygon's rotation, 
      and the circle radius at each polygon vertex.
    - The mask is generated by checking whether points in the FFT frequency space lie inside 
      circles drawn around each polygon vertex. 
    - The resulting mask is stored in the global variable `global_fft_mask` for further use, 
      such as applying it for inverse FFT filtering.
    
    Returns:
    --------
    global_fft_mask : ndarray
        The mask generated by the polygon and circles, stored in the global variable. Can be accessed after interaction.
    """
    global global_fft_mask  # To store the mask globally

    # Extract FFT data and coordinates
    fft_data = ds_fft[ch_fft].values
    freq_X = ds_fft.freq_X.values
    freq_Y = ds_fft.freq_Y.values
    
    center_x = 0
    center_y = 0
    
    def calculate_polygon_vertices(n_sides, radius, rotation_deg):
        theta = np.linspace(0, 2 * np.pi, n_sides, endpoint=False) + np.radians(rotation_deg)
        x_vertices = center_x + radius * np.cos(theta)
        y_vertices = center_y + radius * np.sin(theta)
        return x_vertices, y_vertices
    
    def create_mask(x_vertices, y_vertices, circle_radius):
        mask = np.zeros_like(fft_data, dtype=bool)
        for fy in range(len(freq_Y)):
            for fx in range(len(freq_X)):
                for (vx, vy) in zip(x_vertices, y_vertices):
                    if (freq_X[fx] - vx) ** 2 + (freq_Y[fy] - vy) ** 2 <= circle_radius ** 2:
                        mask[fy, fx] = True
        return mask
    
    def update_plot(vmin_percent, vmax_percent, n_sides, radius, rotation_deg, circle_radius, guide_line_alpha):
        global global_fft_mask
        
        plt.figure(figsize=(8, 6))
        vmin = np.percentile(fft_data, vmin_percent)
        vmax = np.percentile(fft_data, vmax_percent)
        
        plt.imshow(fft_data, cmap=cmap, origin='lower', aspect='auto',
                   extent=[freq_X.min(), freq_X.max(), freq_Y.min(), freq_Y.max()],
                   vmin=vmin, vmax=vmax)
        
        x_vertices, y_vertices = calculate_polygon_vertices(n_sides, radius, rotation_deg)
        polygon = Polygon(np.column_stack([x_vertices, y_vertices]),
                          edgecolor='cyan', fill=False, alpha=guide_line_alpha, linewidth=1)
        plt.gca().add_patch(polygon)
        
        for (x, y) in zip(x_vertices, y_vertices):
            circle = Circle((x, y), radius=circle_radius,
                            edgecolor='magenta', fill=False, alpha=guide_line_alpha, linewidth=1)
            plt.gca().add_patch(circle)
        
        plt.colorbar(label='FFT Amplitude')
        plt.title(f'FFT Plot with Color Range {vmin_percent}% - {vmax_percent}% and Mask')
        plt.xlabel('Frequency X (1/m)')
        plt.ylabel('Frequency Y (1/m)')
        plt.grid(False)
        plt.show()
        
        global_fft_mask = create_mask(x_vertices, y_vertices, circle_radius)
        
        radius_nm = (1 / radius) * 1e9 if radius > 0 else np.inf
        print(f"Radius: {radius:.2f} [1/m] | {radius_nm:.2f} nm")
        
        return global_fft_mask
    
    interact(update_plot,
             vmin_percent=FloatSlider(min=0, max=100, step=0.01, value=vmin_percent, description='Min %'),
             vmax_percent=FloatSlider(min=0, max=100, step=0.01, value=vmax_percent, description='Max %'),
             n_sides=IntSlider(min=3, max=12, step=1, value=6, description='Polygon Sides'),
             radius=FloatSlider(min=0.1, max=freq_X.max(), step=1, value=100, description='Radius [1/m]'),
             rotation_deg=FloatSlider(min=0, max=360, step=1, value=0, description='Rotation (deg)'),
             circle_radius=FloatSlider(min=0, max=freq_X.max()//4, step=1, value=10, description='Circle Radius'),
             guide_line_alpha=FloatSlider(min=0, max=1, step=0.01, value=0.2, description='Guide Line Alpha'))
    
    return global_fft_mask

# Example of running the function interactively for the dataset
#plot_interactive_fft_filtering(ds_fft)

# After interacting, check the final mask value by printing global_fft_mask
#print(global_fft_mask)
# -



# +
# Define a global variable to store the mask
global_fft_mask = None

plot_interactive_fft_filtering(grid_topo_fft+1)
# -


global_fft_mask



# ##### rotation for line profile 

# +

#grid_LDOS_fft_r34= rotate_3D_fft_xr(grid_LDOS_fft,rotation_angle= 34)
#grid_LDOS_fft_r34
# -

#grid_LDOS_fft_r32_rescale= rescale_intensity_xr( rotate_3D_fft_xr(grid_LDOS_fft,rotation_angle= 32), percentile=(0,99.96))#.where(grid_LDOS_fft.freq_Y>6E9,drop= True)
"""grid_LDOS_fft_crop = grid_LDOS_fft_rescale.where(
    grid_LDOS_fft_rescale.freq_X<4.8E9,drop= True).where(
    grid_LDOS_fft_rescale.freq_X>-4.8E9,drop= True).where(
    grid_LDOS_fft_rescale.freq_Y<4.8E9,drop= True).where(
    grid_LDOS_fft_rescale.freq_Y>-4.8E9,drop= True)
grid_LDOS_fft_crop"""

# #####  hv plot after log10 

grid_LDOS_fft_r32

# +
grid_LDOS_fft_log = np.log10(grid_LDOS_fft)
fft_bias_mV_slicing_0= hv_fft_bias_mV_slicing(grid_LDOS_fft_log,ch = 'LDOS_fft',frame_width=400)
fft_bias_mV_slicing_0
# rotate 32 deg 
#grid_LDOS_fft_r32_log = np.log10(grid_LDOS_fft_r32)
#fft_bias_mV_slicing_0= hv_fft_bias_mV_slicing(grid_LDOS_fft_r32_log,ch = 'LDOS_fft')

#fft_bias_mV_slicing_0

# +


fft_XY_slicing_0= hv_fft_XY_slicing(grid_LDOS_fft
                  ,ch = 'LDOS_fft',slicing='freq_X',cmap='GnBu',frame_width=400)
'''
# rotate 34 deg 
fft_XY_slicing_0= hv_fft_XY_slicing(grid_LDOS_fft_r
                  ,ch = 'LDOS_fft',slicing='freq_Y',cmap='Blues',frame_width=400)
                
'''
fft_XY_slicing_0

# -

grid_LDOS_fft_r32

grid_LDOS_fft_dog = filter_diffofgaussians_xr(grid_LDOS_fft,
                              low_sigma=0.5, 
                              high_sigma=20,overwrite = True)
hv_fft_XY_slicing(rescale_intensity_xr(grid_LDOS_fft_dog, percentile=(00,99.5)),
                  ch = 'LDOS_fft',
                  slicing='freq_Y',
                  cmap='Blues',
                  frame_width=400)

# ##### add circle to fft (bias_mV slicing)plot 
#

grid_LDOS

# +


# 원 그리기
x0, y0 = (0,0)  # 튜플 언패킹
diameter =   2  *   (1E9* 1/grid_LDOS.ref_a0nm ) # hv.ellipse use the diameter not radius
diameter_1_2 =   1  *   (1E9* 1/grid_LDOS.ref_a0nm ) # hv.ellipse use the diameter not radius
diameter_1_4 =   0.5  *   (1E9* 1/grid_LDOS.ref_a0nm ) # hv.ellipse use the diameter not radius
diameter_1_8 =   0.25  *   (1E9* 1/grid_LDOS.ref_a0nm ) # hv.ellipse use the diameter not radius

circle = hv.Ellipse(x0, y0, diameter).opts(color='black', line_width=1,line_dash='dashed',  alpha =1)
circle_1_2 = hv.Ellipse(x0, y0, diameter_1_2).opts(color='black', line_width=1,line_dash='dashed',  alpha =1)
circle_1_4 = hv.Ellipse(x0, y0, diameter_1_4).opts(color='black', line_width=1,line_dash='dashed',  alpha =1)
circle_1_8 = hv.Ellipse(x0, y0, diameter_1_8).opts(color='black', line_width=1,line_dash='dashed',  alpha =1)

hv.extension('bokeh')
fft_bias_mV_slicing_0 * circle  # 원을 이미지에 겹치기
fft_bias_mV_slicing_0 * circle *circle_1_2*circle_1_4*circle_1_8  # 원을 이미지에 겹치기
# -


grid_LDOS


# ##### add verticalline to fft_XY_slicing_0


# +
#fft_XY_slicing_0
bragg_q0=  1  *   (1E9* 1/grid_LDOS.ref_a0nm )# r 6

# 수직 가이드 라인 추가
q0vlinePos = hv.VLine(bragg_q0).opts(color='red', line_width=2, line_dash='dashed', alpha = 0.5 )
q0vlineNeg = hv.VLine(-bragg_q0).opts(color='red', line_width=2, line_dash='dashed', alpha = 0.5)

fft_XY_slicing_0*q0vlineNeg*q0vlinePos


# +
# ref_6pts  = ref_lattice_k0 * np.array([[math.cos(pt_i* math.pi/3), math.sin(pt_i* math.pi/3)]for pt_i in range(6)])

# +
# # rotate?
# rotate_3D_fft_xr(grid_LDOS_fft,rotation_angle= 30)

# +
# Plot FFT bias_mV_slicing 
# -


# #### Bias slicing for fft result 
grid_LDOS_fft


bias_mV_slices = np.linspace(-1.2, 1.2, 15)  
bias_mV_slices


def plot_bias_sliced_grid_fft_images(grid_fft, number_of_bias_slice, 
                                     ch='LDOS_fft', height=3,
                                     col_wrap=6, cmap="bwr", 
                                     perc=(0, 99.99), add_q0_circle=False, circle_ratio=1):
    """
    Plots bias-sliced FFT images from a given grid dataset in a grid layout.

    This function creates and displays a series of 2D FFT images sliced by the specified bias voltage 
    from the `grid_fft` dataset. The images are displayed in a grid format, and the function allows 
    customization of the grid layout, color map, percentile range, and optional reference circle to 
    represent a characteristic wavevector (q0).

    Parameters:
    -----------
    grid_fft : xarray.Dataset
        The dataset containing the FFT data with dimensions including 'bias_mV', 'freq_X', and 'freq_Y'.
    number_of_bias_slice : int or array-like
        Either an integer specifying the number of bias slices to create (using linear spacing), 
        or an array-like object specifying exact bias voltage values for slicing.
    ch : str, optional
        The channel name in the dataset to visualize. Default is 'LDOS_fft'.
    height : float, optional
        The height of each subplot in the grid, by default 3.
    col_wrap : int, optional
        The number of columns in the image grid, by default 6.
    cmap : str, optional
        The color map for the images, by default 'bwr'.
    perc : tuple of two floats, optional
        The percentile range for contrast stretching the FFT images, by default (0, 99.99).
    add_q0_circle : bool, optional
        Whether to add a reference wavevector (q0) circle to each subplot, by default False.
    circle_ratio : float, optional
        Scaling factor for the size of the q0 circle, by default 1.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the grid of FFT images.

    Notes:
    ------
    - The function adjusts the grid layout dynamically based on the number of bias slices and col_wrap.
    - The x and y axis tick labels represent the frequency values ('freq_X' and 'freq_Y') from the FFT data.
    - The reference circle (q0) can be added to each subplot to visualize a characteristic wavevector.
    
    Example:
    --------
    plot_bias_sliced_grid_fft_images(grid_fft, number_of_bias_slice=10, add_q0_circle=True)
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import matplotlib.patches as patches
    import seaborn_image as isns

    # Ensure number_of_bias_slice is either an int or a numpy array
    if isinstance(number_of_bias_slice, int):
        min_bias = grid_fft.bias_mV.min()
        max_bias = grid_fft.bias_mV.max()
        bias_mV_slices = np.linspace(min_bias, max_bias, number_of_bias_slice)
    else:
        bias_mV_slices = np.sort(np.unique(number_of_bias_slice))
        bias_mV_slices = grid_fft.bias_mV.sel(bias_mV=bias_mV_slices, method="nearest").values

    # Select the FFT data for the bias slices and ensure it is 3D (freq_Y, freq_X, bias_mV)
    grid_fft_slicing = grid_fft.sel(bias_mV=bias_mV_slices, method="nearest")[ch].transpose('freq_Y', 'freq_X', 'bias_mV').values

    # Create the ImageGrid using the sliced FFT data
    g = isns.ImageGrid(grid_fft_slicing, cbar=True, height=height, col_wrap=col_wrap, cmap=cmap, robust=True, perc=perc)

    # Add a title summarizing the parameters
    plt.suptitle(grid_fft.attrs.get('title', '') + '\n' + ch + '\n' + f'Percentage Range: {perc}' + '\n' + 'q0_circle_ratio = ' + str(circle_ratio),
                 fontsize='large', y=1.05)

    # Calculate the reference wavevector (q0)
    ref_q01overnm = 1E9 / grid_fft.attrs.get('ref_a0nm', 1)

    # Iterate over each bias slice and configure each subplot
    for idx, (ax, bias_val) in enumerate(zip(g.axes.flat, bias_mV_slices)):
        # Set the title of each subplot with the corresponding bias voltage
        ax.set_title(f'{bias_val:.2f} mV', pad=2)
        ax.set_aspect(1)  # Keep aspect ratio square

        # Set xticks and yticks based on the frequency dimensions
        xticks = [0, grid_fft.freq_X.size // 2, grid_fft.freq_X.size - 1]
        yticks = [0, grid_fft.freq_Y.size // 2, grid_fft.freq_Y.size - 1]
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        # Show x-axis labels for subplots in the last row or the second-to-last row if the last row is incomplete
        if idx // col_wrap == (len(bias_mV_slices) - 1) // col_wrap or (
                (len(bias_mV_slices) % col_wrap != 0) and (idx // col_wrap == (len(bias_mV_slices) - 1) // col_wrap - 1) and (idx % col_wrap >= len(bias_mV_slices) % col_wrap)):
            ax.set_xticklabels([f'{grid_fft.freq_X.values[tick]:.2e}' for tick in xticks], rotation=45)
            ax.set_xlabel('freq_X')
        else:
            ax.set_xticklabels([])

        # Show y-axis labels only for subplots in the first column
        if idx % col_wrap == 0:
            ax.set_yticklabels([f'{grid_fft.freq_Y.values[tick]:.2e}' for tick in yticks], rotation=45)
            ax.set_ylabel('freq_Y')
        else:
            ax.set_yticklabels([])

        # Optionally add a q0 circle to the plot
        if add_q0_circle:
            center = (grid_fft.freq_X.size // 2, grid_fft.freq_Y.size // 2)
            radius = ref_q01overnm * grid_fft.freq_X.size / (2 * grid_fft.freq_X.max().values)
            circle = patches.Circle(center, radius * circle_ratio, fill=False, edgecolor='red', linestyle='--')
            ax.add_patch(circle)

    # Display the final grid of FFT images
    plt.tight_layout()
    plt.show()

    return plt.gcf()


plot_bias_sliced_grid_fft_images(grid_fft=np.log10(grid_LDOS_fft),
                                 number_of_bias_slice=bias_mV_slices, cmap='GnBu',
                                 add_q0_circle=True,    circle_ratio=0.1,
                                 col_wrap=4, perc=(50, 99.5595))
#plot_bias_sliced_grid_fft_images(grid_fft=np.log10(grid_LDOS_fft_r32), bias_mV_slices=bias_mV_slices, col_wrap=6, perc=(50, 99.99))


def plot_XYsliced_grid_fft_images(grid_fft, 
                                  number_of_XYslice, 
                                  slicing='freq_X', 
                                  ch='LDOS_fft',
                                  height=3, 
                                  col_wrap=4, 
                                  cmap="bwr", 
                                  perc=(0, 99.99), 
                                  aspect_ratio=1, 
                                  add_q0_line=False, 
                                  q0_ratio=1,
                                  use_individual_vlimits=False, 
                                  cbar=False):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import xarray as xr

    def create_symmetric_slices(min_freq, max_freq, number_of_slices):
        if number_of_slices % 2 == 0:
            number_of_slices += 1
        max_abs_freq = max(abs(min_freq), abs(max_freq))
        slices = np.linspace(-max_abs_freq, max_abs_freq, number_of_slices)
        return np.sort(np.unique(np.concatenate(([0], slices))))

    # Determine the slicing axis and the corresponding labels
    if slicing == 'freq_X':
        slicing_axis = 'freq_X'
        x_label = 'freq_Y'
        y_label = 'Bias_mV'
    elif slicing == 'freq_Y':
        slicing_axis = 'freq_Y'
        x_label = 'freq_X'
        y_label = 'Bias_mV'
    else:
        raise ValueError("slicing must be either 'freq_X' or 'freq_Y'")

    # Determine slice values based on the slicing axis
    if isinstance(number_of_XYslice, int):
        min_freq = grid_fft[slicing_axis].min()
        max_freq = grid_fft[slicing_axis].max()
        slices = create_symmetric_slices(min_freq, max_freq, number_of_XYslice)
    else:  # list or numpy array
        slices = np.sort(np.unique(np.concatenate(([0], number_of_XYslice))))

    slices_v = grid_fft[slicing_axis].sel({slicing_axis: slices}, method="nearest").values

    # Select and transpose grid_fft_slicing for visualization
    if slicing == 'freq_X':
        grid_fft_slicing = grid_fft.sel(freq_X=slices_v, method="nearest")[ch].values
        grid_fft_slicing = np.transpose(grid_fft_slicing, (2, 0, 1))
    elif slicing == 'freq_Y':
        grid_fft_slicing = grid_fft.sel(freq_Y=slices_v, method="nearest")[ch].values
        grid_fft_slicing = np.transpose(grid_fft_slicing, (2, 1, 0))

    # Calculate vmin and vmax if not using individual limits
    if use_individual_vlimits:
        vmin, vmax = None, None
    else:
        vmin, vmax = np.percentile(grid_fft_slicing, perc)

    # Adjust the number of slices
    number_of_XYslice = len(slices_v)

    # Determine the number of rows and columns for subplots
    ncols = min(number_of_XYslice, col_wrap)
    nrows = number_of_XYslice // col_wrap + (1 if number_of_XYslice % col_wrap != 0 else 0)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(height * ncols, height * nrows))
    axes = np.array(axes).reshape(-1)  # Flatten axes for easier iteration

    plt.suptitle(grid_fft.title + '\n' + ch +
                 '\n' + f'grid_fft - {slicing} Slices\n' +
                 f'Percentage Range: {perc}' + '\n' +
                 ' q0_line_ratio = ' + str(q0_ratio), fontsize=16, y=1.02)

    for axes_i in range(number_of_XYslice):
        ax = axes[axes_i]
        img = ax.imshow(grid_fft_slicing[axes_i], cmap=cmap, origin='lower', 
                        vmin=vmin if not use_individual_vlimits else None, 
                        vmax=vmax if not use_individual_vlimits else None)

        ax.set_title(f'slicing {slicing_axis} = {slices_v[axes_i]:.2e}', pad=2)
        ax.set_aspect(aspect_ratio)

        # Set xticks and labels only for the bottom row
        if axes_i >= (nrows - 1) * ncols and (axes_i < number_of_XYslice):
            xticks = [0, grid_fft_slicing.shape[2] // 2, grid_fft_slicing.shape[2] - 1]
            xticks = [tick for tick in xticks if tick < grid_fft[x_label].size]
            ax.xaxis.set_major_locator(ticker.FixedLocator(xticks))
            ax.set_xticklabels([f'{grid_fft[x_label].isel({x_label: tick}).values:.2e}' for tick in xticks], rotation=45)
            ax.set_xlabel(x_label)
        elif axes_i >= (nrows - 2) * ncols and axes_i < (nrows - 1) * ncols and (number_of_XYslice % col_wrap) != 0 and (axes_i >= len(axes) - col_wrap):
            xticks = [0, grid_fft_slicing.shape[2] // 2, grid_fft_slicing.shape[2] - 1]
            xticks = [tick for tick in xticks if tick < grid_fft[x_label].size]
            ax.xaxis.set_major_locator(ticker.FixedLocator(xticks))
            ax.set_xticklabels([f'{grid_fft[x_label].isel({x_label: tick}).values:.2e}' for tick in xticks], rotation=45)
            ax.set_xlabel(x_label)
        else:
            ax.set_xticks([])

        # Set yticks and labels only for the leftmost column
        if axes_i % ncols == 0:
            yticks = [0, grid_fft_slicing.shape[1] // 2, grid_fft_slicing.shape[1] - 1]
            yticks = [tick for tick in yticks if tick < grid_fft.bias_mV.size]
            ax.yaxis.set_major_locator(ticker.FixedLocator(yticks))
            ax.set_yticklabels([f'{grid_fft.bias_mV.isel(bias_mV=tick).values:.2f}' for tick in yticks])
            ax.set_ylabel(y_label)
        else:
            ax.set_yticks([])

        # Add vertical lines if add_q0_line is True
        if add_q0_line:
            ref_q01overnm = q0_ratio / (grid_fft.ref_a0nm * 1E-9)
            center = grid_fft_slicing.shape[2] // 2
            line_offset = ref_q01overnm * grid_fft_slicing.shape[2] / grid_fft[slicing_axis].max().values
            ax.axvline(x=center - line_offset, color='red', linestyle='--', linewidth=1)
            ax.axvline(x=center + line_offset, color='red', linestyle='--', linewidth=1)

    # Hide any unused subplots
    for axes_i in range(number_of_XYslice, len(axes)):
        axes[axes_i].axis('off')

    plt.tight_layout()
    plt.show()
    return plt.gcf()



plot_XYsliced_grid_fft_images(grid_fft=np.log10(grid_LDOS_fft), 
                              number_of_XYslice=12, 
                              slicing='freq_X', perc=(3, 99.0595),
                              height=4, 
                              cmap ='GnBu',
                              col_wrap=5, 
                              aspect_ratio=0.5,
                              add_q0_line=True, 
                              q0_ratio=0.1,
                              use_individual_vlimits = True,cbar=False)
# +
#np.log10(grid_LDOS_fft).sel(freq_X = 1E9, method ='nearest').LDOS_fft.T.plot(robust= True, cmap = 'GnBu')
#plt.show()

# +
#vline = 1/(grid_LDOS_fft.ref_a0nm *1E-9* 10)
#vline
# -

#grid_fft_data_dim_slicing(grid_LDOS_fft_r32, channel= 'LDOS_fft',log_data=True)
#grid_fft_data_dim_slicing(np.log(grid_LDOS_fft+1), channel= 'LDOS_fft',log_data=True)
grid_fft_data_dim_slicing(grid_LDOS_fft, channel= 'LDOS_fft',log_data=True)

# #### line profile bias_mV slicing 

# +
#grid_LDOS_fft_r32
# -


# ### interactive slicing for X, Y, bias_mV

# #### FFT line profile bias_mV slicing 

# ##### line_profile_xr_GUI

# %matplotlib inline


grid_LDOS_fft

# %matplotlib qt5
l_pf_start, l_pf_end, sliced_image_stack_xr, fig   = line_profile_xr_fft_GUI(grid_LDOS_fft,
                                                                             profile_width = 5, 
                                                                             log_plot= True,
                                                                             slicing_bias_mV=0, cmap = 'GnBu', perc =(2,98.5))

# +
#l_pf_start
l_pf_length = sp.spatial.distance.euclidean(l_pf_start, l_pf_end)
l_pf_length

grid_LDOS_fft

# %matplotlib inline
fig
# -

# ##### FFT line profile After crop Freq XY

grid_LDOS_fft

grid_LDOS_fft= grid_LDOS_fft[['LDOS_fft']].copy()
grid_LDOS_fft

# ####  Cropped FFT slicing 

grid_LDOS_fft.ref_a0nm

# +
#ref_q01overnm =  1E9*1/grid_LDOS_fft_r.ref_a0nm 
ref_q01overnm =  1E9*1/grid_LDOS_fft.ref_a0nm 
crop_win_ratio= 0.5

freq_X_max = ref_q01overnm * crop_win_ratio
freq_X_min = -ref_q01overnm * crop_win_ratio
freq_Y_max = ref_q01overnm * crop_win_ratio
freq_Y_min = -ref_q01overnm * crop_win_ratio
'''
grid_LDOS_fft_r_crop = grid_LDOS_fft_r.where((grid_LDOS_fft_r.freq_X<freq_X_max)&
                                                 (grid_LDOS_fft_r.freq_X>freq_X_min)&
                                                 (grid_LDOS_fft_r.freq_Y<freq_Y_max)&
                                                 (grid_LDOS_fft_r.freq_Y>freq_Y_min), drop = True)
'''

grid_LDOS_fft_crop = grid_LDOS_fft.where((grid_LDOS_fft.freq_X<freq_X_max)&
                                         (grid_LDOS_fft.freq_X>freq_X_min)&
                                         (grid_LDOS_fft.freq_Y<freq_Y_max)&
                                         (grid_LDOS_fft.freq_Y>freq_Y_min), drop = True)
grid_LDOS_fft_crop


# +

bias_mV_slices = np.linspace(-1.2,1.2, 15)  
plot_bias_sliced_grid_fft_images(grid_fft=np.log10(grid_LDOS_fft_crop),
                                 number_of_bias_slice=bias_mV_slices, 
                                 col_wrap=5, cmap ="GnBu",
                                 perc=(3, 99.8195),add_q0_circle= True, circle_ratio = 0.5)
#plot_bias_sliced_grid_fft_images(grid_fft=np.log10(grid_LDOS_fft_r32_crop), bias_mV_slices=bias_mV_slices, col_wrap=6, perc=(3, 99.99))
# -
np.linspace(-5, 5, 25)


'''
plot_XYsliced_grid_fft_images(grid_fft=np.log10(grid_LDOS_fft_r_crop).where(
    (grid_LDOS_fft_r_crop.bias_mV>-5)&(grid_LDOS_fft_r_crop.bias_mV<5),
    drop= True), 
                          number_of_XYslice=15, 
                          slicing='freq_Y', perc=(30, 99.699),
                          height=4, cmap ='GnBu',
                          col_wrap=6, 
                          aspect_ratio=1,add_q0_line= True, q0_ratio = 0.5,
                          use_individual_vlimits = False,cbar=False)
'''
plot_XYsliced_grid_fft_images(grid_fft=np.log10(grid_LDOS_fft_crop), 
                          number_of_XYslice=np.linspace(-0.5E9, 0.5E9, 12), 
                          slicing='freq_X', perc=(10, 99.699),
                          height=4, 
                          col_wrap=6,cmap ='GnBu', 
                          aspect_ratio=0.25,
                          use_individual_vlimits = False,cbar=False)

# +
#fft_XY_slicing_crop = hv_fft_XY_slicing(np.log10(grid_LDOS_fft_r32_crop),ch='LDOS_fft', slicing='freq_Y',cmap='Viridis')
fft_XY_slicing_crop = hv_fft_XY_slicing(np.log10(grid_LDOS_fft_crop),ch='LDOS_fft', slicing='freq_Y',cmap='GnBu')


# in case of crop & gaussain avg
'''
fft_XY_slicing_crop = hv_fft_XY_slicing(np.log10( filter_gaussian_xr( grid_LDOS_fft_crop, 
                                                                     sigma=1,
                                                                     overwrite= True) 
                                                ),ch='LDOS_fft', slicing='freq_Y',cmap='Viridis')
'''
#fft_XY_slicing_0




bragg_q0=  1  *   (1E9* 1/grid_LDOS.ref_a0nm ) 

# 수직 가이드 라인 추가

zerovlinePos = hv.VLine(0).opts(color='black', line_width=2, line_dash='solid', alpha = 0.5 )

q0vlinePos = hv.VLine(bragg_q0).opts(color='red', line_width=2, line_dash='dotted', alpha = 0.5 )
q0vlineNeg = hv.VLine(-bragg_q0).opts(color='red', line_width=2, line_dash='dotted', alpha = 0.5)
q0vlinePos_1_10 = hv.VLine(bragg_q0/10).opts(color='red', line_width=2, line_dash='dotted', alpha = 0.5 )
q0vlineNeg_1_10  = hv.VLine(-bragg_q0/10).opts(color='red', line_width=2, line_dash='dotted', alpha = 0.5)

fft_XY_slicing_crop.opts(clim = (-9.5,-7.0))*zerovlinePos*q0vlinePos*q0vlineNeg*q0vlinePos_1_10*q0vlineNeg_1_10

# without clim options
#fft_XY_slicing_crop*zerovlinePos*q0vlinePos*q0vlineNeg*q0vlinePos_1_10*q0vlineNeg_1_10
# -
# #### LDOS profile along grid_LDOS bias_slicing  GUI input

GS_LDOS_0T_002

# +
# %matplotlib qt5

l_pf_start_px,l_pf_end_px,fig =  line_profile_xr_GUI(grid_LDOS.sel(bias_mV = 0, method= 'nearest'), ch_name='LDOS', cmap='viridis', arrow_alpha= 0.3)

# +
l_pf_start_px,l_pf_end_px

#### After GUI 
## use the magic command 'inline'

# %matplotlib inline 
fig
#plt.show()

# -
def line_profile_xr_3d(xrdata, l_pf_start_px, l_pf_end_px, ch_name='LDOS', profile_width=3, bias_mV_slice=0, arrow_alpha=0.5, perc=(0, 98)):
    """
    Extract line profiles from a 3D xarray dataset along the specified line.
    Use it after assigning the l_pf_start_px, l_pf_end_px from the "line_profile_xr_GUI" function.

    Parameters
    ----------
    xrdata : xarray.DataArray
        3D xarray dataset with dimensions (freq_X, freq_Y, bias_mV).
    l_pf_start_px : tuple
        Starting point of the line (x, y) index in pixels.
    l_pf_end_px : tuple
        Ending point of the line (x, y) index in pixels.
    ch_name : str, optional
        The channel name to extract the line profile from. Default is 'LDOS'.
    profile_width : int, optional
        The width of the line profile, perpendicular to the line. Default is 3.
    bias_mV_slice : float, optional
        The bias_mV value to slice the data. Default is 0.
    arrow_alpha : float, optional
        The transparency of the arrow used in the plot. Default is 0.5.
    perc : tuple, optional
        Percentile values to control the display range of the data in the second plot. Default is (0, 98).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the line profile plot.
    line_profiles_xr : xarray.DataArray
        2D xarray.DataArray containing the line profiles for each bias_mV value.
    """
    import matplotlib.ticker as mticker
    import scipy as sp
    import skimage.measure
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    import xarray as xr
    import seaborn_image as isns

    [size_x, size_y] = xrdata.image_size
    print(ch_name)
    l_pf_start_px = l_pf_start_px[::-1]
    l_pf_end_px = l_pf_end_px[::-1]
    l_pf_length = sp.spatial.distance.euclidean(l_pf_start_px, l_pf_end_px) * xrdata.X_spacing
    print('line profile_length = ', l_pf_length)

    line_profiles = []
    bias_mV_values = xrdata.coords['bias_mV'].values
    for bias_mV in bias_mV_values:
        data_slice = xrdata[ch_name].sel(bias_mV=bias_mV).values
        l_pf = skimage.measure.profile_line(data_slice, l_pf_start_px, l_pf_end_px, linewidth=profile_width, reduce_func=np.mean, mode='reflect')
        line_profiles.append(l_pf)
    line_profiles = np.array(line_profiles).T
    pixel_spacing = l_pf_length / (line_profiles.shape[0] - 1)
    distance_values = np.linspace(0, l_pf_length, line_profiles.shape[0])

    # Reverse the distance values to match the line profile direction
    distance_values = distance_values[::-1]

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        
    # Left plot (First plot, restored)
    slice_bias_mV = xrdata.sel(bias_mV=bias_mV_slice, method='nearest').bias_mV.item()
    isns.imshow(xrdata[ch_name].sel(bias_mV=slice_bias_mV).values, robust=True, origin="lower", ax=axs[0])
    axs[0].arrow(l_pf_start_px[0], l_pf_start_px[1], l_pf_end_px[0] - l_pf_start_px[0], l_pf_end_px[1] - l_pf_start_px[1], 
                 width=2, color='red', alpha=arrow_alpha)
    axs[0].set_title(f"{xrdata.title}\nslicing at {slice_bias_mV:.2f} mV", fontsize='medium')
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    
    # Right plot (Second plot, controlled by `perc`)
    im = axs[1].imshow(line_profiles, cmap='viridis', aspect='auto', origin='upper', extent=[bias_mV_values[0], bias_mV_values[-1], 0, l_pf_length], 
                       vmin=np.percentile(line_profiles, perc[0]), vmax=np.percentile(line_profiles, perc[1]))
    axs[1].set_xlabel("Bias (mV)", fontsize=12)
    axs[1].set_ylabel("Length (nm)", fontsize=12)
    if bias_mV_values[0] > bias_mV_values[-1]:
        axs[1].invert_xaxis()
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axs[1])
    
    # Add subtitle for the second plot
    axs[1].set_title(f"Line Profile (perc={perc})", fontsize='medium')

    # Add major ticks and white dashed lines
    axs[1].xaxis.set_major_locator(mticker.AutoLocator())
    axs[1].grid(axis='x', which='major', color='white', linestyle='--', linewidth=0.5)

    # Add vertical arrow to the right side of the colorbar
    cbar_pos = cbar.ax.get_position()
    arrow_x = cbar_pos.x1 + 0.1  # Adjust this value to move the arrow left or right
    arrow_start = (arrow_x, cbar_pos.y0)
    arrow_end = (arrow_x, cbar_pos.y1)
    fig.add_artist(plt.Arrow(arrow_start[0], arrow_start[1], 0, arrow_end[1] - arrow_start[1], 
                         width=0.01, color='red'))
    
    # Add text next to the arrow
    fig.text(arrow_x + 0.02, (cbar_pos.y0 + cbar_pos.y1) / 2, 'Line Profile Direction', 
         rotation=90, va='center', ha='left', color='red')

    plt.tight_layout()
    plt.show()

    line_profiles_xr = xr.DataArray(line_profiles, dims=['distance', 'bias_mV'],
                                    coords={'distance': distance_values, 'bias_mV': bias_mV_values},
                                    attrs={'description': 'Line profile along the specified path',
                                           'l_pf_start_px': l_pf_start_px,
                                           'l_pf_end_px': l_pf_end_px,
                                           'profile_width': profile_width})
    
    print("check line profile arrow direction")
    return fig, line_profiles_xr



fig,GUI_line_profiles_LDOS = line_profile_xr_3d(grid_LDOS, l_pf_start_px=l_pf_start_px,l_pf_end_px=l_pf_end_px, perc=(0, 88))

GUI_line_profiles_LDOS

# #### add define deltaoverEf

# +
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# %matplotlib inline 
# Calculate the 2nd and 98th percentiles
perc  = [2, 78]
vmin, vmax = np.percentile(GUI_line_profiles_LDOS, perc )

# Plot the DataArray with the color limits set to the calculated percentiles
fig, ax = plt.subplots(figsize=(4, 4))
im = GUI_line_profiles_LDOS.plot(
    ax=ax,
    cmap='viridis',
    vmin=vmin,
    vmax=vmax
)

# Add vertical grid lines
ax.xaxis.grid(True, which='major', color='white', linestyle='--', linewidth=0.5)

# Add vertical lines at specific x-coordinates
deltaoverEf  = 0.6
v_lines_int = np.array( [-3,-2,-1,1,2,3]) * deltaoverEf
v_lines_half_int = np.array([-2.5,-1.5,-0.5,0.5,1.5,2.5])* deltaoverEf
for x in v_lines_int:
    ax.axvline(x=x, color='blue', linestyle='-',linewidth=0.5)
for x in v_lines_half_int:
    ax.axvline(x=x, color='red', linestyle='--',linewidth=0.5)


# Show the plot
plt.title('Line Profile: v_perc = '+ str(perc )+ '\n' + ' guile line \n' + 'int (red)  & half int(blue) ' + r'$ \times $ ' +' \n'
 + r' ${\Delta ^{2} }/{E_{f}}$ (' + str(deltaoverEf)+ ' meV)')
plt.show()
# -


# ###  Find the symmetric peaks --> 
# * at the same position, symmetric within bias_mV_tolerance 
# * bias_mV tolerance increase --> # of symmetry peaks --> find the offset and compare to Zerobais conductance peak.
# * 
#



# # Peak detection  , Whole Range, InGapRange

# ### including SC gap. whole area 
#

#grid_topo = GS_topo_0T_002.copy()#
grid_topo = GS_topo_2T_003.copy()
#grid_LDOS = updated_GS_LDOS_0T_002[['LDOS']].copy()
#grid_LDOS = GS_LDOS_0T_002[['LDOS_smoothed']].copy()
grid_LDOS = GS_LDOS_2T_003[['LDOS_smoothed']].copy()
grid_LDOS = grid_LDOS.rename({'LDOS_smoothed': 'LDOS'})

# #### Use the the same area 0T and 2T data 
#

# +
updated_GS_topo_0T_002
updated_GS_LDOS_0T_002

#GS_LDOS_2T_003


# +
## extract the mask from 0T for 2T data 
# -

# ## choose the dataset 
# *  ~~2T_005~~  topo & LDOS data select
# * ~~updated_GS_LDOS_0T_002~~
# * GS_LDOS_2T_003

# ### choose updated (selected) area

# +
#GS_LDOS_2T_003
#GS_LDOS_2T_003

grid_topo = updated_GS_topo_0T_002.copy()#
#grid_topo = GS_topo_2T_005.copy()
#grid_LDOS = updated_GS_LDOS_0T_002[['LDOS']].copy()
grid_LDOS = updated_GS_LDOS_0T_002[['LDOS_smoothed']].copy()


#grid_topo = GS_topo_2T_003.copy()
#GS_LDOS_2T_003


#grid_LDOS = GS_LDOS_2T_005[['LDOS']].copy()
#grid_LDOS = GS_LDOS_2T_005[['LDOS_smoothed']].copy()


#grid_LDOS = GS_LDOS_2T_003[['LDOS']].copy()
#grid_LDOS = GS_LDOS_2T_003[['LDOS_smoothed']].copy()

# use the LDOS smooth? 
grid_LDOS = grid_LDOS.rename({'LDOS_smoothed': 'LDOS'})
# -

# ### LDOS data fitting range and parameter test 

grid_LDOS

# ## +- 0.8mV range fitting

# +
grid_LDOS_SnD = smoothing_and_deriv_LDOS(
    grid_LDOS.sel(bias_mV = slice(0.75,-0.75)),window_length_ratio=0.1)

# zoom in in gap states 
#grid_LDOS_SnD = smoothing_and_deriv_LDOS(grid_LDOS.sel(bias_mV = slice(1,-1)),
#                                         window_length_ratio=0.1 )

#find peaks in grid_LDOS_SnD

# find peaks 
grid_LDOS_SnD_pks = find_pks_grid_LDOS(grid_LDOS_SnD,threshold=0.1E-12, distance=None, prominence = 1E-12)
#grid_LDOS_SnD_pks
plot_ldos_with_pks(grid_LDOS_SnD_pks, wrap_ncols=5, num_random_points=20)
# -

grid_LDOS_SnD_pks
#plt.show()

# ## peak detectio near Zero Bias 

# +
#grid_LDOS_SnD_pks.to_netcdf('grid_LDOS_SnD_pks.nc')
# -

# ## ZB masking 

'''
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import ipywidgets as widgets
from IPython.display import display

def zero_bias_masking_with_threshold(dataset, bias_mV=0):
    """
    Perform Zero Bias analysis and apply various thresholding methods to the dataset,
    allowing the user to interactively select a threshold method to generate a 'ZB_mask'.

    This function does the following:
      - Extracts a bias window around the specified bias value (bias_mV ± 0.5 mV) and computes
        the mean and minimum LDOS (Local Density of States) within that window.
      - Identifies Zero Bias Peak (ZBP) regions based on a superconducting gap criterion,
        where a gap is defined if the minimum LDOS is less than 50% of the mean LDOS.
      - Plots the spatial distribution of the ZBP regions and the zero bias conductance intensity.
      - Applies several externally defined thresholding methods to the data and displays their results
        in subplots.
      - Provides interactive buttons for the user to select one of the threshold methods.
      - When a threshold method is selected, the corresponding threshold mask is computed and added
        to the dataset as the 'ZB_mask' data array. If a 'ZB_mask' already exists, it is overwritten.
      - The selected threshold method is recorded in the dataset's attributes under 'ZB_mask_threshold'.

    Parameters:
    -----------
    dataset : xarray.Dataset
        The input dataset containing the LDOS data and associated spatial coordinates.
    bias_mV : float, optional
        The bias value to use for Zero Bias analysis (default is 0 mV).

    Returns:
    --------
    xarray.Dataset
        The modified dataset including the added 'ZB_mask' data array and the 'ZB_mask_threshold' attribute.
    """
    # Create a copy of the input dataset to avoid modifying the original data.
    ds = dataset.copy()

    # -----------------------------
    # Zero Bias Analysis
    # -----------------------------
    # Select a bias window around the specified bias value.
    # The window is defined from (bias_mV + 0.5) to (bias_mV - 0.5).
    ldos_near_bias = ds.LDOS.sel(bias_mV=slice(bias_mV + 0.5, bias_mV - 0.5))
    
    # Compute the mean LDOS within the selected bias window.
    mean_ldos = ldos_near_bias.mean(dim='bias_mV')
    # Compute the minimum LDOS within the selected bias window.
    min_ldos = ldos_near_bias.min(dim='bias_mV')
    
    # Define the superconducting gap criterion:
    # A gap is assumed if the minimum LDOS is less than 50% of the mean LDOS.
    gap_threshold = mean_ldos * 0.5
    is_gap = min_ldos < gap_threshold
    # Determine the presence of Zero Bias Peaks (ZBP) by inverting the gap condition.
    is_zbp = ~is_gap  
    
    # Select the LDOS value at the specified bias (or the nearest available bias value).
    zero_bias_ldos = ds.LDOS.sel(bias_mV=bias_mV, method='nearest')
    
    # -----------------------------
    # Threshold Methods Setup
    # -----------------------------
    # Define a list of thresholding methods.
    # Each tuple contains a method name and a corresponding function that applies the threshold.
    # These threshold functions (e.g., threshold_mean_xr, threshold_local_xr, etc.) must be defined externally.
    threshold_methods = [
        ("Mean", threshold_mean_xr),
        ("Local", threshold_local_xr),
        ("Isodata", threshold_isodata_xr),
        ("Minimum", threshold_minimum_xr),
        ("Otsu", threshold_otsu_xr),
        ("MultiOtsu", threshold_multiotsu_xr),
        ("Sauvola", threshold_sauvola_xr),
        ("Yen", threshold_yen_xr)
    ]
    
    # Extract the data for threshold analysis at the specified bias.
    # Only the 'LDOS' data array is selected for this purpose.
    data = ds.sel(bias_mV=bias_mV, method='nearest')[['LDOS']]
    
    # Calculate the aspect ratio based on the spatial dimensions 'Y' and 'X'
    # to ensure that plots maintain the correct spatial proportions.
    aspect_ratio = ds.sizes['Y'] / ds.sizes['X']
    
    # -----------------------------
    # Create Combined Figure using GridSpec for Plotting
    # -----------------------------
    # Create a figure with a specified size.
    fig = plt.figure(figsize=(12, 15))
    # Define a GridSpec layout with 3 rows and 4 columns.
    # Here, all rows are assigned equal height ratios.
    gs = gridspec.GridSpec(3, 4, height_ratios=[1, 1, 1])
    
    # Create subplots for the first row:
    # ax1 will display the spatial distribution of Zero Bias Peak regions.
    # ax2 will display the spatial distribution of Zero Bias Conductance Intensity.
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    
    # Plot the Zero Bias Peak regions using a red colormap.
    is_zbp.plot(ax=ax1, cmap='Reds')
    ax1.set_title('Spatial Distribution of Gapped Regions')
    ax1.set_aspect(aspect_ratio)
    
    # Plot the Zero Bias Conductance Intensity using an 'inferno' colormap.
    zero_bias_ldos.plot(ax=ax2, cmap='inferno')
    ax2.set_title("Spatial Distribution of Zero Bias Conductance Intensity")
    ax2.set_aspect(aspect_ratio)
    
    # -----------------------------
    # Plot Threshold Method Results in Subplots
    # -----------------------------
    # Create a list to store the axes for the threshold method plots.
    axes_thresh = []
    # Generate subplots for threshold results in the bottom 2 rows (4 columns each).
    for i in range(2):
        for j in range(4):
            ax = fig.add_subplot(gs[i+1, j])
            axes_thresh.append(ax)
    
    # Loop over each threshold method and its corresponding subplot axis.
    for ax, (th_name, th_func) in zip(axes_thresh, threshold_methods):
        # Apply the threshold method function to the selected data.
        result = th_func(data)
        # Plot the resulting threshold mask using the 'viridis' colormap.
        im = result.LDOS.plot(ax=ax, cmap='viridis', add_colorbar=False)
        ax.set_title(th_name)
        ax.set_aspect(aspect_ratio)
        # Add a horizontal colorbar to the subplot for reference.
        plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.15)
    
    # Adjust the layout to minimize overlapping elements.
    plt.tight_layout()
    # Display the combined figure with all subplots.
    plt.show()
    
    # -----------------------------
    # Interactive Button Selection for Threshold Method
    # -----------------------------
    # Inform the user to select a threshold method via interactive buttons.
    print("Select a threshold method by clicking one of the buttons below:")
    
    # Dictionary to store the selected threshold function and its name.
    selected_method = {'func': None, 'name': None}
    # List to hold the button widgets.
    buttons = []
    
    # Callback function that will be executed when a button is clicked.
    def on_button_clicked(b):
        # Retrieve the name of the threshold method from the button's description.
        selected_name = b.description
        # Find the corresponding threshold function for the selected method.
        for name, func in threshold_methods:
            if name.lower() == selected_name.lower():
                selected_method['func'] = func
                selected_method['name'] = name
                break
        
        # Apply the selected threshold function to the data.
        selected_result = selected_method['func'](data)
        # Extract the threshold mask from the result.
        mask = selected_result.LDOS
        
        # Check if a 'ZB_mask' already exists in the dataset.
        if 'ZB_mask' in ds:
            print("ZB_mask already exists. Overwriting the existing data.")
        # Add (or overwrite) the 'ZB_mask' data array in the dataset with the computed mask.
        ds['ZB_mask'] = mask
        
        # Record the selected threshold method in the dataset's attributes.
        ds.attrs['ZB_mask_threshold'] = selected_method['name']
        
        # Inform the user of the selection and confirm that the mask has been added.
        print(f"Selected threshold method: {selected_method['name']}")
        print(f"'ZB_mask' has been added to the dataset using the {selected_method['name']} threshold method.")
        
        # Disable all buttons to prevent further selections after one method is chosen.
        for btn in buttons:
            btn.disabled = True

    # Create a button widget for each threshold method.
    for th_name, _ in threshold_methods:
        btn = widgets.Button(description=th_name)
        btn.on_click(on_button_clicked)
        buttons.append(btn)
    
    # Arrange the buttons horizontally in a container.
    container = widgets.HBox(buttons)
    # Display the container with the buttons.
    display(container)
    
    # Return the modified dataset that now includes the threshold mask and its corresponding attributes.
    return ds


'''

grid_LDOS_SnD_pks = zero_bias_masking_with_threshold(grid_LDOS_SnD_pks, bias_mV=0)





grid_LDOS_SnD_pks

grid_LDOS_SnD_pks.ZB_mask.plot()
plt.show()

# ## Select multiple points  using  simple Lorentzian fitting test

# # use the LMFIFT package fitting instead of Lorentzian 
#

# +
#grid_LDOS_SnD_pks = xr.open_dataset('grid_LDOS_SnD_pks_2T005.nc')

#grid_LDOS_SnD_pks.to_netcdf('grid_LDOS_SnD_pks_0T002_WholeRange.nc')
grid_LDOS_SnD_pks.to_netcdf('grid_LDOS_SnD_pks_2T003_750uV_0628.nc')

#grid_LDOS_SnD_pks = xr.open_dataset('grid_LDOS_SnD_pks_2T003_800uV_0416.nc')
# -

grid_LDOS_SnD_pks#.data_vars

# ### interactive fitting test 

# +
# ───────────────────────────────────────────────────────────────────────────────
# Interactive multi-peak weighted fitting of LDOS data with inline plots
# ───────────────────────────────────────────────────────────────────────────────

# Ensure matplotlib plots appear below the code cell
# %matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import LorentzianModel, GaussianModel, VoigtModel
from functools import reduce
import ipywidgets as widgets
from IPython.display import display

def interactive_fitting(grid_LDOS_SnD_pks):
    """
    Creates an interactive widget for exploring multi-peak weighted fitting of LDOS data.
    It uses 2nd-derivative peak information (bias and prominence) from the dataset as
    initial guesses for multi-peak fitting. The user can adjust the Y index, X index, and
    the sigma parameter for a Gaussian weight function centered at zero bias. After setting
    the initial values, clicking the 'Calculate' button updates the plots and fits 
    inline, directly beneath the cell.

    Parameters
    ----------
    grid_LDOS_SnD_pks : xarray.Dataset
        An xarray dataset containing:
          - 'LDOS' (the raw LDOS data)
          - 'bias_mV' (bias voltages)
          - 'LDOS_smoothed' (smoothed LDOS data)
          - 'LDOS_2deriv_smoothed' (2nd derivative of the smoothed LDOS)
          - various peak metadata arrays for both smoothed and 2nd-derivative data
    """
    # Copy dataset to avoid mutating original
    ds = grid_LDOS_SnD_pks.copy()

    # -------------------------------------------------------------------------
    # 1) Gaussian Weight Function
    # -------------------------------------------------------------------------
    def weight_function(x, center=0.0, sigma=1.0):
        """Gaussian weights centered at zero bias."""
        return np.exp(-((x - center) ** 2) / (2 * sigma ** 2))

    # -------------------------------------------------------------------------
    # 2) Multi-Peak Weighted Fitting
    # -------------------------------------------------------------------------
    def fit_multi_peak_with_weights(x_data, y_data, weights, cen_init, amp_init, model_class):
        """Builds and fits a sum of single-peak models weighted by 'weights'."""
        if len(cen_init) == 0:
            return None, None
        models = [model_class(prefix=f'peak{i}_') for i in range(len(cen_init))]
        model  = reduce(lambda m1, m2: m1 + m2, models)
        params = model.make_params()
        for i, c in enumerate(cen_init):
            params[f'peak{i}_amplitude'].set(value=amp_init[i], min=0)
            params[f'peak{i}_center']   .set(value=c, min=x_data.min(), max=x_data.max())
            params[f'peak{i}_sigma']    .set(value=0.1, min=0.0)
        result = model.fit(y_data, params, x=x_data, weights=weights)
        comps  = result.eval_components(x=x_data)
        return result, comps

    # -------------------------------------------------------------------------
    # 3) Plotting Routine
    # -------------------------------------------------------------------------
    def plot_fitting(ds, y_idx, x_idx, sigma):
        bias    = ds['bias_mV'].values
        ldos_map= ds['LDOS'].sel(bias_mV=0, method='nearest')
        ldos_sm = ds['LDOS_smoothed'].isel(Y=y_idx, X=x_idx).values
        ldos_2d = ds['LDOS_2deriv_smoothed'].isel(Y=y_idx, X=x_idx).values

        # 2nd-derivative peak info
        pb_2d, pp_2d, ph_2d = (ds['LDOS_2deriv_smoothed_peak_bias']       .isel(Y=y_idx,X=x_idx).values,
                               ds['LDOS_2deriv_smoothed_peak_prominences'].isel(Y=y_idx,X=x_idx).values,
                               ds['LDOS_2deriv_smoothed_peak_heights']    .isel(Y=y_idx,X=x_idx).values)
        pw_2d, pwh_2d = (ds['LDOS_2deriv_smoothed_peak_widths_mV']      .isel(Y=y_idx,X=x_idx).values,
                         ds['LDOS_2deriv_smoothed_peak_width_heights'] .isel(Y=y_idx,X=x_idx).values)

        # smoothed-LDOS peak info
        pb_sm, pp_sm, ph_sm = (ds['LDOS_smoothed_peak_bias']       .isel(Y=y_idx,X=x_idx).values,
                               ds['LDOS_smoothed_peak_prominences'].isel(Y=y_idx,X=x_idx).values,
                               ds['LDOS_smoothed_peak_heights']    .isel(Y=y_idx,X=x_idx).values)
        pw_sm, pwh_sm = (ds['LDOS_smoothed_peak_widths_mV']      .isel(Y=y_idx,X=x_idx).values,
                         ds['LDOS_smoothed_peak_width_heights'] .isel(Y=y_idx,X=x_idx).values)

        # initial guesses from 2nd-derivative peaks
        mask     = (~np.isnan(pb_2d)) & (~np.isnan(pp_2d))
        cen_init = pb_2d[mask]
        amp_init = pp_2d[mask]
        w        = weight_function(bias, sigma=sigma)

        # perform fits
        lorentz_res, lorentz_comps = fit_multi_peak_with_weights(bias, ldos_sm, w, cen_init, amp_init, LorentzianModel)
        gauss_res,   gauss_comps   = fit_multi_peak_with_weights(bias, ldos_sm, w, cen_init, amp_init, GaussianModel)
        voigt_res,   voigt_comps   = fit_multi_peak_with_weights(bias, ldos_sm, w, cen_init, amp_init, VoigtModel)

        # (a) LDOS map + smoothed/2nd-deriv spectrum
        fig1, (ax1, ax2) = plt.subplots(2,1,figsize=(8,10))
        ax1.imshow(ldos_map, origin='lower', cmap='viridis', aspect='equal')
        ax1.scatter(x_idx, y_idx, color='red', s=50)
        ax1.set_title(f"LDOS map @ 0 mV (Y={y_idx}, X={x_idx})")
        ax2.plot(bias, ldos_sm, 'r', label='LDOS_smoothed'); ax2.set_ylabel('LDOS_smoothed', color='r')
        ax2.set_xlabel('Bias (mV)')
        ax3 = ax2.twinx(); ax3.plot(bias, ldos_2d, 'purple', label='LDOS_2deriv')
        ax3.set_ylabel('LDOS_2deriv', color='purple')
        # overlay peaks...
        for b_,h_,p_,w_,wh_ in zip(pb_2d,ph_2d,pp_2d,pw_2d,pwh_2d):
            if not np.isnan(b_): ax3.plot(b_,h_,'^m'); ax3.vlines(b_,h_-p_,h_,linestyle='--',color='m')
        for b_,h_,p_,w_,wh_ in zip(pb_sm,ph_sm,pp_sm,pw_sm,pwh_sm):
            if not np.isnan(b_): ax2.plot(b_,h_,'or'); ax2.vlines(b_,h_-p_,h_,linestyle='--',color='r')
        fig1.tight_layout(); plt.show()

        # (b-d) Lorentzian, Gaussian, Voigt fits
        for res,comps,color,name in [
            (lorentz_res,lorentz_comps,'r','Lorentz'),
            (gauss_res,  gauss_comps,  'b','Gaussian'),
            (voigt_res,  voigt_comps,  'g','Voigt')
        ]:
            fig, ax = plt.subplots(figsize=(6,5))
            ax.plot(bias,ldos_sm,'k',alpha=0.6,label='LDOS_smoothed')
            if res is not None:
                ax.plot(bias,res.best_fit, color, label=f'{name} Fit (χ²={res.chisqr:.2f})')
                for i, comp in enumerate(comps.values()):
                    ax.plot(bias,comp,'--', label=f'Peak {i}')
            ax.axvline(0, color='gray', linestyle=':')
            ax.set_title(f"{name} Weighted Fit (peaks={len(cen_init)}, σ={sigma})")
            ax.legend(); fig.tight_layout(); plt.show()

    # -------------------------------------------------------------------------
    # 4) Build Widgets + Output Container
    # -------------------------------------------------------------------------
    ny, nx = ds.dims['Y'], ds.dims['X']
    slider_y     = widgets.IntSlider(min=0,   max=ny-1,  step=1,   description='Y index', value=0)
    slider_x     = widgets.IntSlider(min=0,   max=nx-1,  step=1,   description='X index', value=0)
    slider_sigma = widgets.FloatSlider(min=0.1, max=2.0, step=0.1, description='Sigma',   value=1.0)
    button_calc  = widgets.Button(description='Calculate', button_style='info')

    output = widgets.Output()

    def on_button_clicked(b):
        with output:
            output.clear_output(wait=True)
            plot_fitting(ds, slider_y.value, slider_x.value, slider_sigma.value)

    button_calc.on_click(on_button_clicked)

    # display controls + output inline
    display(widgets.VBox([slider_y, slider_x, slider_sigma, button_calc, output]))



# -

# %matplotlib inline
#matplotlib widget

interactive_fitting(grid_LDOS_SnD_pks)
# check results  at the log window






# ## multi peak fitting for single XY 

'''
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from lmfit.models import LorentzianModel, GaussianModel, VoigtModel, ConstantModel
from functools import reduce

def multi_peak_IGS_fitting_singleXY(
    grid_LDOS_SnD_pks,
    y_idx=None,
    x_idx=None,
    weight_sigma=1.0,
    model_type=['Lorentzian','Gaussian','Voigt'],
    visualize=True,
    margin2zerobias=None,
    margin2levels=None,
    max_peaks=None,  # Maximum number of peaks to consider
    background_offset=False,
    background_offset_adjust=False,
    exclude_negative=True,       # Exclude negative values flag
    Ef=4.4,
    SCgap=1.8,
    show_initial_guess=False,    # Show initial guess markers (vertical line, text, and detailed 2nd derivative markers)
    show_weight_function=False   # Show weight function on plot
):
    """
    Perform multi-peak weighted fitting on a single LDOS spectrum with detailed debug and visualization.
    
    This function performs the following steps:
      1) Uses the 2nd derivative peak information from 'grid_LDOS_SnD_pks' to guess initial peak parameters 
         (center, amplitude, sigma). The initial guess is refined using local maximum information.
      2) Optionally excludes negative LDOS data; if background_offset is True, negative values can be adjusted.
      3) Applies a Gaussian weight centered at bias=0 if weight_sigma is provided.
      4) Limits the number of peaks based on 'max_peaks' and the available data points to avoid lmfit errors.
      5) Tries each specified model in 'model_type' (e.g. Lorentzian, Gaussian, Voigt) and selects the best fit 
         based on chi-square.
      6) (Optional) Visualizes:
            (a) The LDOS map at bias=0 with a marker at the chosen (y_idx,x_idx).
            (b) A two-panel plot: 
                - The top panel displays the LDOS map.
                - The bottom panel displays the smoothed LDOS and its 2nd derivative.
                  If show_initial_guess is True, the initial guess peaks are indicated with vertical dashed lines 
                  (on the LDOS curve) and detailed markers (on the 2nd derivative curve) showing the peak position, 
                  amplitude, and width (via horizontal lines).
            (c) Individual fit results for each model.
            (d) A combined comparison plot with weight function (if requested) and CdGM level reference lines.
      7) Returns an xarray.Dataset containing the fitted peak parameters, the best-fit curve, background, 
         and detailed parameter information.
      8) Debug information such as number of data points and peak counts are stored in ds_out.attrs.

    Parameters
    ----------
    grid_LDOS_SnD_pks : xarray.Dataset
        Input dataset containing LDOS and pre-computed 2nd derivative peak information.
    y_idx, x_idx : int, optional
        Pixel indices to select the spectrum; if None, random indices are chosen.
    weight_sigma : float, optional
        Standard deviation of the Gaussian weight function centered at bias=0. If None, no weighting is applied.
    model_type : list or str, optional
        List (or single string) specifying which model(s) to use for fitting (e.g. 'Lorentzian', 'Gaussian', 'Voigt').
    visualize : bool, optional
        If True, display various plots for debugging and visualization.
    margin2zerobias, margin2levels : float, optional
        Thresholds for judging proximity of a peak to zero bias or candidate levels.
    max_peaks : int, optional
        Maximum number of peaks to consider in the fitting.
    background_offset : bool, optional
        If True, process the negative LDOS values separately.
    background_offset_adjust : bool, optional
        If True and background_offset is True, adjust the data so that the minimum value becomes zero.
    exclude_negative : bool, optional
        If True, exclude negative LDOS data from the fitting (default: True).
    Ef : float, optional
        Fermi energy used for CdGM energy level calculation.
    SCgap : float, optional
        Superconducting gap value used for CdGM energy level calculation.
    show_initial_guess : bool, optional
        If True, display the initial guess markers:
            - Vertical dashed lines on the smoothed LDOS plot.
            - Detailed markers on the 2nd derivative plot indicating peak positions, amplitudes, and widths.
    show_weight_function : bool, optional
        If True, display the weight function on the combined fit comparison plot.

    Returns
    -------
    xarray.Dataset
        Dataset containing:
            - 'peak_center', 'peak_amplitude', 'peak_sigma': Fitted peak parameters.
            - 'level_proximity': Proximity of each peak to candidate energy levels.
            - 'ldos_sm', 'best_fit', 'background_fit', 'peak_fit': Fitting curves.
            - Parameter details: 'param_value', 'param_stderr', 'param_min', 'param_max', 'param_vary', 'param_expr'.
        Additionally, debug and metadata information is stored in ds_out.attrs.
    """

    ds = grid_LDOS_SnD_pks.copy()

    def weight_function(x, center=0.0, sigma=1.0):
        """Return a Gaussian weight centered at 'center' with standard deviation 'sigma'."""
        return np.exp(-0.5 * ((x - center) / sigma)**2)

    def fit_multi_peak_with_weights(x_data, y_data, weights, cen_init, amp_init, sigma_init, model_class):
        """
        Build and fit a composite model (constant background + multiple peaks) to the data.

        Background is constrained so that:
          - It does not exceed the median of y_data.
          - It does not exceed the minimum of the peak amplitudes.
        """
        if len(cen_init) == 0:
            return None, None

        peak_models = [model_class(prefix=f'peak{i}_') for i in range(len(cen_init))]
        peaks_model = reduce(lambda m1, m2: m1 + m2, peak_models)
        bg_model = ConstantModel(prefix='bg_')
        model = bg_model + peaks_model

        params = model.make_params()

        median_ldos = np.median(y_data)
        min_peak_amp = min(amp_init) if len(amp_init) > 0 else 0.0
        bg_max = min(median_ldos, min_peak_amp)
        params['bg_c'].set(value=np.min(y_data), min=0.0, max=bg_max)

        for i in range(len(cen_init)):
            params[f'peak{i}_amplitude'].set(value=amp_init[i], min=0)
            params[f'peak{i}_center'].set(value=cen_init[i], min=x_data.min(), max=x_data.max())
            params[f'peak{i}_sigma'].set(value=sigma_init[i], min=0.0)

        result = model.fit(y_data, params, x=x_data, weights=weights)
        comps = result.eval_components(x=x_data)
        return result, comps

    # 2) Determine pixel indices; if not provided, choose random indices
    ny, nx = ds.dims['Y'], ds.dims['X']
    if y_idx is None:
        y_idx = np.random.randint(0, ny)
    if x_idx is None:
        x_idx = np.random.randint(0, nx)

    # 3) Extract data and 2nd derivative peak information for the chosen pixel
    bias = ds['bias_mV'].values
    ldos_sm = ds['LDOS_smoothed'].isel(Y=y_idx, X=x_idx).values
    ldos_2d = ds['LDOS_2deriv_smoothed'].isel(Y=y_idx, X=x_idx).values

    pb_2d = ds['LDOS_2deriv_smoothed_peak_bias'].isel(Y=y_idx, X=x_idx).values
    pp_2d = ds['LDOS_2deriv_smoothed_peak_prominences'].isel(Y=y_idx, X=x_idx).values
    ph_2d = ds['LDOS_2deriv_smoothed_peak_heights'].isel(Y=y_idx, X=x_idx).values   
    pw_2d = ds['LDOS_2deriv_smoothed_peak_widths_mV'].isel(Y=y_idx, X=x_idx).values
    pwh_2d = ds['LDOS_2deriv_smoothed_peak_width_heights'].isel(Y=y_idx, X=x_idx).values

    # Remove NaN values from the peak information arrays
    valid_mask = (~np.isnan(pb_2d)) & (~np.isnan(pp_2d))
    cen_init_orig = pb_2d[valid_mask]
    width_init_orig = pw_2d[valid_mask]

    # (3-1) Compute initial amplitudes based on the closest bias value for each peak
    amp_init_raw = []
    for c in cen_init_orig:
        idx_closest = np.argmin(np.abs(bias - c))
        amp_init_raw.append(ldos_sm[idx_closest])
    amp_init_raw = np.array(amp_init_raw)
    
    # (3-2) Refine initial guesses using local maximum within the peak width window
    new_cen_init = []
    new_amp_init = []
    new_sigma_init = []
    for i, c in enumerate(cen_init_orig):
        window = width_init_orig[i] if width_init_orig[i] > 0 else 0.5
        in_window = (bias >= (c - window/2)) & (bias <= (c + window/2))
        if np.any(in_window):
            local_bias = bias[in_window]
            local_ldos = ldos_sm[in_window]
            max_idx = np.argmax(local_ldos)
            new_center = local_bias[max_idx]
            new_amp = local_ldos[max_idx]
        else:
            new_center = c
            new_amp = amp_init_raw[i]
        new_amp = new_amp if new_amp > 0 else 0.0
        new_cen_init.append(new_center)
        new_amp_init.append(new_amp)
        new_sigma_init.append(width_init_orig[i])
    new_cen_init = np.array(new_cen_init)
    new_amp_init = np.array(new_amp_init)
    new_sigma_init = np.array(new_sigma_init)

    # 4) Exclude negative data if flag is set; otherwise adjust background if requested
    if exclude_negative:
        valid_idx = (ldos_sm >= 0)
        bias = bias[valid_idx]
        ldos_sm = ldos_sm[valid_idx]
        ldos_2d = ldos_2d[valid_idx]
    else:
        if background_offset:
            if background_offset_adjust:
                min_val = np.min(ldos_sm)
                if min_val < 0:
                    offset = abs(min_val)
                    ldos_sm += offset
                    ldos_2d += offset
            else:
                valid_idx = (ldos_sm >= 0)
                bias = bias[valid_idx]
                ldos_sm = ldos_sm[valid_idx]
                ldos_2d = ldos_2d[valid_idx]

    # 5) Construct weight function if weight_sigma is provided
    if weight_sigma is None:
        weights = None
    else:
        weights = weight_function(bias, center=0.0, sigma=weight_sigma)

    # 6) Ensure model_type is a list
    if isinstance(model_type, str):
        model_list = [model_type]
    else:
        model_list = model_type

    # 7) Perform fitting for each specified model type
    fit_results = {}
    for mtype in model_list:
        if len(new_cen_init) == 0:
            continue

        if mtype.lower() == 'lorentzian':
            model_class = LorentzianModel
        elif mtype.lower() == 'gaussian':
            model_class = GaussianModel
        elif mtype.lower() == 'voigt':
            model_class = VoigtModel
        else:
            continue

        res, comps = fit_multi_peak_with_weights(
            bias, ldos_sm, weights,
            new_cen_init, new_amp_init, new_sigma_init,
            model_class
        )
        if res is not None:
            fit_results[mtype] = (res, comps)

    # If no successful fit was achieved, return an empty dataset with debug info
    if len(fit_results) == 0:
        ds_out = xr.Dataset()
        ds_out.attrs['best_model'] = None
        ds_out.attrs['chisqr'] = np.nan
        ds_out.attrs['redchi'] = np.nan
        ds_out.attrs['chosen_y_idx'] = y_idx
        ds_out.attrs['chosen_x_idx'] = x_idx
        ds_out.attrs['debug_n_data_points'] = len(ldos_sm)
        ds_out.attrs['debug_peaks_before'] = len(cen_init_orig)
        ds_out.attrs['debug_peaks_after'] = 0
        ds_out.attrs['debug_param_count'] = 1
        return ds_out

    # 8) Select the best model based on chi-square (chisqr)
    best_model = None
    best_result = None
    best_comps = None
    best_chi = None
    for mtype, (res, comps) in fit_results.items():
        if best_chi is None or res.chisqr < best_chi:
            best_chi = res.chisqr
            best_model = mtype
            best_result = res
            best_comps = comps

    # 9) Visualization
    if visualize:
        # (a) Plot LDOS map at bias=0 with marker at the selected pixel and
        #     display the LDOS_smoothed and 2nd derivative spectrum with initial guess markers.
        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(8,8))
        ldos_map = ds['LDOS'].sel(bias_mV=0, method='nearest')
        ax_top.imshow(ldos_map, origin='lower', cmap='viridis', aspect='equal')
        ax_top.scatter(x_idx, y_idx, color='red', s=50)
        ax_top.set_title(f"LDOS map (bias=0) | (Y={y_idx}, X={x_idx})")
        
        # Plot smoothed LDOS on the bottom left
        ax_bottom.plot(bias, ldos_sm, 'r-', label='LDOS_smoothed')
        ax_bottom.set_xlabel("Bias (mV)")
        ax_bottom.set_ylabel("LDOS_smoothed", color='r')
        
        # Create a twin axis for the 2nd derivative spectrum
        ax2 = ax_bottom.twinx()
        ax2.plot(bias, ldos_2d, 'purple', label='LDOS_2deriv')
        ax2.set_ylabel("LDOS_2deriv", color='purple')
        
        if show_initial_guess:
            # On the smoothed LDOS plot, mark the initial guess peak positions with vertical dashed lines and text.
            for (b_val, a_val, s_val) in zip(new_cen_init, new_amp_init, new_sigma_init):
                ax_bottom.axvline(b_val, color='orange', linestyle='--', alpha=0.6)
                ax_bottom.text(b_val, np.max(ldos_sm)*0.9, f'{b_val:.1f}', rotation=90,
                               va='center', color='orange')
            # On the 2nd derivative plot, add detailed markers:
            # For each 2nd derivative peak, plot a marker at the peak (as a triangle),
            # a vertical dashed line representing the prominence, and a horizontal line representing the width.
            for (b_val, p_val, h_val, w_val, wh_val) in zip(pb_2d, pp_2d, ph_2d, pw_2d, pwh_2d):
                if not np.isnan(b_val) and not np.isnan(h_val):
                    ax2.plot(b_val, h_val, '^', color='purple')
                    ax2.vlines(b_val, h_val - p_val, h_val, color='purple', linestyles='dashed', alpha=0.6)
                    ax2.hlines(h_val - wh_val, b_val - w_val/2, b_val + w_val/2, color='purple', alpha=0.6)
        
        ax_bottom.set_title("LDOS_smoothed & 2nd derivative peaks (initial guesses)")
        plt.tight_layout()
        plt.show()

        # (b) Plot individual fit results for each model type
        colors_map = {'lorentzian':'r','gaussian':'b','voigt':'g'}
        for mtype, (res, comps) in fit_results.items():
            fig_fit, ax_fit = plt.subplots(figsize=(6,5))
            ax_fit.plot(bias, ldos_sm, 'k-', alpha=0.6, label='LDOS_smoothed')
            fit_color = colors_map.get(mtype.lower(), 'orange')
            ax_fit.plot(bias, res.best_fit, fit_color+'-', linewidth=2,
                        label=f'{mtype.capitalize()} Fit (chi={res.chisqr:.2f})')
            if "bg_" in comps:
                ax_fit.plot(bias, comps["bg_"], color='gray', linestyle='--', label='Background')
            peak_i = 0
            for comp_name, comp_val in comps.items():
                if comp_name.startswith("bg_"):
                    continue
                ax_fit.plot(bias, comp_val, '--', label=f'Peak{peak_i}')
                peak_i += 1
            ax_fit.axvline(0, color='gray', linestyle=':')
            ax_fit.set_title(f"{mtype.capitalize()} Weighted Fit\n(# of init peaks={len(new_cen_init)}, weight_sigma={weight_sigma})")
            ax_fit.legend()
            fig_fit.tight_layout()
            plt.show()

        # (c) Combined fit comparison plot with weight function and CdGM level reference lines
        fig_comb, ax_comb = plt.subplots(figsize=(8,6))
        ax_comb.plot(bias, ldos_sm, 'k-', alpha=0.6, label='LDOS_smoothed')
        for mtype, (res, comps) in fit_results.items():
            fit_color = colors_map.get(mtype.lower(), 'orange')
            ax_comb.plot(bias, res.best_fit, fit_color+'-', linewidth=2,
                         label=f'{mtype.capitalize()} Fit (chi={res.chisqr:.2f})')
        if show_weight_function and (weights is not None):
            scaled_weights = weights * np.max(ldos_sm)
            ax_comb.plot(bias, scaled_weights, 'grey--', label='Weight function')
        E_mu = (SCgap**2)/Ef
        n_min = int(np.floor(np.min(bias)/E_mu))
        n_max = int(np.ceil(np.max(bias)/E_mu))
        levels_arr = np.array([n*E_mu for n in range(n_min, n_max+1)])
        half_arr  = np.array([(n+0.5)*E_mu for n in range(n_min, n_max)])
        cands = np.sort(np.concatenate([levels_arr, half_arr]))
        legend_plotted_int = False
        legend_plotted_half = False
        for lvl in cands:
            frac = abs((lvl/E_mu)-np.round(lvl/E_mu))
            if np.isclose(frac, 0, atol=1e-2):
                color_line = 'magenta'
                label_line = 'Integer E_mu' if not legend_plotted_int else None
                legend_plotted_int = True
            elif np.isclose(frac, 0.5, atol=1e-2):
                color_line = 'cyan'
                label_line = 'Half-integer E_mu' if not legend_plotted_half else None
                legend_plotted_half = True
            else:
                continue
            ax_comb.axvline(lvl, color=color_line, linestyle=':', label=label_line)
        ax_comb.axvline(0, color='gray', linestyle=':')
        ax_comb.set_xlabel("Bias (mV)")
        ax_comb.set_ylabel("LDOS_smoothed")
        ax_comb.legend()
        ax_comb.set_title("Combined Fit Comparison with Weight Function")
        plt.tight_layout()
        plt.show()

    # 10) Extract fitted peak parameters from the best fit result
    peak_centers = []
    peak_amplitudes = []
    peak_sigmas = []
    i = 0
    while True:
        prefix = f'peak{i}_'
        if prefix + 'center' in best_result.params:
            peak_centers.append(best_result.params[prefix + 'center'].value)
            peak_amplitudes.append(best_result.params[prefix + 'amplitude'].value)
            peak_sigmas.append(best_result.params[prefix + 'sigma'].value)
            i += 1
        else:
            break

    # 11) Collect full parameter information from the fit
    param_names = list(best_result.params.keys())
    param_values = []
    param_stderr = []
    param_min = []
    param_max = []
    param_vary = []
    param_expr = []
    for pname in param_names:
        p = best_result.params[pname]
        param_values.append(p.value)
        param_stderr.append(p.stderr if p.stderr is not None else np.nan)
        param_min.append(p.min if p.min is not None else np.nan)
        param_max.append(p.max if p.max is not None else np.nan)
        param_vary.append(p.vary)
        param_expr.append(p.expr if p.expr is not None else "")

    # 12) Calculate CdGM candidate energy levels and determine level proximity for each peak
    E_mu = (SCgap**2)/Ef
    n_min = int(np.floor(np.min(bias)/E_mu))
    n_max = int(np.ceil(np.max(bias)/E_mu))
    levels = [n*E_mu for n in range(n_min, n_max+1)]
    half = [(n+0.5)*E_mu for n in range(n_min, n_max)]
    cands = sorted(levels+half)

    level_proximities = []
    for i, center in enumerate(peak_centers):
        sigma_val = best_result.params[f'peak{i}_sigma'].value
        if best_model.lower() == 'gaussian':
            half_width = 1.1775 * sigma_val
        else:
            half_width = sigma_val
        if margin2zerobias is not None and margin2levels is not None:
            if abs(center) <= margin2zerobias:
                level_prox = 0
            else:
                diffs = [abs(center - l) for l in cands]
                valid_candidates = [l for l, d in zip(cands, diffs) if d <= margin2levels]
                if valid_candidates:
                    level_prox = min(valid_candidates, key=lambda xx: abs(center - xx))
                else:
                    level_prox = np.nan
        else:
            if (center - half_width) <= 0 <= (center + half_width):
                level_prox = 0
            else:
                valids = [l for l in cands if (center - half_width) <= l <= (center + half_width)]
                if valids:
                    level_prox = min(valids, key=lambda xx: abs(center - xx))
                else:
                    level_prox = np.nan
        level_proximities.append(level_prox)

    # 13) Retrieve individual peak and background components from the best fit result
    peak_keys = [k for k in best_comps.keys() if not k.startswith("bg_")]
    peak_keys = sorted(peak_keys, key=lambda x: int(x.replace("peak", "").split("_")[0]))
    peak_fit_list = []
    for k in peak_keys:
        peak_fit_list.append(best_comps[k])
    peak_fit = np.array(peak_fit_list)
    background_fit = best_comps.get("bg_", np.full_like(bias, np.nan))


    # ---------------------------- PATCH START ----------------------------
    # Ensure all fit-related outputs are expanded to match original bias_mV shape.
    full_bias = ds['bias_mV'].values
    full_length = len(full_bias)
    
    # Mapping mask: True where bias was not excluded
    bias_mask = np.isin(full_bias, bias)
    
    # Initialize full arrays with NaNs
    full_ldos_sm = np.full(full_length, np.nan)
    full_best_fit = np.full(full_length, np.nan)
    full_background_fit = np.full(full_length, np.nan)
    full_peak_fit = np.full((peak_fit.shape[0], full_length), np.nan)
    
    # Assign valid fitted values to their corresponding bias positions
    full_ldos_sm[bias_mask] = ldos_sm
    full_best_fit[bias_mask] = best_result.best_fit
    full_background_fit[bias_mask] = background_fit
    for i in range(peak_fit.shape[0]):
        full_peak_fit[i, bias_mask] = peak_fit[i]
    
    # Replace bias and fit arrays to ensure consistent shape
    bias = full_bias
    best_result.data = full_ldos_sm
    best_result.best_fit = full_best_fit
    background_fit = full_background_fit
    peak_fit = full_peak_fit
    # ---------------------------- PATCH END ----------------------------
    # 14) Build the final output dataset with all fitting results and curves

    # 14) Build the final output dataset with all fitting results and curves
    ds_out = xr.Dataset(
        {
            'peak_center':    (['peak'], peak_centers),
            'peak_amplitude': (['peak'], peak_amplitudes),
            'peak_sigma':     (['peak'], peak_sigmas),
            'level_proximity':(['peak'], level_proximities),
            'param_value':    (['param_name'], param_values),
            'param_stderr':   (['param_name'], param_stderr),
            'param_min':      (['param_name'], param_min),
            'param_max':      (['param_name'], param_max),
            'param_vary':     (['param_name'], param_vary),
            'param_expr':     (['param_name'], param_expr),
            'ldos_sm':        (['bias'], best_result.data),
            'best_fit':       (['bias'], best_result.best_fit),
            'background_fit': (['bias'], background_fit),
            'peak_fit':       (['peak','bias'], peak_fit)
        },
        coords={
            'peak': np.arange(len(peak_centers)),
            'param_name': param_names,
            'bias': bias
        }
    )

    # If a background parameter is present, store its value and error
    if 'bg_c' in best_result.params:
        bg_param = best_result.params['bg_c']
        ds_out['background_value']  = ([], bg_param.value)
        ds_out['background_stderr'] = ([], bg_param.stderr if bg_param.stderr else np.nan)

    # 15) Merge original metadata and store fitting/debug information in attrs
    ds_out.attrs.update(ds.attrs)
    ds_out.attrs['best_model'] = best_model
    ds_out.attrs['chisqr']     = best_result.chisqr
    ds_out.attrs['redchi']     = best_result.redchi
    ds_out.attrs['chosen_y_idx'] = y_idx
    ds_out.attrs['chosen_x_idx'] = x_idx
    ds_out.attrs['weight_sigma'] = weight_sigma
    ds_out.attrs['Ef']         = Ef
    ds_out.attrs['SCgap']      = SCgap

    # Debug info: data points and peak counts before and after auto-reduction
    ds_out.attrs['debug_n_data_points'] = len(ldos_sm)
    ds_out.attrs['debug_peaks_before']  = len(cen_init_orig)
    # Calculate number of peaks remaining after limiting by max_peaks and margin adjustment
    n_peaks_after = len(new_cen_init)
    ds_out.attrs['debug_peaks_after']   = n_peaks_after
    ds_out.attrs['debug_param_count']   = 1 + 3*n_peaks_after

    return ds_out
    '''


# +
grid_LDOS_SnD_pks_fit_singleP= multi_peak_IGS_fitting_singleXY(grid_LDOS_SnD_pks,
                                                               #y_idx=None,
                                                               y_idx=24,
                                                               #x_idx=None,
                                                               x_idx= 48,
                                                               weight_sigma=0.3,
                                                               #model_type=['Lorentzian','Gaussian','Voigt'],
                                                               model_type='Lorentzian',
                                                               visualize=True,
                                                              show_initial_guess=True,  
                                                               show_weight_function=False )

grid_LDOS_SnD_pks_fit_singleP
#grid_LDOS_SnD_pks_fit_singleP.to_netcdf('grid_LDOS_SnD_pks_fit_singleP_eg.nc')
# -

# ### single point plot again

def plot_singleXY_fitting_result_from_dsout(ds_out, ds):
    """
    Plot the best-fit spectrum and its individual peak/background components 
    for a specific spatial location, using the output from 
    `multi_peak_IGS_fitting_singleXY` or 'multi_peak_IGS_fitting_region_parallel'.

    This function reconstructs the composite model fit—including multiple 
    peaks and a constant background—at a selected pixel (Y, X) using only 
    the parameter values stored in `ds_out`. It then plots:
        - The original smoothed LDOS spectrum at that pixel
        - The total best-fit curve
        - Individual fitted peak components
        - The constant background component (if available)

    This function relies on the following information stored in `ds_out`:
        - 'param_value' and 'param_name': Fitted parameters for all peaks 
          and background
        - Coordinate metadata: 'peak' and 'bias' dimensions
        - Attribute metadata: 
            - 'best_model' (str): Which model type was chosen as optimal 
              (e.g. "Lorentzian", "Gaussian", "Voigt")
            - 'chosen_y_idx', 'chosen_x_idx' (int): Pixel coordinates used 
              for the fitting
            - 'chisqr' (float): Chi-square of the best-fit model
        - Other metadata (e.g. 'redchi', debug info) is not required for plotting 
          but may appear in the legend or title.

    Parameters
    ----------
    ds_out : xarray.Dataset
        The output from `multi_peak_IGS_fitting_singleXY`, which contains:
            - The best-fit parameters in 'param_value' (with names in 'param_name')
            - Dimensional coordinates for bias and peak index
            - Attribute metadata describing the model type, pixel location, and fit statistics
    
    ds : xarray.Dataset
        The full LDOS dataset that contains at least:
            - 'LDOS_smoothed': 3D DataArray with dimensions (bias, Y, X)
            - 'bias_mV': 1D coordinate for bias voltage
        This is used to extract the raw data at the fitted (Y, X) location 
        for comparison with the fit.

    Returns
    -------
    None
        Displays a Matplotlib plot showing the raw spectrum, the best-fit 
        curve, and individual peak/background components. No data is returned.
    """
    import matplotlib.pyplot as plt
    from lmfit.models import LorentzianModel, GaussianModel, VoigtModel, ConstantModel
    from functools import reduce
    
    # 1) Extract basic info from ds_out
    best_model_name = ds_out.attrs.get('best_model', None)
    y_idx = ds_out.attrs.get('chosen_y_idx', 0)
    x_idx = ds_out.attrs.get('chosen_x_idx', 0)

    # 2) Extract the 1D data for plotting
    bias = ds['bias_mV'].values
    ldos_sm = ds['LDOS_smoothed'].isel(Y=y_idx, X=x_idx).values
    
    # 3) Rebuild the multi-peak + background model using param names from ds_out.
    # Get the number of peaks from ds_out dims ('peak')
    n_peaks = ds_out.dims['peak']  # number of fitted peaks

    if best_model_name is None:
        print("No valid fit found in ds_out.")
        return
    elif best_model_name.lower() == 'lorentzian':
        model_class = LorentzianModel
    elif best_model_name.lower() == 'gaussian':
        model_class = GaussianModel
    elif best_model_name.lower() == 'voigt':
        model_class = VoigtModel
    else:
        print(f"Unknown best_model in ds_out: {best_model_name}")
        return

    # Build constant background model and multi-peak model
    bg_model = ConstantModel(prefix='bg_')
    peak_models = [model_class(prefix=f'peak{i}_') for i in range(n_peaks)]
    peaks_model = reduce(lambda m1, m2: m1 + m2, peak_models)
    model = bg_model + peaks_model

    # 4) Create parameters and set values from ds_out.
    params = model.make_params()
    param_names = ds_out.coords['param_name'].values
    param_values = ds_out['param_value'].values

    # Create mapping from parameter name to value.
    param_map = {pname: pval for pname, pval in zip(param_names, param_values)}

    for pname, pval in param_map.items():
        if pname in params:
            params[pname].set(value=pval)
    
    # 5) Evaluate best-fit curve and each component (components will include background).
    best_fit = model.eval(params=params, x=bias)
    comps = model.eval_components(params=params, x=bias)

    # 6) Plot the data vs the best-fit curve.
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(bias, ldos_sm, 'k-', alpha=0.6, label='LDOS_smoothed')

    if best_model_name.lower() == 'lorentzian':
        fit_color = 'r'
    elif best_model_name.lower() == 'gaussian':
        fit_color = 'b'
    else:
        fit_color = 'g'

    ax.plot(bias, best_fit, fit_color+'-', linewidth=2,
            label=f'{best_model_name.capitalize()} Fit (chi={ds_out.attrs.get("chisqr", -1):.2f})')
    
    # 7) Plot individual peak components and background.
    # Plot background curve from comps dict, key likely "bg_"
    if 'bg_c' in params:
        # Evaluate constant background as a horizontal line.
        bg_val = params['bg_c'].value
        ax.plot(bias, np.full_like(bias, bg_val), 'm--', label='Background')

    # For each peak, sum up components that start with 'peak{i}_'
    for i in range(n_peaks):
        prefix = f'peak{i}_'
        comp_keys = [k for k in comps.keys() if k.startswith(prefix)]
        if len(comp_keys) == 1:
            comp_curve = comps[comp_keys[0]]
            ax.plot(bias, comp_curve, '--', label=f'Peak{i}')
        else:
            combined = sum(comps[k] for k in comp_keys)
            ax.plot(bias, combined, '--', label=f'Peak{i}')
    
    ax.axvline(0, color='gray', linestyle=':')
    ax.set_title(f"{best_model_name.capitalize()} Weighted Fit (n_peaks={n_peaks})")
    ax.legend()
    plt.tight_layout()
    plt.show()

plot_singleXY_fitting_result_from_dsout(grid_LDOS_SnD_pks_fit_singleP, grid_LDOS_SnD_pks)

# ## Region fitting with ZB mask
# ### No parallel computing 
# * use the for loop

# ####  grid_LDOS_SnD_pks crop area test

grid_LDOS_SnD_pks_crop =  grid_LDOS_SnD_pks.isel(X=slice(0,15),Y=slice(0,15))

grid_LDOS_SnD_pks_crop

'''
import xarray as xr
import numpy as np
import time
from tqdm.notebook import tqdm
from joblib import Parallel, delayed

def multi_peak_IGS_fitting_region_parallel(
    ds,
    weight_sigma=1.0,
    model_type=['Lorentzian', 'Gaussian', 'Voigt'],
    n_jobs=-1,
    Ef=4.4,
    SCgap=1.8,
    margin2zerobias=0.1,
    margin2levels=0.1,
    exclude_negative=False,
    ZB_masking=False
):
    """
    [Parallel Version]
    Perform multi-peak weighted fitting for each (Y, X) pixel by calling 
    multi_peak_IGS_fitting_singleXY in parallel, and save the fitting results.
    
    This function performs the following steps:
      1) Copies the input dataset to avoid modifying the original.
      2) Reads or creates the 'Y' and 'X' coordinate arrays (if missing) to index the pixels.
      3) Pre-allocates arrays to store the final results, including:
         - peak parameters ('peak_center', 'peak_amplitude', 'peak_sigma')
         - parameter details ('param_data' with [value, stderr, min, max, vary], 'param_expr')
         - reduced chi-square ('redchi'), level proximity ('level_proximity'), background values
         - single-pixel fit outputs: 'ldos_sm', 'best_fit', 'background_fit', 'peak_fit'
         - best model tracking output: **model_type** (2D string array of best-fitting model at each (Y, X))
      4) Builds a list of pixel indices to process, optionally filtering based on 'ZB_mask' if 'ZB_masking' is True.
      5) Processes all pixels in parallel by calling multi_peak_IGS_fitting_singleXY with the specified arguments,
         and stores the returned results in the pre-allocated arrays. If an exception occurs for a pixel, 
         a debug message is printed and that pixel is skipped.
      6) Constructs a final xarray.Dataset (ds_out) with the collected results, assigning coordinates and 
         copying any relevant variables (e.g. 'LDOS', 'ZB_mask') from the original dataset.
      7) Saves metadata in ds_out.attrs describing the fitting procedure (e.g., weight_sigma, model_type, etc.).
      8) Returns ds_out, which contains all aggregated fitting results and the original 'LDOS'/'ZB_mask' if present.

    Returns
    -------
    ds_out : xarray.Dataset
        Output dataset with dimensions (Y, X). Includes all peak fitting outputs and:
        - model_type (Y, X): best model used at each pixel (as str: 'Lorentzian', 'Gaussian', 'Voigt')
        - All fitting curve data and parameters
    """

    start_time = time.time()

    # -------------------------------------------------------------------
    # (1) Copy dataset and setup coordinates
    # -------------------------------------------------------------------
    ds_copy = ds.copy()
    Y_coords = ds_copy.coords["Y"].values if "Y" in ds_copy.coords else np.arange(ds_copy.dims['Y'])
    X_coords = ds_copy.coords["X"].values if "X" in ds_copy.coords else np.arange(ds_copy.dims['X'])
    ny, nx = len(Y_coords), len(X_coords)

    print(f"Starting multi-peak fitting for {ny} x {nx} = {ny * nx} pixels...")

    # -------------------------------------------------------------------
    # (2) Setup arrays and dimensions
    # -------------------------------------------------------------------
    peak_dim = ds_copy.dims.get("peak", 16)
    param_name_dim = ds_copy.dims.get("param_name", 80)
    param_attr_list = ["value", "stderr", "min", "max", "vary"]
    n_attr = len(param_attr_list)

    bias_common = ds_copy["bias_mV"].values
    n_bias = len(bias_common)

    peak_vars = ["peak_center", "peak_amplitude", "peak_sigma"]
    peak_data = {var: np.full((ny, nx, peak_dim), np.nan, dtype=float) for var in peak_vars}
    param_data = np.full((ny, nx, param_name_dim, n_attr), np.nan, dtype=float)
    param_expr = np.empty((ny, nx, param_name_dim), dtype=object)
    param_expr[:] = ""

    redchi_data = np.full((ny, nx), np.nan, dtype=float)
    level_proximity_data = np.full((ny, nx, peak_dim), np.nan, dtype=float)
    background_data = np.full((ny, nx), np.nan, dtype=float)

    ldos_sm_data = np.full((ny, nx, n_bias), np.nan, dtype=float)
    best_fit_data = np.full((ny, nx, n_bias), np.nan, dtype=float)
    background_fit_data = np.full((ny, nx, n_bias), np.nan, dtype=float)
    peak_fit_data = np.full((ny, nx, peak_dim, n_bias), np.nan, dtype=float)

    # ✅ model type map 초기화
    model_type_map = np.empty((ny, nx), dtype=object)

    # -------------------------------------------------------------------
    # (3) CdGM 후보 에너지 스케일 계산
    # -------------------------------------------------------------------
    E_CdGM = (SCgap ** 2) / Ef
    n_min = int(np.floor(bias_common.min() / E_CdGM))
    n_max = int(np.ceil(bias_common.max() / E_CdGM))
    levels = [n * E_CdGM for n in range(n_min, n_max + 1)]
    half_levels = [(n + 0.5) * E_CdGM for n in range(n_min, n_max)]
    candidate_levels = levels + half_levels

    # -------------------------------------------------------------------
    # (4) 개별 픽셀 fitting 함수
    # -------------------------------------------------------------------
    def fit_pixel(y, x):
        try:
            return multi_peak_IGS_fitting_singleXY(
                grid_LDOS_SnD_pks=ds_copy,
                y_idx=y,
                x_idx=x,
                weight_sigma=weight_sigma,
                model_type=model_type,
                visualize=False,
                margin2zerobias=margin2zerobias,
                margin2levels=margin2levels,
                Ef=Ef,
                SCgap=SCgap,
                exclude_negative=exclude_negative,
                background_offset=False,
                background_offset_adjust=False,
                show_initial_guess=False,
                show_weight_function=False
            )
        except Exception as e:
            print(f"[DEBUG] Pixel (Y={y}, X={x}) error: {e}")
            return None

    # -------------------------------------------------------------------
    # (5) 마스크 기반 좌표 선택
    # -------------------------------------------------------------------
    if ZB_masking:
        if "ZB_mask" not in ds_copy:
            raise ValueError("ZB_mask is not found, but ZB_masking=True.")
        valid_mask = ~np.isnan(ds_copy["ZB_mask"].values)
        pixel_indices = [(y, x) for y in range(ny) for x in range(nx) if valid_mask[y, x]]
    else:
        pixel_indices = [(y, x) for y in range(ny) for x in range(nx)]

    # -------------------------------------------------------------------
    # (6) 병렬 처리 실행
    # -------------------------------------------------------------------
    results = Parallel(n_jobs=n_jobs)(
        delayed(fit_pixel)(y, x) for (y, x) in tqdm(pixel_indices, desc="Multi-peak fitting", unit="pixel")
    )

    # -------------------------------------------------------------------
    # (7) 결과 채우기
    # -------------------------------------------------------------------
    for idx, (y, x) in enumerate(pixel_indices):
        ds_fit = results[idx]
        if ds_fit is not None and isinstance(ds_fit, xr.Dataset):
            num_peaks = ds_fit.dims.get("peak", 0)

            # ✅ 정확한 모델 이름 저장
            best_model = ds_fit.attrs.get("best_model", None)
            model_type_map[y, x] = best_model if best_model else "unknown"

            for var in peak_vars:
                if var in ds_fit:
                    arr = ds_fit[var].values
                    peak_data[var][y, x, :min(num_peaks, peak_dim)] = arr[:min(num_peaks, peak_dim)]

            for i, attr in enumerate(param_attr_list):
                key = f"param_{attr}"
                if key in ds_fit:
                    val = ds_fit[key].values
                    if attr == "vary":
                        val = [1.0 if v else 0.0 for v in val]
                    param_data[y, x, :min(len(val), param_name_dim), i] = val[:min(len(val), param_name_dim)]

            if "param_expr" in ds_fit:
                param_expr[y, x, :min(len(ds_fit["param_expr"]), param_name_dim)] = ds_fit["param_expr"].values

            redchi_data[y, x] = ds_fit.attrs.get("redchi", np.nan)

            if num_peaks > 0 and "peak_center" in ds_fit:
                centers = ds_fit["peak_center"].values
                for i_peak in range(min(num_peaks, peak_dim)):
                    cen = centers[i_peak]
                    closest_level = min(candidate_levels, key=lambda l: abs(cen - l))
                    if abs(cen - closest_level) <= margin2levels:
                        level_proximity_data[y, x, i_peak] = closest_level

            if "background_value" in ds_fit:
                background_data[y, x] = np.atleast_1d(ds_fit["background_value"].values).item()

            for name, target in zip(["ldos_sm", "best_fit", "background_fit"],
                                    [ldos_sm_data, best_fit_data, background_fit_data]):
                if name in ds_fit and ds_fit[name].shape[0] == n_bias:
                    target[y, x, :] = ds_fit[name].values

            if "peak_fit" in ds_fit:
                pf = ds_fit["peak_fit"].values
                peak_fit_data[y, x, :min(pf.shape[0], peak_dim), :] = pf[:min(pf.shape[0], peak_dim), :]

    # -------------------------------------------------------------------
    # (8) Dataset 구성
    # -------------------------------------------------------------------
    ds_out = xr.Dataset({var: (["Y", "X", "peak"], peak_data[var]) for var in peak_vars})
    ds_out["param_data"] = (["Y", "X", "param_name", "param_attr"], param_data)
    ds_out["param_expr"] = (["Y", "X", "param_name"], param_expr)
    ds_out["redchi"] = (["Y", "X"], redchi_data)
    ds_out["level_proximity"] = (["Y", "X", "peak"], level_proximity_data)
    ds_out["background_value"] = (["Y", "X"], background_data)
    ds_out["ldos_sm"] = (["Y", "X", "bias_mV"], ldos_sm_data)
    ds_out["best_fit"] = (["Y", "X", "bias_mV"], best_fit_data)
    ds_out["background_fit"] = (["Y", "X", "bias_mV"], background_fit_data)
    ds_out["peak_fit"] = (["Y", "X", "peak", "bias_mV"], peak_fit_data)
    ds_out["model_type"] = (["Y", "X"], model_type_map)

    ds_out = ds_out.assign_coords({
        "bias_mV": bias_common,
        "peak": np.arange(peak_dim),
        "param_name": np.arange(param_name_dim),
        "param_attr": param_attr_list,
        "Y": Y_coords,
        "X": X_coords
    })

    if "LDOS" in ds_copy:
        ds_out["LDOS"] = ds_copy["LDOS"]
    if "ZB_mask" in ds_copy:
        ds_out["ZB_mask"] = ds_copy["ZB_mask"]

    # -------------------------------------------------------------------
    # (9) 메타데이터 기록
    # -------------------------------------------------------------------
    ds_out.attrs.update({
        "description": "Parallel multi-peak fitting results with model type map and full param structure.",
        "weight_sigma": weight_sigma,
        "model_type_input": str(model_type),
        "Ef": Ef,
        "SCgap": SCgap,
        "margin2zerobias": margin2zerobias,
        "margin2levels": margin2levels,
        "exclude_negative": int(exclude_negative),
        "ZB_masking": int(ZB_masking)
    })

    end_time = time.time()
    print(f"Multi-peak fitting completed in {end_time - start_time:.2f} sec.")

    return ds_out
'''
# +
import xarray as xr
import numpy as np
import time
from tqdm.notebook import tqdm
from joblib import Parallel, delayed

def multi_peak_IGS_fitting_region_parallel(
    ds,
    weight_sigma=1.0,
    model_type=['Lorentzian', 'Gaussian', 'Voigt'],
    n_jobs=-1,
    Ef=4.4,
    SCgap=1.8,
    margin2zerobias=0.1,
    margin2levels=0.1,
    exclude_negative=False,
    ZB_masking=False
):
    """
    [Parallel Version]
    Perform multi-peak weighted fitting for each (Y, X) pixel by calling 
    multi_peak_IGS_fitting_singleXY in parallel, and save the fitting results.
    
    This function performs the following steps:
      1) Copies the input dataset to avoid modifying the original.
      2) Reads or creates the 'Y' and 'X' coordinate arrays (if missing) to index the pixels.
      3) Pre-allocates arrays to store the final results, including:
         - peak parameters ('peak_center', 'peak_amplitude', 'peak_sigma')
         - parameter details ('param_data' with [value, stderr, min, max, vary], 'param_expr')
         - reduced chi-square ('redchi'), level proximity ('level_proximity'), background values
         - single-pixel fit outputs: 'ldos_sm', 'best_fit', 'background_fit', 'peak_fit'
         - best model tracking output: **model_type** (2D string array of best-fitting model at each (Y, X))
      4) Builds a list of pixel indices to process, optionally filtering based on 'ZB_mask' if 'ZB_masking' is True.
      5) Processes all pixels in parallel by calling multi_peak_IGS_fitting_singleXY with the specified arguments,
         and stores the returned results in the pre-allocated arrays. If an exception occurs for a pixel, 
         a debug message is printed and that pixel is skipped.
      6) Constructs a final xarray.Dataset (ds_out) with the collected results, assigning coordinates and 
         copying any relevant variables (e.g. 'LDOS', 'ZB_mask') from the original dataset.
      7) Saves metadata in ds_out.attrs describing the fitting procedure (e.g., weight_sigma, model_type, etc.).
      8) Returns ds_out, which contains all aggregated fitting results and the original 'LDOS'/'ZB_mask' if present.

    Returns
    -------
    ds_out : xarray.Dataset
        Output dataset with dimensions (Y, X). Includes all peak fitting outputs and:
        - model_type (Y, X): best model used at each pixel (as str: 'Lorentzian', 'Gaussian', 'Voigt')
        - All fitting curve data and parameters
    """

    start_time = time.time()

    # (1) Copy dataset and set up coordinates
    ds_copy = ds.copy()
    Y_coords = ds_copy.coords["Y"].values if "Y" in ds_copy.coords else np.arange(ds_copy.dims['Y'])
    X_coords = ds_copy.coords["X"].values if "X" in ds_copy.coords else np.arange(ds_copy.dims['X'])
    ny, nx = len(Y_coords), len(X_coords)

    print(f"Starting multi-peak fitting for {ny} x {nx} = {ny * nx} pixels...")

    # (2) Set up arrays and dimensions for storing output
    peak_dim = ds_copy.dims.get("peak", 16)
    param_name_dim = ds_copy.dims.get("param_name", 80)
    param_attr_list = ["value", "stderr", "min", "max", "vary"]
    n_attr = len(param_attr_list)

    bias_common = ds_copy["bias_mV"].values
    n_bias = len(bias_common)

    peak_vars = ["peak_center", "peak_amplitude", "peak_sigma"]
    peak_data = {var: np.full((ny, nx, peak_dim), np.nan, dtype=float) for var in peak_vars}
    param_data = np.full((ny, nx, param_name_dim, n_attr), np.nan, dtype=float)
    param_expr = np.empty((ny, nx, param_name_dim), dtype=object)
    param_expr[:] = ""

    redchi_data = np.full((ny, nx), np.nan, dtype=float)
    level_proximity_data = np.full((ny, nx, peak_dim), np.nan, dtype=float)
    background_data = np.full((ny, nx), np.nan, dtype=float)

    ldos_sm_data = np.full((ny, nx, n_bias), np.nan, dtype=float)
    best_fit_data = np.full((ny, nx, n_bias), np.nan, dtype=float)
    background_fit_data = np.full((ny, nx, n_bias), np.nan, dtype=float)
    peak_fit_data = np.full((ny, nx, peak_dim, n_bias), np.nan, dtype=float)

    # ✅ Initialize model type map to store the best model at each (Y, X)
    model_type_map = np.empty((ny, nx), dtype=object)

    # (3) Compute CdGM candidate energy scale
    E_CdGM = (SCgap ** 2) / Ef
    n_min = int(np.floor(bias_common.min() / E_CdGM))
    n_max = int(np.ceil(bias_common.max() / E_CdGM))
    levels = [n * E_CdGM for n in range(n_min, n_max + 1)]
    half_levels = [(n + 0.5) * E_CdGM for n in range(n_min, n_max)]
    candidate_levels = levels + half_levels

    # (4) Define pixel-wise fitting function
    def fit_pixel(y, x):
        try:
            return multi_peak_IGS_fitting_singleXY(
                grid_LDOS_SnD_pks=ds_copy,
                y_idx=y,
                x_idx=x,
                weight_sigma=weight_sigma,
                model_type=model_type,
                visualize=False,
                margin2zerobias=margin2zerobias,
                margin2levels=margin2levels,
                Ef=Ef,
                SCgap=SCgap,
                exclude_negative=exclude_negative,
                background_offset=False,
                background_offset_adjust=False,
                show_initial_guess=False,
                show_weight_function=False
            )
        except Exception as e:
            print(f"[DEBUG] Pixel (Y={y}, X={x}) error: {e}")
            return None

    # (5) Choose pixel coordinates based on mask (if ZB_masking is enabled)
    if ZB_masking:
        if "ZB_mask" not in ds_copy:
            raise ValueError("ZB_mask is not found, but ZB_masking=True.")
        valid_mask = ~np.isnan(ds_copy["ZB_mask"].values)
        pixel_indices = [(y, x) for y in range(ny) for x in range(nx) if valid_mask[y, x]]
    else:
        pixel_indices = [(y, x) for y in range(ny) for x in range(nx)]

    # (6) Run parallel fitting across selected pixel list
    results = Parallel(n_jobs=n_jobs)(
        delayed(fit_pixel)(y, x) for (y, x) in tqdm(pixel_indices, desc="Multi-peak fitting", unit="pixel")
    )

    # (7) Fill in results into preallocated arrays
    for idx, (y, x) in enumerate(pixel_indices):
        ds_fit = results[idx]
        if ds_fit is not None and isinstance(ds_fit, xr.Dataset):
            num_peaks = ds_fit.dims.get("peak", 0)

            # ✅ Save the exact best model name to the model type map
            best_model = ds_fit.attrs.get("best_model", None)
            model_type_map[y, x] = best_model if best_model else "unknown"

            for var in peak_vars:
                if var in ds_fit:
                    arr = ds_fit[var].values
                    peak_data[var][y, x, :min(num_peaks, peak_dim)] = arr[:min(num_peaks, peak_dim)]

            for i, attr in enumerate(param_attr_list):
                key = f"param_{attr}"
                if key in ds_fit:
                    val = ds_fit[key].values
                    if attr == "vary":
                        val = [1.0 if v else 0.0 for v in val]
                    param_data[y, x, :min(len(val), param_name_dim), i] = val[:min(len(val), param_name_dim)]

            if "param_expr" in ds_fit:
                param_expr[y, x, :min(len(ds_fit["param_expr"]), param_name_dim)] = ds_fit["param_expr"].values

            redchi_data[y, x] = ds_fit.attrs.get("redchi", np.nan)

            if num_peaks > 0 and "peak_center" in ds_fit:
                centers = ds_fit["peak_center"].values
                for i_peak in range(min(num_peaks, peak_dim)):
                    cen = centers[i_peak]
                    closest_level = min(candidate_levels, key=lambda l: abs(cen - l))
                    if abs(cen - closest_level) <= margin2levels:
                        level_proximity_data[y, x, i_peak] = closest_level

            if "background_value" in ds_fit:
                background_data[y, x] = np.atleast_1d(ds_fit["background_value"].values).item()

            for name, target in zip(["ldos_sm", "best_fit", "background_fit"],
                                    [ldos_sm_data, best_fit_data, background_fit_data]):
                if name in ds_fit and ds_fit[name].shape[0] == n_bias:
                    target[y, x, :] = ds_fit[name].values

            if "peak_fit" in ds_fit:
                pf = ds_fit["peak_fit"].values
                peak_fit_data[y, x, :min(pf.shape[0], peak_dim), :] = pf[:min(pf.shape[0], peak_dim), :]

    # (8) Construct output xarray.Dataset
    ds_out = xr.Dataset({var: (["Y", "X", "peak"], peak_data[var]) for var in peak_vars})
    ds_out["param_data"] = (["Y", "X", "param_name", "param_attr"], param_data)
    ds_out["param_expr"] = (["Y", "X", "param_name"], param_expr)
    ds_out["redchi"] = (["Y", "X"], redchi_data)
    ds_out["level_proximity"] = (["Y", "X", "peak"], level_proximity_data)
    ds_out["background_value"] = (["Y", "X"], background_data)
    ds_out["ldos_sm"] = (["Y", "X", "bias_mV"], ldos_sm_data)
    ds_out["best_fit"] = (["Y", "X", "bias_mV"], best_fit_data)
    ds_out["background_fit"] = (["Y", "X", "bias_mV"], background_fit_data)
    ds_out["peak_fit"] = (["Y", "X", "peak", "bias_mV"], peak_fit_data)
    ds_out["model_type"] = (["Y", "X"], model_type_map)

    ds_out = ds_out.assign_coords({
        "bias_mV": bias_common,
        "peak": np.arange(peak_dim),
        "param_name": np.arange(param_name_dim),
        "param_attr": param_attr_list,
        "Y": Y_coords,
        "X": X_coords
    })

    if "LDOS" in ds_copy:
        ds_out["LDOS"] = ds_copy["LDOS"]
    if "ZB_mask" in ds_copy:
        ds_out["ZB_mask"] = ds_copy["ZB_mask"]

    # (9) Record metadata for reproducibility
    ds_out.attrs.update({
        "description": "Parallel multi-peak fitting results with model type map and full param structure.",
        "weight_sigma": weight_sigma,
        "model_type_input": str(model_type),
        "Ef": Ef,
        "SCgap": SCgap,
        "margin2zerobias": margin2zerobias,
        "margin2levels": margin2levels,
        "exclude_negative": int(exclude_negative),
        "ZB_masking": int(ZB_masking)
    })

    end_time = time.time()
    print(f"Multi-peak fitting completed in {end_time - start_time:.2f} sec.")

    return ds_out



# -


grid_LDOS_SnD_pks_crop_results = multi_peak_IGS_fitting_region_parallel(
    grid_LDOS_SnD_pks_crop,
    weight_sigma=0.5,
    #model_type=['Lorentzian', 'Gaussian', 'Voigt'],
    model_type=['Lorentzian'],
    n_jobs=-1,
    Ef=4.4,
    SCgap=1.8,
    margin2levels=0.1,
    margin2zerobias=0.1,
    exclude_negative=True,
    ZB_masking=True
)

plot_region_fitting_result_from_dsout(ds_out= grid_LDOS_SnD_pks_crop_results, ds=grid_LDOS_SnD_pks_crop,y_idx=14, x_idx=10)#model_type='Gaussian',  #model_type='Lorentzian')#, 

grid_LDOS_SnD_pks_crop_results

# # parallel computing  full range 


grid_LDOS_SnD_pks_results = multi_peak_IGS_fitting_region_parallel(
    grid_LDOS_SnD_pks,
    weight_sigma=0.5,
    #model_type=['Lorentzian', 'Gaussian', 'Voigt'],
    model_type=['Lorentzian'],
    n_jobs=-1,
    Ef=4.4,
    SCgap=1.8,
    margin2levels=0.1,
    margin2zerobias=0.1,
    exclude_negative=True,
    ZB_masking=True
)

grid_LDOS_SnD_pks_results

plot_region_fitting_result_from_dsout(ds_out= grid_LDOS_SnD_pks_results, ds=grid_LDOS_SnD_pks,y_idx=15, x_idx=55)#model_type='Gaussian',  #model_type='Lorentzian')#, 

grid_LDOS_SnD_pks_results.to_netcdf('grid_LDOS_SnD_pks_2T003_mask_fit_750uVRange_LGV_0629.nc')


# +
#grid_LDOS_SnD_pks_results= xr.open_dataset('grid_LDOS_SnD_pks_2T005_mask_fit_800uVRange_LGV_0522.nc')
# -

grid_LDOS_SnD_pks_results

#
# ## plot fitting results (within ZB_mask)

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from lmfit.models import LorentzianModel, GaussianModel, VoigtModel, ConstantModel
from functools import reduce

def plot_region_fitting_result_from_dsout(
        ds_out, ds,
        model_type=None,
        allowed_models=['Lorentzian', 'Gaussian', 'Voigt'],
        y_idx=None, x_idx=None,
        weight_function_show=False,
        use_zb_mask=False,
        zb_mask_key='ZB_mask',
        show_shade=True,
        return_fig=False):
    """
    Reconstruct and plot the best-fit curve for a selected pixel,
    including optional CdGM level proximity shading and guide lines,
    and collect all curves into a pandas DataFrame.

    Parameters
    ----------
    ds_out : xarray.Dataset
        Fitting output with dims (Y, X, peak) and variables:
          - peak_center, peak_amplitude, peak_sigma, redchi
          - optional: background_value, model_type, ZB_mask, level_proximity
          - attrs: weight_sigma, Ef, SCgap
    ds : xarray.Dataset
        Contains 'bias_mV' and 'LDOS_smoothed' with dims (Y, X, bias_mV).
    model_type : None | str
        If None, per-pixel model from ds_out['model_type'] is used.
        If str, forces a fixed model for all pixels (must be in allowed_models).
    allowed_models : list of str
        Supported model types when model_type=None.
    y_idx, x_idx : int, optional
        Pixel indices. If None and use_zb_mask=True, picks a random valid pixel.
    weight_function_show : bool
        If True, also plot convoluted fit and weight function.
    use_zb_mask : bool
        If True, applies ZB_mask to bias axis selection.
    zb_mask_key : str
        Name of mask variable in ds_out for zero-bias filtering.
    show_shade : bool
        If True, plots CdGM level proximity shading and guide lines.
    return_fig : bool
        If True, returns (fig, df). Otherwise shows the plot and returns df.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame indexed by bias_mV, with columns:
          - 'LDOS_smoothed', 'best_fit', each 'peak{i}', 'bkg' if present
          - 'convoluted_fit', 'weight_function' if weight_function_show=True
    fig : matplotlib.figure.Figure, optional
        If return_fig=True, also returns the Figure object.
    """
    # 1) Determine pixel indices
    ny, nx = ds_out.dims['Y'], ds_out.dims['X']
    if use_zb_mask and y_idx is None and x_idx is None and zb_mask_key in ds_out:
        m = ds_out[zb_mask_key].values
        valid_mask = np.any(~np.isnan(m), axis=2) if m.ndim == 3 else m.astype(bool)
        rc = ds_out['redchi'].values
        valid_fit = ~np.isnan(rc)
        ys, xs = np.where(valid_mask & valid_fit)
        if len(ys) == 0:
            raise RuntimeError("No valid pixel found with ZB mask and redchi")
        idx = np.random.randint(len(ys))
        y_idx, x_idx = int(ys[idx]), int(xs[idx])
    if y_idx is None:
        y_idx = np.random.randint(ny)
    if x_idx is None:
        x_idx = np.random.randint(nx)

    print(f"Using pixel Y={y_idx}, X={x_idx} for plotting")

    # 2) Extract raw data for this pixel
    bias = ds['bias_mV'].values
    ldos = ds['LDOS_smoothed'].isel(Y=y_idx, X=x_idx).values

    # 3) Decide which model to use with fallback
    if model_type is None:
        raw_model = ds_out['model_type'].isel(Y=y_idx, X=x_idx).item()
        if raw_model is None or not isinstance(raw_model, str):
            chosen = allowed_models[0]
            print(f"Warning: model_type at pixel Y={y_idx}, X={x_idx} is None; defaulting to '{chosen}' model.")
        else:
            chosen = raw_model.capitalize()
    else:
        chosen = model_type.capitalize()
    if chosen not in allowed_models:
        raise ValueError(f"Model '{chosen}' not supported. Choose from {allowed_models}.")

    # 4) Map model name to lmfit class & color
    if chosen == 'Lorentzian':
        mc, fit_color = LorentzianModel, 'r'
    elif chosen == 'Gaussian':
        mc, fit_color = GaussianModel,   'b'
    else:
        mc, fit_color = VoigtModel,      'g'

    # 5) Build mask on bias axis if requested
    mask = np.ones_like(bias, bool)
    if use_zb_mask and zb_mask_key in ds_out:
        raw = ds_out[zb_mask_key].isel(Y=y_idx, X=x_idx).values
        if isinstance(raw, np.ndarray) and raw.shape == bias.shape:
            mask = ~np.isnan(raw) if np.issubdtype(raw.dtype, np.floating) else raw.astype(bool)
        elif np.ndim(raw) == 0:
            mask = np.full_like(bias, bool(raw), bool)
    mask = mask.astype(bool)

    # 6) Load and filter peak parameters
    redchi  = ds_out['redchi'].isel(Y=y_idx, X=x_idx).item()
    n_peaks = ds_out.dims['peak']
    centers = ds_out['peak_center'].isel(Y=y_idx, X=x_idx).values
    amps    = ds_out['peak_amplitude'].isel(Y=y_idx, X=x_idx).values
    sigmas  = ds_out['peak_sigma'].isel(Y=y_idx, X=x_idx).values
    idxs = [i for i in range(n_peaks)
            if not (np.isnan(centers[i]) or np.isnan(amps[i]) or np.isnan(sigmas[i]))]
    print("Valid peak indices:", idxs)

    # 7) Construct composite model
    models = []
    if 'background_value' in ds_out:
        models.append(ConstantModel(prefix='bkg_'))
        bgv = ds_out['background_value'].isel(Y=y_idx, X=x_idx).item()
    for i in idxs:
        models.append(mc(prefix=f'peak{i}_'))
    comp = reduce(lambda a, b: a + b, models)
    params = comp.make_params()
    if 'background_value' in ds_out:
        params['bkg_c'].set(value=bgv)
    for i in idxs:
        params[f'peak{i}_center'].set(value=centers[i])
        params[f'peak{i}_amplitude'].set(value=amps[i])
        params[f'peak{i}_sigma'].set(value=sigmas[i])

    # 8) Evaluate fit and components
    best_fit   = comp.eval(params=params, x=bias)
    comps_vals = comp.eval_components(params=params, x=bias)

    # 9) Optional convolution with weight function
    if weight_function_show:
        wsig = ds_out.attrs.get('weight_sigma', 1.0)
        wfunc = np.exp(-bias**2 / (2 * wsig**2))
        conv = np.convolve(best_fit, wfunc, mode='same') / np.sum(wfunc)

    # 10) Plotting
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(bias, ldos, 'k-', lw=1.5, alpha=0.8, label='LDOS_smoothed', zorder=1)
    ax.plot(bias, best_fit, fit_color+'-', lw=4, alpha=1.0,
            label=f"{chosen} Fit (redchi={redchi:.2e})", zorder=10)

    # Shade CdGM levels if requested
    if show_shade and 'level_proximity' in ds_out:
        level_prox = ds_out['level_proximity'].isel(Y=y_idx, X=x_idx).values
        Ef_val = ds_out.attrs.get('Ef', 4.4)
        SCgap = ds_out.attrs.get('SCgap', 1.8)
        E_mu = (SCgap**2) / Ef_val
        # peak-specific lines and shading
        for i in idxs:
            lp = level_prox[i]
            if not np.isnan(lp) and abs(lp) < SCgap:
                if lp == 0:
                    lc, ls, fill = 'gray', '-', True
                else:
                    frac = abs((lp / E_mu) % 1)
                    frac = 1 - frac if frac>0.5 else frac
                    if np.isclose(frac,0,atol=1e-2): lc, ls, fill = 'cyan', '--', True
                    elif np.isclose(frac,0.5,atol=1e-2): lc, ls, fill = 'magenta', '--', True
                    else: lc, ls, fill = 'gray', '--', False
                ax.axvline(lp, color=lc, linestyle=ls, linewidth=1)
                if fill:
                    ax.fill_between(bias, comps_vals[f'peak{i}_'], color=lc, alpha=0.3)
        # integer and half-integer guide lines
        cand = []
        n_min = int(np.floor(bias.min()/E_mu)); n_max = int(np.ceil(bias.max()/E_mu))
        for n in range(n_min, n_max+1):
            for lvl in [n*E_mu, (n+0.5)*E_mu]:
                if abs(lvl) < SCgap: cand.append(lvl)
        for lvl in sorted(cand):
            frac = abs((lvl/E_mu)%1)
            frac = 1 - frac if frac>0.5 else frac
            col = 'cyan' if np.isclose(frac,0,atol=1e-2) else 'magenta'
            ax.axvline(lvl, color=col, ls='--', lw=1,
                       label=('Integer CdGM Level' if col=='cyan' else 'Half-Integer CdGM Level'))
    
    # weight function and peak components
    if weight_function_show:
        ax.plot(bias, conv, fit_color+'-', lw=3, alpha=0.8, label='Convoluted Fit', zorder=9)
        ax2 = ax.twinx()
        ax2.plot(bias, wfunc, '--', lw=1.5, alpha=0.5, color='gray', label='Weight Function')
        ax2.set_ylabel('Weight Function', color='gray')

    # plot individual peaks and background
    for i in idxs:
        arr = comps_vals[f'peak{i}_']
        ax.plot(bias, arr, '--', lw=1.5, alpha=0.4, label=f'Peak {i}', zorder=2)
    if 'bkg_' in comps_vals:
        ax.plot(bias, comps_vals['bkg_'], '--', lw=1.5, alpha=0.4, color='gray', label='Background', zorder=2)

    ax.axvline(0, color='gray', ls='-', lw=1)
    ax.set_xlabel('Bias (mV)')
    ax.set_ylabel('LDOS_smoothed')

    # title with physical coordinates
    y_phys = ds_out['Y'].isel(Y=y_idx).item()*1e9
    x_phys = ds_out['X'].isel(X=x_idx).item()*1e9
    ax.set_title(f'Y_idx={y_idx}, X_idx={x_idx}, model={chosen}\n'
                 f'Y={y_phys:.2f} nm, X={x_phys:.2f} nm', fontsize=10)

    # legend handling
    if weight_function_show:
        h1,l1 = ax.get_legend_handles_labels(); h2,l2 = ax2.get_legend_handles_labels()
        ax.legend(h1+h2, l1+l2, loc='upper left')
    else:
        ax.legend(loc='best')
    plt.tight_layout()

    # 11) Collect all curves into DataFrame
    data = {'LDOS_smoothed': ldos, 'best_fit': best_fit}
    for name, arr in comps_vals.items():
        data[name.rstrip('_')] = arr
    if weight_function_show:
        data['convoluted_fit'] = conv
        data['weight_function'] = wfunc
    df = pd.DataFrame(data, index=bias)
    df.index.name = 'bias_mV'

    # 12) Return
    if return_fig:
        return fig, df
    else:
        plt.show()
        return df



# -

fig,_ = plot_region_fitting_result_from_dsout(ds_out= grid_LDOS_SnD_pks_results, ds=grid_LDOS_SnD_pks,
                                            weight_function_show=False, 
                                            use_zb_mask=True,
                                            #y_idx=14, x_idx=19,
                                             y_idx=104, x_idx=149,
                                              show_shade=False,
                                            return_fig=True)#,model_type='Gaussian')#model_type='Gaussian',  #model_type='Lorentzian')#, 

fig,_ = plot_region_fitting_result_from_dsout(ds_out= grid_LDOS_SnD_pks_results, ds=grid_LDOS_SnD_pks,
                                            weight_function_show=False, 
                                            use_zb_mask=True,
                                            #y_idx=14, x_idx=19,
                                              #y_idx=104, x_idx=149,
                                              show_shade=False,
                                            return_fig=True)#,model_type='Gaussian')#model_type='Gaussian',  #model_type='Lorentzian')#, 
fig





# ## export graph & data fro Dr. shin

# +
import matplotlib.pyplot as plt

# 1) 좌표 리스트
# for 2T 003
'''select_coords = [
    (24, 50),
    (29, 152),
    (53, 97),
    (81, 16),
    (104, 149),
    (125, 32),
    (131, 118),
    (145, 99)
]
'''
# for 2T 003
select_coords = [
    (24, 50),
    (14, 49),
    (25, 32),
    (31, 18),
    (45, 59)
]
# 2) Y값 기준으로 위→아래 정렬 및 번호 매기기
sorted_coords = sorted(select_coords, key=lambda yx: yx[0])
numbers = list(range(1, len(sorted_coords) + 1))

# 3) 그림 생성 및 지도 플롯
fig, ax = plt.subplots(figsize=(6,6))
grid_LDOS_SnD_pks.LDOS.sel(bias_mV=0).plot(ax=ax)

# 4) 좌표별로 점 찍고 번호 텍스트는 점 위쪽에 배치
for (y, x), num in zip(sorted_coords, numbers):
    # 실제 축 좌표값
    x_val = grid_LDOS_SnD_pks['X'].isel(X=x).values
    y_val = grid_LDOS_SnD_pks['Y'].isel(Y=y).values

    # 반투명 붉은 점
    ax.scatter(
        x_val, y_val,
        color='red', alpha=0.5, s=50,
        edgecolors='none', zorder=10
    )
    ax.set_aspect('equal', adjustable='box')
    
    # 번호 텍스트를 점의 위쪽으로 오프셋
    ax.annotate(
        str(num),
        xy=(x_val, y_val),
        xytext=(0, 5),            # y축으로 5포인트만큼 위쪽으로 이동
        textcoords='offset points',
        ha='center', va='bottom', # 중앙 정렬, 텍스트의 하단이 기준점
        color='white',
        fontsize=12,
        fontweight='bold',
        zorder=11
    )

# 5) 최종 레이아웃
ax.set_title('Bias=0 LDOS Map with Selected Points')
plt.tight_layout()
plt.show()


# +
import matplotlib.pyplot as plt

# 1) 저장할 좌표 리스트
# for 2T 003
'''select_coords = [
    (24, 50),
    (29, 152),
    (53, 97),
    (81, 16),
    (104, 149),
    (125, 32),
    (131, 118),
    (145, 99)
]
'''
# for 2T 003
select_coords = [
    (24, 50),
    (14, 49),
    (25, 32),
    (31, 18),
    (45, 59)
]


# 2) 순회하면서 그림과 DataFrame 생성·저장
for idx, (y, x) in enumerate(select_coords, start=1):
    label = f"{idx}_Y{y}X{x}"
    
    # a) 함수 호출: fig, df 반환
    fig, df = plot_region_fitting_result_from_dsout(
        ds_out=grid_LDOS_SnD_pks_results,
        ds=grid_LDOS_SnD_pks,
        weight_function_show=False,
        use_zb_mask=True,
        y_idx=y,
        x_idx=x,
        return_fig=True
    )
    
    # b) SVG로 그림 저장
    fig.savefig(f"{label}.svg", format='svg', bbox_inches='tight')
    
    # **c) 화면에 그림 출력**
    plt.show()
    
    # d) CSV로 DataFrame 저장
    df.to_csv(f"df_{label}.csv", index=True)
    
    # e) 메모리 해제
    plt.close(fig)


# +
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from lmfit.models import LorentzianModel, GaussianModel, VoigtModel, ConstantModel
from functools import reduce

def plot_fitting_result_facet_grid(
        ds_out,
        model_type=['Lorentzian','Gaussian','Voigt'],
        point_list=20,
        ncol=4,
        weight_function_show=False,
        ZB_masking=True,
        show_legend=True,
        share_xy=True,
        show_shapes=True):
    """
    Version v1.9.1:
    - Default model_type is a list of allowed models; per-pixel best model taken from ds_out['model_type'].

    Parameters
    ----------
    ds_out : xarray.Dataset
        Must include dims Y,X,peak and variables:
            - 'LDOS', 'bias_mV', 'peak_center', 'peak_amplitude', 'peak_sigma'
            - 'redchi', 'level_proximity', 'model_type'
            - Optional: 'background_value', 'best_fit', 'ZB_mask'
    model_type : list of str or str
        If list: allowed models and per-pixel model chosen from ds_out['model_type'].
        If str: override and use this fixed model for all pixels.
    point_list : int or list of (Y, X)
        Pixels to plot.
    ncol : int
        Number of columns in grid.
    weight_function_show : bool
        Show convolution and weight function.
    ZB_masking : bool
        Restrict pixel selection by 'ZB_mask'.
    show_legend : bool
        Toggle legend in subplots.
    share_xy : bool
        Share axes among subplots.
    show_shapes : bool
        Toggle filled shapes for CdGM peaks.
    """
    Ny, Nx = ds_out.dims['Y'], ds_out.dims['X']
    # select points
    if isinstance(point_list, int):
        if ZB_masking and 'ZB_mask' in ds_out:
            mask = ~np.isnan(ds_out['ZB_mask'].values)
            ys, xs = np.where(mask)
            coords = list(zip(ys,xs))
        else:
            coords = [(y,x) for y in range(Ny) for x in range(Nx)]
        idxs = np.random.choice(len(coords), min(point_list,len(coords)), replace=False)
        selected = [coords[i] for i in idxs]
    else:
        selected = point_list
    # setup grid
    n_points = len(selected)
    nrow = int(np.ceil(n_points/ncol))
    fig, axes = plt.subplots(nrow,ncol,figsize=(5*ncol,5*nrow), sharex=share_xy, sharey=share_xy)
    axes = axes.flatten()
    # constants
    bias = ds_out['bias_mV'].values
    w_sigma = ds_out.attrs.get('weight_sigma',1.0)
    Ef = ds_out.attrs.get('Ef',4.4)
    SCgap = ds_out.attrs.get('SCgap',1.8)
    E_mu = SCgap**2/Ef
    for i,(y,x) in enumerate(selected):
        ax = axes[i]
        ldos = ds_out['LDOS'].isel(Y=y,X=x).values
        n_peaks = ds_out.dims['peak']
        centers = ds_out['peak_center'].isel(Y=y,X=x).values
        amps = ds_out['peak_amplitude'].isel(Y=y,X=x).values
        sigmas = ds_out['peak_sigma'].isel(Y=y,X=x).values
        level_prox = ds_out['level_proximity'].isel(Y=y,X=x).values
        redchi = ds_out['redchi'].isel(Y=y,X=x).item()
        # determine per-pixel model
        if isinstance(model_type, list):
            per = ds_out['model_type'].isel(Y=y,X=x).item().capitalize()
        else:
            per = model_type.capitalize()
        if per not in (model_type if isinstance(model_type,list) else [model_type]):
            raise ValueError(f"Model '{per}' not in allowed model_type list.")
        # map to class and color
        if per=='Lorentzian': MC, col = LorentzianModel, 'r'
        elif per=='Gaussian': MC, col = GaussianModel, 'b'
        else:               MC, col = VoigtModel,      'g'
        # plot LDOS
        ax.plot(bias,ldos,'k-',alpha=0.6,label='LDOS')
        # background
        if 'background_value' in ds_out:
            bgv = ds_out['background_value'].isel(Y=y,X=x).item()
            if not np.isnan(bgv):
                ax.plot(bias, np.full_like(bias,bgv),'--',color='gray',label='Background')
        # build and eval composite
        models=[]
        if 'background_value' in ds_out: models.append(ConstantModel(prefix='bkg_'))
        for j in range(n_peaks):
            if not np.isnan(centers[j]): models.append(MC(prefix=f'peak{j}_'))
        comp = reduce(lambda a,b: a+b, models)
        params=comp.make_params()
        if 'background_value' in ds_out: params['bkg_c'].set(value=bgv)
        for j in range(n_peaks):
            if not np.isnan(centers[j]):
                params[f'peak{j}_center'].set(value=centers[j])
                params[f'peak{j}_amplitude'].set(value=amps[j])
                params[f'peak{j}_sigma'].set(value=sigmas[j])
        best = comp.eval(params=params,x=bias)
        ax.plot(bias,best,'-',linewidth=4,color=col,alpha=0.5,label=f'{per} Fit (χ²={redchi:.2e})')
        # convolution
        if weight_function_show:
            wfunc = np.exp(-0.5*(bias/w_sigma)**2)
            conv = np.convolve(best,wfunc,mode='same')/np.sum(wfunc)
            ax.plot(bias,conv,'-',linewidth=3,color=col,alpha=0.8,label='Convoluted')
        # peaks and shapes
        cmap=get_cmap('tab20')(np.linspace(0,1,max(n_peaks,10)))
        alpha_p=0.5 if show_shapes else 1.0
        for j in range(n_peaks):
            key=f'peak{j}_'
            if key in comp.eval_components(params=params,x=bias):
                pk=comp.eval_components(params=params,x=bias)[key]
                ax.plot(bias,pk,'--',color=cmap[j],alpha=alpha_p,label=f'Peak {j}')
                if show_shapes and not np.isnan(level_prox[j]) and abs(level_prox[j])<SCgap:
                    lp=level_prox[j]; frac=abs((lp/E_mu)%1)
                    if frac>0.5: frac=1-frac
                    if lp==0: c,ls,fill='gray','-',True
                    elif np.isclose(frac,0): c,ls,fill='cyan','--',True
                    elif np.isclose(frac,0.5): c,ls,fill='magenta','--',True
                    else: c,ls,fill='gray','--',False
                    ax.axvline(lp,color=c,linestyle=ls)
                    if fill: ax.fill_between(bias,pk,color=c,alpha=0.3)
        ax.axvline(0,color='gray',linestyle=':')
        #ax.set_title(f'(Y={y},X={x},model={per})',fontsize=10)
        y_phys = ds_out['Y'].isel(Y=y).item()*1E9
        x_phys = ds_out['X'].isel(X=x).item()*1E9
        ax.set_title(
            f'Y_idx={y}, X_idx={x}, model={per}\n Y={y_phys:.2f} nm, X={x_phys:.2f} nm',
            fontsize=10
        )       
        if show_legend: ax.legend(fontsize=8,loc='upper right')
        else:
            v=~np.isnan(ldos)
            ax.set_xlim(bias[v].min(),bias[v].max()); ax.set_ylim(ldos[v].min(),ldos[v].max())
        ax.grid(True)
    # disable extras
    for k in range(n_points,nrow*ncol): axes[k].axis('off')
    plt.suptitle('Fit Result per Pixel',fontsize=16)
    plt.tight_layout()
    plt.show()



# +
#grid_LDOS_SnD_pks_results.sel( X = slice(0.5E-7,0.7E-7), Y = slice(2.0E-7,2.2E-7))
# -

plot_fitting_result_facet_grid(grid_LDOS_SnD_pks_results,#.sel( X = slice(0.5E-7,0.7E-7), Y = slice(2.0E-7,2.2E-7)), 
                              model_type=['Lorentzian', 'Gaussian', 'Voigt'], 
                                   point_list=60, 
                                   ncol=6, 
                                   weight_function_show=False,
                                   ZB_masking=True,
                                   show_legend=True,
                                   share_xy=False,show_shapes=False)





# # load the fitting results 

# +
#updated_GS_LDOS_0T_002 = xr.open_dataset('updated_GS_LDOS_0T_002.nc')
#updated_GS_topo_0T_002 = xr.open_dataset('updated_GS_topo_0T_002.nc')
GS_topo_2T_003 = xr.open_dataset('GS_topo_2T_003.nc')
GS_LDOS_2T_003 = xr.open_dataset('GS_LDOS_2T_003.nc')

#GS_topo_2T_005 = xr.open_dataset('GS_topo_2T_005.nc')
#GS_LDOS_2T_005 = xr.open_dataset('GS_LDOS_2T_005.nc')
# -

# ## load fitting results & update fitted_curves



#grid_LDOS_SnD_pks_Lorentzian_fit = xr.open_dataset('grid_LDOS_SnD_2T_003_smth_LDOS_L_fit_2mV.nc')
#grid_LDOS_SnD_pks_fit_results =  xr.open_dataset('grid_LDOS_SnD_pks_updated0T002_fit_gaussian_IngapRange.nc')
grid_LDOS_SnD_pks_results


# +
#grid_LDOS  = updated_GS_LDOS_0T_002[['LDOS']].copy()
#grid_topo  = updated_GS_topo_0T_002.copy()

# +
#grid_2T_003_fit = xr.merge([grid_LDOS_SnD_pks_Lorentzian_fit,grid_LDOS,grid_topo])
#grid_0T_002_fit = xr.merge([grid_LDOS_SnD_pks_Lorentzian_fit,grid_LDOS,grid_topo])

grid_2T003_fit = grid_LDOS_SnD_pks_results.update(grid_LDOS).update(grid_topo)
grid_2T003_fit.to_netcdf('grid_2T003_fit.nc')

#grid_2T005_fit = grid_LDOS_SnD_pks_results.update(grid_LDOS).update(grid_topo)
#grid_2T005_fit.to_netcdf('grid_2T005_fit.nc')
# -

# # OPEN(update)_dataset_with_fit_result 

ds = xr.open_dataset('grid_2T003_fit.nc')
#ds = xr.open_dataset('grid_2T005_fit.nc')

ds

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from lmfit.models import LorentzianModel, GaussianModel, VoigtModel, ConstantModel
from functools import reduce

def plot_region_fitting_result_from_dsfit(
        ds_fit,
        model_type=None,
        allowed_models=['Lorentzian', 'Gaussian', 'Voigt'],
        y_idx=None, x_idx=None,
        weight_function_show=False,
        use_zb_mask=False,
        zb_mask_key='ZB_mask',
        show_shade=True,
        show_legend=True,
        return_fig=False):
    """
    Reconstruct and plot the best-fit curve for a selected pixel from a combined dataset,
    including optional CdGM level proximity and guide lines, and collect all curves into a pandas DataFrame.

    Parameters
    ----------
    ds_fit : xarray.Dataset
        Combined dataset with dims (Y, X, bias_mV, peak) containing:
          - Variables: 'bias_mV', 'LDOS', 'peak_center', 'peak_amplitude', 'peak_sigma', 'redchi'
          - Optional: 'background_value', 'model_type', zb_mask_key
        Attributes: 'weight_sigma', 'Ef', 'SCgap'
    model_type : {None, str}
        If None, uses per-pixel model from ds_fit['model_type']; else forces given model.
    allowed_models : list of str
    y_idx, x_idx : int
        Pixel indices.
    weight_function_show : bool
    use_zb_mask : bool
    zb_mask_key : str
    show_shade : bool
    show_legend : bool
        If False, suppress legend display.
    return_fig : bool
        If True, return (fig, df); else show plot and return df.

    Returns
    -------
    df : pandas.DataFrame
    fig : matplotlib.figure.Figure, optional
    """
    # 1) Determine pixel indices
    ny, nx = ds_fit.dims['Y'], ds_fit.dims['X']
    if use_zb_mask and y_idx is None and x_idx is None and zb_mask_key in ds_fit:
        raw_mask = ds_fit[zb_mask_key].values
        valid_mask = np.any(~np.isnan(raw_mask), axis=2) if raw_mask.ndim == 3 else raw_mask.astype(bool)
        rc = ds_fit['redchi'].values
        valid_fit = ~np.isnan(rc)
        ys, xs = np.where(valid_mask & valid_fit)
        if len(ys) == 0:
            raise RuntimeError("No valid pixel found with ZB mask and redchi")
        sel = np.random.randint(len(ys))
        y_idx, x_idx = int(ys[sel]), int(xs[sel])
    if y_idx is None:
        y_idx = np.random.randint(ny)
    if x_idx is None:
        x_idx = np.random.randint(nx)

    print(f"Using pixel Y={y_idx}, X={x_idx} for plotting")

    # 2) Extract data
    bias = ds_fit['bias_mV'].values
    ldos = ds_fit['LDOS'].isel(Y=y_idx, X=x_idx).values

    # 3) Model selection
    if model_type is None:
        chosen = ds_fit['model_type'].isel(Y=y_idx, X=x_idx).item().capitalize()
    else:
        chosen = model_type.capitalize()
    if chosen not in allowed_models:
        raise ValueError(f"Model '{chosen}' not supported. Choose from {allowed_models}.")
    if chosen == 'Lorentzian':
        mc, fit_color = LorentzianModel, 'r'
    elif chosen == 'Gaussian':
        mc, fit_color = GaussianModel, 'b'
    else:
        mc, fit_color = VoigtModel, 'g'

    # 4) Bias mask
    mask = np.ones_like(bias, bool)
    if use_zb_mask and zb_mask_key in ds_fit:
        raw = ds_fit[zb_mask_key].isel(Y=y_idx, X=x_idx).values
        if isinstance(raw, np.ndarray) and raw.shape == bias.shape:
            mask = ~np.isnan(raw) if np.issubdtype(raw.dtype, np.floating) else raw.astype(bool)
        elif np.ndim(raw) == 0:
            mask = np.full_like(bias, bool(raw), bool)

    # 5) Load fit parameters
    redchi = ds_fit['redchi'].isel(Y=y_idx, X=x_idx).item()
    n_peaks = ds_fit.dims['peak']
    centers = ds_fit['peak_center'].isel(Y=y_idx, X=x_idx).values
    amps    = ds_fit['peak_amplitude'].isel(Y=y_idx, X=x_idx).values
    sigmas  = ds_fit['peak_sigma'].isel(Y=y_idx, X=x_idx).values
    idxs = [i for i in range(n_peaks)
            if not (np.isnan(centers[i]) or np.isnan(amps[i]) or np.isnan(sigmas[i]))]
    print("Valid peak indices:", idxs)

    # 6) Construct composite model
    models = []
    if 'background_value' in ds_fit:
        models.append(ConstantModel(prefix='bkg_'))
        bgv = ds_fit['background_value'].isel(Y=y_idx, X=x_idx).item()
    for i in idxs:
        models.append(mc(prefix=f'peak{i}_'))
    comp = reduce(lambda a, b: a + b, models)
    params = comp.make_params()
    if 'background_value' in ds_fit:
        params['bkg_c'].set(value=bgv)
    for i in idxs:
        params[f'peak{i}_center'].set(value=centers[i])
        params[f'peak{i}_amplitude'].set(value=amps[i])
        params[f'peak{i}_sigma'].set(value=sigmas[i])

    # 7) Evaluate fit & components
    best_fit  = comp.eval(params=params, x=bias)
    comps_vals = comp.eval_components(params=params, x=bias)

    # 8) Optional weight convolution
    if weight_function_show:
        wsig = ds_fit.attrs.get('weight_sigma', 1.0)
        wfunc = np.exp(-bias**2 / (2 * wsig**2))
        conv = np.convolve(best_fit, wfunc, mode='same') / np.sum(wfunc)

    # 9) Plot
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(bias, ldos, 'k-', lw=1.5, alpha=0.8, label='LDOS', zorder=1)
    ax.plot(bias, best_fit, fit_color+'-', lw=4, alpha=1.0,
            label=f"{chosen} Fit (redchi={redchi:.2e})", zorder=10)
    if weight_function_show:
        ax.plot(bias, conv, fit_color+'-', lw=3, alpha=0.8, label='Convoluted Fit', zorder=9)
        ax2 = ax.twinx()
        ax2.plot(bias, wfunc, '--', lw=1.5, alpha=0.5, color='gray', label='Weight Function')
        ax2.set_ylabel('Weight Function', color='gray')
    for i in idxs:
        ax.plot(bias, comps_vals[f'peak{i}_'], '--', lw=1.5, alpha=0.4,
                label=f'Peak {i}', zorder=2)
    if 'bkg_' in comps_vals:
        ax.plot(bias, comps_vals['bkg_'], '--', lw=1.5, alpha=0.4,
                color='gray', label='Background', zorder=2)

    # CdGM levels & shading
    if show_shade:
        lvl_prox = ds_fit['level_proximity'].isel(Y=y_idx, X=x_idx).values
        Ef = ds_fit.attrs.get('Ef', 1.0)
        SCgap = ds_fit.attrs.get('SCgap', 1.0)
        E_mu = SCgap**2 / Ef
        for i in idxs:
            lp = lvl_prox[i]
            if not np.isnan(lp) and abs(lp) < SCgap:
                if lp == 0:
                    lc, ls, fill = 'gray', '-', True
                else:
                    frac = abs((lp/E_mu) % 1)
                    frac = 1 - frac if frac > 0.5 else frac
                    if np.isclose(frac, 0, atol=1e-2):
                        lc, ls, fill = 'cyan', '--', True
                    elif np.isclose(frac, 0.5, atol=1e-2):
                        lc, ls, fill = 'magenta', '--', True
                    else:
                        lc, ls, fill = 'gray', '--', False
                ax.axvline(lp, color=lc, linestyle=ls, linewidth=1)
                if fill:
                    ax.fill_between(bias, comps_vals[f'peak{i}_'], color=lc, alpha=0.3)
        cand = []
        nmin = int(np.floor(bias.min()/E_mu))
        nmax = int(np.ceil(bias.max()/E_mu))
        for n in range(nmin, nmax+1):
            for lvl in (n*E_mu, (n+0.5)*E_mu):
                if abs(lvl) < SCgap:
                    cand.append(lvl)
        for lvl in sorted(cand):
            frac = abs((lvl/E_mu) % 1)
            frac = 1 - frac if frac > 0.5 else frac
            if np.isclose(frac, 0, atol=1e-2):
                col, lab = 'cyan', 'Integer CdGM Level'
            elif np.isclose(frac, 0.5, atol=1e-2):
                col, lab = 'magenta', 'Half-Integer CdGM Level'
            else:
                col, lab = 'gray', None
            ax.axvline(lvl, color=col, ls='--', lw=1, label=lab)
    else:
        valid_bias = bias[mask]
        if valid_bias.size > 0:
            ax.set_xlim(valid_bias.min(), valid_bias.max())

    ax.axvline(0, color='gray', ls='-', lw=1)
    ax.set_xlabel('Bias (mV)')
    ax.set_ylabel('LDOS')

    # Title with pixel indices and physical coordinates
    y_phys = ds_fit['Y'].isel(Y=y_idx).item() * 1e9
    x_phys = ds_fit['X'].isel(X=x_idx).item() * 1e9
    ax.set_title(
        f'Y_idx={y_idx}, X_idx={x_idx}, model={chosen}\n'
        f'Y={y_phys:.2f} nm, X={x_phys:.2f} nm',
        fontsize=10
    )

    # Legend
    if show_legend:
        if weight_function_show:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc='upper left')
        else:
            h, l = ax.get_legend_handles_labels()
            unique = dict(zip(l, h))
            ax.legend(unique.values(), unique.keys(), loc='best')

    plt.tight_layout()

    # 10) Collect into DataFrame
    data = {'LDOS': ldos, 'best_fit': best_fit}
    for name, arr in comps_vals.items():
        data[name.rstrip('_')] = arr
    if weight_function_show:
        data['convoluted_fit'] = conv
        data['weight_function'] = wfunc
    df = pd.DataFrame(data, index=bias)
    df.index.name = 'bias_mV'

    if return_fig:
        return fig, df
    else:
        plt.show()
        return df



# -

fig,_ = plot_region_fitting_result_from_dsfit(ds,use_zb_mask=True,
        zb_mask_key='ZB_mask',
        show_shade=False,
                                              show_legend=False,
                                              
        return_fig=True)
#fig



# # ZBP and symmetric peak search withou Applying AI/ML 
# ## After LMfit package fitting  Classification test 

# +
import os
import numpy as np
import xarray as xr
import ipywidgets as widgets
import seaborn_image as isns
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from IPython.display import display, FileLink, clear_output

def process_and_plot_fit_results(
    ds_out: xr.Dataset,
    selected_bias: float = 0,
    fit_quality: list = ['R_squared', 'redchi', 'residuals']
) -> xr.Dataset:
    """
    Compute residuals (best_fit - LDOS), redchi, and R² per pixel (Y, X), store them 
    in ds_out, and visualize selected quantities using seaborn-image with scale bars
    in nm. Colorbars are placed next to each subplot automatically.

    Parameters
    ----------
    ds_out : xarray.Dataset
        Dataset with fitting results including 'best_fit', 'LDOS', and 'redchi'.
    selected_bias : float, optional
        Bias value at which to plot the residuals. Default is 0 mV.
    fit_quality : list of str, optional
        List of which plots to show. Options: 'R_squared', 'redchi', 'residuals'

    Returns
    -------
    xarray.Dataset
        Dataset with added variables: 'residuals', 'fit_quality', 'r_squared'
    """
    # 1. Compute derived maps
    residuals = ds_out["best_fit"] - ds_out["LDOS"]
    residuals = residuals.where(~np.isnan(ds_out["best_fit"]))

    fit_quality_map = ds_out["redchi"]

    y_true = ds_out["LDOS"]
    y_fit = ds_out["best_fit"]
    ss_residual = ((y_true - y_fit) ** 2).sum(dim="bias_mV")
    ss_total = ((y_true - y_true.mean(dim="bias_mV")) ** 2).sum(dim="bias_mV")
    r_squared = 1 - (ss_residual / ss_total)
    r_squared = r_squared.where(~np.isnan(y_fit.isel(bias_mV=0)))

    ds_new = ds_out.copy()
    ds_new["residuals"] = residuals
    ds_new["fit_quality"] = fit_quality_map
    ds_new["r_squared"] = r_squared

    bias = ds_out["bias_mV"].values
    selected_bias_idx = np.argmin(np.abs(bias - selected_bias))
    print(f"Selected bias value: {bias[selected_bias_idx]:.4f} mV (closest to {selected_bias} mV)")

    # 2. Determine which plots to show
    plot_flags = {
        'R_squared': 'R_squared' in fit_quality,
        'redchi': 'redchi' in fit_quality,
        'residuals': 'residuals' in fit_quality
    }
    n_plots = sum(plot_flags.values())

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]
    plt.suptitle("Fitting Results Validation", fontsize=16)

    ax_idx = 0

    # 3. Plot R²
    if plot_flags['R_squared']:
        isns.imshow(
            r_squared,
            ax=axes[ax_idx],
            cmap="Grays",
            vmin=0, vmax=1,
            cbar_label="R² Score",
            dx=1, dy=1,
            units="nm"
        )
        axes[ax_idx].set_title("R² Mapping")
        axes[ax_idx].set_xlabel("X")
        axes[ax_idx].set_ylabel("Y")
        ax_idx += 1

    # 4. Plot redchi
    if plot_flags['redchi']:
        isns.imshow(
            fit_quality_map,
            ax=axes[ax_idx],
            cmap="winter",
            cbar_label="reduced χ²",
            dx=1, dy=1,
            units="nm"
        )
        axes[ax_idx].set_title("Fit Quality (redchi)")
        axes[ax_idx].set_xlabel("X")
        axes[ax_idx].set_ylabel("Y")
        ax_idx += 1

    # 5. Plot residuals
    if plot_flags['residuals']:
        try:
            residuals_bias = residuals.isel(bias_mV=selected_bias_idx)
        except IndexError as e:
            raise ValueError(f"Selected bias index {selected_bias_idx} is out of range.") from e

        vmax = np.nanmax(np.abs(residuals_bias.values))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

        isns.imshow(
            residuals_bias,
            ax=axes[ax_idx],
            cmap="bwr",
            norm=norm,
            cbar_label="Residuals",
            dx=1, dy=1,
            units="nm"
        )
        axes[ax_idx].set_title(f"Residuals at {bias[selected_bias_idx]:.4f} mV")
        axes[ax_idx].set_xlabel("X")
        axes[ax_idx].set_ylabel("Y")
        ax_idx += 1

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # 6. Save as SVG widget
    btn = widgets.Button(description="Save as SVG", button_style='success')
    out = widgets.Output()

    def _on_save_clicked(_):
        with out:
            clear_output()
            base_dir = os.getcwd()
            target_dir = os.path.join(base_dir, 'output_figures')
            os.makedirs(target_dir, exist_ok=True)
            filepath = os.path.join(target_dir, 'fitting_results.svg')
            fig.savefig(filepath, format='svg')
            print(f"✔ Figure saved to: {filepath}")
            display(FileLink(filepath, result_html_prefix="Download SVG: "))

    btn.on_click(_on_save_clicked)
    display(widgets.HBox([btn]), out)

    return ds_new



# -

ds = process_and_plot_fit_results(ds, selected_bias=0,fit_quality = ['R_squared', 'residuals'])

# +
import xarray as xr
import numpy as np
from lmfit.models import LorentzianModel, ConstantModel

def update_peak_categories_and_convolution(
    ds: xr.Dataset,
    bias_tolerance: float = None,
    amplitude_tolerance_ratio: float = None,
    margin_to_comparison: str = 'FWHM'
) -> xr.Dataset:
    """
    Analyze and categorize STM spectrum peaks, then compute Lorentzian convolution maps.

    This function performs three main operations on the input xarray Dataset:

    1. Zero-Bias Peak (ZBP) Detection
       - For each (Y, X, peak):
         · If margin_to_comparison == 'FWHM', a peak is a ZBP candidate when its FWHM interval
           [center - sigma/2, center + sigma/2] includes 0 mV.
         · Otherwise, a peak whose |center| is less than the numeric margin is a candidate.
       - If multiple candidates exist, select the one minimizing |center|/amplitude,
         excluding any peak whose center lies closer to the superconducting gap (SC_gap)
         than to zero.

    2. Symmetric Peak Classification
       - Divide peaks at each pixel into positive- and negative-bias lists.
       - For each positive peak, find the best matching negative peak:
         · If bias_tolerance and amplitude_tolerance_ratio are both provided:
             • A pair is symmetric if (|pos_center + neg_center| < bias_tolerance OR
               their FWHM intervals overlap)
               AND their relative amplitude difference < amplitude_tolerance_ratio.
         · If neither tolerance is provided, only require FWHM overlap.
       - Pairs satisfying symmetry are further classified:
         · If |pos_center - SC_gap| < |pos_center|, label as SC_peak.
         · Otherwise, label as in_gap_sym.

    3. Lorentzian Convolution Computation
       - For each category mask (ZBP, SC_peak, in_gap_sym) at each pixel:
         · Build a composite model by summing LorentzianModel instances
           (one per selected peak), plus an optional ConstantModel background.
         · Evaluate the composite over the bias_mV axis to produce a convolution curve.
         · If no peaks are selected, produce a zero array.

    New DataArrays added to ds:
      - ZBP_mask         : Boolean mask (Y, X, peak) marking Zero-Bias Peaks
      - SC_peak_mask     : Boolean mask marking superconducting peaks
      - in_gap_sym_mask  : Boolean mask marking in-gap symmetric peak pairs
      - ZBP_conv         : Convolution map (Y, X, bias_mV) for ZBP
      - SC_peak_conv     : Convolution map for SC peaks
      - in_gap_sym_conv  : Convolution map for in-gap symmetric peaks

    Parameters
    ----------
    ds : xarray.Dataset
        Must contain DataArrays 'peak_center', 'peak_amplitude', 'peak_sigma',
        and coordinate 'bias_mV'. Optional DataArray 'background_value' and
        attribute 'SCgap' may be present.
    bias_tolerance : float or None
        Maximum allowed |pos_center + neg_center| for symmetric pairing.
    amplitude_tolerance_ratio : float or None
        Maximum relative amplitude difference for symmetric pairing.
    margin_to_comparison : {'FWHM'} or float
        Criterion for selecting zero-bias peaks.

    Returns
    -------
    xr.Dataset
        A copy of the input dataset with added masks and convolution maps.
    """
    # Make a copy to avoid modifying the original dataset
    ds = ds.copy()

    # Extract numeric arrays
    center    = ds['peak_center'].values      # shape: (Y, X, peak)
    amplitude = ds['peak_amplitude'].values   # shape: (Y, X, peak)
    sigma     = ds['peak_sigma'].values       # shape: (Y, X, peak)
    bias_mV   = ds['bias_mV'].values          # shape: (bias_dim,)
    Y_dim, X_dim, peak_dim = center.shape
    bias_dim = bias_mV.size

    # Retrieve superconducting gap from attributes (default 0)
    SC_gap = ds.attrs.get('SCgap', 0.0)

    # Initialize boolean masks
    ZBP_mask        = np.zeros((Y_dim, X_dim, peak_dim), dtype=bool)
    SC_peak_mask    = np.zeros((Y_dim, X_dim, peak_dim), dtype=bool)
    in_gap_sym_mask = np.zeros((Y_dim, X_dim, peak_dim), dtype=bool)

    # Iterate over each spatial pixel
    for y in range(Y_dim):
        for x in range(X_dim):
            # --- Zero-Bias Peak (ZBP) Detection ---
            candidates = []
            for p in range(peak_dim):
                cen = center[y, x, p]
                wid = sigma[y, x, p]
                if margin_to_comparison == 'FWHM':
                    # FWHM interval covers zero?
                    if (cen - wid/2 <= 0) and (cen + wid/2 >= 0):
                        candidates.append(p)
                else:
                    # Absolute center within numeric margin
                    if abs(cen) < margin_to_comparison:
                        candidates.append(p)

            # Select best ZBP: minimize |center|/amplitude, exclude peaks nearer to SC_gap
            best_score = np.inf
            selected_zbp = None
            for p in candidates:
                cen = center[y, x, p]
                amp = amplitude[y, x, p]
                # Skip if peak is closer to SC gap than to zero
                if abs(cen - SC_gap) < abs(cen):
                    continue
                score = abs(cen) / amp
                if score < best_score:
                    best_score = score
                    selected_zbp = p
            if selected_zbp is not None:
                ZBP_mask[y, x, selected_zbp] = True

            # --- Symmetric Peak Classification ---
            pos_idxs = [p for p in range(peak_dim) if center[y, x, p] > 0]
            neg_idxs = [p for p in range(peak_dim) if center[y, x, p] < 0]

            for pos in pos_idxs:
                pos_cen = center[y, x, pos]
                pos_amp = amplitude[y, x, pos]
                pos_wid = sigma[y, x, pos]
                pos_fwhm = (pos_cen - pos_wid/2, pos_cen + pos_wid/2)

                best_match = None
                min_amp_diff = np.inf

                for neg in neg_idxs:
                    neg_cen = center[y, x, neg]
                    neg_amp = amplitude[y, x, neg]
                    neg_wid = sigma[y, x, neg]
                    neg_fwhm = (neg_cen - neg_wid/2, neg_cen + neg_wid/2)

                    # Condition 1: FWHM overlap
                    overlap = (pos_fwhm[0] <= neg_fwhm[1]) and (pos_fwhm[1] >= neg_fwhm[0])

                    if bias_tolerance is None and amplitude_tolerance_ratio is None:
                        condition = overlap
                        amp_diff_ratio = None
                    else:
                        # Condition 2: center sum within tolerance
                        center_cond = abs(pos_cen + neg_cen) < bias_tolerance if bias_tolerance is not None else False
                        amp_diff_ratio = abs(pos_amp - neg_amp) / max(pos_amp, neg_amp)
                        condition = (center_cond or overlap) and (amp_diff_ratio < amplitude_tolerance_ratio)

                    if condition:
                        if amp_diff_ratio is None:
                            best_match = neg
                        elif amp_diff_ratio < min_amp_diff:
                            min_amp_diff = amp_diff_ratio
                            best_match = neg

                # Assign to SC_peak or in_gap_sym based on distance to SC_gap
                if best_match is not None:
                    dist_zero = abs(pos_cen)
                    dist_sc   = abs(pos_cen - SC_gap)
                    if dist_sc < dist_zero:
                        SC_peak_mask[y, x, pos] = True
                        SC_peak_mask[y, x, best_match] = True
                    else:
                        in_gap_sym_mask[y, x, pos] = True
                        in_gap_sym_mask[y, x, best_match] = True

    # --- Convolution Calculation ---
    def compute_convolution(y, x, mask):
        """Builds and evaluates composite Lorentzian (+ background) model."""
        model = None
        params = None
        for p in range(peak_dim):
            if not mask[y, x, p]:
                continue
            lm = LorentzianModel(prefix=f'p{p}_')
            prm = lm.make_params(
                amplitude=amplitude[y, x, p],
                center=center[y, x, p],
                sigma=sigma[y, x, p]
            )
            if model is None:
                model, params = lm, prm
            else:
                model += lm
                params.update(prm)

        # Add background constant if present
        if "background_value" in ds:
            bkg_val = ds["background_value"].values[y, x]
            bkg = ConstantModel(prefix="bkg_")
            bkg_prm = bkg.make_params(c=bkg_val)
            if model is None:
                model, params = bkg, bkg_prm
            else:
                model += bkg
                params.update(bkg_prm)

        if model is not None:
            return model.eval(params, x=bias_mV)
        else:
            return np.zeros(bias_dim, dtype=float)

    # Allocate arrays for convolution maps
    ZBP_conv_arr        = np.zeros((Y_dim, X_dim, bias_dim), dtype=np.float32)
    SC_peak_conv_arr    = np.zeros_like(ZBP_conv_arr)
    in_gap_sym_conv_arr = np.zeros_like(ZBP_conv_arr)

    for y in range(Y_dim):
        for x in range(X_dim):
            ZBP_conv_arr[y, x]        = compute_convolution(y, x, ZBP_mask)
            SC_peak_conv_arr[y, x]    = compute_convolution(y, x, SC_peak_mask)
            in_gap_sym_conv_arr[y, x] = compute_convolution(y, x, in_gap_sym_mask)

    # Attach new DataArrays to the dataset
    ds = ds.assign(
        ZBP_mask        = (['Y','X','peak'], ZBP_mask),
        SC_peak_mask    = (['Y','X','peak'], SC_peak_mask),
        in_gap_sym_mask = (['Y','X','peak'], in_gap_sym_mask),
        ZBP_conv        = (['Y','X','bias_mV'], ZBP_conv_arr),
        SC_peak_conv    = (['Y','X','bias_mV'], SC_peak_conv_arr),
        in_gap_sym_conv = (['Y','X','bias_mV'], in_gap_sym_conv_arr)
    )

    # Return the enriched dataset
    return ds



# -

ds

ds = update_peak_categories_and_convolution(ds,
                                            bias_tolerance=0.1,
                                            amplitude_tolerance_ratio=0.5,
                                            margin_to_comparison='FWHM')

# ### Extract the ZB_peaks & symmetric peaks from fitting data

# +
#ds.to_netcdf('GS_0T002_fit.nc')

#ds.to_netcdf('GS_2T003_fit_20250518.nc')

ds.to_netcdf('GS_2T003_fit_20250629.nc')

# +
#hv_bias_mV_slicing(ds, ch= 'in_gap_sym_conv')

#hv_bias_mV_slicing(ds, ch= 'ZBP_conv')
#hv_XY_slicing(ds, ch= 'in_gap_sym_conv')

#hv_XY_slicing(ds, ch= 'SC_peak_conv').opts(clim=(0, 0.2E-11))

#hv_XY_slicing(ds, ch= 'ZBP_conv').opts(clim=(0, 0.8E-11))
# -

ds

# + jp-MarkdownHeadingCollapsed=true
##### import os
import numpy as np
import panel as pn
import holoviews as hv
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Initialize HoloViews for interactive display
hv.extension('bokeh')

# Assume `ds` is already loaded as an xarray.Dataset
#ds = ds.copy()
ds = ds_cluster.copy()

# ————— Widgets —————
bias_slider = pn.widgets.FloatSlider(
    name="Bias (mV)",
    start=float(ds['bias_mV'].values.min()),
    end=float(ds['bias_mV'].values.max()),
    step=0.02,
    value=float(ds['bias_mV'].values[0])
)
lower_pct_slider = pn.widgets.FloatSlider(
    name="Lower Percentile (%)", start=0.0, end=100.0, step=0.1, value=5.0
)
upper_pct_slider = pn.widgets.FloatSlider(
    name="Upper Percentile (%)", start=0.0, end=100.0, step=0.1, value=95.0
)
convert_button = pn.widgets.Button(
    name="Convert Percentiles to Actual Values", button_type="primary"
)
mode_selector = pn.widgets.RadioButtonGroup(
    name="Color Mode", options=["Global", "Local"], value="Global"
)
clim_text_ldos    = pn.widgets.StaticText(name="LDOS Clim", value="")
clim_text_bestfit = pn.widgets.StaticText(name="Best Fit Clim", value="")
clim_text_zbp     = pn.widgets.StaticText(name="ZBP Conv Clim", value="")
clim_text_ingap   = pn.widgets.StaticText(name="In Gap Sym Conv Clim", value="")

# ————— Helper Functions —————
def safe_percentile(data, lower, upper):
    valid = data[~np.isnan(data)]
    if valid.size > 0:
        return (np.percentile(valid, lower), np.percentile(valid, upper))
    else:
        return (0.0, 0.0)

def compute_actual_range(event=None):
    bias_arr = ds['bias_mV'].values
    idx = np.abs(bias_arr - bias_slider.value).argmin()
    ldos     = ds['LDOS'].isel(bias_mV=idx).values
    best_fit = ds['best_fit'].isel(bias_mV=idx).values
    zbp      = ds['ZBP_conv'].isel(bias_mV=idx).values
    ingap    = ds['in_gap_sym_conv'].isel(bias_mV=idx).values

    if mode_selector.value == "Global":
        all_vals = np.concatenate([ldos.ravel(), best_fit.ravel(),
                                   zbp.ravel(), ingap.ravel()])
        clim = safe_percentile(all_vals,
                               lower_pct_slider.value,
                               upper_pct_slider.value)
        for text in (clim_text_ldos, clim_text_bestfit,
                     clim_text_zbp, clim_text_ingap):
            text.value = f"{clim[0]:.2e} ~ {clim[1]:.2e}"
    else:
        clim_text_ldos.value    = f"{safe_percentile(ldos,     lower_pct_slider.value, upper_pct_slider.value)[0]:.2e} ~ {safe_percentile(ldos,     lower_pct_slider.value, upper_pct_slider.value)[1]:.2e}"
        clim_text_bestfit.value = f"{safe_percentile(best_fit, lower_pct_slider.value, upper_pct_slider.value)[0]:.2e} ~ {safe_percentile(best_fit, lower_pct_slider.value, upper_pct_slider.value)[1]:.2e}"
        clim_text_zbp.value     = f"{safe_percentile(zbp,      lower_pct_slider.value, upper_pct_slider.value)[0]:.2e} ~ {safe_percentile(zbp,      lower_pct_slider.value, upper_pct_slider.value)[1]:.2e}"
        clim_text_ingap.value   = f"{safe_percentile(ingap,    lower_pct_slider.value, upper_pct_slider.value)[0]:.2e} ~ {safe_percentile(ingap,    lower_pct_slider.value, upper_pct_slider.value)[1]:.2e}"

convert_button.on_click(compute_actual_range)

# ————— Plot Function —————
def plot_bias_mV_slice_2x2(bias_value, lower_pct, upper_pct, mode, ds_input):
    """
    Return a 2×2 grid of HoloViews Images:
      - LDOS
      - Best Fit
      - ZBP Convolution
      - In-Gap Symmetric Convolution
    at the bias closest to bias_value, with axes in nanometers (nm) formatted
    to two decimal places.
    """
    ds_local = ds_input.copy()
    bias_arr = ds_local['bias_mV'].values
    idx      = np.abs(bias_arr - bias_value).argmin()
    sel_bias = bias_arr[idx]

    # Extract 2D slices
    ldos     = ds_local['LDOS'].isel(bias_mV=idx).values
    best_fit = ds_local['best_fit'].isel(bias_mV=idx).values
    zbp      = ds_local['ZBP_conv'].isel(bias_mV=idx).values
    ingap    = ds_local['in_gap_sym_conv'].isel(bias_mV=idx).values

    # Scale coordinates to nm
    x_vals = ds_local['X'].values * 1e9
    y_vals = ds_local['Y'].values * 1e9

    # Determine color limits
    if mode == "Global":
        all_data = np.concatenate([ldos.ravel(),
                                   best_fit.ravel(),
                                   zbp.ravel(),
                                   ingap.ravel()])
        clim = safe_percentile(all_data, lower_pct, upper_pct)
        opts = dict(clim=clim)
        ldos_opts = best_fit_opts = zbp_opts = ingap_opts = opts
    else:
        ldos_opts     = dict(clim=safe_percentile(ldos,     lower_pct, upper_pct))
        best_fit_opts = dict(clim=safe_percentile(best_fit, lower_pct, upper_pct))
        zbp_opts      = dict(clim=safe_percentile(zbp,      lower_pct, upper_pct))
        ingap_opts    = dict(clim=safe_percentile(ingap,    lower_pct, upper_pct))

    # Build HoloViews Images with corrected Bokeh formatters
    def make_img(data, title, opts):
        return hv.Image((x_vals, y_vals, data),
                        kdims=['X (nm)', 'Y (nm)'], vdims=[title]).opts(
            title=f"{title} @ {sel_bias:.2f} mV",
            #cmap='bwr',
            cmap='viridis',
            colorbar=True,
            xlabel='X (nm)', ylabel='Y (nm)',
            xformatter="%.2f", yformatter="%.2f",  # two-decimal tick labels
            frame_width=300, frame_height=300,
            **opts
        )

    ldos_img  = make_img(ldos,     'LDOS',     ldos_opts)
    best_img  = make_img(best_fit, 'Best Fit', best_fit_opts)
    zbp_img   = make_img(zbp,      'ZBP Conv', zbp_opts)
    ingap_img = make_img(ingap,    'In Gap Sym Conv', ingap_opts)

    return (ldos_img + best_img + zbp_img + ingap_img).cols(2)

# Create the interactive HoloViews panel
interactive_panel = pn.panel(
    pn.bind(
        plot_bias_mV_slice_2x2,
        bias_value=bias_slider,
        lower_pct=lower_pct_slider,
        upper_pct=upper_pct_slider,
        mode=mode_selector,
        ds_input=ds
    )
)

# ————— Save Button (reconstruct and save with Matplotlib) —————
save_button = pn.widgets.Button(name="Save as SVG", button_type="primary")

def save_current_svg(event=None):
    # read current widget values
    bias_value = bias_slider.value
    lower_pct  = lower_pct_slider.value
    upper_pct  = upper_pct_slider.value
    mode       = mode_selector.value

    # find index closest to bias_value
    bias_arr = ds['bias_mV'].values
    idx      = np.abs(bias_arr - bias_value).argmin()
    sel_bias = bias_arr[idx]

    # extract data slices
    ldos     = ds['LDOS'].isel(bias_mV=idx).values
    best_fit = ds['best_fit'].isel(bias_mV=idx).values
    zbp      = ds['ZBP_conv'].isel(bias_mV=idx).values
    ingap    = ds['in_gap_sym_conv'].isel(bias_mV=idx).values

    # determine color limits exactly as displayed
    if mode == "Global":
        all_data = np.concatenate([ldos.ravel(),
                                   best_fit.ravel(),
                                   zbp.ravel(),
                                   ingap.ravel()])
        clim = safe_percentile(all_data, lower_pct, upper_pct)
    else:
        ldos_clim     = safe_percentile(ldos,     lower_pct, upper_pct)
        best_fit_clim = safe_percentile(best_fit, lower_pct, upper_pct)
        zbp_clim      = safe_percentile(zbp,      lower_pct, upper_pct)
        ingap_clim    = safe_percentile(ingap,    lower_pct, upper_pct)

    # build Matplotlib figure
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    im_kwargs = dict(interpolation='none', 
                     #cmap='bwr',
                     cmap='viridis',
                    )

    if mode == "Global":
        vmin, vmax = clim
        axes[0,0].imshow(ldos, origin='lower', vmin=vmin, vmax=vmax, **im_kwargs)
        axes[0,1].imshow(best_fit, origin='lower', vmin=vmin, vmax=vmax, **im_kwargs)
        axes[1,0].imshow(zbp, origin='lower', vmin=vmin, vmax=vmax, **im_kwargs)
        axes[1,1].imshow(ingap, origin='lower', vmin=vmin, vmax=vmax, **im_kwargs)
    else:
        axes[0,0].imshow(ldos, origin='lower', vmin=ldos_clim[0], vmax=ldos_clim[1], **im_kwargs)
        axes[0,1].imshow(best_fit, origin='lower', vmin=best_fit_clim[0], vmax=best_fit_clim[1], **im_kwargs)
        axes[1,0].imshow(zbp, origin='lower', vmin=zbp_clim[0], vmax=zbp_clim[1], **im_kwargs)
        axes[1,1].imshow(ingap, origin='lower', vmin=ingap_clim[0], vmax=ingap_clim[1], **im_kwargs)

    # format axes in nm with two decimals
    for ax, title in zip(axes.flatten(),
                         ['LDOS', 'Best Fit', 'ZBP Conv', 'In Gap Sym Conv']):
        ax.set_title(f"{title} @ {sel_bias:.2f} mV")
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, p: f"{v:.2f}")
        )
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, p: f"{v:.2f}")
        )
        ax.set_xlabel("X (nm)")
        ax.set_ylabel("Y (nm)")
        fig.colorbar(ax.images[0], ax=ax)

    plt.tight_layout()

    # save to output_figures folder
    out_dir = os.path.join(os.getcwd(), 'output_figures')
    os.makedirs(out_dir, exist_ok=True)
    filename = f"slice_{sel_bias:.2f}mV.svg"
    filepath = os.path.join(out_dir, filename)
    fig.savefig(filepath, format='svg')
    plt.close(fig)

    print(f"✔ Saved SVG to: {filepath}")

save_button.on_click(save_current_svg)

# ————— Layout and Serve —————
layout = pn.Column(
    pn.Row(bias_slider, mode_selector),
    pn.Row(lower_pct_slider, upper_pct_slider),
    convert_button,
    pn.Row(clim_text_ldos, clim_text_bestfit),
    pn.Row(clim_text_zbp, clim_text_ingap),
    pn.Row(save_button),
    interactive_panel
)

layout.servable()

# -





# +
###  compare original LDOS  vs fitted curves 
# -



# # AI/ML analysis 

# ## load fitted dataset 

# +
### fitting result + LDOS + topo  + fitted curve+ difference 

# +
#grid_0T_002_fit_results.to_netcdf('grid_0T_002_fit_results.nc')
#ds = xr.open_dataset('GS_2T003_fit.nc')

#ds = xr.open_dataset('GS_2T003_fit_20250518.nc')

#ds = xr.open_dataset('GS_2T005_fit_20250522.nc')
ds = xr.open_dataset('GS_2T003_fit_20250629.nc')
# -


ds


# # Define ML clustering functions 

# +
def select_ML_features_interactive(ds: xr.Dataset, callback=None, **kwargs) -> None:
    """
    Launches an interactive widget for selecting features to use in downstream
    machine learning or clustering workflows.

    This function:
      - Reads `ds.data_vars` and identifies any variable whose dimensions include 'peak'.
      - Adds 'X_coordinate' and 'Y_coordinate' to the list of selectable features.
      - Displays a SelectMultiple widget where the user can select one or more features.
      - Upon confirmation, updates the module-level `selected_features_global` list
        and prints the selected features.
      - Calls `callback(selected_features, **kwargs)` if a callback is provided.

    Parameters
    ----------
    ds : xr.Dataset
        xarray Dataset containing at minimum 'X' and 'Y' coordinates and one or more
        data_vars with dimension 'peak'.
    callback : callable, optional
        Function to be invoked after feature selection completes. It will be passed
        the list of selected feature names and any additional keyword arguments.
    **kwargs : dict
        Additional keyword arguments forwarded to the callback.

    Notes
    -----
    - `selected_features_global` is a module-level list reused across calls.
    - Ensure this function is called before initiating clustering routines that
      depend on selected features.
    """
    global selected_features_global

    # Confirm global state variable presence
    print("[select_ML_features_interactive] Using global variable: selected_features_global ->", selected_features_global)
    selected_features_global = []

    # Identify feature candidates: X, Y, plus any 'peak' variables
    base_features = [var for var in ds.data_vars if 'peak' in ds[var].dims]
    candidate_features = ['X_coordinate', 'Y_coordinate'] + base_features

    # Build widgets
    selector = widgets.SelectMultiple(
        options=candidate_features,
        description='Features to use:',
        rows=min(len(candidate_features), 10),
        style={'description_width': 'initial'}
    )
    confirm_btn = widgets.Button(description='Confirm Selection', button_style='primary')
    output = widgets.Output()

    def _on_confirm(_):
        global selected_features_global
        with output:
            clear_output()
            sel = list(selector.value)
            if not sel:
                print("⚠️ Please select at least one feature before confirming.")
                return
            selected_features_global = sel
            print(f"✅ selected_features_global updated to: {selected_features_global}")
            if callback:
                callback(selected_features_global, **kwargs)

    confirm_btn.on_click(_on_confirm)
    display(widgets.VBox([selector, confirm_btn, output]))


def select_cluster_data_var_interactive(ds: xr.Dataset, callback=None, **kwargs) -> None:
    """
    Launches an interactive widget for choosing which cluster variable to use.

    This function:
      - Scans `ds.data_vars` for any variables whose names include 'cluster'.
      - Presents a RadioButtons widget for selection.
      - On confirmation, sets the module-level `selected_cluster_var_global` string
        and prints the chosen variable name.
      - If provided, invokes `callback(selected_cluster_var, **kwargs)`.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing at least one data_var with 'cluster' in its name.
    callback : callable, optional
        Function called after a cluster variable is selected.
    **kwargs : dict
        Additional parameters forwarded to the callback.

    Notes
    -----
    - `selected_cluster_var_global` persists across calls.
    - Subsequent label-selection routines depend on this variable.
    """
    global selected_cluster_var_global

    # Confirm global state variable presence
    print("[select_cluster_data_var_interactive] Using global variable: selected_cluster_var_global ->", selected_cluster_var_global)

    # Find cluster-named variables
    options = [v for v in ds.data_vars if 'cluster' in v]
    if not options:
        raise ValueError("No data_vars containing 'cluster' found in the dataset.")

    selector = widgets.RadioButtons(
        options=options,
        description='Cluster var:',
        style={'description_width': 'initial'}
    )
    confirm_btn = widgets.Button(description='Confirm', button_style='primary')
    output = widgets.Output()

    def _on_confirm(_):
        global selected_cluster_var_global
        with output:
            clear_output()
            selected_cluster_var_global = selector.value
            print(f"✅ selected_cluster_var_global set to: '{selected_cluster_var_global}'")
            if callback:
                callback(selected_cluster_var_global, **kwargs)

    confirm_btn.on_click(_on_confirm)
    display(widgets.VBox([selector, confirm_btn, output]))


def select_labels_in_cluster_interactive(ds: xr.Dataset, callback=None, **kwargs) -> None:
    """
    Launches an interactive widget to select which cluster labels to keep.

    This function:
      - Requires that `selected_cluster_var_global` is already set.
      - Extracts unique labels from the selected cluster variable.
      - Displays a SelectMultiple widget for label selection.
      - Creates a boolean mask for the chosen labels and applies it to all
        compatible data_vars, producing `filtered_ds`.
      - Prints the chosen labels and, if provided, calls
        `callback(filtered_ds, cluster_var, clusters_to_keep, **kwargs)`.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset that must contain the previously selected cluster variable.
    callback : callable, optional
        Function that receives `(filtered_ds, cluster_var, clusters_to_keep)`.
    **kwargs : dict
        Additional arguments forwarded to the callback.

    Raises
    ------
    ValueError
        If `selected_cluster_var_global` is unset or no labels found.
    """
    global selected_cluster_labels_global

    # Confirm global state variable presence
    print("[select_labels_in_cluster_interactive] Using global variable: selected_cluster_labels_global ->", selected_cluster_labels_global)

    if not selected_cluster_var_global:
        raise ValueError("No cluster variable selected. Run select_cluster_data_var_interactive first.")

    # Get unique labels
    da = ds[selected_cluster_var_global]
    unique = np.unique(da.values)
    labels = sorted([int(l) for l in unique if not np.isnan(l)])
    if not labels:
        raise ValueError("No valid labels found in the selected cluster variable.")

    selector = widgets.SelectMultiple(
        options=labels,
        description='Labels to keep:',
        style={'description_width': 'initial'}
    )
    confirm_btn = widgets.Button(description='Confirm', button_style='success')
    output = widgets.Output()

    def _on_confirm(_):
        global selected_cluster_labels_global
        with output:
            clear_output()
            sel = list(selector.value)
            if not sel:
                print("⚠️ Please select at least one label.")
                return
            selected_cluster_labels_global = sel
            print(f"✅ selected_cluster_labels_global updated to: {selected_cluster_labels_global}")

            # Apply mask and filter dataset
            mask = xr.DataArray(
                np.isin(da.values, selected_cluster_labels_global),
                dims=da.dims,
                coords=da.coords
            )
            filtered_ds = ds.copy(deep=True)
            for var in ds.data_vars:
                if set(da.dims).issubset(ds[var].dims):
                    filtered_ds[var] = ds[var].where(mask)

            if callback:
                callback(
                    filtered_ds,
                    cluster_var=selected_cluster_var_global,
                    clusters_to_keep=selected_cluster_labels_global,
                    **kwargs
                )

    confirm_btn.on_click(_on_confirm)
    display(widgets.VBox([selector, confirm_btn, output]))



# -

# ## PCA & KNN clustering 

# +
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import itertools
from tqdm.notebook import tqdm

def cluster_with_pca_knn(
    ds: xr.Dataset,
    selected_features: list[str] = None,
    auto_select_best_k: bool = True,
    k_range: tuple[int,int] = (2,11)
) -> xr.Dataset:
    """
    Perform PCA-based KMeans clustering on ds in-place.

    If selected_features is None, launches interactive feature selection,
    then re-assigns the updated ds in the callback. Returns ds immediately
    so that the first call never overwrites it with None.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with 'X','Y' coords and 'peak' data_vars.
    selected_features : list[str], optional
        Features to use ('X_coordinate','Y_coordinate', or any 'peak' var).
    auto_select_best_k : bool
        If True, pick K with highest silhouette score automatically.
    k_range : tuple(int,int)
        Range of K to test: [start, end).

    Returns
    -------
    xr.Dataset
        The same ds object, now containing the new cluster variable.
    """
    # ——————————————————————————————————————————————————————————————
    # 0) Interactive feature selection을 위해 전역 변수 정의
    global selected_features_global
    try:
        # 이미 정의되어 있으면 그대로 사용
        selected_features_global
    except NameError:
        # 정의되지 않았으면 빈 리스트로 초기화
        selected_features_global = []

    if not selected_features:
        def cont(feats):
            global ds
            # 재귀 호출로 clustering 수행
            ds = cluster_with_pca_knn(ds, feats, auto_select_best_k, k_range)
            print("🔄 Features selected; ds has been updated.")
        select_ML_features_interactive(ds, callback=cont)
        return ds

    # ——————————————————————————————————————————————————————————————
    # 1) Build spatial grid
    x, y = ds['X'].values, ds['Y'].values
    Xg, Yg = np.meshgrid(x, y)
    nY, nX = Xg.shape
    peak_vars = [v for v in ds.data_vars if 'peak' in ds[v].dims]
    nP = ds[peak_vars[0]].shape[2] if peak_vars else 1

    # 2) Construct feature matrix
    arrs = []
    for f in selected_features:
        if f == 'X_coordinate':
            a = np.repeat(Xg.flatten()[:, None], nP, axis=1)
        elif f == 'Y_coordinate':
            a = np.repeat(Yg.flatten()[:, None], nP, axis=1)
        else:
            data = ds[f].values
            if f == 'peak_amplitude' and 'background_value' in ds:
                bg = ds['background_value'].values
                data = data - np.repeat(bg[..., None], data.shape[2], axis=2)
            a = data.reshape(-1,)
        arrs.append(a.flatten())
    A = np.stack(arrs, axis=1)

    # 3) Apply mask
    if 'ZB_mask' in ds:
        mask_flat = np.repeat(
            ds['ZB_mask'].values.astype(bool)[..., None],
            nP, axis=2
        ).reshape(-1)
    else:
        mask_flat = np.ones(A.shape[0], bool)
    valid = (~np.isnan(A).any(axis=1)) & mask_flat
    F = A[valid]

    # 4) PCA transform
    pca = PCA(n_components=F.shape[1])
    S = pca.fit_transform(F)

    # 5) Evaluate clustering for each K
    Ks = list(range(k_range[0], k_range[1]))
    inertias, silhs = [], []
    for k in tqdm(Ks, desc='Clustering'):
        km = KMeans(n_clusters=k, random_state=42).fit(S)
        inertias.append(km.inertia_)
        silhs.append(silhouette_score(S, km.labels_))

    # 6) Diagnostic plots
    for data, title, xl, yl in [
        (pca.explained_variance_ratio_, 'PCA Var Ratio', 'PC', 'Variance'),
        (inertias,                'Inertia vs K',   'K',  'Inertia'),
        (silhs,                   'Silhouette vs K','K',  'Score')
    ]:
        plt.figure(figsize=(6,4))
        plt.plot(range(1, len(data)+1), data, 'o-')
        plt.title(title)
        plt.xlabel(xl); plt.ylabel(yl)
        plt.grid(True); plt.tight_layout(); plt.show()

    # 7) Pick best K
    if auto_select_best_k:
        k_best = Ks[int(np.argmax(silhs))]
        print(f"Selected K = {k_best} (auto)")
    else:
        print(f"Ks tested: {Ks}")
        from IPython.display import display, clear_output
        import ipywidgets as widgets

        selector = widgets.Dropdown(
            options=Ks,
            description='Select K:',
            style={'description_width': 'initial'}
        )
        confirm_btn = widgets.Button(description='Confirm', button_style='primary')
        output = widgets.Output()

        def _on_confirm(_):
            nonlocal S, valid, A, nY, nX, nP
            with output:
                clear_output()
                k = selector.value
                print(f"✅ Selected K = {k}")

                # 8) Final clustering and reshape
                kmf = KMeans(n_clusters=k, random_state=42).fit(S)
                labels = np.full(A.shape[0], -1, int)
                labels[valid] = kmf.labels_
                C3 = labels.reshape((nY, nX, nP))

                # 9) In-place save to ds
                var = 'cluster_PCA_knn0'; idx = 0
                while var in ds.data_vars:
                    idx += 1
                    var = f'cluster_PCA_knn{idx}'
                ds[var] = (('Y','X','peak'), C3)
                ds.attrs['feature_vars_used'] = selected_features

                # 10) Plot PC scatter grid
                pairs = sorted(
                    itertools.combinations(range(pca.n_components_), 2),
                    key=lambda p: pca.explained_variance_ratio_[p[0]] +
                                  pca.explained_variance_ratio_[p[1]],
                    reverse=True
                )[:5]
                plt.figure(figsize=(12,8))
                for i,(a,b) in enumerate(pairs):
                    ax = plt.subplot(2,3,i+1)
                    ax.scatter(S[:,a], S[:,b], c=labels[valid], s=5)
                    ax.set_xlabel(f'PC{a+1}'); ax.set_ylabel(f'PC{b+1}'); ax.grid(True)
                plt.tight_layout(); plt.show()

                print(f"✅ Done. Added '{var}' to dataset.")

        confirm_btn.on_click(_on_confirm)
        display(widgets.VBox([selector, confirm_btn, output]))
        return ds

    # ——————————————————————————————————————————————————————————————
    # 자동 모드 이후 계속 실행되는 부분
    print(f"Selected K = {k_best}")

    # 8) Final clustering and reshape
    kmf = KMeans(n_clusters=k_best, random_state=42).fit(S)
    labels = np.full(A.shape[0], -1, int)
    labels[valid] = kmf.labels_
    C3 = labels.reshape((nY, nX, nP))

    # 9) In-place save to ds
    var = 'cluster_PCA_knn0'; idx = 0
    while var in ds.data_vars:
        idx += 1
        var = f'cluster_PCA_knn{idx}'
    ds[var] = (('Y','X','peak'), C3)
    ds.attrs['feature_vars_used'] = selected_features

    # 10) Plot PC scatter grid
    pairs = sorted(
        itertools.combinations(range(pca.n_components_), 2),
        key=lambda p: pca.explained_variance_ratio_[p[0]] +
                      pca.explained_variance_ratio_[p[1]],
        reverse=True
    )[:5]
    plt.figure(figsize=(12,8))
    for i, (a,b) in enumerate(pairs):
        ax = plt.subplot(2,3,i+1)
        ax.scatter(S[:,a], S[:,b], c=labels[valid], s=5)
        ax.set_xlabel(f'PC{a+1}'); ax.set_ylabel(f'PC{b+1}'); ax.grid(True)
    plt.tight_layout(); plt.show()

    print(f"✅ Done. Added '{var}' to dataset.")
    return ds


# +
#ds = cluster_with_pca_knn(ds, auto_select_best_k= False , k_range=(2, 14))

# +
#ds
# -


# ##  After PCA, cluster statistics 

# +
import os
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
import matplotlib.pyplot as plt
from lmfit.models import LorentzianModel, GaussianModel, VoigtModel

def plot_cluster_statistics(
    ds: xr.Dataset,
    cluster_var: str = None,
    clusters_to_keep: list[int] = None,
    model_type: str = None,
    remove_background: bool = False,
    remove_neg_amp: bool = False,
    remove_outliers: bool = False
) -> None:
    """
    Interactive visualization of peak statistics by cluster, with a Save-as-SVG button.

    This function operates in three stages:

      Stage 1: If `cluster_var` is None, displays a widget for the user to choose
               which dataset variable contains the cluster labels. When the user
               confirms, the function is re-invoked with `cluster_var` set.

      Stage 2: If `clusters_to_keep` is None, displays a widget for the user to select
               which cluster labels to include in the analysis. When confirmed, it
               re-invokes itself with both `cluster_var` and `clusters_to_keep`.

      Stage 3: Once both `cluster_var` and `clusters_to_keep` are provided, it:
        - Optionally subtracts the 'background_value' from peak amplitudes.
        - Optionally zeros out negative amplitudes.
        - Optionally removes outliers per cluster using the 1.5×IQR rule.
        - Fits and evaluates the chosen peak model ('lorentzian', 'gaussian', or 'voigt')
          to generate per-pixel curves.
        - Plots, for each cluster:
            * The mean peak curve ± 95% confidence interval.
            * Histograms of peak centers, amplitudes, and widths.
        - Finally, displays an interactive "Save as SVG" button that writes the
          current figure to `output_figures/cluster_<var>_<labels>.svg`.

    Parameters
    ----------
    ds : xarray.Dataset
        Must contain:
          - 'peak_center', 'peak_amplitude', 'peak_sigma' variables (dims Y, X, peak)
          - Optional 'background_value' variable
          - 'bias_mV' coordinate for the bias axis
        The cluster label variable will be chosen interactively.
    cluster_var : str, optional
        Name of the cluster label variable in `ds`. Default None triggers Stage 1.
    clusters_to_keep : list of int, optional
        List of cluster labels to include. Default None triggers Stage 2.
    model_type : {'lorentzian','gaussian','voigt'} or None
        Peak model to use. If None, reads `ds.attrs['peak_model_type']` or defaults to 'lorentzian'.
    remove_background : bool, default False
        If True, subtracts `ds['background_value']` from amplitudes.
    remove_neg_amp : bool, default False
        If True, resets negative amplitudes to zero.
    remove_outliers : bool, default False
        If True, removes outliers per cluster using the 1.5×IQR rule.

    Returns
    -------
    None
        Displays the interactive widgets, the resulting Matplotlib figure,
        and a Save-as-SVG button when complete.

    Usage
    -----
    >>> plot_cluster_statistics(ds)
    """
    # Preserve any previous selections
    global selected_cluster_var_global, selected_cluster_labels_global
    selected_cluster_var_global = globals().get('selected_cluster_var_global', None)
    selected_cluster_labels_global = globals().get('selected_cluster_labels_global', None)

    # Stage 1: select the cluster variable
    if cluster_var is None:
        def _on_var(var):
            print(f"Selected cluster variable: {var}")
            plot_cluster_statistics(
                ds, var, None, model_type,
                remove_background, remove_neg_amp, remove_outliers
            )
        select_cluster_data_var_interactive(ds, callback=_on_var)
        return

    # Stage 2: select which clusters to keep
    if clusters_to_keep is None:
        def _on_labels(filtered_ds, cluster_var, clusters_to_keep, **kwargs):
            print(f"Selected labels: {clusters_to_keep}")
            plot_cluster_statistics(
                filtered_ds, cluster_var, clusters_to_keep,
                model_type, remove_background, remove_neg_amp, remove_outliers
            )
        select_labels_in_cluster_interactive(ds, callback=_on_labels)
        return

    # Stage 3: all inputs provided, build the figure
    # Determine peak model
    if model_type is None:
        model_type = ds.attrs.get('peak_model_type', 'lorentzian')
    if model_type == 'lorentzian':
        model = LorentzianModel()
    elif model_type == 'gaussian':
        model = GaussianModel()
    elif model_type == 'voigt':
        model = VoigtModel()
    else:
        raise ValueError("model_type must be 'lorentzian', 'gaussian', or 'voigt'")

    # Extract data arrays
    bias = ds['bias_mV'].values
    centers = ds['peak_center'].values
    amps = ds['peak_amplitude'].values
    sigs = ds['peak_sigma'].values
    Y, X, P = centers.shape

    # Optional background subtraction
    if remove_background and 'background_value' in ds:
        bg3d = np.broadcast_to(ds['background_value'].values[:, :, None], (Y, X, P))
        amps = amps - bg3d

    # Optional zero‐negative amplitudes
    if remove_neg_amp:
        amps = np.where(amps < 0, 0, amps)

    # Set up plotting
    n = len(clusters_to_keep)
    colors = sns.color_palette('tab10', n)
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n), squeeze=False)

    # Loop through each selected cluster
    for i, cl in enumerate(clusters_to_keep):
        c = colors[i]
        mask = (ds[cluster_var].values == cl)
        fc = centers[mask]
        fa = amps[mask]
        fs = sigs[mask]

        # Optional outlier removal per cluster
        if remove_outliers and fc.size:
            df = pd.DataFrame({'center': fc, 'amplitude': fa, 'width': fs})
            for col in df:
                q1, q3 = df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                df = df[df[col].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)]
            fc, fa, fs = df['center'].values, df['amplitude'].values, df['width'].values

        count = fc.size
        title = f"Cluster {cl} (n={count})"
        if model_type:
            title += f", model={model_type}"

        # Plot mean curve ±95% CI
        ax0 = axes[i, 0]
        if count:
            curves = []
            for cen, amp, sig in zip(fc, fa, fs):
                params = model.make_params(center=cen, amplitude=amp, sigma=sig)
                curves.append(model.eval(params=params, x=bias))
            arr = np.vstack(curves)
            mu = arr.mean(axis=0)
            ci = 1.96 * arr.std(axis=0) / np.sqrt(count) if count > 1 else np.zeros_like(mu)
            ax0.plot(bias, mu, color=c)
            if count > 1:
                ax0.fill_between(bias, mu - ci, mu + ci, color=c, alpha=0.3)
        else:
            ax0.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax0.set_title(title)
        ax0.grid(True)

        # Plot histograms of center, amplitude, width
        for j, (data, lbl) in enumerate(zip([fc, fa, fs], ['Center', 'Amplitude', 'Width'])):
            ax = axes[i, j + 1]
            if data.size:
                sns.histplot(data,
                             bins=max(5, min(30, data.size)),
                             kde=(data.size > 1),
                             color=c, ax=ax)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(lbl)

    # Final layout adjustments
    fig.suptitle('Cluster Peak Statistics', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # ────────────────────────────────────────────────────────────────────
    # Add a Save-as-SVG button using ipywidgets
    try:
        from ipywidgets import Button
        from IPython.display import display

        save_button = Button(description='Save as SVG', button_style='success')
        def _on_save(btn):
            out_dir = os.path.join(os.getcwd(), 'output_figures')
            os.makedirs(out_dir, exist_ok=True)
            fname = f"cluster_{cluster_var}_{'_'.join(map(str, clusters_to_keep))}.svg"
            path = os.path.join(out_dir, fname)
            fig.savefig(path, format='svg')
            print(f"✔ Saved SVG to: {path}")
        save_button.on_click(_on_save)
        display(save_button)
    except ImportError:
        # If ipywidgets is not available, silently skip the button
        pass

# -



'''
plot_cluster_statistics(
    ds, 
    cluster_var = None,
    clusters_to_keep = None,
    model_type = None,
    remove_background = False,
    remove_neg_amp = False,
    remove_outliers = False

)'''


# ##  interactive3D after cluster selection

def interactive_3d_cluster_plot(ds: xr.Dataset) -> None:
    """
    Create an interactive 3D scatter plot of peak centers for selected clusters.

    Workflow (always re-select every time):
    1) Launch select_cluster_data_var_interactive to choose the cluster variable.
    2) In its callback, launch select_labels_in_cluster_interactive on the same ds.
    3) In its callback, perform the 3D scatter plot on the filtered dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Must contain 'X', 'Y', 'peak_center', and at least one 'cluster_*' data_var.
    """
    # Stage 1: select the cluster variable
    def _on_var(var_name: str, **kwargs):
        print(f"✅ Selected cluster variable: {var_name}")
        # Stage 2: select which labels to keep
        def _on_labels(filtered_ds: xr.Dataset, cluster_var: str, clusters_to_keep: list[int], **kwargs):
            print(f"✅ Selected labels: {clusters_to_keep}")
            # Stage 3: build and show the 3D plot
            import plotly.graph_objects as go
            import numpy as np
            from matplotlib import pyplot as plt

            # Flatten coordinates and cluster labels
            X = filtered_ds['X'].values
            Y = filtered_ds['Y'].values
            Z = filtered_ds['peak_center'].values  # shape (Y, X, peak)
            C = filtered_ds[cluster_var].values
            Y_dim, X_dim, peak_dim = Z.shape

            X_flat = np.repeat(X, Y_dim * peak_dim)
            Y_flat = np.tile(np.repeat(Y, peak_dim), X_dim)
            Z_flat = Z.reshape(-1)
            C_flat = C.reshape(-1)

            # Prepare tab10 colors and marker symbols
            tab10_colors = plt.get_cmap('tab10').colors
            marker_symbols = [
                'circle','circle-open','cross','diamond',
                'diamond-open','square','square-open','x'
            ]

            fig = go.Figure()
            # Add each cluster trace with color and symbol based on cluster number
            for lbl in clusters_to_keep:
                sym = marker_symbols[lbl % len(marker_symbols)]
                # Convert matplotlib RGB to hex for Plotly
                rgb = tuple(int(255 * c) for c in tab10_colors[lbl % len(tab10_colors)])
                color_hex = '#%02x%02x%02x' % rgb
                mask = (C_flat == lbl)
                fig.add_trace(go.Scatter3d(
                    x=X_flat[mask],
                    y=Y_flat[mask],
                    z=Z_flat[mask],
                    mode='markers',
                    marker=dict(size=4, symbol=sym, opacity=0.1, color=color_hex),
                    name=f"Cluster {lbl}"
                ))

            # Set default camera view to better center the clusters
            camera = dict(
                eye=dict(x=1.25, y=1.25, z=0),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            )
            fig.update_layout(
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Peak Center',
                    camera=camera
                ),
                title='3D Cluster Scatter',
                margin=dict(l=0, r=0, b=0, t=30)
            )
            fig.show()

        # always re-run label selection on the original ds
        select_labels_in_cluster_interactive(ds, callback=_on_labels)

    # always start with variable selection
    select_cluster_data_var_interactive(ds, callback=_on_var)


# +
# interactive_3d_cluster_plot(ds)
# -


# ## cluster filtering function 

# +
import numpy as np
import xarray as xr
import ipywidgets as widgets
from IPython.display import display, clear_output

# Global variable to hold the final filtered dataset when using interactive mode
ds_filtered_global: xr.Dataset | None = None


def filter_ds_by_cluster(
    ds: xr.Dataset,
    cluster_var: str,
    clusters_to_keep
) -> xr.Dataset:
    """
    Return a new Dataset containing only data points belonging to the specified
    cluster label(s) in all variables with a 'peak' dimension, dropping all
    other cluster-related variables.

    This function performs a deep copy of the input `ds` and does not modify it
    in-place.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset, which must include at least one data variable named
        like 'cluster_*' and one or more data variables whose dimensions include
        'peak' (e.g. variables of shape (Y, X, peak)).
    cluster_var : str
        The name of the cluster data variable to use for filtering (must be
        present in `ds.data_vars` and contain integer labels).
    clusters_to_keep : int or list[int]
        One or multiple integer labels indicating which clusters to retain. A
        single integer will be converted to a one‐element list internally.

    Returns
    -------
    xr.Dataset
        A new Dataset in which:
          - All data variables with a 'peak' dimension have been masked so that
            only points with label ∈ `clusters_to_keep` remain.
          - Any other variables whose name contains 'cluster' (except
            `cluster_var`) have been removed.

    Raises
    ------
    ValueError
        If `cluster_var` is not found among `ds.data_vars`.

    Examples
    --------
    >>> # Keep clusters 0 and 2 from variable 'cluster_PCA_knn0'
    >>> ds_filtered = filter_ds_by_cluster(ds, 'cluster_PCA_knn0', [0, 2])
    >>> # Keep only cluster 1
    >>> ds_filtered = filter_ds_by_cluster(ds, 'cluster_PCA_knn0', 1)
    """
    # Ensure the specified cluster variable exists
    if cluster_var not in ds.data_vars:
        raise ValueError(f"{cluster_var!r} not found in dataset data_vars.")

    # Normalize the clusters_to_keep argument into a list of ints
    if isinstance(clusters_to_keep, (int, np.integer)):
        labels = [int(clusters_to_keep)]
    else:
        labels = [int(l) for l in clusters_to_keep]

    # Build a boolean mask DataArray from the cluster labels
    arr = ds[cluster_var].values
    mask = xr.DataArray(
        np.isin(arr, labels),
        dims=ds[cluster_var].dims,
        coords=ds[cluster_var].coords
    )

    # Deep copy the dataset so as not to modify the input in-place
    ds_out = ds.copy(deep=True)

    # Apply the boolean mask to every variable that includes 'peak' in its dims
    for var in ds_out.data_vars:
        if 'peak' in ds_out[var].dims:
            ds_out[var] = ds_out[var].where(mask)

    # Identify and drop any other cluster_* variables besides the one we used
    other_clusters = [
        v for v in ds_out.data_vars
        if v != cluster_var and 'cluster' in v
    ]
    if other_clusters:
        ds_out = ds_out.drop_vars(other_clusters)

    return ds_out


def cluster_filter_interactive(
    ds: xr.Dataset,
    cluster_var: str | None = None,
    clusters_to_keep: list[int] | None = None
) -> xr.Dataset | None:
    """
    Filter a Dataset by cluster labels, either directly or via a two‐step UI.

    Modes
    -----
    1. Direct mode:
       If both `cluster_var` and `clusters_to_keep` are provided, immediately
       returns a filtered Dataset (using `filter_ds_by_cluster`).

    2. Interactive mode:
       If either input is None, launches two widget prompts in sequence:
         a) RadioButtons to choose `cluster_var`.
         b) SelectMultiple to choose `clusters_to_keep`.
       After the second confirmation, stores the result in the global
       `ds_filtered_global` and prints a completion message.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing at least one 'cluster_*' variable and one or
        more data_vars with a 'peak' dimension.
    cluster_var : str or None
        Name of the cluster variable to filter by. If None, user is prompted.
    clusters_to_keep : list[int] or None
        List of integer labels to retain. If None, user is prompted.

    Returns
    -------
    xr.Dataset or None
        - In direct mode: returns the filtered Dataset.
        - In interactive mode: returns None (check `ds_filtered_global` after
          completing the prompts).

    Examples
    --------
    Direct mode:
    >>> ds_filtered = cluster_filter_interactive(
    ...     ds,
    ...     cluster_var='cluster_PCA_knn0',
    ...     clusters_to_keep=[0, 3]
    ... )

    Interactive mode:
    >>> cluster_filter_interactive(ds)
    # → Step 1 widget: choose cluster variable
    # → Step 2 widget: choose cluster labels
    >>> ds_filtered = ds_filtered_global
    """
    global ds_filtered_global

    # ----- Direct mode: both args provided -----
    if cluster_var is not None and clusters_to_keep is not None:
        return filter_ds_by_cluster(ds, cluster_var, clusters_to_keep)

    # ----- Interactive mode step 1: pick cluster_var -----
    if cluster_var is None:
        def _on_var(var_name: str, **_):
            # Print confirmation and proceed to label selection
            print(f"✅ cluster_var selected: {var_name}")
            cluster_filter_interactive(ds, cluster_var=var_name)
        select_cluster_data_var_interactive(ds, callback=_on_var)
        return None

    # ----- Interactive mode step 2: pick clusters_to_keep -----
    if clusters_to_keep is None:
        def _on_labels(filtered_ds: xr.Dataset,
                       cluster_var: str,
                       clusters_to_keep: list[int],
                       **_):
            # Store the filtered result globally and notify
            global ds_filtered_global
            ds_filtered_global = filtered_ds
            print(f"✅ clusters_to_keep selected: {clusters_to_keep}")
            print("📌 Filtered dataset is now in `ds_filtered_global`.")
        select_labels_in_cluster_interactive(ds, callback=_on_labels)
        return None

    # Should never reach here
    return None



# +
# cluster_filter_interactive(ds)
#ds_filtered = ds_filtered_global

# +
#ds_filter = ds_filtered_global.copy()

# +
#ds_filter

# + [markdown] jp-MarkdownHeadingCollapsed=true
#
#
# -

# #### non-interactive case 

'''ds_filtered = cluster_filter_interactive(
    ds,
    cluster_var='cluster_PCA_knn0',
    clusters_to_keep=[0]
)'''




# ### PCA + UMAP  + HDBSCAN

# +
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import umap
import hdbscan
import itertools
import pandas as pd
from IPython.display import display
from tqdm.notebook import tqdm

# Assumes select_ML_features_interactive(ds, callback) is defined elsewhere

def cluster_umap_HDBSCAN(
    ds: xr.Dataset,
    selected_features: list[str] = None,
    param_grid: dict | None = None,
    umap_kwargs: dict | None = None,
    variance_threshold: float = 0.95
) -> xr.Dataset:
    """
    Perform UMAP + HDBSCAN clustering on ds, in-place.

    If selected_features is None, launches an interactive feature‐selection
    widget; once the user confirms, clustering is run and the original ds is
    updated with the new cluster variable.

    If selected_features is provided, clustering is run immediately (no callback).

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with 1D coords 'X','Y' and data_vars having a 'peak' dim.
    selected_features : list[str], optional
        Features to use: 'X_coordinate','Y_coordinate', or any 'peak' data_var.
        If None, will prompt interactively.
    param_grid : dict, optional
        HDBSCAN grid search options. Defaults to:
        {'min_cluster_size':[20,50], 'min_samples':[5,10],
         'cluster_selection_epsilon':[0.0,0.2]}
    umap_kwargs : dict, optional
        Arguments for UMAP. Defaults to:
        {'n_neighbors':30,'min_dist':0.3,'random_state':42}
    variance_threshold : float, default 0.95
        PCA cumulative explained variance threshold.

    Returns
    -------
    xr.Dataset
        The same ds, mutated in-place with a new variable
        'cluster_umap_HDBSCAN0','cluster_umap_HDBSCAN1',… containing
        integer cluster labels with dims ('Y','X','peak').
    """
    # ─── Interactive branch ────────────────────────────────────────────────
    if selected_features is None:
        def _continue(feats):
            # Once features selected, rerun clustering immediately
            cluster_umap_HDBSCAN(
                ds, feats, param_grid, umap_kwargs, variance_threshold
            )
            print("🔄 UMAP+HDBSCAN clustering complete; ds updated in-place.")
        select_ML_features_interactive(ds, callback=_continue)
        return ds
    # ─────────────────────────────────────────────────────────────────────────

    # Echo chosen features
    print(f"[select_ML_features_interactive] selected_features_global -> {selected_features}")

    # Set defaults
    if param_grid is None:
        param_grid = {
            'min_cluster_size': [20, 50],
            'min_samples': [5, 10],
            'cluster_selection_epsilon': [0.0, 0.2]
        }
    if umap_kwargs is None:
        umap_kwargs = {'n_neighbors': 30, 'min_dist': 0.3, 'random_state': 42}

    # 1) Build spatial grid
    x_vals = ds['X'].values
    y_vals = ds['Y'].values
    Xg, Yg = np.meshgrid(x_vals, y_vals)
    nY, nX = Xg.shape

    # 2) Construct feature matrix
    peak_vars = [v for v in ds.data_vars if 'peak' in ds[v].dims]
    nPeak = ds[peak_vars[0]].shape[-1] if peak_vars else 1
    arrs = []
    for feat in selected_features:
        if feat == 'X_coordinate':
            arr = np.repeat(Xg.flatten()[:,None], nPeak, axis=1)
        elif feat == 'Y_coordinate':
            arr = np.repeat(Yg.flatten()[:,None], nPeak, axis=1)
        else:
            data = ds[feat].values
            if feat == 'peak_amplitude' and 'background_value' in ds:
                bg = ds['background_value'].values
                data = data - np.repeat(bg[...,None], data.shape[-1], axis=2)
            arr = data.reshape(-1)
        arrs.append(arr.flatten())
    all_feats = np.stack(arrs, axis=1)

    # 3) Mask invalid points
    if 'ZB_mask' in ds:
        zb = ds['ZB_mask'].values.astype(bool)
        mask_flat = np.repeat(zb[...,None], nPeak, axis=2).reshape(-1)
    else:
        mask_flat = np.ones(all_feats.shape[0], bool)
    valid = (~np.isnan(all_feats).any(axis=1)) & mask_flat
    features = all_feats[valid]

    # 4) Standardize + PCA
    X_scaled = StandardScaler().fit_transform(features)
    pca = PCA(random_state=42)
    scores = pca.fit_transform(X_scaled)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_comp = int(np.searchsorted(cumvar, variance_threshold) + 1)
    print(f"PCA: {n_comp} components explain ≥ {variance_threshold*100:.1f}% variance")
    plt.figure(figsize=(6,4))
    plt.plot(np.arange(1,len(cumvar)+1), cumvar, 'o-')
    plt.axhline(variance_threshold, linestyle='--', color='gray')
    plt.title('PCA Cumulative Explained Variance')
    plt.xlabel('Components'); plt.ylabel('Cumulative Variance')
    plt.grid(True); plt.tight_layout(); plt.show()
    X_pca = scores[:,:n_comp]

    # 5) UMAP embedding
    X_emb = umap.UMAP(**umap_kwargs).fit_transform(X_pca)

    # 6) HDBSCAN grid search
    combos = list(itertools.product(
        param_grid['min_cluster_size'],
        param_grid['min_samples'],
        param_grid['cluster_selection_epsilon']
    ))
    results, label_store = [], []
    for mcs, ms, eps in tqdm(combos, desc="HDBSCAN grid search"):
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            cluster_selection_epsilon=eps
        )
        lbls = clusterer.fit_predict(X_emb)
        n_clusters = len(np.unique(lbls[lbls>=0]))
        noise = np.mean(lbls==-1)
        sil = silhouette_score(X_emb, lbls) if n_clusters>1 else -1
        results.append({
            'min_cluster_size': mcs,
            'min_samples': ms,
            'epsilon': eps,
            'n_clusters': n_clusters,
            'noise_ratio': noise,
            'silhouette': sil
        })
        label_store.append((lbls.copy(), (mcs, ms, eps)))
    df = pd.DataFrame(results)

    # 7) Select best configuration
    positive = df[df['silhouette']>0]
    if not positive.empty:
        best = positive.sort_values('silhouette',ascending=False).iloc[0]
    else:
        best = df.sort_values(['noise_ratio','n_clusters']).iloc[0]
    best_lbls = next(
        lbl for lbl,cfg in label_store
        if cfg==(best['min_cluster_size'],best['min_samples'],best['epsilon'])
    )

    # 8) Plot best UMAP clustering
    plt.figure(figsize=(6,5))
    sc = plt.scatter(X_emb[:,0], X_emb[:,1], c=best_lbls, cmap='tab20', s=10)
    plt.title(f"HDBSCAN best: mcs={best['min_cluster_size']}, ms={best['min_samples']}, eps={best['epsilon']}")
    plt.xlabel('UMAP1'); plt.ylabel('UMAP2')
    plt.colorbar(sc, label='Cluster'); plt.grid(True); plt.tight_layout(); plt.show()
    print("\nTop 5 configs by silhouette:")
    display(df.sort_values('silhouette',ascending=False).head())

    # 9) Save clusters back into ds in-place
    full = np.full(all_feats.shape[0], -1, dtype=int)
    full[valid] = best_lbls
    clusters3d = full.reshape((nY, nX, nPeak))
    prefix = "cluster_umap_HDBSCAN"
    idx = 0
    name = f"{prefix}{idx}"
    while name in ds.data_vars:
        idx += 1
        name = f"{prefix}{idx}"
    ds[name] = (('Y','X','peak'), clusters3d)

    ds.attrs.update({
        'feature_vars_used': selected_features,
        'param_grid': param_grid,
        'umap_params': umap_kwargs,
        'variance_threshold': variance_threshold
    })
    print(f"✅ UMAP+HDBSCAN complete. Added '{name}'.")
    return ds



# -

# ## After clustering, creating convolution map

# +
import numpy as np
import xarray as xr
from lmfit.models import LorentzianModel, GaussianModel, VoigtModel
from typing import Optional, List, Union, Callable

def create_cluster_convolution_maps(
    ds: xr.Dataset,
    model_type: str = 'lorentzian',
    model_type_var: str = 'model_type',
    cluster_var: Optional[str] = None,
    clusters_to_keep: Optional[List[Union[int, float]]] = None,
    on_complete: Optional[Callable[[xr.Dataset], None]] = None
) -> xr.Dataset:
    """
    Compute and attach per-cluster convolution maps to `ds` in-place and return it.

    Direct mode:
      - If both `cluster_var` and `clusters_to_keep` are provided, computes and attaches maps and returns ds.

    Interactive mode:
      - If cluster_var is None or clusters_to_keep is None, launches UI;
        the function returns `ds` immediately, and callbacks attach
        maps to the same `ds` object asynchronously.

    Returns
    -------
    xr.Dataset
        The same `ds` (possibly with new data appended), always returned.
    """
    def stage_cluster_var(picked_var: str):
        # After picking cluster_var, prompt for labels
        create_cluster_convolution_maps(
            ds,
            model_type=model_type,
            model_type_var=model_type_var,
            cluster_var=picked_var,
            clusters_to_keep=None,
            on_complete=on_complete
        )

    def stage_cluster_label(
        filtered_ds: xr.Dataset,
        cluster_var: str,
        clusters_to_keep: List[Union[int, float]]
    ):
        # After picking labels, compute & attach maps directly on the original ds
        _compute_and_attach(
            ds,
            cluster_var,
            clusters_to_keep,
            model_type,
            model_type_var
        )
        if on_complete:
            on_complete(ds)

    # Interactive mode: choose cluster variable
    if cluster_var is None:
        select_cluster_data_var_interactive(ds, callback=stage_cluster_var)
        return ds

    # Interactive mode: choose labels for chosen cluster
    if clusters_to_keep is None:
        select_labels_in_cluster_interactive(ds, callback=stage_cluster_label)
        return ds

    # Direct mode: both inputs provided, compute and attach synchronously
    _compute_and_attach(
        ds,
        cluster_var,
        clusters_to_keep,
        model_type,
        model_type_var
    )
    if on_complete:
        on_complete(ds)
    return ds

def _compute_and_attach(
    ds: xr.Dataset,
    cluster_var: str,
    labels: List[Union[int, float]],
    model_type: str,
    model_type_var: str
) -> None:
    """
    Internal helper: for each label, sum up peak-models into a (Y,X,bias_mV) map
    and attach it as "{cluster_var}_L<label>" on `ds`.
    """
    clusters_arr = ds[cluster_var].values
    centers       = ds['peak_center'].values
    amplitudes    = ds['peak_amplitude'].values
    sigmas        = ds['peak_sigma'].values
    background    = ds['background_value'].values
    bias_axis     = ds['bias_mV'].values
    Y, X, P       = centers.shape

    model_map = None
    if model_type == 'per-pixel':
        model_map = ds[model_type_var].values

    for label in labels:
        var_name = f"{cluster_var}_L{int(label)}"
        # Initialize convolved map with background
        conv = np.broadcast_to(
            background[..., None],
            (Y, X, bias_axis.size)
        ).copy()

        # Matching function for label
        if np.isnan(label):
            match_fn = lambda c: np.isnan(c)
        else:
            match_fn = lambda c, lab=label: (not np.isnan(c)) and int(c) == lab

        # Loop through pixels and peaks
        for i in range(Y):
            for j in range(X):
                mtype = (
                    str(model_map[i, j]).lower().strip()
                    if model_map is not None
                    else model_type
                )
                if mtype not in ('lorentzian', 'gaussian', 'voigt'):
                    conv[i, j, :] = np.nan
                    continue

                ModelClass = {
                    'lorentzian': LorentzianModel,
                    'gaussian':   GaussianModel,
                    'voigt':      VoigtModel
                }[mtype]

                for k in range(P):
                    if not match_fn(clusters_arr[i, j, k]):
                        continue
                    cen = centers[i, j, k]
                    amp = amplitudes[i, j, k]
                    sig = sigmas[i, j, k]
                    model = ModelClass(prefix='m_')
                    if ModelClass is VoigtModel:
                        params = model.make_params(
                            m_amplitude=amp,
                            m_center=cen,
                            m_sigma=sig,
                            m_gamma=sig
                        )
                    else:
                        params = model.make_params(
                            m_amplitude=amp,
                            m_center=cen,
                            m_sigma=sig
                        )
                    conv[i, j, :] += model.eval(params, x=bias_axis)

        # Attach convolved map to dataset
        ds[var_name] = (('Y', 'X', 'bias_mV'), conv)
        print(f"Attached '{var_name}' with shape {conv.shape}")



# -








ds

# # After function loading, apply PCA & KNN
#

## PCA & KNN clustering 
ds = cluster_with_pca_knn(ds, auto_select_best_k= False , k_range=(2, 19))

# +
## plot cluzstering results 

plot_cluster_statistics(
    ds, 
    cluster_var = None,
    clusters_to_keep = None,
    model_type = None,
    remove_background = False,
    remove_neg_amp = False,
    remove_outliers = False

)
# -
# filtering unusual widths --> pretreatments for the data 



# +
## check clusters in 3D 

interactive_3d_cluster_plot(ds)
# -



# +
# flitering the clustered data 
##  interactively  or manually 

cluster_filter_interactive(ds)
# -




# +
## incase of interactively select clustering 
## rename the filtered results a

ds_filter = ds_filtered_global.copy()
# -

# # 2nd PCA & KNN 
#

## PCA & KNN clustering 
ds = cluster_with_pca_knn(ds_filter, auto_select_best_k= False , k_range=(2, 19))

# +
## plot cluzstering results 

plot_cluster_statistics(
    ds, 
    cluster_var = None,
    clusters_to_keep = None,
    model_type = None,
    remove_background = False,
    remove_neg_amp = False,
    remove_outliers = False

)

# +
## check clusters in 3D 

interactive_3d_cluster_plot(ds)

# +
# flitering the clustered data 
##  interactively  or manually 

cluster_filter_interactive(ds)


# +
## incase of interactively select clustering 
## rename the filtered results a

ds_filter = ds_filtered_global.copy()
# -

# ## 3rd PCA &KNN



## PCA & KNN clustering 
ds = cluster_with_pca_knn(ds_filter, auto_select_best_k= False , k_range=(2, 19))

# +
## plot cluzstering results 

plot_cluster_statistics(
    ds, 
    cluster_var = None,
    clusters_to_keep = None,
    model_type = None,
    remove_background = False,
    remove_neg_amp = False,
    remove_outliers = False

)
# -







# #  PCA , UMAP,  HDBSCAN  results

# ##   PCA + UMAP + HDBSCAN

ds_filter

# +
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import umap
import hdbscan
import itertools
import pandas as pd
import json
from IPython.display import display, FileLink
from tqdm.notebook import tqdm
from ipywidgets import Button, HBox

# Assumes select_ML_features_interactive(ds, callback) is defined elsewhere

def cluster_umap_HDBSCAN(
    ds: xr.Dataset,
    selected_features: list[str] = None,
    param_grid: dict | None = None,
    umap_kwargs: dict | None = None,
    variance_threshold: float = 0.95
) -> xr.Dataset:
    """
    Perform UMAP + HDBSCAN clustering on ds, in-place.
    Adds discrete-colormap scatter with cluster-center annotations
    and provides an SVG save button with download link.
    """
    # Ensure output directory exists
    output_dir = 'output_figures'
    os.makedirs(output_dir, exist_ok=True)

    # Interactive branch
    if selected_features is None:
        def _continue(feats):
            cluster_umap_HDBSCAN(ds, feats, param_grid, umap_kwargs, variance_threshold)
            print("🔄 UMAP+HDBSCAN clustering complete; ds updated in-place.")
        select_ML_features_interactive(ds, callback=_continue)
        return ds

    # Defaults
    if param_grid is None:
        param_grid = {'min_cluster_size':[20,50],'min_samples':[5,10],'cluster_selection_epsilon':[0.0,0.2]}
    if umap_kwargs is None:
        umap_kwargs = {'n_neighbors':30,'min_dist':0.3,'random_state':42}

    # Build spatial grid
    x_vals, y_vals = ds['X'].values, ds['Y'].values
    Xg, Yg = np.meshgrid(x_vals, y_vals)
    nY, nX = Xg.shape

    # Construct feature matrix
    peak_vars = [v for v in ds.data_vars if 'peak' in ds[v].dims]
    nPeak = ds[peak_vars[0]].shape[-1] if peak_vars else 1
    arrs = []
    for feat in selected_features:
        if feat == 'X_coordinate':
            arr = np.repeat(Xg.flatten()[:, None], nPeak, axis=1)
        elif feat == 'Y_coordinate':
            arr = np.repeat(Yg.flatten()[:, None], nPeak, axis=1)
        else:
            data = ds[feat].values
            if feat == 'peak_amplitude' and 'background_value' in ds:
                bg = ds['background_value'].values
                data = data - np.repeat(bg[..., None], data.shape[-1], axis=2)
            arr = data.reshape(-1)
        arrs.append(arr.flatten())
    all_feats = np.stack(arrs, axis=1)

    # Mask invalid points
    if 'ZB_mask' in ds:
        zb = ds['ZB_mask'].values.astype(bool)
        mask_flat = np.repeat(zb[..., None], nPeak, axis=2).reshape(-1)
    else:
        mask_flat = np.ones(all_feats.shape[0], bool)
    valid = (~np.isnan(all_feats).any(axis=1)) & mask_flat
    features = all_feats[valid]

    # Standardize + PCA
    X_scaled = StandardScaler().fit_transform(features)
    pca = PCA(random_state=42)
    scores = pca.fit_transform(X_scaled)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_comp = int(np.searchsorted(cumvar, variance_threshold) + 1)
    X_pca = scores[:, :n_comp]

    # UMAP embedding
    X_emb = umap.UMAP(**umap_kwargs).fit_transform(X_pca)

    # HDBSCAN grid search
    combos = list(itertools.product(
        param_grid['min_cluster_size'],
        param_grid['min_samples'],
        param_grid['cluster_selection_epsilon']
    ))
    results, label_store = [], []
    for mcs, ms, eps in tqdm(combos, desc="HDBSCAN grid search"):
        lbls = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            cluster_selection_epsilon=eps
        ).fit_predict(X_emb)
        n_clusters = len(np.unique(lbls[lbls >= 0]))
        noise = np.mean(lbls == -1)
        sil = silhouette_score(X_emb, lbls) if n_clusters > 1 else -1
        results.append({'min_cluster_size':mcs, 'min_samples':ms, 'epsilon':eps,
                        'n_clusters':n_clusters, 'noise_ratio':noise, 'silhouette':sil})
        label_store.append((lbls.copy(), (mcs, ms, eps)))
    df = pd.DataFrame(results)

    # Select best configuration
    positive = df[df['silhouette'] > 0]
    best = positive.sort_values('silhouette', ascending=False).iloc[0] if not positive.empty else df.sort_values(['noise_ratio', 'n_clusters']).iloc[0]
    best_lbls = next(lbl for lbl, cfg in label_store if cfg == (best['min_cluster_size'], best['min_samples'], best['epsilon']))

    # Plot with discrete tab10-based colormap matching label values
    from matplotlib.colors import ListedColormap
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = best_lbls
    unique_labels = np.unique(labels)
    max_label = int(unique_labels.max())
    # Use base tab10 colors, cycling if labels > 9
    base_colors = plt.get_cmap('tab10').colors
    colors_list = [base_colors[i % len(base_colors)] for i in range(max_label + 1)]
    cmap = ListedColormap(colors_list)
    # Scatter, mapping label value directly to color index
    sc = ax.scatter(
        X_emb[:, 0], X_emb[:, 1],
        c=labels, cmap=cmap, vmin=0, vmax=max_label, s=10
    )
    ax.set_title(f"HDBSCAN best: mcs={best['min_cluster_size']}, ms={best['min_samples']}, eps={best['epsilon']}")
    ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2')
    ax.grid(True)
    # Colorbar with original labels
    cbar = fig.colorbar(sc, ax=ax, ticks=unique_labels)
    cbar.set_ticklabels(unique_labels)
    # Annotate cluster centers
    for lbl in unique_labels:
        mask = labels == lbl
        cx, cy = X_emb[mask, 0].mean(), X_emb[mask, 1].mean()
        ax.text(cx, cy, str(int(lbl)), ha='center', va='center', fontsize=12, weight='bold')
    plt.tight_layout()

        # Save SVG button and link
    save_btn = Button(description='Save UMAP as SVG')
    def _save_svg(b):
        path = os.path.join(output_dir, 'umap_clusters.svg')
        fig.savefig(path, format='svg')
        display(FileLink(path))
    save_btn.on_click(_save_svg)
    display(HBox([save_btn]))

    plt.show()
    print("\nTop 5 configs by silhouette:")
    display(df.sort_values('silhouette', ascending=False).head())

    # Save clusters back into dataset
    full = np.full(all_feats.shape[0], -1, dtype=int)
    full[valid] = labels
    clusters3d = full.reshape((nY, nX, nPeak))
    prefix = "cluster_umap_HDBSCAN"
    idx = 0; name = f"{prefix}{idx}"
    while name in ds.data_vars:
        idx += 1; name = f"{prefix}{idx}"
    ds[name] = (('Y', 'X', 'peak'), clusters3d)

    # Serializable attrs
    ds.attrs['feature_vars_used'] = json.dumps(selected_features)
    ds.attrs['param_grid'] = json.dumps(param_grid)
    ds.attrs['umap_params'] = json.dumps(umap_kwargs)
    ds.attrs['variance_threshold'] = float(variance_threshold)

    print(f"✅ UMAP+HDBSCAN complete. Added '{name}'.")
    return ds



# +
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import umap
import hdbscan
import itertools
import pandas as pd
import json
from IPython.display import display, FileLink
from tqdm.notebook import tqdm
from ipywidgets import Button, HBox
from matplotlib import patheffects
from matplotlib.colors import ListedColormap

# Assumes select_ML_features_interactive(ds, callback) is defined elsewhere

def cluster_umap_HDBSCAN(
    ds: xr.Dataset,
    selected_features: list[str] = None,
    param_grid: dict | None = None,
    umap_kwargs: dict | None = None,
    variance_threshold: float = 0.95,
    plot_sample_frac: float = 0.25,
    rasterize_pts: bool = True
) -> xr.Dataset:
    """
    Perform UMAP + HDBSCAN clustering on a 3D Dataset, plot the result with
    optional sampling and rasterization, and add cluster labels back into ds.

    This function does the following:
      1. If `selected_features` is None, launches an interactive selector.
      2. Builds a per-pixel feature matrix from the chosen features.
      3. Masks invalid pixels (NaN or optional ZB_mask).
      4. Standardizes and reduces dimensionality via PCA.
      5. Embeds reduced features into 2D via UMAP.
      6. Performs an HDBSCAN grid search over `param_grid` and selects
         the best configuration by silhouette score (or minimal noise).
      7. Plots the UMAP embedding:
         - Down-samples points by `plot_sample_frac` to reduce file size.
         - Uses a discrete `tab10` colormap cycled for labels >10.
         - Optionally rasterizes the scatter points (`rasterize_pts=True`)
           so that only axes/text remain vector and points become a bitmap.
         - Annotates each cluster’s centroid with a black label outlined
           in white for maximum contrast.
      8. Provides a “Save UMAP as SVG” button that writes to
         `output_figures/umap_clusters.svg` and displays a download link.
      9. Stores the cluster labels back into `ds` as a new data variable.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with 1D coords 'X','Y' and data_vars having a 'peak' dim.
    selected_features : list[str], optional
        List of feature names to use. If None, an interactive selector appears.
    param_grid : dict, optional
        HDBSCAN grid search options; default
        {'min_cluster_size':[20,50],'min_samples':[5,10],
         'cluster_selection_epsilon':[0.0,0.2]}.
    umap_kwargs : dict, optional
        Passed to UMAP(); default {'n_neighbors':30,'min_dist':0.3,'random_state':42}.
    variance_threshold : float, default 0.95
        PCA cumulative explained variance threshold.
    plot_sample_frac : float, default 0.25
        Fraction of points to plot in the UMAP scatter to reduce SVG size.
    rasterize_pts : bool, default True
        If True, scatter points are rasterized (bitmap) inside the SVG.

    Returns
    -------
    xr.Dataset
        The same dataset, mutated in-place with a new data variable
        'cluster_umap_HDBSCAN0' (or next available index) containing cluster labels.
    """
    # Ensure output directory exists for saving figures
    output_dir = 'output_figures'
    os.makedirs(output_dir, exist_ok=True)

    # ─── Interactive feature-selection ─────────────────────────────────────
    if selected_features is None:
        def _continue(feats):
            cluster_umap_HDBSCAN(
                ds, feats, param_grid, umap_kwargs,
                variance_threshold, plot_sample_frac, rasterize_pts
            )
            print("🔄 UMAP+HDBSCAN clustering complete; ds updated in-place.")
        select_ML_features_interactive(ds, callback=_continue)
        return ds

    # Set default grids if not provided
    if param_grid is None:
        param_grid = {
            'min_cluster_size': [20, 50],
            'min_samples': [5, 10],
            'cluster_selection_epsilon': [0.0, 0.2]
        }
    if umap_kwargs is None:
        umap_kwargs = {
            'n_neighbors': 30,
            'min_dist': 0.3,
            'random_state': 42
        }

    # 1) Build spatial coordinate grid
    x_vals = ds['X'].values
    y_vals = ds['Y'].values
    Xg, Yg = np.meshgrid(x_vals, y_vals)
    nY, nX = Xg.shape

    # 2) Stack chosen features into a 2D array [pixels × features]
    peak_vars = [v for v in ds.data_vars if 'peak' in ds[v].dims]
    nPeak = ds[peak_vars[0]].shape[-1] if peak_vars else 1
    arrs = []
    for feat in selected_features:
        if feat == 'X_coordinate':
            arr = np.repeat(Xg.flatten()[:, None], nPeak, axis=1)
        elif feat == 'Y_coordinate':
            arr = np.repeat(Yg.flatten()[:, None], nPeak, axis=1)
        else:
            data = ds[feat].values
            if feat == 'peak_amplitude' and 'background_value' in ds:
                bg = ds['background_value'].values
                data = data - np.repeat(bg[..., None], data.shape[-1], axis=2)
            arr = data.reshape(-1)
        arrs.append(arr.flatten())
    all_feats = np.stack(arrs, axis=1)

    # 3) Mask out invalid points (NaNs or ZB_mask)
    if 'ZB_mask' in ds:
        zb = ds['ZB_mask'].values.astype(bool)
        mask_flat = np.repeat(zb[..., None], nPeak, axis=2).reshape(-1)
    else:
        mask_flat = np.ones(all_feats.shape[0], bool)
    valid = (~np.isnan(all_feats).any(axis=1)) & mask_flat
    features = all_feats[valid]

    # 4) Standardize & apply PCA
    X_scaled = StandardScaler().fit_transform(features)
    pca = PCA(random_state=42)
    scores = pca.fit_transform(X_scaled)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_comp = int(np.searchsorted(cumvar, variance_threshold) + 1)
    X_pca = scores[:, :n_comp]

    # 5) UMAP embedding
    X_emb = umap.UMAP(**umap_kwargs).fit_transform(X_pca)

    # 6) HDBSCAN grid search
    combos = list(itertools.product(
        param_grid['min_cluster_size'],
        param_grid['min_samples'],
        param_grid['cluster_selection_epsilon']
    ))
    results, label_store = [], []
    for mcs, ms, eps in tqdm(combos, desc="HDBSCAN grid search"):
        lbls = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            cluster_selection_epsilon=eps
        ).fit_predict(X_emb)
        n_clusters = len(np.unique(lbls[lbls >= 0]))
        noise = np.mean(lbls == -1)
        sil = silhouette_score(X_emb, lbls) if n_clusters > 1 else -1
        results.append({
            'min_cluster_size': mcs,
            'min_samples': ms,
            'epsilon': eps,
            'n_clusters': n_clusters,
            'noise_ratio': noise,
            'silhouette': sil
        })
        label_store.append((lbls.copy(), (mcs, ms, eps)))
    df = pd.DataFrame(results)

    # 7) Choose best configuration
    positive = df[df['silhouette'] > 0]
    if not positive.empty:
        best = positive.sort_values('silhouette', ascending=False).iloc[0]
    else:
        best = df.sort_values(['noise_ratio', 'n_clusters']).iloc[0]
    best_lbls = next(
        lbl for lbl, cfg in label_store
        if cfg == (best['min_cluster_size'], best['min_samples'], best['epsilon'])
    )

    # 8) Plot UMAP embedding with sampling & rasterization
    fig, ax = plt.subplots(figsize=(6, 5))

    # Down-sample points to reduce file size
    N = X_emb.shape[0]
    if 0 < plot_sample_frac < 1.0:
        sample_size = int(N * plot_sample_frac)
        idx = np.random.choice(N, size=sample_size, replace=False)
        X_plot = X_emb[idx]
        labels_plot = best_lbls[idx]
    else:
        X_plot = X_emb
        labels_plot = best_lbls

    unique_labels = np.unique(labels_plot)
    max_label = int(unique_labels.max())

    # Build a ListedColormap from tab10, cycling if >10 labels
    base_colors = plt.get_cmap('tab10').colors
    colors_list = [base_colors[i % len(base_colors)] for i in range(max_label + 1)]
    cmap = ListedColormap(colors_list)

    # Scatter: optionally rasterize points
    sc = ax.scatter(
        X_plot[:, 0], X_plot[:, 1],
        c=labels_plot,
        cmap=cmap,
        vmin=0,
        vmax=max_label,
        s=10,
        rasterized=rasterize_pts
    )

    # Set title without line continuation backslash
    ax.set_title(
        f"HDBSCAN best: mcs={best['min_cluster_size']}, ms={best['min_samples']}, eps={best['epsilon']}"
    )
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.grid(True)

    # Colorbar keyed to actual cluster labels
    cbar = fig.colorbar(sc, ax=ax, ticks=unique_labels)
    cbar.set_ticklabels(unique_labels)

    # Annotate cluster centers with black text + white outline
    for lbl in unique_labels:
        mask = labels_plot == lbl
        cx, cy = X_plot[mask, 0].mean(), X_plot[mask, 1].mean()
        txt = ax.text(
            cx, cy, str(int(lbl)),
            ha='center', va='center',
            fontsize=12, weight='bold', color='black'
        )
        txt.set_path_effects([
            patheffects.Stroke(linewidth=3, foreground='white'),
            patheffects.Normal()
        ])

    plt.tight_layout()
    plt.show()

    # 9) Save-as-SVG button
    save_btn = Button(description='Save UMAP as SVG')
    def _save_svg(btn):
        out_path = os.path.join(output_dir, 'umap_clusters.svg')
        fig.savefig(out_path, format='svg')
        display(FileLink(out_path))
    save_btn.on_click(_save_svg)
    display(save_btn)

    # 10) Display top configurations
    print("\nTop 5 configs by silhouette:")
    display(df.sort_values('silhouette', ascending=False).head())

    # 11) Write cluster labels back into the dataset
    full = np.full(all_feats.shape[0], -1, dtype=int)
    full[valid] = best_lbls
    clusters3d = full.reshape((nY, nX, nPeak))
    prefix = "cluster_umap_HDBSCAN"
    idx = 0
    name = f"{prefix}{idx}"
    while name in ds.data_vars:
        idx += 1
        name = f"{prefix}{idx}"
    ds[name] = (('Y', 'X', 'peak'), clusters3d)

    # 12) Serialize metadata attributes
    ds.attrs.update({
        'feature_vars_used': json.dumps(selected_features),
        'param_grid':         json.dumps(param_grid),
        'umap_params':        json.dumps(umap_kwargs),
        'variance_threshold': float(variance_threshold)
    })

    print(f"✅ UMAP+HDBSCAN complete. Added '{name}'.")
    return ds

# -



ds = cluster_umap_HDBSCAN(        
    ds_filter,
    selected_features=None,
    param_grid={"min_cluster_size":[ 100,500,800,1000, 2000, 2500, 3000,], 
                "min_samples":[30,50,80, 100,160,200], 
                "cluster_selection_epsilon":[0,0.1]},
    umap_kwargs={"n_neighbors":40,
                 "min_dist":0.2, 
                 "random_state":0},
    variance_threshold=0.90,
    plot_sample_frac = 1
)

# +
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import umap
import hdbscan
import itertools
import pandas as pd
import json
from IPython.display import display, FileLink
from tqdm.notebook import tqdm
from ipywidgets import Button, HBox
from matplotlib import patheffects
from matplotlib.colors import ListedColormap

# Assumes select_ML_features_interactive(ds, callback) is defined elsewhere

def cluster_umap_HDBSCAN(
    ds: xr.Dataset,
    selected_features: list[str] = None,
    param_grid: dict | None = None,
    umap_kwargs: dict | None = None,
    variance_threshold: float = 0.95,
    plot_sample_frac: float = 0.25,
    rasterize_pts: bool = True
) -> xr.Dataset:
    """
    Perform UMAP + HDBSCAN clustering on a 3D Dataset, plot the result with
    optional sampling and rasterization, and add cluster labels back into ds.

    This function does the following:
      1. If `selected_features` is None, launches an interactive selector.
      2. Builds a per-pixel feature matrix from the chosen features.
      3. Masks invalid pixels (NaN or optional ZB_mask).
      4. Standardizes and reduces dimensionality via PCA.
      5. Embeds reduced features into 2D via UMAP.
      6. Performs an HDBSCAN grid search over `param_grid` and selects
         the best configuration by silhouette score (or minimal noise).
      7. Plots the UMAP embedding:
         - Down-samples points by `plot_sample_frac` to reduce file size.
         - Uses a discrete `tab10` colormap cycled for labels >10.
         - Optionally rasterizes the scatter points (`rasterize_pts=True`)
           so that only axes/text remain vector and points become a bitmap.
         - Annotates each cluster’s centroid with a black label outlined
           in white for maximum contrast.
      8. Provides a “Save UMAP as SVG” button that writes to
         `output_figures/umap_clusters.svg` and displays a download link.
      9. Stores the cluster labels back into `ds` as a new data variable.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with 1D coords 'X','Y' and data_vars having a 'peak' dim.
    selected_features : list[str], optional
        List of feature names to use. If None, an interactive selector appears.
    param_grid : dict, optional
        HDBSCAN grid search options; default
        {'min_cluster_size':[20,50],'min_samples':[5,10],
         'cluster_selection_epsilon':[0.0,0.2]}.
    umap_kwargs : dict, optional
        Passed to UMAP(); default {'n_neighbors':30,'min_dist':0.3,'random_state':42}.
    variance_threshold : float, default 0.95
        PCA cumulative explained variance threshold.
    plot_sample_frac : float, default 0.25
        Fraction of points to plot in the UMAP scatter to reduce SVG size.
    rasterize_pts : bool, default True
        If True, scatter points are rasterized (bitmap) inside the SVG.

    Returns
    -------
    xr.Dataset
        The same dataset, mutated in-place with a new data variable
        'cluster_umap_HDBSCAN0' (or next available index) containing cluster labels.
    """
    # Ensure output directory exists for saving figures
    output_dir = 'output_figures'
    os.makedirs(output_dir, exist_ok=True)

    # ─── Interactive feature-selection ─────────────────────────────────────
    if selected_features is None:
        def _continue(feats):
            cluster_umap_HDBSCAN(
                ds, feats, param_grid, umap_kwargs,
                variance_threshold, plot_sample_frac, rasterize_pts
            )
            print("🔄 UMAP+HDBSCAN clustering complete; ds updated in-place.")
        select_ML_features_interactive(ds, callback=_continue)
        return ds

    # Set default grids if not provided
    if param_grid is None:
        param_grid = {
            'min_cluster_size': [20, 50],
            'min_samples': [5, 10],
            'cluster_selection_epsilon': [0.0, 0.2]
        }
    if umap_kwargs is None:
        umap_kwargs = {
            'n_neighbors': 30,
            'min_dist': 0.3,
            'random_state': 42
        }

    # 1) Build spatial coordinate grid
    x_vals = ds['X'].values
    y_vals = ds['Y'].values
    Xg, Yg = np.meshgrid(x_vals, y_vals)
    nY, nX = Xg.shape

    # 2) Stack chosen features into a 2D array [pixels × features]
    peak_vars = [v for v in ds.data_vars if 'peak' in ds[v].dims]
    nPeak = ds[peak_vars[0]].shape[-1] if peak_vars else 1
    arrs = []
    for feat in selected_features:
        if feat == 'X_coordinate':
            arr = np.repeat(Xg.flatten()[:, None], nPeak, axis=1)
        elif feat == 'Y_coordinate':
            arr = np.repeat(Yg.flatten()[:, None], nPeak, axis=1)
        else:
            data = ds[feat].values
            if feat == 'peak_amplitude' and 'background_value' in ds:
                bg = ds['background_value'].values
                data = data - np.repeat(bg[..., None], data.shape[-1], axis=2)
            arr = data.reshape(-1)
        arrs.append(arr.flatten())
    all_feats = np.stack(arrs, axis=1)

    # 3) Mask out invalid points (NaNs or ZB_mask)
    if 'ZB_mask' in ds:
        zb = ds['ZB_mask'].values.astype(bool)
        mask_flat = np.repeat(zb[..., None], nPeak, axis=2).reshape(-1)
    else:
        mask_flat = np.ones(all_feats.shape[0], bool)
    valid = (~np.isnan(all_feats).any(axis=1)) & mask_flat
    features = all_feats[valid]

    # 4) Standardize & apply PCA
    X_scaled = StandardScaler().fit_transform(features)
    pca = PCA(random_state=42)
    scores = pca.fit_transform(X_scaled)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_comp = int(np.searchsorted(cumvar, variance_threshold) + 1)
    X_pca = scores[:, :n_comp]

    # 5) UMAP embedding
    X_emb = umap.UMAP(**umap_kwargs).fit_transform(X_pca)

    # 6) HDBSCAN grid search
    combos = list(itertools.product(
        param_grid['min_cluster_size'],
        param_grid['min_samples'],
        param_grid['cluster_selection_epsilon']
    ))
    results, label_store = [], []
    for mcs, ms, eps in tqdm(combos, desc="HDBSCAN grid search"):
        lbls = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            cluster_selection_epsilon=eps
        ).fit_predict(X_emb)
        n_clusters = len(np.unique(lbls[lbls >= 0]))
        noise = np.mean(lbls == -1)
        sil = silhouette_score(X_emb, lbls) if n_clusters > 1 else -1
        results.append({
            'min_cluster_size': mcs,
            'min_samples': ms,
            'epsilon': eps,
            'n_clusters': n_clusters,
            'noise_ratio': noise,
            'silhouette': sil
        })
        label_store.append((lbls.copy(), (mcs, ms, eps)))
    df = pd.DataFrame(results)

    # 7) Choose best configuration
    positive = df[df['silhouette'] > 0]
    if not positive.empty:
        best = positive.sort_values('silhouette', ascending=False).iloc[0]
    else:
        best = df.sort_values(['noise_ratio', 'n_clusters']).iloc[0]
    best_lbls = next(
        lbl for lbl, cfg in label_store
        if cfg == (best['min_cluster_size'], best['min_samples'], best['epsilon'])
    )

    # 8) Plot UMAP embedding with sampling & rasterization
    fig, ax = plt.subplots(figsize=(6, 5))

    # Down-sample points to reduce file size
    N = X_emb.shape[0]
    if 0 < plot_sample_frac < 1.0:
        sample_size = int(N * plot_sample_frac)
        idx = np.random.choice(N, size=sample_size, replace=False)
        X_plot = X_emb[idx]
        labels_plot = best_lbls[idx]
    else:
        X_plot = X_emb
        labels_plot = best_lbls

    unique_labels = np.unique(labels_plot)
    max_label = int(unique_labels.max())

    # Build a ListedColormap from tab10, cycling if >10 labels
    base_colors = plt.get_cmap('tab10').colors
    colors_list = [base_colors[i % len(base_colors)] for i in range(max_label + 1)]
    cmap = ListedColormap(colors_list)

    # Scatter: optionally rasterize points and set alpha for density visualization
    sc = ax.scatter(
        X_plot[:, 0], X_plot[:, 1],  # UMAP coordinates
        c=labels_plot,                # cluster labels for coloring
        cmap=cmap,
        vmin=0,
        vmax=max_label,
        s=10,
        alpha=0.3,                    # 투명도 설정: 점 밀도 강조
        rasterized=rasterize_pts
    )

    # Set title without line continuation backslash
    ax.set_title(
        f"HDBSCAN best: mcs={best['min_cluster_size']}, ms={best['min_samples']}, eps={best['epsilon']}"
    )
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.grid(True)

    # Colorbar keyed to actual cluster labels
    cbar = fig.colorbar(sc, ax=ax, ticks=unique_labels)
    cbar.set_ticklabels(unique_labels)

    # Annotate cluster centers with black text + white outline
    for lbl in unique_labels:
        mask = labels_plot == lbl
        cx, cy = X_plot[mask, 0].mean(), X_plot[mask, 1].mean()
        txt = ax.text(
            cx, cy, str(int(lbl)),
            ha='center', va='center',
            fontsize=12, weight='bold', color='black'
        )
        txt.set_path_effects([
            patheffects.Stroke(linewidth=3, foreground='white'),
            patheffects.Normal()
        ])

    plt.tight_layout()
    plt.show()

    # 9) Save-as-SVG button
    save_btn = Button(description='Save UMAP as SVG')
    def _save_svg(btn):
        out_path = os.path.join(output_dir, 'umap_clusters.svg')
        fig.savefig(out_path, format='svg')
        display(FileLink(out_path))
    save_btn.on_click(_save_svg)
    display(save_btn)

    # 10) Display top configurations
    print("\nTop 5 configs by silhouette:")
    display(df.sort_values('silhouette', ascending=False).head())

    # 11) Write cluster labels back into the dataset
    full = np.full(all_feats.shape[0], -1, dtype=int)
    full[valid] = best_lbls
    clusters3d = full.reshape((nY, nX, nPeak))
    prefix = "cluster_umap_HDBSCAN"
    idx = 0
    name = f"{prefix}{idx}"
    while name in ds.data_vars:
        idx += 1
        name = f"{prefix}{idx}"
    ds[name] = (('Y', 'X', 'peak'), clusters3d)

    # 12) Serialize metadata attributes
    ds.attrs.update({
        'feature_vars_used': json.dumps(selected_features),
        'param_grid':         json.dumps(param_grid),
        'umap_params':        json.dumps(umap_kwargs),
        'variance_threshold': float(variance_threshold)
    })

    print(f"✅ UMAP+HDBSCAN complete. Added '{name}'.")
    return ds

# -



ds = cluster_umap_HDBSCAN(        
    ds_filter,
    selected_features=None,
    param_grid={"min_cluster_size":[ 300, 500, 1000, 1200], 
                "min_samples":[30, 50, 80, 100, ], 
                "cluster_selection_epsilon":[0,0.1]},
    umap_kwargs={"n_neighbors":50,
                 "min_dist":0.2, 
                 "random_state":0},
    variance_threshold=0.90,
    plot_sample_frac = 1
)





ds

plot_cluster_statistics(ds) 

interactive_3d_cluster_plot(ds)

ds.to_netcdf('grid_2T_003_fit_HDBSCAN_cluster3_20250629.nc')

# +
# open cluster data set & re draw 
# +
#ds = xr.open_dataset('grid_2T_003_fit_cluster_20250523.nc')
# -


# ## other  (advanced) clustering test 


ds_filter = ds_filtered_global.copy()

number_of_filtered_peaks  = ( ~np.isnan( ds_filter.peak_center.values)).sum()
number_of_total_peaks = ( ~np.isnan( ds.peak_center.values)).sum()
filtered_ratio  = number_of_filtered_peaks/number_of_total_peaks
print (filtered_ratio)

# +
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import umap
import hdbscan
import pandas as pd
from IPython.display import display
from tqdm.notebook import tqdm
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hdbscan import validity as hdb_validity

def cluster_umap_HDBSCAN_Bayesian_opt(
    ds: xr.Dataset,
    selected_features: list[str] = None,
    umap_kwargs: dict | None = None,
    variance_threshold: float = 0.95,
    noise_range: tuple[float,float] = (0.05, 0.20),
    max_evals: int = 50,
    sample_fraction: float = 0.1,
    hdbscan_space: dict = {
        'min_cluster_size': hp.quniform('min_cluster_size', 10, 500, 10),
        'min_samples':      hp.quniform('min_samples', 2, 100, 1),
        'cluster_selection_epsilon': hp.uniform('cluster_selection_epsilon', 0.0, 0.5)
    }
) -> xr.Dataset:
    """
    Perform UMAP + HDBSCAN clustering with Bayesian optimization of HDBSCAN
    hyperparameters, then append the best clustering to the Dataset.

    If `selected_features` is None, launches an interactive feature‐selection
    widget; once the user confirms, re-invokes itself with the chosen features.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with 1D coords 'X','Y' and data_vars having a 'peak' dim.
    selected_features : list[str], optional
        Features to use. If None, prompts interactively.
    umap_kwargs : dict, optional
        UMAP parameters. Defaults to {'n_neighbors':30,'min_dist':0.3,'random_state':42}.
    variance_threshold : float, default=0.95
        PCA variance cutoff.
    noise_range : tuple, default=(0.05,0.20)
        Desired noise ratio interval.
    max_evals : int, default=50
        Hyperopt max evaluations.
    sample_fraction : float, default=0.1
        Fraction of points to sample for optimization.
    hdbscan_space : dict
        Hyperopt search space for HDBSCAN.

    Returns
    -------
    xr.Dataset
        Copy of ds with new var 'cluster_umap_HDBSCAN_optX' of dims ('Y','X','peak').
    """
    if not selected_features:
        def _cont(feats):
            global ds_opt
            ds_opt = cluster_umap_HDBSCAN_Bayesian_opt(
                ds, feats, umap_kwargs, variance_threshold,
                noise_range, max_evals, sample_fraction, hdbscan_space
            )
            print("🔄 Resumed Bayesian-optimized UMAP+HDBSCAN clustering.")
        select_ML_features_interactive(ds, callback=_cont)
        return ds

    # 1) Build grid & feature matrix
    x_vals, y_vals = ds['X'].values, ds['Y'].values
    Xg, Yg = np.meshgrid(x_vals, y_vals)
    nY, nX = Xg.shape
    peak_vars = [v for v in ds.data_vars if 'peak' in ds[v].dims]
    nP = ds[peak_vars[0]].shape[-1] if peak_vars else 1

    feat_list = []
    for feat in selected_features:
        if feat == 'X_coordinate':
            arr = np.repeat(Xg.flatten()[:,None], nP, axis=1)
        elif feat == 'Y_coordinate':
            arr = np.repeat(Yg.flatten()[:,None], nP, axis=1)
        else:
            data = ds[feat].values
            if feat == 'peak_amplitude' and 'background_value' in ds:
                bg = ds['background_value'].values
                data = data - np.repeat(bg[...,None], data.shape[-1], axis=2)
            arr = data.reshape(-1)
        feat_list.append(arr.flatten())
    all_feats = np.stack(feat_list, axis=1)

    # 2) Mask invalid
    if 'ZB_mask' in ds:
        zb = ds['ZB_mask'].values.astype(bool)
        mask_flat = np.repeat(zb[...,None], nP, axis=2).reshape(-1)
    else:
        mask_flat = np.ones(all_feats.shape[0], bool)
    valid = (~np.isnan(all_feats).any(axis=1)) & mask_flat
    X = all_feats[valid]

    # 3) Standardize + PCA
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(random_state=42)
    scores = pca.fit_transform(Xs)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_comp = int(np.searchsorted(cumvar, variance_threshold) + 1)
    print(f"PCA: using {n_comp} components for ≥{variance_threshold*100:.1f}% variance")
    plt.figure(figsize=(6,4))
    plt.plot(np.arange(1,len(cumvar)+1), cumvar, 'o-')
    plt.axhline(variance_threshold, linestyle='--', color='gray')
    plt.xlabel('Components'); plt.ylabel('Cumulative Variance')
    plt.title('PCA Variance'); plt.grid(True); plt.show()
    X_red = scores[:,:n_comp]

    # 4) UMAP embed
    if umap_kwargs is None:
        umap_kwargs = {'n_neighbors':30,'min_dist':0.3,'random_state':42}
    X_emb = umap.UMAP(**umap_kwargs).fit_transform(X_red)

    # 5) Subsample
    n_pts = X_emb.shape[0]
    size = max(1,int(n_pts*sample_fraction))
    idx = np.random.default_rng(42).choice(n_pts, size, replace=False)
    X_emb_sample = X_emb[idx]

    # 6) Objective
    def objective(params):
        mcs = max(2, int(params['min_cluster_size']))
        ms  = max(1, int(params['min_samples']))
        eps = float(params['cluster_selection_epsilon'])
        lbls = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            cluster_selection_epsilon=eps
        ).fit_predict(X_emb_sample)

        noise = float(np.mean(lbls==-1))
        nclus = len(np.unique(lbls[lbls>=0]))
        if nclus<=1:
            return {'loss':1.0,'status':STATUS_OK}
        sil = silhouette_score(X_emb_sample,lbls)
        try: dbcv = hdb_validity.validity_index(X_emb_sample,lbls)
        except: dbcv = sil
        low,high = noise_range
        penalty = 0 if (low<=noise<=high) else abs(noise-np.clip(noise,low,high))
        score = sil*dbcv - penalty
        return {'loss':-score,'status':STATUS_OK,'noise':noise,'silhouette':sil,'dbcv':dbcv}

    trials = Trials()
    best = fmin(
        fn=objective,
        space=hdbscan_space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(42),
        show_progressbar=False
    )

    # ← HERE IS THE FIX: use Hyperopt's best dict directly
    best_params = {
        'min_cluster_size': max(2, int(best['min_cluster_size'])),
        'min_samples':      max(1, int(best['min_samples'])),
        'cluster_selection_epsilon': float(best['cluster_selection_epsilon'])
    }
    print("🔍 Best HDBSCAN params:", best_params)

    # Top-5 candidates
    recs = []
    for t in trials.trials:
        r, p = t['result'], t['misc']['vals']
        recs.append({
            'min_cluster_size': p['min_cluster_size'][0],
            'min_samples':      p['min_samples'][0],
            'epsilon':          p['cluster_selection_epsilon'][0],
            'silhouette':       r.get('silhouette',np.nan),
            'noise_ratio':      r.get('noise',np.nan),
            'dbcv':             r.get('dbcv',np.nan)
        })
    df_top = pd.DataFrame(recs).sort_values('silhouette',ascending=False).head()
    print("\nTop 5 hyperparameter candidates by silhouette:")
    display(df_top)

    # 7) Final clustering
    final_lbls = hdbscan.HDBSCAN(**best_params).fit_predict(X_emb)
    full = np.full(all_feats.shape[0], -1, int)
    full[valid] = final_lbls
    clusters3d = full.reshape((nY,nX,nP))

    # 8) Plot final UMAP
    plt.figure(figsize=(6,5))
    plt.scatter(X_emb[:,0],X_emb[:,1],c=final_lbls,cmap='tab20',s=8)
    plt.title("Final UMAP + HDBSCAN")
    plt.xlabel("UMAP1"); plt.ylabel("UMAP2"); plt.colorbar(label="Cluster ID")
    plt.show()

    # 9) Save result
    prefix="cluster_umap_HDBSCAN_opt"
    i=0; name=f"{prefix}{i}"
    ds_out = ds.copy(deep=True)
    while name in ds_out.data_vars:
        i+=1; name=f"{prefix}{i}"
    ds_out[name] = (('Y','X','peak'), clusters3d)
    ds_out.attrs.update({
        'feature_vars_used':    selected_features,
        'umap_kwargs':          umap_kwargs,
        'variance_threshold':   variance_threshold,
        'best_hdbscan_params':  best_params,
        'noise_range':          noise_range,
        'sample_fraction':      sample_fraction
    })
    print(f"✅ Done; added '{name}'.")
    return ds_out



# -

# 예제 1: 직접 모드 — 피처 리스트를 명시하고 Bayesian 최적화 수행
ds_opt = cluster_umap_HDBSCAN_Bayesian_opt(
    ds,
    #selected_features=['peak_center', 'peak_amplitude', 'peak_sigma'],
    selected_features=None,
    umap_kwargs={
        'n_neighbors': 50,
        'min_dist':    0.2,
        'random_state': 0
    },
    variance_threshold=0.90,
    noise_range=(0.05, 0.20),    # 허용할 노이즈 비율 5~20%
    max_evals=30,                # 최대 30회 탐색
    sample_fraction=0.10,         # 전체 포인트의 10%만 샘플링
    hdbscan_space = {
        'min_cluster_size': hp.quniform('min_cluster_size', 10, 1000, 100),
        'min_samples':      hp.quniform('min_samples', 1, 1000, 100),
        'cluster_selection_epsilon': hp.uniform('cluster_selection_epsilon', 0.0, 0.5)
    })
print(ds_opt)
# → ds_opt.data_vars 에 'cluster_umap_HDBSCAN_opt0' 변수가 추가됩니다.

ds_opt

ds_opt

plot_cluster_statistics(ds_opt)

interactive_3d_cluster_plot(ds_opt)

cluster_filter_interactive(ds_opt)


ds_opt



# 예제 2: 대화식 모드 — 처음에는 피처를 None 으로 두고 위젯으로 선택
cluster_umap_HDBSCAN_Bayesian_opt(
    ds,
    selected_features=None,     # None일 경우 피처 선택 위젯이 표시됩니다
    umap_kwargs={'n_neighbors':50,'min_dist':0.2,'random_state':0},
    variance_threshold=0.90,
    noise_range=(0.05,0.20),# 허용할 노이즈 비율 5~20%
    max_evals=20,# 최대 30회 탐색
    sample_fraction=0.10 # 전체 포인트의 10%만 샘플링
)
# — 위젯에서 피처를 선택 & Confirm 하신 후,
#   최적 파라미터로 클러스터링된 결과가 전역 변수 ds_opt 에 저장됩니다.



# ## after UMAP cluster statistics 

ds

# ### PCA UMAP KNN 

ds_filtered

# +
import warnings

def cluster_umap_knn(
    ds: xr.Dataset,
    selected_features: list[str],
    k_range: tuple[int, int] = (2, 14),
    auto_select_best_k: bool = True,
    umap_kwargs: dict | None = None,
    variance_threshold: float = 0.95
) -> xr.Dataset:
    """
    Perform UMAP + KMeans clustering and append the results to the Dataset.
    (…docstring 생략…)
    """
    # 경고 무시
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # 입력 유효성 검사
    if ds is None:
        raise ValueError("Input dataset 'ds' is None.")
    if not isinstance(ds, xr.Dataset):
        raise TypeError(f"Expected xr.Dataset, got {type(ds)}.")
    if not selected_features:
        raise ValueError("No features selected for clustering.")

    # UMAP 파라미터 기본값
    if umap_kwargs is None:
        umap_kwargs = {'n_neighbors': 30, 'min_dist': 0.3, 'random_state': 42}

    # 좌표 추출
    try:
        x_vals = ds['X'].values
        y_vals = ds['Y'].values
    except KeyError as e:
        raise KeyError(f"Coordinates missing: {e}")
    Xgrid, Ygrid = np.meshgrid(x_vals, y_vals)
    nY, nX = Xgrid.shape

    # 피크 차원 크기 결정
    peak_vars = [v for v in ds.data_vars if 'peak' in ds[v].dims]
    nPeak = ds[peak_vars[0]].shape[-1] if peak_vars else 1

    # 특징 행렬 구성
    feature_list = []
    for feat in selected_features:
        if feat == 'X_coordinate':
            arr = np.repeat(Xgrid.flatten()[:, None], nPeak, axis=1)
            feature_list.append(arr.flatten())
        elif feat == 'Y_coordinate':
            arr = np.repeat(Ygrid.flatten()[:, None], nPeak, axis=1)
            feature_list.append(arr.flatten())
        else:
            if feat not in ds.data_vars:
                raise KeyError(f"Feature '{feat}' not found.")
            data = ds[feat].values
            if feat == 'peak_amplitude' and 'background_value' in ds:
                bg = ds['background_value'].values
                data = data - np.repeat(bg[..., None], data.shape[-1], axis=2)
            feature_list.append(data.reshape(-1))
    all_feats = np.stack(feature_list, axis=1)

    # 마스크 적용
    if 'ZB_mask' in ds:
        mask = ds['ZB_mask'].values.astype(bool)
        mask_flat = np.repeat(mask[..., None], nPeak, axis=2).reshape(-1)
    else:
        mask_flat = np.ones(all_feats.shape[0], dtype=bool)
    valid = (~np.isnan(all_feats).any(axis=1)) & mask_flat
    features = all_feats[valid]

    # 표준화
    X_scaled = StandardScaler().fit_transform(features)

    # PCA
    pca = PCA(n_components=variance_threshold, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA reduced to {X_pca.shape[1]} components to explain ≥ {variance_threshold*100:.1f}% variance")

    # UMAP 임베딩
    X_umap = umap.UMAP(**umap_kwargs).fit_transform(X_pca)

    # KMeans 탐색
    Ks = list(range(k_range[0], k_range[1] + 1))
    inertias, silhouettes, label_store = [], [], []
    for k in Ks:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        lbl = km.fit_predict(X_umap)
        inertias.append(km.inertia_)
        try:
            sil = silhouette_score(X_umap, lbl)
        except:
            sil = -1
        silhouettes.append(sil)
        label_store.append(lbl)

    # Elbow & Silhouette 플롯
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(Ks, inertias, 'o-')
    axes[0].set_title("Elbow Method (Inertia)")
    axes[0].set_xlabel("K"); axes[0].set_ylabel("Inertia"); axes[0].grid(True)
    axes[1].plot(Ks, silhouettes, 'o-')
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("K"); axes[1].set_ylabel("Score"); axes[1].grid(True)
    plt.tight_layout(); plt.show()

    # 로컬 맥스 찾기
    def find_local_maxima(arr):
        return [i for i in range(1, len(arr)-1) if arr[i] > arr[i-1] and arr[i] > arr[i+1]]
    local_max = find_local_maxima(silhouettes)
    top_idxs = sorted(local_max, key=lambda i: silhouettes[i], reverse=True)[:3]
    top_ks = [Ks[i] for i in top_idxs]
    print("Top 3 silhouette-local-max Ks:", top_ks)

    # k 선택
    if auto_select_best_k:
        best_k = top_ks[0]
        print(f"Auto-selected best k = {best_k}")
    else:
        print(f"Tested Ks: {Ks}")
        user_in = input(f"Enter desired k from above, or press Enter to auto-select ({top_ks[0]}): ")
        if user_in.strip() == "":
            best_k = top_ks[0]
            print(f"No input → auto-selected best k = {best_k}")
        else:
            try:
                best_k = int(user_in)
            except ValueError:
                raise ValueError(f"Invalid input '{user_in}'. Integer required.")
            if best_k not in Ks:
                raise ValueError(f"Chosen k={best_k} not in tested range {Ks}.")

    # 최종 레이블 및 시각화
    best_labels = label_store[Ks.index(best_k)]

    # --- 수정된 부분: axs가 단일 Axes일 때를 처리 ---
    fig, axs = plt.subplots(1, len(top_ks), figsize=(5*len(top_ks), 5))
    if len(top_ks) == 1:
        axs = [axs]  # 단일 Axes 객체를 리스트로 감싸서 일관된 인덱싱 지원
    for ax, k in zip(axs, top_ks):
        lbls = label_store[Ks.index(k)]
        ax.scatter(X_umap[:,0], X_umap[:,1], c=lbls, cmap="tab20", s=20)
        ax.set_title(f"K={k}, Silhouette={silhouettes[Ks.index(k)]:.3f}")
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2"); ax.grid(True)
    plt.tight_layout(); plt.show()
    # --- 수정된 부분 끝 ---

    # 결과 저장
    full_lbl = np.full(all_feats.shape[0], -1, dtype=int)
    full_lbl[valid] = best_labels
    clusters3d = full_lbl.reshape((nY, nX, nPeak))

    ds_out = ds.copy(deep=True)
    base = 'cluster_umap_knn'
    idx = 0
    var_name = f"{base}{idx}"
    while var_name in ds_out.data_vars:
        idx += 1
        var_name = f"{base}{idx}"
    ds_out[var_name] = (('Y','X','peak'), clusters3d)

    # 메타데이터 기록
    ds_out.attrs.update({
        'feature_vars_used': selected_features,
        'k_range': k_range,
        'auto_select_best_k': auto_select_best_k,
        'umap_params': umap_kwargs,
        'variance_threshold': variance_threshold
    })

    print(f"✅ Clustering complete. Added variable '{var_name}'.")
    return ds_out



# -

#cluster_select_features_interactive(ds_filtered)
cluster_select_features_interactive(ds)

selected_features_global

ds_clustered = cluster_umap_knn(
    ds=ds,
    selected_features=selected_features_global,
    k_range=(2, 14),
    auto_select_best_k=False,   # False일 때 콘솔에서 k 직접 입력
    umap_kwargs={"n_neighbors": 30, "min_dist": 0.3, "random_state": 42},
    variance_threshold=0.95
)



# Step 1: Select which cluster variable to use
cluster_select_var_interactive(ds_filtered)


cluster_select_labels_interactive(
    ds_filtered,
    callback=plot_cluster_statistics_grid,  # 그냥 함수명만 전달
    **dict(
        remove_background=True,
        remove_neg_amp=True,
        remove_outliers=True
    )
)

cluster_select_labels_interactive(
    ds_filtered,
    # Use a wrapper lambda that accepts and ignores extra keyword arguments
    # (cluster_var, clusters_to_keep, etc.) passed by the selector,
    # then calls our 3D plot function with just the dataset.
    callback=lambda ds, **kwargs: interactive_3d_cluster_plot(ds)
)



# Step 1: Select the cluster variable
cluster_filter_interactive(ds_filtered)

# 👉 After selecting the variable and labels via the UI,
# run this cell separately to retrieve the filtered dataset:
ds_filtered = ds_filtered_global.copy(deep=True)
ds_filtered

# ### save other ideas for later , better clustering 

# +
#PCA UMAP Random Forest?

# +
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap
import warnings

# Step 0: Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Step 1: Preprocessing (assume ds is loaded already)
# → Your dataset must provide: X_raw = [peak_center, amp, sigma]
# → metadata = [(y, x, peak)] for mapping back

# 1. Standardize features
X_scaled = StandardScaler().fit_transform(X_raw)

# 2. PCA (95% explained variance)
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA reduced to {X_pca.shape[1]} components")

# 3. UMAP embedding
X_umap = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42).fit_transform(X_pca)

# 4. Test multiple K for KMeans
K_range = range(2, 20)
inertias = []
silhouettes = []
all_labels = []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = km.fit_predict(X_umap)
    all_labels.append(labels)
    inertias.append(km.inertia_)
    try:
        sil = silhouette_score(X_umap, labels)
    except:
        sil = -1
    silhouettes.append(sil)

# 5. Plot Elbow and Silhouette curves
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].plot(K_range, inertias, 'o-')
axs[0].set_title("Elbow Method (Inertia)")
axs[0].set_xlabel("K")
axs[0].set_ylabel("Inertia")
axs[0].grid(True)

axs[1].plot(K_range, silhouettes, 'o-')
axs[1].set_title("Silhouette Score")
axs[1].set_xlabel("K")
axs[1].set_ylabel("Score")
axs[1].grid(True)
plt.tight_layout()
plt.show()

# 6. Find local maxima from silhouette scores
def find_local_maxima(arr):
    return [i for i in range(1, len(arr)-1) if arr[i] > arr[i-1] and arr[i] > arr[i+1]]

local_max = find_local_maxima(silhouettes)
top_indices = sorted(local_max, key=lambda i: silhouettes[i], reverse=True)[:3]
top_ks = [K_range[i] for i in top_indices]

print(f"Top 3 local max silhouette K values: {top_ks}")

# 7. Visualize Top 3 K clustering results
fig, axs = plt.subplots(1, len(top_ks), figsize=(5 * len(top_ks), 5))
for i, k in enumerate(top_ks):
    labels = all_labels[K_range.index(k)]
    axs[i].scatter(X_umap[:, 0], X_umap[:, 1], c=labels, cmap="tab20", s=20)
    axs[i].set_title(f"K = {k}, Silhouette = {silhouettes[K_range.index(k)]:.3f}")
    axs[i].set_xlabel("UMAP-1")
    axs[i].set_ylabel("UMAP-2")
    axs[i].grid(True)
plt.tight_layout()
plt.show()

# 8. Store best (top-1) result to dataset
best_k = top_ks[0]
best_labels = all_labels[K_range.index(best_k)]

# ✅ FIXED: Get shape from ds["peak_center"]
nY, nX, nP = ds["peak_center"].shape
cluster_array = np.full((nY, nX, nP), -1, dtype=np.int32)
for i, (y, x, p) in enumerate(metadata):
    cluster_array[y, x, p] = best_labels[i]

ds["clusters_umap_kmeans_best"] = (("Y", "X", "peak"), cluster_array)

# Optional: save
# ds.to_netcdf("clustered_output_umap_kmeans_best.nc")

# 9. Print summary
print("Top 3 K values:", top_ks)
print("Corresponding silhouette scores:", [silhouettes[K_range.index(k)] for k in top_ks])



# +
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap
import warnings

# Step 0: Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Step 1: Preprocessing (ds already loaded)

# Required variables from ds
x_vals = ds["X"].values
y_vals = ds["Y"].values
peak_center = ds["peak_center"].values
amp = ds["peak_amplitude"].values
sigma = ds["peak_sigma"].values
background = ds["background_value"].values
zb_mask = ds["ZB_mask"].values

# Background subtraction
amp = amp - np.repeat(background[..., np.newaxis], amp.shape[2], axis=2)

# Build valid mask: only use (Y,X,peak) where all 3 features are valid
valid_mask = (~np.isnan(peak_center)) & (~np.isnan(amp)) & (~np.isnan(sigma)) & (~np.isnan(zb_mask[..., np.newaxis]))
valid_indices = np.argwhere(valid_mask)

# Step 2: Sampling (e.g., 1%)
np.random.seed(42)
n_sample = max(1, int(len(valid_indices) * 0.9))
sample_indices = valid_indices[np.random.choice(len(valid_indices), size=n_sample, replace=False)]

# Step 3: Build feature vectors — exclude X, Y!
features = []
metadata = []

for y, x, p in sample_indices:
    features.append([
        peak_center[y, x, p],
        amp[y, x, p],
        sigma[y, x, p]
    ])
    metadata.append((int(y), int(x), int(p)))

X_raw = np.stack(features)

# Step 4: Standardize features
X_scaled = StandardScaler().fit_transform(X_raw)

# Step 5: PCA
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA reduced to {X_pca.shape[1]} components")

# Step 6: UMAP
X_umap = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42).fit_transform(X_pca)

# Step 7: Try multiple K in KMeans
K_range = range(2, 20)
inertias = []
silhouettes = []
all_labels = []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X_umap)
    all_labels.append(labels)
    inertias.append(km.inertia_)
    try:
        sil = silhouette_score(X_umap, labels)
    except:
        sil = -1
    silhouettes.append(sil)

# Step 8: Plot Elbow and Silhouette
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].plot(K_range, inertias, 'o-')
axs[0].set_title("Elbow Method (Inertia)")
axs[0].set_xlabel("K")
axs[0].set_ylabel("Inertia")
axs[0].grid(True)

axs[1].plot(K_range, silhouettes, 'o-')
axs[1].set_title("Silhouette Score")
axs[1].set_xlabel("K")
axs[1].set_ylabel("Score")
axs[1].grid(True)

plt.tight_layout()
plt.show()

# Step 9: Top 3 local max based on silhouette
def find_local_maxima(arr):
    return [i for i in range(1, len(arr) - 1) if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]]

local_max = find_local_maxima(silhouettes)
top_indices = sorted(local_max, key=lambda i: silhouettes[i], reverse=True)[:3]
top_ks = [K_range[i] for i in top_indices]

print("Top 3 silhouette-local-max Ks:", top_ks)

# Step 10: Visualize Top 3 clustering results
fig, axs = plt.subplots(1, len(top_ks), figsize=(5 * len(top_ks), 5))
for i, k in enumerate(top_ks):
    labels = all_labels[K_range.index(k)]
    axs[i].scatter(X_umap[:, 0], X_umap[:, 1], c=labels, cmap="tab20", s=20)
    axs[i].set_title(f"K={k}, Silhouette={silhouettes[K_range.index(k)]:.3f}")
    axs[i].set_xlabel("UMAP-1")
    axs[i].set_ylabel("UMAP-2")
    axs[i].grid(True)
plt.tight_layout()
plt.show()

# Step 11: Store best result (K=top_ks[0]) in xarray.Dataset
best_k = top_ks[0]
best_labels = all_labels[K_range.index(best_k)]

nY, nX, nP = peak_center.shape
cluster_array = np.full((nY, nX, nP), -1, dtype=np.int32)
for i, (y, x, p) in enumerate(metadata):
    cluster_array[y, x, p] = best_labels[i]

ds["clusters_umap_kmeans_best"] = (("Y", "X", "peak"), cluster_array)

# Optional save
# ds.to_netcdf("clustered_output_umap_kmeans_best.nc")

# Step 12: Summary
print("Best K =", best_k)
print("Unique Clusters:", np.unique(best_labels))
# -


# ## zero bias weight 

# +
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Step 1: 데이터 전처리 (ds는 이미 로드되었다고 가정)
x_vals = ds["X"].values
y_vals = ds["Y"].values
peak_center = ds["peak_center"].values
amp = ds["peak_amplitude"].values
sigma = ds["peak_sigma"].values
background = ds["background_value"].values
zb_mask = ds["ZB_mask"].values

# Step 2: Background subtraction
amp = amp - np.repeat(background[..., np.newaxis], amp.shape[2], axis=2)

# Step 3: 유효한 피크 마스크
valid_mask = (~np.isnan(peak_center)) & (~np.isnan(amp)) & (~np.isnan(sigma)) & (~np.isnan(zb_mask[..., np.newaxis]))
valid_indices = np.argwhere(valid_mask)

# Step 4: 샘플링 (예: 90%)
np.random.seed(42)
n_sample = max(1, int(len(valid_indices) * 0.9))
sample_indices = valid_indices[np.random.choice(len(valid_indices), size=n_sample, replace=False)]

# Step 5: Feature 생성 (X, Y 제외) + 메타데이터 저장
features = []
metadata = []

for y, x, p in sample_indices:
    features.append([
        peak_center[y, x, p],
        amp[y, x, p],
        sigma[y, x, p]
    ])
    metadata.append((int(y), int(x), int(p)))

X_raw = np.stack(features)

# Step 6: Feature weighting
# 가중치: peak_center 5배 강조
feature_weights = np.array([5.0, 1.0, 1.0])
X_weighted = X_raw * feature_weights

# Step 7: 정규화 → PCA
X_scaled = StandardScaler().fit_transform(X_weighted)
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA reduced to {X_pca.shape[1]} components")

# Step 8: UMAP
X_umap = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42).fit_transform(X_pca)

# Step 9: KMeans for multiple K
K_range = range(2, 20)
inertias, silhouettes, all_labels = [], [], []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X_umap)
    all_labels.append(labels)
    inertias.append(km.inertia_)
    try:
        sil = silhouette_score(X_umap, labels)
    except:
        sil = -1
    silhouettes.append(sil)

# Step 10: Plot Elbow & Silhouette
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].plot(K_range, inertias, 'o-')
axs[0].set_title("Elbow Method (Inertia)")
axs[0].set_xlabel("K")
axs[0].set_ylabel("Inertia")
axs[0].grid(True)

axs[1].plot(K_range, silhouettes, 'o-')
axs[1].set_title("Silhouette Score")
axs[1].set_xlabel("K")
axs[1].set_ylabel("Score")
axs[1].grid(True)
plt.tight_layout()
plt.show()

# Step 11: Top 3 local maxima of silhouette score
def find_local_maxima(arr):
    return [i for i in range(1, len(arr)-1) if arr[i] > arr[i-1] and arr[i] > arr[i+1]]

local_max = find_local_maxima(silhouettes)
top_indices = sorted(local_max, key=lambda i: silhouettes[i], reverse=True)[:3]
top_ks = [K_range[i] for i in top_indices]

print("Top 3 silhouette-local-max Ks:", top_ks)

# Step 12: Visualize Top 3 clustering results
fig, axs = plt.subplots(1, len(top_ks), figsize=(5 * len(top_ks), 5))
for i, k in enumerate(top_ks):
    labels = all_labels[K_range.index(k)]
    axs[i].scatter(X_umap[:, 0], X_umap[:, 1], c=labels, cmap="tab20", s=20)
    axs[i].set_title(f"K={k}, Silhouette={silhouettes[K_range.index(k)]:.3f}")
    axs[i].set_xlabel("UMAP-1")
    axs[i].set_ylabel("UMAP-2")
    axs[i].grid(True)
plt.tight_layout()
plt.show()

# Step 13: Store best (top-1) result
best_k = top_ks[0]
best_labels = all_labels[K_range.index(best_k)]

nY, nX, nP = peak_center.shape
cluster_array = np.full((nY, nX, nP), -1, dtype=np.int32)
for i, (y, x, p) in enumerate(metadata):
    cluster_array[y, x, p] = best_labels[i]

ds["clusters_umap_kmeans_weighted"] = (("Y", "X", "peak"), cluster_array)

# Optional save
# ds.to_netcdf("clustered_output_umap_kmeans_weighted.nc")

# Step 14: Summary
print("Best K =", best_k)
print("Unique Clusters:", np.unique(best_labels))

# -



# ### zero-bias proximity feature 추가 방식

# +
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Step 1: 데이터 로딩 및 전처리 (ds 가 이미 로드된 상태라고 가정)
x_vals = ds["X"].values
y_vals = ds["Y"].values
peak_center = ds["peak_center"].values
amp = ds["peak_amplitude"].values
sigma = ds["peak_sigma"].values
background = ds["background_value"].values
zb_mask = ds["ZB_mask"].values

# Step 2: 배경 제거
amp = amp - np.repeat(background[..., np.newaxis], amp.shape[2], axis=2)

# Step 3: 유효한 피크 선택
valid_mask = (~np.isnan(peak_center)) & (~np.isnan(amp)) & (~np.isnan(sigma)) & (~np.isnan(zb_mask[..., np.newaxis]))
valid_indices = np.argwhere(valid_mask)

# Step 4: 샘플링 (예: 1%)
np.random.seed(42)
n_sample = max(1, int(len(valid_indices) * 0.91))
sample_indices = valid_indices[np.random.choice(len(valid_indices), size=n_sample, replace=False)]

# Step 5: feature 벡터 생성 + proximity 추가
features = []
metadata = []

for y, x, p in sample_indices:
    pc = peak_center[y, x, p]
    a = amp[y, x, p]
    s = sigma[y, x, p]
    prox = 1 / (1 + abs(pc))  # zero-bias proximity
    features.append([pc, a, s, prox])
    metadata.append((int(y), int(x), int(p)))

X_raw = np.stack(features)

# Step 6: 정규화
X_scaled = StandardScaler().fit_transform(X_raw)

# Step 7: PCA (설명력 95% 유지)
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA reduced to {X_pca.shape[1]} components")

# Step 8: UMAP
X_umap = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42).fit_transform(X_pca)

# Step 9: 다양한 K에 대해 KMeans 실행
K_range = range(2, 20)
inertias, silhouettes, all_labels = [], [], []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X_umap)
    all_labels.append(labels)
    inertias.append(km.inertia_)
    try:
        sil = silhouette_score(X_umap, labels)
    except:
        sil = -1
    silhouettes.append(sil)

# Step 10: 로컬 최대값 기준 top 3 선택
def find_local_maxima(arr):
    return [i for i in range(1, len(arr)-1) if arr[i] > arr[i-1] and arr[i] > arr[i+1]]

local_max = find_local_maxima(silhouettes)
top_indices = sorted(local_max, key=lambda i: silhouettes[i], reverse=True)[:3]
top_ks = [K_range[i] for i in top_indices]

print("Top 3 silhouette-local-max Ks:", top_ks)

# Step 11: 시각화
fig, axs = plt.subplots(1, len(top_ks), figsize=(5 * len(top_ks), 5))
for i, k in enumerate(top_ks):
    labels = all_labels[K_range.index(k)]
    axs[i].scatter(X_umap[:, 0], X_umap[:, 1], c=labels, cmap="tab20", s=20)
    axs[i].set_title(f"K={k}, Silhouette={silhouettes[K_range.index(k)]:.3f}")
    axs[i].set_xlabel("UMAP-1")
    axs[i].set_ylabel("UMAP-2")
    axs[i].grid(True)
plt.tight_layout()
plt.show()

# Step 12: best K 결과 저장
best_k = top_ks[0]
best_labels = all_labels[K_range.index(best_k)]

nY, nX, nP = peak_center.shape
cluster_array = np.full((nY, nX, nP), -1, dtype=np.int32)
for i, (y, x, p) in enumerate(metadata):
    cluster_array[y, x, p] = best_labels[i]

ds["clusters_umap_kmeans_zero_proximity"] = (("Y", "X", "peak"), cluster_array)

# Step 13: 결과 요약
print("Best K =", best_k)
print("Unique Clusters:", np.unique(best_labels))
print("Feature used: [center, amplitude, sigma, 1 / (1 + |center|)]")

# -

# #### Zero-Bias Proximity Feature에 다양한 가중치 적용 + UMAP + KMeans

# +
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap
import warnings

# 경고 무시 설정
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Step 1: Load Dataset
ds
# Step 2: Extract Variables
x_vals = ds["X"].values
y_vals = ds["Y"].values
peak_center = ds["peak_center"].values
amp = ds["peak_amplitude"].values
sigma = ds["peak_sigma"].values
background = ds["background_value"].values
zb_mask = ds["ZB_mask"].values

# Step 3: Background Subtraction
amp = amp - np.repeat(background[..., np.newaxis], amp.shape[2], axis=2)

# Step 4: Build valid mask
valid_mask = (~np.isnan(peak_center)) & (~np.isnan(amp)) & (~np.isnan(sigma)) & (~np.isnan(zb_mask[..., np.newaxis]))
valid_indices = np.argwhere(valid_mask)

# Step 5: Sample 1%
np.random.seed(42)
n_sample = max(1, int(len(valid_indices) * 0.01))
sample_indices = valid_indices[np.random.choice(len(valid_indices), size=n_sample, replace=False)]

# Step 6: Define weights to test for proximity feature
proximity_weights = [1.0, 2.0, 5.0, 10.0, 20.0]

results = []

for w in proximity_weights:
    # Step 7: Construct features with proximity
    features = []
    for y, x, p in sample_indices:
        pc = peak_center[y, x, p]
        a = amp[y, x, p]
        s = sigma[y, x, p]
        prox = 1 / (1 + abs(pc))
        features.append([pc, a, s, prox])
    X_raw = np.stack(features)

    # Step 8: Apply weight to proximity
    weights = np.array([1.0, 1.0, 1.0, w])
    X_weighted = X_raw * weights

    # Step 9: Normalize and PCA
    X_scaled = StandardScaler().fit_transform(X_weighted)
    X_pca = PCA(n_components=0.95, random_state=42).fit_transform(X_scaled)

    # Step 10: UMAP
    X_umap = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42).fit_transform(X_pca)

    # Step 11: KMeans for multiple K
    K_range = range(2, 20)
    silhouettes = []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(X_umap)
        try:
            sil = silhouette_score(X_umap, labels)
        except:
            sil = -1
        silhouettes.append(sil)

    # Step 12: Find best K from silhouette local maxima
    def find_local_maxima(arr):
        return [i for i in range(1, len(arr)-1) if arr[i] > arr[i-1] and arr[i] > arr[i+1]]

    local_max = find_local_maxima(silhouettes)
    if local_max:
        best_index = local_max[np.argmax([silhouettes[i] for i in local_max])]
    else:
        best_index = np.argmax(silhouettes)

    best_k = K_range[best_index]
    best_sil = silhouettes[best_index]

    results.append({
        "proximity_weight": w,
        "best_k": best_k,
        "silhouette_score": best_sil
    })

# Step 13: 결과 요약 출력
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("proximity_weight").reset_index(drop=True)

# Step 14: 시각화
plt.figure(figsize=(8, 5))
plt.plot(results_df["proximity_weight"], results_df["silhouette_score"], 'o-', label='Silhouette Score')
plt.xlabel("Proximity Feature Weight")
plt.ylabel("Best Silhouette Score")
plt.title("Effect of Zero-Bias Proximity Weight on Clustering")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Step 15: 표 출력
print("Weight vs Clustering Quality Summary:")
print(results_df)

# +
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap
import warnings

# 경고 무시
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Step 1: 데이터 로딩 및 전처리 (ds 는 이미 로드된 상태라고 가정)
x_vals = ds["X"].values
y_vals = ds["Y"].values
peak_center = ds["peak_center"].values
amp = ds["peak_amplitude"].values
sigma = ds["peak_sigma"].values
background = ds["background_value"].values
zb_mask = ds["ZB_mask"].values

# Step 2: 배경 제거
amp = amp - np.repeat(background[..., np.newaxis], amp.shape[2], axis=2)

# Step 3: 유효한 피크 선택
valid_mask = (~np.isnan(peak_center)) & (~np.isnan(amp)) & (~np.isnan(sigma)) & (~np.isnan(zb_mask[..., np.newaxis]))
valid_indices = np.argwhere(valid_mask)

# Step 4: 샘플링 (예: 1%)
np.random.seed(42)
n_sample = max(1, int(len(valid_indices) * 0.91))
sample_indices = valid_indices[np.random.choice(len(valid_indices), size=n_sample, replace=False)]

# Step 5: feature 벡터 생성 + proximity 추가
features = []
metadata = []

for y, x, p in sample_indices:
    pc = peak_center[y, x, p]
    a = amp[y, x, p]
    s = sigma[y, x, p]
    prox = 1 / (1 + abs(pc))  # zero-bias proximity
    features.append([pc, a, s, prox])
    metadata.append((int(y), int(x), int(p)))

X_raw = np.stack(features)

# Step 6: 정규화
X_scaled = StandardScaler().fit_transform(X_raw)

# Step 7: PCA (설명력 95% 유지)
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA reduced to {X_pca.shape[1]} components")

# Step 8: UMAP
X_umap = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42).fit_transform(X_pca)

# Step 9: 다양한 K에 대해 KMeans 실행
K_range = range(2, 20)
inertias, silhouettes, all_labels = [], [], []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X_umap)
    all_labels.append(labels)
    inertias.append(km.inertia_)
    try:
        sil = silhouette_score(X_umap, labels)
    except:
        sil = -1
    silhouettes.append(sil)

# Step 10-1: Elbow plot & Silhouette score 시각화
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].plot(K_range, inertias, 'o-')
axs[0].set_title("Elbow Method (Inertia)")
axs[0].set_xlabel("K")
axs[0].set_ylabel("Inertia")
axs[0].grid(True)

axs[1].plot(K_range, silhouettes, 'o-')
axs[1].set_title("Silhouette Score")
axs[1].set_xlabel("K")
axs[1].set_ylabel("Score")
axs[1].grid(True)

plt.suptitle("KMeans Clustering Evaluation", fontsize=14)
plt.tight_layout()
plt.show()

# Step 10: 로컬 최대값 기준 top 3 선택
def find_local_maxima(arr):
    return [i for i in range(1, len(arr)-1) if arr[i] > arr[i-1] and arr[i] > arr[i+1]]

local_max = find_local_maxima(silhouettes)
top_indices = sorted(local_max, key=lambda i: silhouettes[i], reverse=True)[:3]
top_ks = [K_range[i] for i in top_indices]

print("Top 3 silhouette-local-max Ks:", top_ks)

# Step 11: UMAP 결과 시각화
fig, axs = plt.subplots(1, len(top_ks), figsize=(5 * len(top_ks), 5))
for i, k in enumerate(top_ks):
    labels = all_labels[K_range.index(k)]
    axs[i].scatter(X_umap[:, 0], X_umap[:, 1], c=labels, cmap="tab20", s=20)
    axs[i].set_title(f"K={k}, Silhouette={silhouettes[K_range.index(k)]:.3f}")
    axs[i].set_xlabel("UMAP-1")
    axs[i].set_ylabel("UMAP-2")
    axs[i].grid(True)
plt.tight_layout()
plt.show()

# Step 12: best K 결과 저장
best_k = top_ks[0]
best_labels = all_labels[K_range.index(best_k)]5




nY, nX, nP = peak_center.shape
cluster_array = np.full((nY, nX, nP), -1, dtype=np.int32)
for i, (y, x, p) in enumerate(metadata):
    cluster_array[y, x, p] = best_labels[i]

ds["clusters_umap_kmeans_zero_proximity"] = (("Y", "X", "peak"), cluster_array)

# Step 13: 결과 요약
print("Best K =", best_k)
print("Unique Clusters:", np.unique(best_labels))
print("Feature used: [center, amplitude, sigma, 1 / (1 + |center|)]")

# -

ds



# ## UMAP(Uniform Manifold Approximation and Projection) 활용
# Manifold Learning 기반:
# UMAP는 데이터가 고차원 공간에서 어떤 저차원 매니폴드(다양체)에 분포한다고 가정합니다. 즉, 데이터의 중요한 구조는 저차원 공간에 보존될 수 있다는 전제에서 시작합니다.
#
# 국소적(neighborhood) 구조 보존:
# UMAP는 먼저 각 데이터 포인트 주변의 이웃 관계(국소적 구조)를 파악합니다. 이를 위해 사용자로부터 n_neighbors와 같은 하이퍼파라미터를 받아, 각 포인트에 대해 일정 개수의 가까운 이웃을 찾습니다.
#
# 그래프 및 위상수학적 접근:
# 각 포인트와 그 이웃 사이의 관계를 기반으로, 데이터의 국소적 구조를 표현하는 그래프(또는 퍼지 집합)를 구성합니다. 이 그래프는 데이터의 위상수학적 구조를 반영하도록 설계됩니다.
#
# 저차원 임베딩 최적화:
# 고차원에서 구성된 국소적 관계를 저차원에서도 최대한 보존하도록 임베딩을 최적화합니다. 이 과정은 고차원 그래프와 저차원 그래프 간의 차이를 최소화하는 비용 함수를 정의하고, 이를 최적화하는 방식으로 이루어집니다. 결과적으로 데이터의 글로벌 및 국소 구조가 균형 있게 반영된 저차원 공간을 얻게 됩니다.
#
# 계산 효율성과 확장성:
# UMAP는 계산 효율성이 뛰어나고, 대규모 데이터셋에 대해서도 상대적으로 빠른 임베딩이 가능합니다. 이는 특히 t-SNE에 비해 큰 데이터셋에서 유리한 점입니다.
#
# 하이퍼파라미터 예시:
#
# n_neighbors: 각 데이터 포인트에서 고려할 이웃의 수. 국소적 구조의 해상도를 조정합니다.
# min_dist: 저차원 임베딩에서 데이터 포인트들 사이의 최소 거리. 군집의 뭉침 정도나 분산 정도를 결정합니다.
# metric: 고차원 데이터 간 거리를 측정하는 방식 (예: euclidean, cosine 등).

# +
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import itertools

# 1. Data Loading and Preprocessing
# Assume 'ds' is an xarray Dataset already loaded.
# Variables names are in uppercase.
x = ds.variables['X'][:]  # 1D: X coordinates (loaded but not used as features)
y = ds.variables['Y'][:]  # 1D: Y coordinates (loaded but not used as features)

# Peak-related variables (shape: (Y, X, peak))
bias_center = ds.variables['peak_center'][:]   # bias_center
amp = ds.variables['peak_amplitude'][:]        # raw amplitude (not background-corrected)
sigma = ds.variables['peak_sigma'][:]          # peak sigma

# Use background_value (shape: (Y, X)) for background subtraction
background_value = ds.variables['background_value'][:]  # background for each (X, Y)
# Convert to numpy arrays
background_np = background_value.to_numpy()
amp_np = amp.to_numpy()
# Expand background_np to shape (Y, X, peak) and subtract from amp
amp_np = amp_np - np.repeat(background_np[..., np.newaxis], amp_np.shape[2], axis=2)
print("Background subtracted from amplitude.")

# Use ZB_mask (e.g., shape: (Y, X)) for filtering
zb_mask = ds.variables['ZB_mask'][:]

# Convert remaining xarray objects to numpy arrays using to_numpy()
bias_center_np = bias_center.to_numpy()
sigma_np = sigma.to_numpy()
zb_mask_np = zb_mask.to_numpy()

# Create meshgrid for X and Y (for shape checking only)
Xgrid, Ygrid = np.meshgrid(x, y)
print("Xgrid shape:", Xgrid.shape)
print("bias_center shape:", bias_center_np.shape)
print("ZB_mask shape:", zb_mask_np.shape)

# 2. Reshape Observations
# Use only bias_center, background-subtracted amp, and sigma (3 features)
nY, nX, nPeak = bias_center_np.shape

# Flatten the arrays
bias_center_flat = bias_center_np.reshape(-1)
amp_flat = amp_np.reshape(-1)
sigma_flat = sigma_np.reshape(-1)

# Expand zb_mask (shape: (Y, X)) to each peak and flatten
zb_mask_expanded = np.repeat(zb_mask_np[..., np.newaxis], nPeak, axis=2)
zb_mask_flat = zb_mask_expanded.reshape(-1)

# Select valid observations: non-NaN values and zb_mask True
valid = (~np.isnan(bias_center_flat)) & (~np.isnan(amp_flat)) & (~np.isnan(sigma_flat)) & (zb_mask_flat.astype(bool))
print("Total observations:", bias_center_flat.size, "/ Valid observations:", np.sum(valid))

# Final feature matrix: (N, 3)
features = np.column_stack((bias_center_flat[valid], amp_flat[valid], sigma_flat[valid]))

# 3. UMAP Embedding
umap_embedder = umap.UMAP(n_components=3, random_state=42)
embedding = umap_embedder.fit_transform(features)

# Create one figure with 3 subplots: Elbow, Silhouette, and UMAP 2D Projection (first two dims)
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Plot Elbow Method (Inertia)
inertias = []
silhouette_scores = []
K_range = range(2, 11)  # Try k from 2 to 10

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(embedding)
    inertias.append(kmeans.inertia_)
    sil_score = silhouette_score(embedding, cluster_labels)
    silhouette_scores.append(sil_score)
    print(f"k={k}: inertia={kmeans.inertia_:.2f}, silhouette_score={sil_score:.3f}")

axs[0].plot(list(K_range), inertias, 'o-', linewidth=2)
axs[0].set_title("Elbow Method: Inertia vs K")
axs[0].set_xlabel("Number of Clusters (K)")
axs[0].set_ylabel("Inertia")
axs[0].grid(True)

# Plot Silhouette Scores
axs[1].plot(list(K_range), silhouette_scores, 'o-', linewidth=2)
axs[1].set_title("Silhouette Score vs K")
axs[1].set_xlabel("Number of Clusters (K)")
axs[1].set_ylabel("Silhouette Score")
axs[1].grid(True)

# Plot UMAP 2D Projection (first 2 dims)
axs[2].scatter(embedding[:, 0], embedding[:, 1], s=10, cmap='viridis')
axs[2].set_title("UMAP 2D Projection (First 2 Dimensions)")
axs[2].set_xlabel("UMAP1")
axs[2].set_ylabel("UMAP2")
axs[2].grid(True)

plt.tight_layout()
plt.show()

# Automatically select best K based on highest silhouette score
auto_best_k = K_range[np.argmax(silhouette_scores)]
print("Auto-selected optimal number of clusters:", auto_best_k)

# 4. Ask User for Best K and Perform Clustering
user_input = input(f"Enter desired number of clusters (press Enter to use {auto_best_k}): ")
if user_input.strip() != "":
    best_k = int(user_input.strip())
else:
    best_k = auto_best_k

print("Final selected number of clusters:", best_k)

kmeans_final = KMeans(n_clusters=best_k, random_state=42)
clusters = kmeans_final.fit_predict(embedding)

# Create full cluster array with the same shape as bias_center (invalid obs set to -1)
clusters_full_flat = np.full(bias_center_flat.shape, -1, dtype=np.int32)
clusters_full_flat[valid] = clusters
clusters_full = clusters_full_flat.reshape(bias_center_np.shape)

# Save cluster array to the xarray Dataset (drop existing variable if needed)
if 'clusters' in ds:
    ds = ds.drop_vars('clusters')
ds['clusters'] = (ds['peak_center'].dims, clusters_full)

# 5. Scatter Plots for UMAP Embedding Pairs with Cluster Center Annotations
# For 3 components, possible pairs: (UMAP1,UMAP2), (UMAP1,UMAP3), (UMAP2,UMAP3)
pairs = list(itertools.combinations(range(3), 2))
plt.figure(figsize=(15, 5))
for i, (dim1, dim2) in enumerate(pairs):
    ax = plt.subplot(1, 3, i+1)
    sc = ax.scatter(embedding[:, dim1], embedding[:, dim2], c=clusters, cmap='viridis', s=10)
    ax.set_xlabel(f"UMAP{dim1+1}")
    ax.set_ylabel(f"UMAP{dim2+1}")
    ax.set_title(f"UMAP{dim1+1} vs UMAP{dim2+1}")
    ax.grid(True)
    
    # Annotate cluster centers on each plot.
    unique_clusters = np.unique(clusters)
    for cl in unique_clusters:
        # Select points belonging to cluster cl.
        idx = np.where(clusters == cl)[0]
        if len(idx) > 0:
            center_x = np.mean(embedding[idx, dim1])
            center_y = np.mean(embedding[idx, dim2])
            ax.text(center_x, center_y, str(cl), fontsize=12, fontweight='bold',
                    color='red', horizontalalignment='center', verticalalignment='center')
plt.tight_layout()
plt.show()


# +
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import itertools

# 1. Data Loading and Preprocessing
# Assume 'ds' is an xarray Dataset already loaded.
# Variables names are in uppercase.
x = ds.variables['X'][:]  # 1D: X coordinates (loaded but not used as features)
y = ds.variables['Y'][:]  # 1D: Y coordinates (loaded but not used as features)

# Peak-related variables (shape: (Y, X, peak))
bias_center = ds.variables['peak_center'][:]   # bias_center
amp = ds.variables['peak_amplitude'][:]        # raw amplitude (not background-corrected)
sigma = ds.variables['peak_sigma'][:]          # peak sigma


# Use background_value (shape: (Y, X)) for background subtraction
background_value = ds.variables['background_value'][:]  # background for each (X, Y)
# Convert to numpy arrays
background_np = background_value.to_numpy()
amp_np = amp.to_numpy()
# Expand background_np to shape (Y, X, peak) and subtract from amp
amp_np = amp_np - np.repeat(background_np[..., np.newaxis], amp_np.shape[2], axis=2)
print("Background subtracted from amplitude.")

# Use ZB_mask (e.g., shape: (Y, X)) for filtering
zb_mask = ds.variables['ZB_mask'][:]

# Convert remaining xarray objects to numpy arrays using to_numpy()
bias_center_np = bias_center.to_numpy()
sigma_np = sigma.to_numpy()
zb_mask_np = zb_mask.to_numpy()

# (X and Y are loaded for checking purposes, but they won't be used in the features.)
print("bias_center shape:", bias_center_np.shape)
print("ZB_mask shape:", zb_mask_np.shape)

# 2. Reshape Observations
# Here we exclude X and Y. Only use bias_center, background-subtracted amp, and sigma.
nY, nX, nPeak = bias_center_np.shape

# Flatten the arrays for the three features
bias_center_flat = bias_center_np.reshape(-1)
amp_flat = amp_np.reshape(-1)
sigma_flat = sigma_np.reshape(-1)

# Expand zb_mask (shape: (Y, X)) to each peak and flatten
zb_mask_expanded = np.repeat(zb_mask_np[..., np.newaxis], nPeak, axis=2)
zb_mask_flat = zb_mask_expanded.reshape(-1)

# Select valid observations: non-NaN values and zb_mask True
valid = (~np.isnan(bias_center_flat)) & (~np.isnan(amp_flat)) & (~np.isnan(sigma_flat)) & (zb_mask_flat.astype(bool))
print("Total observations:", bias_center_flat.size, "/ Valid observations:", np.sum(valid))

# Final feature matrix: (N, 3)
features = np.column_stack((bias_center_flat[valid], amp_flat[valid], sigma_flat[valid]))

# 3. UMAP Embedding
umap_embedder = umap.UMAP(n_components=3, random_state=42)
embedding = umap_embedder.fit_transform(features)

# Create one figure with 3 subplots: Elbow, Silhouette, and UMAP 2D Projection (first two dims)
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# 4. Elbow and Silhouette Analysis for Optimal Clusters (using UMAP embedding)
inertias = []
silhouette_scores = []
K_range = range(2, 11)  # Try k from 2 to 10

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(embedding)
    inertias.append(kmeans.inertia_)
    sil_score = silhouette_score(embedding, cluster_labels)
    silhouette_scores.append(sil_score)
    print(f"k={k}: inertia={kmeans.inertia_:.2f}, silhouette_score={sil_score:.3f}")

axs[0].plot(list(K_range), inertias, 'o-', linewidth=2)
axs[0].set_title("Elbow Method: Inertia vs K")
axs[0].set_xlabel("Number of Clusters (K)")
axs[0].set_ylabel("Inertia")
axs[0].grid(True)

axs[1].plot(list(K_range), silhouette_scores, 'o-', linewidth=2)
axs[1].set_title("Silhouette Score vs K")
axs[1].set_xlabel("Number of Clusters (K)")
axs[1].set_ylabel("Silhouette Score")
axs[1].grid(True)

axs[2].scatter(embedding[:, 0], embedding[:, 1], s=10, cmap='viridis')
axs[2].set_title("UMAP 2D Projection (First 2 Dimensions)")
axs[2].set_xlabel("UMAP1")
axs[2].set_ylabel("UMAP2")
axs[2].grid(True)

plt.tight_layout()
plt.show()

# Automatically select best K based on highest silhouette score
auto_best_k = K_range[np.argmax(silhouette_scores)]
print("Auto-selected optimal number of clusters:", auto_best_k)

# 5. Ask User for Best K and Perform Clustering
user_input = input(f"Enter desired number of clusters (press Enter to use {auto_best_k}): ")
if user_input.strip() != "":
    best_k = int(user_input.strip())
else:
    best_k = auto_best_k

print("Final selected number of clusters:", best_k)

kmeans_final = KMeans(n_clusters=best_k, random_state=42)
clusters = kmeans_final.fit_predict(embedding)

# Create a full cluster array with the same shape as bias_center (invalid observations set to -1)
clusters_full_flat = np.full(bias_center_flat.shape, -1, dtype=np.int32)
clusters_full_flat[valid] = clusters
clusters_full = clusters_full_flat.reshape(bias_center_np.shape)

# Save cluster array to the xarray Dataset (drop existing variable if needed)
if 'clusters' in ds:
    ds = ds.drop_vars('clusters')
ds['clusters'] = (ds['peak_center'].dims, clusters_full)

# 6. Scatter Plots for UMAP Embedding Pairs with Cluster Center Annotations
# For 3 components, possible pairs: (UMAP1, UMAP2), (UMAP1, UMAP3), (UMAP2, UMAP3)
pairs = list(itertools.combinations(range(3), 2))
plt.figure(figsize=(15, 5))
for i, (dim1, dim2) in enumerate(pairs):
    ax = plt.subplot(1, 3, i+1)
    sc = ax.scatter(embedding[:, dim1], embedding[:, dim2], c=clusters, cmap='viridis', s=10)
    ax.set_xlabel(f"UMAP{dim1+1}")
    ax.set_ylabel(f"UMAP{dim2+1}")
    ax.set_title(f"UMAP{dim1+1} vs UMAP{dim2+1}")
    ax.grid(True)
    
    # Annotate cluster centers on each plot.
    unique_clusters = np.unique(clusters)
    for cl in unique_clusters:
        # Select points belonging to cluster cl.
        idx = np.where(clusters == cl)[0]
        if len(idx) > 0:
            center_x = np.mean(embedding[idx, dim1])
            center_y = np.mean(embedding[idx, dim2])
            ax.text(center_x, center_y, str(cl), fontsize=12, fontweight='bold',
                    color='red', horizontalalignment='center', verticalalignment='center')
plt.tight_layout()
plt.show()

# -




# +
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import itertools

# Load Dataset
#file_path = "/mnt/data/GS_2T003_fit.nc"
#ds = xr.open_dataset(file_path)

# Select a smaller region (optional)
#ds = ds.isel(X=slice(60, 100), Y=slice(60, 100))

# Load and preprocess variables
x = ds['X'].to_numpy()
y = ds['Y'].to_numpy()
bias_center = ds['peak_center'].to_numpy()
amp = ds['peak_amplitude'].to_numpy()
sigma = ds['peak_sigma'].to_numpy()
background = ds['background_value'].to_numpy()
zb_mask = ds['ZB_mask'].to_numpy()
ldos = ds['LDOS'].to_numpy()
bias_mV = ds['bias_mV'].to_numpy()

# Background subtraction
amp = amp - np.repeat(background[..., np.newaxis], amp.shape[2], axis=2)
print("Background subtracted from amplitude.")

# Shape info
nY, nX, nPeak = bias_center.shape
print("bias_center shape:", bias_center.shape)
print("ZB_mask shape:", zb_mask.shape)

# Expand ZB mask
zb_mask_expanded = np.repeat(zb_mask[..., np.newaxis], nPeak, axis=2)

# Flatten variables
bias_center_flat = bias_center.reshape(-1)
amp_flat = amp.reshape(-1)
sigma_flat = sigma.reshape(-1)
zb_mask_flat = zb_mask_expanded.reshape(-1)

# Prepare indices for LDOS indexing
YY, XX, PP = np.meshgrid(np.arange(nY), np.arange(nX), np.arange(nPeak), indexing='ij')
yy = YY.reshape(-1)
xx = XX.reshape(-1)

# Match bias_center to LDOS
bias_center_clipped = np.clip(bias_center_flat, bias_mV.min(), bias_mV.max())
bias_idx = np.abs(bias_mV[np.newaxis, :] - bias_center_clipped[:, np.newaxis]).argmin(axis=1)
ldos_at_center_flat = ldos[yy, xx, bias_idx]

# Select valid entries
valid = (~np.isnan(bias_center_flat)) & (~np.isnan(amp_flat)) & (~np.isnan(sigma_flat)) & (~np.isnan(ldos_at_center_flat)) & (zb_mask_flat.astype(bool))
print("Total observations:", bias_center_flat.size, "/ Valid observations:", np.sum(valid))

# Feature matrix: (N, 4)
features = np.column_stack((bias_center_flat[valid], amp_flat[valid], sigma_flat[valid], ldos_at_center_flat[valid]))

# UMAP embedding
umap_embedder = umap.UMAP(n_components=3, random_state=42)
embedding = umap_embedder.fit_transform(features)

# Plot Elbow, Silhouette, UMAP 2D
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
inertias, silhouette_scores = [], []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(embedding)
    inertias.append(kmeans.inertia_)
    sil_score = silhouette_score(embedding, labels)
    silhouette_scores.append(sil_score)
    print(f"k={k}: inertia={kmeans.inertia_:.2f}, silhouette_score={sil_score:.3f}")

axs[0].plot(list(K_range), inertias, 'o-', linewidth=2)
axs[0].set_title("Elbow Method: Inertia vs K")
axs[0].set_xlabel("Number of Clusters (K)")
axs[0].set_ylabel("Inertia")
axs[0].grid(True)

axs[1].plot(list(K_range), silhouette_scores, 'o-', linewidth=2)
axs[1].set_title("Silhouette Score vs K")
axs[1].set_xlabel("Number of Clusters (K)")
axs[1].set_ylabel("Silhouette Score")
axs[1].grid(True)

axs[2].scatter(embedding[:, 0], embedding[:, 1], s=10, cmap='viridis')
axs[2].set_title("UMAP 2D Projection (First 2 Dimensions)")
axs[2].set_xlabel("UMAP1")
axs[2].set_ylabel("UMAP2")
axs[2].grid(True)

plt.tight_layout()
plt.show()

# Determine best K and ask user
auto_best_k = K_range[np.argmax(silhouette_scores)]
print("Auto-selected optimal number of clusters:", auto_best_k)
user_input = input(f"Enter desired number of clusters (press Enter to use {auto_best_k}): ")
best_k = int(user_input.strip()) if user_input.strip() else auto_best_k
print("Final selected number of clusters:", best_k)

# Final clustering
kmeans_final = KMeans(n_clusters=best_k, random_state=42)
clusters = kmeans_final.fit_predict(embedding)

# Full cluster array
clusters_full_flat = np.full(bias_center_flat.shape, -1, dtype=np.int32)
clusters_full_flat[valid] = clusters
clusters_full = clusters_full_flat.reshape(bias_center.shape)
if 'clusters' in ds:
    ds = ds.drop_vars('clusters')
ds['clusters'] = (ds['peak_center'].dims, clusters_full)

# UMAP pairwise cluster plots
pairs = list(itertools.combinations(range(3), 2))
plt.figure(figsize=(15, 5))
for i, (dim1, dim2) in enumerate(pairs):
    ax = plt.subplot(1, 3, i+1)
    sc = ax.scatter(embedding[:, dim1], embedding[:, dim2], c=clusters, cmap='viridis', s=10)
    ax.set_xlabel(f"UMAP{dim1+1}")
    ax.set_ylabel(f"UMAP{dim2+1}")
    ax.set_title(f"UMAP{dim1+1} vs UMAP{dim2+1}")
    ax.grid(True)

    unique_clusters = np.unique(clusters)
    for cl in unique_clusters:
        idx = np.where(clusters == cl)[0]
        if len(idx) > 0:
            center_x = np.mean(embedding[idx, dim1])
            center_y = np.mean(embedding[idx, dim2])
            ax.text(center_x, center_y, str(cl), fontsize=12, fontweight='bold',
                    color='red', horizontalalignment='center', verticalalignment='center')

plt.tight_layout()
plt.show()
# -

# # After Clustering, extract clustered peaks only 

ds = xr.open_dataset('grid_2T_003_fit_HDBSCAN_cluster3_20250629.nc')
#ds_filtered
ds

ds.LDOS


# +
import numpy as np
import xarray as xr
from lmfit.models import LorentzianModel, GaussianModel, VoigtModel
from typing import Optional, List, Union, Callable

def create_cluster_convolution_maps(
    ds: xr.Dataset,
    model_type: str = 'lorentzian',
    model_type_var: str = 'model_type',
    cluster_var: Optional[str] = None,
    clusters_to_keep: Optional[List[Union[int, float]]] = None,
    on_complete: Optional[Callable[[xr.Dataset], None]] = None
) -> xr.Dataset:
    """
    Compute and attach per-cluster convolution maps to `ds` in-place and return it.

    Direct mode:
      - If both `cluster_var` and `clusters_to_keep` are provided, computes and attaches maps and returns ds.

    Interactive mode:
      - If cluster_var is None or clusters_to_keep is None, launches UI;
        the function returns `ds` immediately, and callbacks attach
        maps to the same `ds` object asynchronously.

    Returns
    -------
    xr.Dataset
        The same `ds` (possibly with new data appended), always returned.
    """
    def stage_cluster_var(picked_var: str):
        # After picking cluster_var, prompt for labels
        create_cluster_convolution_maps(
            ds,
            model_type=model_type,
            model_type_var=model_type_var,
            cluster_var=picked_var,
            clusters_to_keep=None,
            on_complete=on_complete
        )

    def stage_cluster_label(
        filtered_ds: xr.Dataset,
        cluster_var: str,
        clusters_to_keep: List[Union[int, float]]
    ):
        # After picking labels, compute & attach maps directly on the original ds
        _compute_and_attach(
            ds,
            cluster_var,
            clusters_to_keep,
            model_type,
            model_type_var
        )
        if on_complete:
            on_complete(ds)

    # Interactive mode: choose cluster variable
    if cluster_var is None:
        select_cluster_data_var_interactive(ds, callback=stage_cluster_var)
        return ds

    # Interactive mode: choose labels for chosen cluster
    if clusters_to_keep is None:
        select_labels_in_cluster_interactive(ds, callback=stage_cluster_label)
        return ds

    # Direct mode: both inputs provided, compute and attach synchronously
    _compute_and_attach(
        ds,
        cluster_var,
        clusters_to_keep,
        model_type,
        model_type_var
    )
    if on_complete:
        on_complete(ds)
    return ds

def _compute_and_attach(
    ds: xr.Dataset,
    cluster_var: str,
    labels: List[Union[int, float]],
    model_type: str,
    model_type_var: str
) -> None:
    """
    Internal helper: for each label, sum up peak-models into a (Y,X,bias_mV) map
    and attach it as "{cluster_var}_L<label>" on `ds`.
    """
    clusters_arr = ds[cluster_var].values
    centers       = ds['peak_center'].values
    amplitudes    = ds['peak_amplitude'].values
    sigmas        = ds['peak_sigma'].values
    background    = ds['background_value'].values
    bias_axis     = ds['bias_mV'].values
    Y, X, P       = centers.shape

    model_map = None
    if model_type == 'per-pixel':
        model_map = ds[model_type_var].values

    for label in labels:
        var_name = f"{cluster_var}_L{int(label)}"
        # Initialize convolved map with background
        conv = np.broadcast_to(
            background[..., None],
            (Y, X, bias_axis.size)
        ).copy()

        # Matching function for label
        if np.isnan(label):
            match_fn = lambda c: np.isnan(c)
        else:
            match_fn = lambda c, lab=label: (not np.isnan(c)) and int(c) == lab

        # Loop through pixels and peaks
        for i in range(Y):
            for j in range(X):
                mtype = (
                    str(model_map[i, j]).lower().strip()
                    if model_map is not None
                    else model_type
                )
                if mtype not in ('lorentzian', 'gaussian', 'voigt'):
                    conv[i, j, :] = np.nan
                    continue

                ModelClass = {
                    'lorentzian': LorentzianModel,
                    'gaussian':   GaussianModel,
                    'voigt':      VoigtModel
                }[mtype]

                for k in range(P):
                    if not match_fn(clusters_arr[i, j, k]):
                        continue
                    cen = centers[i, j, k]
                    amp = amplitudes[i, j, k]
                    sig = sigmas[i, j, k]
                    model = ModelClass(prefix='m_')
                    if ModelClass is VoigtModel:
                        params = model.make_params(
                            m_amplitude=amp,
                            m_center=cen,
                            m_sigma=sig,
                            m_gamma=sig
                        )
                    else:
                        params = model.make_params(
                            m_amplitude=amp,
                            m_center=cen,
                            m_sigma=sig
                        )
                    conv[i, j, :] += model.eval(params, x=bias_axis)

        # Attach convolved map to dataset
        ds[var_name] = (('Y', 'X', 'bias_mV'), conv)
        print(f"Attached '{var_name}' with shape {conv.shape}")



# -

#ds_opt
ds_opt2 = ds.copy()

ds_opt2

# +
# import numpy as np
import xarray as xr
from lmfit.models import LorentzianModel, GaussianModel, VoigtModel
from typing import Optional, List, Union, Callable

# Module-level globals for interactive state
selected_cluster_var_global: Optional[str] = None
selected_cluster_labels_global: List[Union[int, float]] = []


def create_cluster_convolution_maps(
    ds: xr.Dataset,
    model_type: Union[str, List[str]] = ['Lorentzian', 'Gaussian', 'Voigt'],
    model_type_var: str = 'model_type',
    cluster_var: Optional[str] = None,
    clusters_to_keep: Optional[List[Union[int, float]]] = None,
    on_complete: Optional[Callable[[xr.Dataset], None]] = None
) -> xr.Dataset:
    """
    Compute and attach per-cluster convolution maps to `ds` in-place and return it.

    This function supports both interactive and direct modes:
      - Interactive mode: if `cluster_var` or `clusters_to_keep` is None, prompts user via widgets.
      - Direct mode: when both are provided, computes immediately.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing variables:
        - 'peak_center', 'peak_amplitude', 'peak_sigma', 'background_value', 'bias_mV'
        - Optional per-pixel model types in `model_type_var`.
    model_type : str or list of str
        A single model name (applied globally) or list of allowed model names
        (enabling per-pixel selection via `model_type_var`).
    model_type_var : str
        Name of the variable in `ds` storing per-pixel model types.
    cluster_var : str, optional
        Name of the variable in `ds` representing cluster labels. If None, user is prompted.
    clusters_to_keep : list of int/float, optional
        Labels to compute maps for. If None, user is prompted.
    on_complete : callable, optional
        Callback invoked with `ds` after computation completes.

    Returns
    -------
    xr.Dataset
        The same dataset `ds`, augmented with new variables named
        "{cluster_var}_L<label>" for each label.
    """
    # Import UI components for interactive mode
    import ipywidgets as widgets
    from IPython.display import display, clear_output

    def select_cluster_data_var_interactive(
        ds: xr.Dataset,
        callback: Callable[[str], None] = None,
        **kwargs
    ) -> None:
        """
        Launch a widget for choosing which cluster variable to use.

        Scans `ds.data_vars` for names including 'cluster' and
        displays a radio button list. On confirmation, updates
        `selected_cluster_var_global` and invokes `callback`.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset containing cluster variables.
        callback : callable, optional
            Function called as `callback(selected_var, **kwargs)`.
        **kwargs : dict
            Additional arguments forwarded to `callback`.
        """
        global selected_cluster_var_global
        # Identify cluster variables
        options = [v for v in ds.data_vars if 'cluster' in v]
        if not options:
            raise ValueError("No data_vars containing 'cluster' found.")

        selector = widgets.RadioButtons(
            options=options,
            description='Cluster var:',
            style={'description_width': 'initial'}
        )
        confirm_btn = widgets.Button(description='Confirm', button_style='primary')
        output = widgets.Output()

        def _on_confirm(_):
            global selected_cluster_var_global
            with output:
                clear_output()
                selected_cluster_var_global = selector.value
                print(f"Selected cluster variable: '{selected_cluster_var_global}'")
                if callback:
                    callback(selected_cluster_var_global, **kwargs)

        confirm_btn.on_click(_on_confirm)
        display(widgets.VBox([selector, confirm_btn, output]))

    def select_labels_in_cluster_interactive(
        ds: xr.Dataset,
        callback: Callable[[xr.Dataset, str, List[Union[int, float]]], None] = None,
        **kwargs
    ) -> None:
        """
        Launch a widget to select which labels of the chosen cluster to compute.

        Extracts unique labels from `selected_cluster_var_global`, displays
        a multi-select widget, builds a filtered dataset, and invokes `callback`.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset containing the selected cluster variable.
        callback : callable, optional
            Function called as `callback(filtered_ds, cluster_var, labels, **kwargs)`.
        **kwargs : dict
            Additional arguments forwarded to `callback`.
        """
        global selected_cluster_labels_global, selected_cluster_var_global
        if not selected_cluster_var_global:
            raise ValueError("Cluster variable not selected.")

        da = ds[selected_cluster_var_global]
        unique = np.unique(da.values)
        labels = sorted([int(l) for l in unique if not np.isnan(l)])
        if not labels:
            raise ValueError("No valid labels found.")

        selector = widgets.SelectMultiple(
            options=labels,
            description='Labels to keep:',
            style={'description_width': 'initial'}
        )
        confirm_btn = widgets.Button(description='Confirm', button_style='success')
        output = widgets.Output()

        def _on_confirm(_):
            global selected_cluster_labels_global
            with output:
                clear_output()
                sel = list(selector.value)
                if not sel:
                    print("Please select at least one label.")
                    return
                selected_cluster_labels_global = sel
                print(f"Selected labels: {selected_cluster_labels_global}")

                # Build masked dataset for selected labels
                mask = xr.DataArray(
                    np.isin(da.values, selected_cluster_labels_global),
                    dims=da.dims,
                    coords=da.coords
                )
                filtered = ds.copy(deep=True)
                for var in ds.data_vars:
                    if set(da.dims).issubset(ds[var].dims):
                        filtered[var] = ds[var].where(mask)

                if callback:
                    callback(
                        filtered,
                        selected_cluster_var_global,
                        selected_cluster_labels_global,
                        **kwargs
                    )

        confirm_btn.on_click(_on_confirm)
        display(widgets.VBox([selector, confirm_btn, output]))

    def stage_cluster_var(picked_var: str) -> None:
        """
        Callback invoked after cluster variable selection.

        Parameters
        ----------
        picked_var : str
            The name of the chosen cluster variable.
        """
        # Recurse into main function with chosen variable
        create_cluster_convolution_maps(
            ds,
            model_type=model_type,
            model_type_var=model_type_var,
            cluster_var=picked_var,
            clusters_to_keep=None,
            on_complete=on_complete
        )

    def stage_cluster_label(
        filtered_ds: xr.Dataset,
        cluster_var: str,
        clusters_to_keep: List[Union[int, float]]
    ) -> None:
        """
        Callback invoked after cluster label selection.

        Parameters
        ----------
        filtered_ds : xr.Dataset
            Dataset masked to selected cluster labels.
        cluster_var : str
            Name of the cluster variable.
        clusters_to_keep : list
            Labels to compute.
        """
        # Perform the actual convolution computation
        _compute_and_attach(
            ds,
            cluster_var,
            clusters_to_keep,
            model_type,
            model_type_var
        )
        if on_complete:
            on_complete(ds)

    # Interactive mode: prompt for cluster variable
    if cluster_var is None:
        select_cluster_data_var_interactive(ds, callback=stage_cluster_var)
        return ds

    # Interactive mode: prompt for labels
    if clusters_to_keep is None:
        select_labels_in_cluster_interactive(ds, callback=stage_cluster_label)
        return ds

    # Direct mode: compute and attach maps
    _compute_and_attach(
        ds,
        cluster_var,
        clusters_to_keep,
        model_type,
        model_type_var
    )
    if on_complete:
        on_complete(ds)
    return ds


def _compute_and_attach(
    ds: xr.Dataset,
    cluster_var: str,
    labels: List[Union[int, float]],
    model_type: Union[str, List[str]],
    model_type_var: str
) -> None:
    """
    Internal helper: sums peak models into a (Y, X, bias_mV) map per label
    and attaches them to `ds` as new variables.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to augment.
    cluster_var : str
        Name of the cluster label variable.
    labels : list of int/float
        Labels for which to compute maps.
    model_type : str or list
        Model name(s) specifying which fit function to use.
    model_type_var : str
        Name of the per-pixel model type variable, if any.
    """
    clusters_arr = ds[cluster_var].values
    centers      = ds['peak_center'].values
    amplitudes   = ds['peak_amplitude'].values
    sigmas       = ds['peak_sigma'].values
    background   = ds['background_value'].values
    bias_axis    = ds['bias_mV'].values
    Y, X, P      = centers.shape

    # Determine if per-pixel model map is used
    if isinstance(model_type, list):
        allowed_models = [m.lower() for m in model_type]
        model_map = ds[model_type_var].values
    else:
        allowed_models = None
        model_map = None
        model_type = str(model_type).lower().strip()

    for label in labels:
        var_name = f"{cluster_var}_L{int(label)}"
        # Initialize the convolved map with background offsets
        conv = np.broadcast_to(
            background[..., None],
            (Y, X, bias_axis.size)
        ).copy()

        # Define how to match this label
        match_fn = (
            lambda c: np.isnan(c)
        ) if np.isnan(label) else (
            lambda c, lab=label: (not np.isnan(c)) and int(c) == lab
        )

        for i in range(Y):
            for j in range(X):
                # Select model type for this pixel
                if model_map is not None:
                    mtype = str(model_map[i, j]).lower().strip()
                    # Skip if not in allowed list
                    if allowed_models and mtype not in allowed_models:
                        conv[i, j, :] = np.nan
                        continue
                else:
                    mtype = model_type

                if mtype not in ('lorentzian', 'gaussian', 'voigt'):
                    conv[i, j, :] = np.nan
                    continue

                ModelClass = {
                    'lorentzian': LorentzianModel,
                    'gaussian':   GaussianModel,
                    'voigt':      VoigtModel
                }[mtype]

                for k in range(P):
                    if not match_fn(clusters_arr[i, j, k]):
                        continue
                    cen = centers[i, j, k]
                    amp = amplitudes[i, j, k]
                    sig = sigmas[i, j, k]
                    model = ModelClass(prefix='m_')
                    # Configure parameters for Voigt vs others
                    if ModelClass is VoigtModel:
                        params = model.make_params(
                            m_amplitude=amp,
                            m_center=cen,
                            m_sigma=sig,
                            m_gamma=sig
                        )
                    else:
                        params = model.make_params(
                            m_amplitude=amp,
                            m_center=cen,
                            m_sigma=sig
                        )
                    # Evaluate and accumulate
                    conv[i, j, :] += model.eval(params, x=bias_axis)

        # Attach result to dataset
        ds[var_name] = (('Y', 'X', 'bias_mV'), conv)
        print(f"Attached '{var_name}' with shape {conv.shape}")



# -

ds_opt2

ds_opt2 = create_cluster_convolution_maps(
    ds_opt2,
    model_type=['Lorentzian', 'Gaussian', 'Voigt']
)

ds_opt2

'''#not-interactievly select
ds_opt2 = create_cluster_convolution_maps(
    ds_opt2,
    cluster_var='cluster_umap_HDBSCAN0',
    clusters_to_keep=[0, 1, 2, 3, 4]
)'''

# +
#ds_opt2.cluster_umap_HDBSCAN9_L0.to_dataframe().describe()

#ds_opt2.cluster_umap_HDBSCAN0_L2.mean(dim = ['X','Y']).plot()
#ds_opt2.cluster_umap_HDBSCAN9_L0.sel(bias_mV =0).plot()

#grid_data_dim_slicing(ds_opt2, channel = 'cluster_umap_HDBSCAN9_L2')

#ds_opt2.to_netcdf('grid_2T_003_fit_cluster9_20250523.nc')

# +
import re
import xarray as xr

def combine_cluster_maps(ds: xr.Dataset, new_var_name: str = "cluster_maps") -> xr.Dataset:
    """
    Collect all DataArrays in `ds` whose names contain 'cluster' and end with '_L<digit>',
    then concatenate them along a new dimension 'cluster_label'.

    The extracted <digit> becomes the coordinate values for 'cluster_label'.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing per‐cluster maps named like '<something>_L0', '<something>_L1', etc.
    new_var_name : str, default "cluster_maps"
        Name for the combined DataArray in the output dataset.

    Returns
    -------
    xr.Dataset
        The same dataset with an added DataArray `ds[new_var_name]` of dims
        ('cluster_label', ...) containing all the collected maps.
    """
    # Define regex to match names ending with '_L<digits>' and containing 'cluster'
    pattern = re.compile(r".*cluster.*_L(\d+)$")

    # Find all matching variable names and extract their label indices
    matches = []
    for var in ds.data_vars:
        m = pattern.match(var)
        if m:
            label = int(m.group(1))
            matches.append((label, var))

    if not matches:
        raise ValueError("No variables matching '*cluster*_L<digit>' found in dataset.")

    # Sort by the numeric label
    matches.sort(key=lambda x: x[0])
    labels, var_names = zip(*matches)

    # Gather the DataArrays in order
    arrays = [ds[v] for v in var_names]

    # Concatenate along new dimension 'cluster_label'
    combined = xr.concat(arrays, dim="cluster_label")

    # Assign coordinate values for 'cluster_label' and sort
    combined = combined.assign_coords(cluster_label=("cluster_label", list(labels)))
    combined = combined.sortby("cluster_label")

    # Attach the combined DataArray back to the dataset
    ds[new_var_name] = combined

    return ds


# ── Usage Example ────────────────────────────────────────────────────────────
# Assuming `ds` already contains variables like:
#   'cluster_umap_HDBSCAN_opt0_L0', 'cluster_umap_HDBSCAN_opt0_L1', ...
#
# >>> ds = combine_cluster_maps(ds, new_var_name="cluster_maps")
# >>> print(ds["cluster_maps"])
# DataArray 'cluster_maps' with dimensions: (cluster_label, Y, X, bias_mV)
# Coordinates:
#   * cluster_label  (cluster_label) int64 0 1 ...

# -

ds_opt2 = combine_cluster_maps(ds_opt2, new_var_name="cluster_maps")

# +
import os
import xarray as xr

def interactive_cluster_map(cluster_da: xr.DataArray):
    """
    Launch an interactive cluster map viewer in a Jupyter notebook.

    This function provides:
      - Slider to select cluster label (int).
      - Slider to select bias voltage (float, 2-decimal).
      - Toggle between Global/Local color-scaling modes.
      - Toggle between percentage-based or absolute-based color limits.
      - Sliders to choose lower/upper color limits.
      - Dropdown to select matplotlib colormap.
      - “Confirm” button to render the HoloViews plot with explicit X/Y tick labels in nm.
      - “Save SVG” button to write the current view to `output_figures/cluster_map.svg`
        using Matplotlib’s `imshow`, preserving true data aspect ratio and adding a 10 nm scale bar.

    Parameters
    ----------
    cluster_da : xr.DataArray
        A 4D DataArray with dims ('cluster_label','Y','X','bias_mV').
    """
    import numpy as np
    import hvplot.xarray     # for interactive on-screen plots
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display, clear_output

    def compute_global_stats():
        """
        Compute the global minimum and maximum values across all clusters and biases.
        
        Returns
        -------
        (gmin, gmax) : tuple of floats
            Global minimum and maximum.
        """
        gmin = float(cluster_da.min(skipna=True).values)
        gmax = float(cluster_da.max(skipna=True).values)
        return gmin, gmax

    def compute_clim_limits(slice2d, gmin, gmax, mode, ctype,
                            lower_pct, upper_pct, lower_val, upper_val):
        """
        Compute the color limits (cmin, cmax) for a 2D slice.

        Parameters
        ----------
        slice2d : np.ndarray
            2D data slice.
        gmin, gmax : float
            Global min/max.
        mode : {'Global','Local'}
            Whether to use global or local data range.
        ctype : {'Percent','Absolute'}
            Use percentage or absolute limits.
        lower_pct, upper_pct : float
            Percent limits (0–100) if ctype=='Percent'.
        lower_val, upper_val : float
            Absolute limits if ctype=='Absolute'.

        Returns
        -------
        (cmin, cmax) : tuple of floats
        """
        if ctype == 'Percent':
            lp, up = lower_pct/100, upper_pct/100
            if mode == 'Global':
                cmin = gmin + (gmax - gmin) * lp
                cmax = gmin + (gmax - gmin) * up
            else:
                loc_min, loc_max = float(np.nanmin(slice2d)), float(np.nanmax(slice2d))
                cmin = loc_min + (loc_max - loc_min) * lp
                cmax = loc_min + (loc_max - loc_min) * up
        else:
            cmin, cmax = lower_val, upper_val
        return cmin, cmax

    def generate_slice(cluster_label, bias_val):
        """
        Extract a 2D slice for the given cluster_label and bias voltage,
        and convert its X/Y coordinates from meters to nanometers.

        Returns
        -------
        (slice2d, X_nm, Y_nm) : tuple
        """
        sl = cluster_da.sel(cluster_label=cluster_label,
                            bias_mV=bias_val, method='nearest')
        X_nm = sl.coords['X'].values * 1e9
        Y_nm = sl.coords['Y'].values * 1e9
        return sl.values, X_nm, Y_nm

    def create_widgets():
        """
        Instantiate all ipywidgets needed for interaction, arrange them
        in a VBox layout, and return a dict containing widgets and state.
        """
        gmin, gmax = compute_global_stats()
        labels = sorted(cluster_da.coords['cluster_label'].values.tolist())
        biases = sorted(cluster_da.coords['bias_mV'].values.tolist())
        step   = round(biases[1] - biases[0], 4) if len(biases)>1 else 0.01
        cmaps  = ['bwr','viridis','plasma','inferno','magma','cividis']

        w = {}
        w['cluster_slider'] = widgets.IntSlider(
            description='Cluster Label',
            min=int(labels[0]), max=int(labels[-1]), step=1,
            value=int(labels[0])
        )
        w['bias_slider'] = widgets.FloatSlider(
            description='Bias (mV)',
            min=biases[0], max=biases[-1], step=step,
            value=biases[0], readout_format='.2f'
        )
        w['global_label'] = widgets.HTML(
            value=f"<b>Global min:</b> {gmin:.3e}, <b>Global max:</b> {gmax:.3e}"
        )
        w['mode_widget'] = widgets.RadioButtons(
            options=['Global','Local'],
            description='Color Mode', value='Global'
        )
        w['clim_type'] = widgets.ToggleButtons(
            options=['Percent','Absolute'],
            description='Clim Type', value='Percent'
        )
        w['clim_lower_pct'] = widgets.FloatSlider(
            description='Clim Lower (%)',
            min=0.0, max=100.0, step=0.1, value=0.0, readout_format='.1f'
        )
        w['clim_upper_pct'] = widgets.FloatSlider(
            description='Clim Upper (%)',
            min=0.0, max=100.0, step=0.1, value=100.0, readout_format='.1f'
        )
        w['clim_lower_val'] = widgets.FloatSlider(
            description='Clim Lower',
            min=gmin, max=gmax,
            step=(gmax-gmin)/100, value=gmin, readout_format='.2e'
        )
        w['clim_upper_val'] = widgets.FloatSlider(
            description='Clim Upper',
            min=gmin, max=gmax,
            step=(gmax-gmin)/100, value=gmax, readout_format='.2e'
        )
        w['cmap_dropdown'] = widgets.Dropdown(
            options=cmaps, value='bwr', description='Colormap'
        )
        w['confirm_btn'] = widgets.Button(
            description='Confirm', button_style='primary'
        )
        w['save_btn'] = widgets.Button(
            description='Save SVG', button_style='success'
        )
        w['cmin_label'] = widgets.Label(value="Cmin: ")
        w['cmax_label'] = widgets.Label(value="Cmax: ")
        w['output'] = widgets.Output()

        def toggle_clim(evt=None):
            use_pct = (w['clim_type'].value=='Percent')
            w['clim_lower_pct'].layout.display = None if use_pct else 'none'
            w['clim_upper_pct'].layout.display = None if use_pct else 'none'
            w['clim_lower_val'].layout.display = None if not use_pct else 'none'
            w['clim_upper_val'].layout.display = None if not use_pct else 'none'
        w['clim_type'].observe(toggle_clim, names='value')
        toggle_clim()

        ui = widgets.VBox([
            w['cluster_slider'],
            w['bias_slider'],
            w['global_label'],
            w['mode_widget'],
            w['clim_type'],
            widgets.VBox([w['clim_lower_pct'], w['clim_upper_pct'],
                          w['clim_lower_val'], w['clim_upper_val']]),
            w['cmap_dropdown'],
            w['cmin_label'], w['cmax_label'],
            widgets.HBox([w['confirm_btn'], w['save_btn']]),
            w['output']
        ])
        w['ui'] = ui
        w['last'] = {}
        return w

    def plot_hv(slice2d, X_nm, Y_nm, cmin, cmax, cmap, cl, bv, mode):
        """
        Render the 2D slice interactively with explicit X/Y tick labels in nm.
        """
        da2d = xr.DataArray(slice2d, dims=('Y','X'),
                            coords={'X':X_nm,'Y':Y_nm})
        title = f"Cluster {cl}, Bias={bv:.2f} mV ({mode})"
        xt = np.linspace(X_nm.min(), X_nm.max(), 5)
        yt = np.linspace(Y_nm.min(), Y_nm.max(), 5)
        xticks = [(float(v), f"{v:.0f}") for v in xt]
        yticks = [(float(v), f"{v:.0f}") for v in yt]
        return da2d.hvplot.image(
            x='X', y='Y',
            cmap=cmap, colorbar=True, clim=(cmin,cmax),
            xlabel='X (nm)', ylabel='Y (nm)',
            title=title, aspect='equal', width=500
        ).opts(xticks=xticks, yticks=yticks)

    def save_svg(slice2d, X_nm, Y_nm, cmin, cmax, cmap, cl, bv):
        """
        Save the 2D slice as SVG into 'output_figures/cluster_map.svg',
        using Matplotlib imshow for correct aspect ratio and adding a 10 nm scale bar.
        """
        folder = 'output_figures'
        os.makedirs(folder, exist_ok=True)
        fname = os.path.join(folder, 'cluster_map.svg')

        # Compute spans and figure size
        x_span = X_nm.max() - X_nm.min()
        y_span = Y_nm.max() - Y_nm.min()
        aspect = x_span / y_span
        height = 6  # inches
        width = height * aspect

        fig, ax = plt.subplots(figsize=(width, height))
        # Display with proper extent and aspect
        extent = [X_nm.min(), X_nm.max(), Y_nm.min(), Y_nm.max()]
        im = ax.imshow(
            slice2d,
            extent=extent,
            origin='lower',
            cmap=cmap,
            vmin=cmin, vmax=cmax,
            aspect='equal'
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Intensity')

        # Labels and title
        ax.set_xlabel('X (nm)')
        ax.set_ylabel('Y (nm)')
        ax.set_title(f"Cluster {cl}, Bias={bv:.2f} mV")

        # Add 10 nm scale bar
        x0 = X_nm.min() + 0.05 * x_span
        y0 = Y_nm.min() + 0.05 * y_span
        length = 10.0  # nm
        ax.hlines(y=y0, xmin=x0, xmax=x0+length, colors='white', linewidth=3)
        ax.text(x0 + length/2, y0 + 0.02 * y_span,
                '10 nm', color='white', ha='center', va='bottom',
                fontsize=12, weight='bold')

        # Format tick labels as integers
        ax.set_xticks(xt := ax.get_xticks())
        ax.set_yticks(yt := ax.get_yticks())
        ax.set_xticklabels([f"{v:.0f}" for v in xt], rotation=45, ha='right')
        ax.set_yticklabels([f"{v:.0f}" for v in yt])

        fig.savefig(fname, format='svg', bbox_inches='tight')
        plt.close(fig)
        return fname

    # Build UI and bind callbacks
    app = create_widgets()

    def on_confirm(_):
        """Render the interactive HoloViews plot."""
        with app['output']:
            clear_output()
            cl = app['cluster_slider'].value
            bv = app['bias_slider'].value
            sl, X_nm, Y_nm = generate_slice(cl, bv)
            gmin, gmax = compute_global_stats()
            cmin, cmax = compute_clim_limits(
                sl, gmin, gmax,
                app['mode_widget'].value,
                app['clim_type'].value,
                app['clim_lower_pct'].value,
                app['clim_upper_pct'].value,
                app['clim_lower_val'].value,
                app['clim_upper_val'].value
            )
            app['cmin_label'].value = f"Cmin: {cmin:.2e}"
            app['cmax_label'].value = f"Cmax: {cmax:.2e}"
            plot = plot_hv(
                sl, X_nm, Y_nm, cmin, cmax,
                app['cmap_dropdown'].value,
                cl, bv, app['mode_widget'].value
            )
            display(plot)
            app['last'] = {
                'slice2d': sl, 'X_nm': X_nm, 'Y_nm': Y_nm,
                'cmin': cmin, 'cmax': cmax,
                'cl': cl, 'bv': bv, 'cmap': app['cmap_dropdown'].value
            }

    def on_save(_):
        """Save the current view as an SVG file."""
        with app['output']:
            clear_output()
            if not app['last']:
                print("⚠️ Generate a plot first.")
                return
            p = app['last']
            fname = save_svg(
                p['slice2d'], p['X_nm'], p['Y_nm'],
                p['cmin'], p['cmax'],
                p['cmap'], p['cl'], p['bv']
            )
            print(f"✅ Saved SVG to '{fname}'")

    app['confirm_btn'].on_click(on_confirm)
    app['save_btn'].on_click(on_save)

    display(app['ui'])



# -

ds_opt2

#ds = ds_opt2.copy()
interactive_cluster_map(ds_opt2.cluster_maps)



# ## save clustering result

ds_opt2.to_netcdf('grid_2T_003_fit_HDBSCAN_cluster5_20250702.nc')

ds_opt2 = xr.open_dataset('grid_2T_003_fit_HDBSCAN_cluster5_20250702.nc')

#ds = ds_opt2.copy()
interactive_cluster_map(ds_opt2.cluster_maps)

grid_data_dim_slicing(ds_opt2,channel='cluster_umap_HDBSCAN1_L1')

hv_bias_mV_slicing(ds_opt2, ch= 'cluster_umap_HDBSCAN1_L0')

ds_opt2.cluster_umap_HDBSCAN1_L1


# +
import numpy as np
import plotly.graph_objects as go

def plot_top_percent_3d(
    ds, 
    var_name, 
    top_percent=20, 
    cmap='Viridis', 
    opacity=0.8, 
    max_size=5,
    camera_eye=(1.25, 1.25, 1.25)
):
    """
    ds: xarray.Dataset
        DataSet containing the variable to plot.
    var_name: str
        이름이 var_name인 DataArray를 ds에서 가져옵니다.
    top_percent: float
        상위 몇 %의 데이터를 그릴지 설정합니다 (0 < top_percent <= 100).
    cmap: str
        Plotly colorscale 이름을 지정합니다 (예: 'Viridis', 'Cividis', 'Plasma' 등).
    opacity: float
        마커의 투명도를 0.0 (완전 투명) ~ 1.0 (불투명) 사이로 설정합니다.
    max_size: float
        마커의 최대 크기를 설정합니다.
    camera_eye: tuple of 3 floats
        카메라의 초기 'eye' 위치를 지정 (x, y, z).
    
    전체 데이터 값 범위를 기준으로 각 지점의 값 백분율을 계산하여
    마커 크기를 조정하고, 지정된 카메라 위치로 초기 뷰를 설정합니다.
    배경은 흰색입니다.
    """
    # 1. DataArray 선택
    data = ds[var_name]
    
    # 2. 문턱값 계산 (100 - top_percent 백분위수)
    threshold = np.nanpercentile(data.values, 100.0 - top_percent)
    
    # 3. 상위 top_percent%에 해당하는 값의 마스크 생성
    mask = data.values >= threshold
    
    # 4. 마스크에 따라 좌표 및 값 추출
    indices = np.where(mask)
    coords = {
        dim: data.coords[dim].values[idx]
        for dim, idx in zip(data.dims, indices)
    }
    x_coords = coords['X']
    y_coords = coords['Y']
    z_coords = coords['bias_mV']
    values = data.values[mask]
    
    # 5. 전체 데이터 범위에서 값 백분율 계산 -> 마커 크기 (0 ~ max_size)
    global_min = np.nanmin(data.values)
    global_max = np.nanmax(data.values)
    rel = (values - global_min) / (global_max - global_min + 1e-12)
    sizes = rel * max_size
    
    # 6. 데이터 중앙값 계산 (카메라 중심)
    x_center = float(np.nanmedian(x_coords))
    y_center = float(np.nanmedian(y_coords))
    z_center = float(np.nanmedian(z_coords))
    
    # 7. Plotly 3D 산점도 생성
    scatter = go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode='markers',
        marker=dict(
            size=sizes,
            color=values,
            colorscale=cmap,
            colorbar=dict(title='Value'),
            opacity=opacity
        )
    )
    
    # 8. 레이아웃 및 카메라 설정
    camera = dict(
        center=dict(x=x_center, y=y_center, z=z_center),
        eye=dict(x=camera_eye[0], y=camera_eye[1], z=camera_eye[2])
    )
    
    layout = go.Layout(
            width=600,     # ← 여기
    height=600,     # ← 그리고 여기
        title=f"'{var_name}' 3D Scatter",
        paper_bgcolor='white',
        plot_bgcolor='white',
        scene=dict(
            camera=camera,
            xaxis=dict(
                title='X',
                backgroundcolor='white',
                gridcolor='lightgray',
                zerolinecolor='lightgray',
            ),
            yaxis=dict(
                title='Y',
                backgroundcolor='white',
                gridcolor='lightgray',
                zerolinecolor='lightgray',
            ),
            zaxis=dict(
                title='bias_mV',
                backgroundcolor='white',
                gridcolor='lightgray',
                zerolinecolor='lightgray',
            )
        )
    )
    
    fig = go.Figure(data=[scatter], layout=layout)
    fig.show()

# 사용 예시:
# plot_top_percent_3d(
#     ds_opt2,
#     'cluster_umap_HDBSCAN1_L1',
#     top_percent=1,
#     cmap='Cividis',
#     opacity=0.6,
#     max_size=5,
#     camera_eye=(1.5, 1.5, 1.5)
# )



# +

plot_top_percent_3d(ds_opt2, 'cluster_umap_HDBSCAN1_L1', top_percent=0.5, cmap='Greens', opacity=0.1,max_size=10, camera_eye=(2, 2, 0.5))
# -

plot_top_percent_3d(ds_opt2, 'LDOS', top_percent=10, cmap='viridis', opacity=0.1,max_size=10, camera_eye=(2, 2, 0.5))





# +
import numpy as np
import plotly.graph_objects as go

def plot_multi_top_percent_3d_logscale(
    ds,
    var_names,
    top_percent=20,
    opacity=0.8,
    max_size=5,
    camera_eye=(1.25, 1.25, 1.25)
):
    """
    ds: xarray.Dataset
        DataSet containing the variables to plot.
    var_names: list of str
        DataArray 이름 목록 (예: ['cluster_umap_HDBSCAN1_L0', ..., 'cluster_umap_HDBSCAN1_L5']).
    top_percent: float
        상위 몇 %의 데이터를 표시할지 설정합니다 (0 < top_percent <= 100).
    opacity: float
        마커의 투명도 (0.0 ~ 1.0).
    max_size: float
        마커의 최대 크기.
    camera_eye: tuple of 3 floats
        카메라의 초기 'eye' 위치를 지정 (x, y, z).

    각 변수에 대해 상위 top_percent% 포인트만 추출하여,
    흰색에서 tab10 컬러로 이어지는 2색 스케일을 적용하고,
    로그 스케일로 마커 크기를 매핑하여 작은 값은 거의 보이지 않고 큰 값은 강조합니다.
    배경 흰색, 테두리 없는 마커 설정.
    """
    # Tab10 첫 6가지 색상
    tab10_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c',
        '#d62728', '#9467bd', '#8c564b'
    ]
    
    traces = []
    all_x, all_y, all_z = [], [], []
    
    for idx, var in enumerate(var_names):
        data = ds[var]
        threshold = np.nanpercentile(data.values, 100.0 - top_percent)
        mask = data.values >= threshold
        
        indices = np.where(mask)
        coords = {
            dim: data.coords[dim].values[i]
            for dim, i in zip(data.dims, indices)
        }
        x = coords['X']
        y = coords['Y']
        z = coords['bias_mV']
        vals = data.values[mask]
        
        all_x.append(x); all_y.append(y); all_z.append(z)
        
        # 로그 스케일 크기 매핑
        global_min = np.nanmin(data.values)
        global_max = np.nanmax(data.values)
        rel = (vals - global_min) / (global_max - global_min + 1e-12)
        sizes = max_size * np.log10(rel * 9 + 1)
        
        # 흰색 → 지정 색상
        color = tab10_colors[idx % len(tab10_colors)]
        colorscale = [[0, 'white'], [1, color]]
        
        trace = go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=sizes,
                color=vals,
                colorscale=colorscale,
                opacity=opacity,
                line=dict(width=0)  # 테두리 제거
            ),
            name=var
        )
        traces.append(trace)
    
    # 중앙값 기반 카메라 중심
    all_x_arr = np.concatenate(all_x)
    all_y_arr = np.concatenate(all_y)
    all_z_arr = np.concatenate(all_z)
    center = dict(
        x=float(np.nanmedian(all_x_arr)),
        y=float(np.nanmedian(all_y_arr)),
        z=float(np.nanmedian(all_z_arr))
    )
    
    camera = dict(
        center=center,
        eye=dict(x=camera_eye[0], y=camera_eye[1], z=camera_eye[2])
    )
    
    layout = go.Layout(
        title=f"상위 {top_percent}% 3D Scatter (Log‑scale size): {', '.join(var_names)}",
        paper_bgcolor='white',
        plot_bgcolor='white',
        scene=dict(
            camera=camera,
            xaxis=dict(
                title='X', backgroundcolor='white',
                gridcolor='lightgray', zerolinecolor='lightgray'
            ),
            yaxis=dict(
                title='Y', backgroundcolor='white',
                gridcolor='lightgray', zerolinecolor='lightgray'
            ),
            zaxis=dict(
                title='bias_mV', backgroundcolor='white',
                gridcolor='lightgray', zerolinecolor='lightgray'
            )
        )
    )
    
    fig = go.Figure(data=traces, layout=layout)
    fig.show()



# -

# 사용 예시:
plot_multi_top_percent_3d_logscale(
    ds_opt2,
    ['cluster_umap_HDBSCAN1_L0', 'cluster_umap_HDBSCAN1_L1',
     'cluster_umap_HDBSCAN1_L2', 'cluster_umap_HDBSCAN1_L3',
     'cluster_umap_HDBSCAN1_L4',],# 'cluster_umap_HDBSCAN1_L5'],
    top_percent=1,
    opacity=1,
    max_size=10,
    camera_eye=(1.5, 1.5, 0.5)
)


# +
import numpy as np
import plotly.graph_objects as go

def plot_multi_size_by_value(
    ds,
    var_names,
    top_percent=20,
    opacity=0.8,
    max_size=5,
    camera_eye=(1.25, 1.25, 1.25)
):
    """
    Plot the top X% of points for multiple variables in a 3D scatter,
    with marker size proportional to the actual data values.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing DataArrays to plot.
    var_names : list of str
        Names of the DataArrays (e.g., ['cluster_umap_HDBSCAN1_L0', ..., 'cluster_umap_HDBSCAN1_L5']).
    top_percent : float, optional
        Percentage of highest values to include (0 < top_percent <= 100).
    opacity : float, optional
        Marker opacity between 0.0 (transparent) and 1.0 (opaque).
    max_size : float, optional
        Maximum marker size.
    camera_eye : tuple of float, optional
        Initial 3D camera eye position (x, y, z).
    """
    # Define the first six colors from the Tab10 palette
    tab10_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c',
        '#d62728', '#9467bd', '#8c564b'
    ]

    traces = []
    xs, ys, zs = [], [], []

    for idx, var in enumerate(var_names):
        data = ds[var]
        thresh = np.nanpercentile(data.values, 100 - top_percent)
        mask = data.values >= thresh

        # Extract coordinates and values
        idxs = np.where(mask)
        coords = {dim: data.coords[dim].values[i] for dim, i in zip(data.dims, idxs)}
        x, y, z = coords['X'], coords['Y'], coords['bias_mV']
        vals = data.values[mask]

        xs.append(x); ys.append(y); zs.append(z)

        # Map actual values to marker sizes (linear scale)
        vmin, vmax = np.nanmin(data.values), np.nanmax(data.values)
        rel = (vals - vmin) / (vmax - vmin + 1e-12)
        sizes = rel * max_size

        # Use a single color per variable
        color = tab10_colors[idx % len(tab10_colors)]

        traces.append(
            go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=color,
                    opacity=opacity,
                    line=dict(width=0)
                ),
                name=var
            )
        )

    # Compute global center for camera
    all_x = np.concatenate(xs)
    all_y = np.concatenate(ys)
    all_z = np.concatenate(zs)
    center = dict(
        x=float(np.nanmedian(all_x)),
        y=float(np.nanmedian(all_y)),
        z=float(np.nanmedian(all_z))
    )

    camera = dict(
        center=center,
        eye=dict(x=camera_eye[0], y=camera_eye[1], z=camera_eye[2])
    )

    layout = go.Layout(
        title=f"Top {top_percent}% 3D Scatter",
        paper_bgcolor='white',
        plot_bgcolor='white',
        scene=dict(
            camera=camera,
            xaxis=dict(title='X', backgroundcolor='white',
                       gridcolor='lightgray', zerolinecolor='lightgray'),
            yaxis=dict(title='Y', backgroundcolor='white',
                       gridcolor='lightgray', zerolinecolor='lightgray'),
            zaxis=dict(title='Bias (mV)', backgroundcolor='white',
                       gridcolor='lightgray', zerolinecolor='lightgray'),
        )
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.show()

# Example usage:
# plot_multi_size_by_value(
#     ds_opt2,
#     ['cluster_umap_HDBSCAN1_L0', 'cluster_umap_HDBSCAN1_L1',
#      'cluster_umap_HDBSCAN1_L2', 'cluster_umap_HDBSCAN1_L3',
#      'cluster_umap_HDBSCAN1_L4', 'cluster_umap_HDBSCAN1_L5'],
#     top_percent=1,
#     opacity=0.6,
#     max_size=5,
#     camera_eye=(1.5, 1.5, 1.5)
# )



# -

plot_multi_size_by_value(
    ds_opt2,
    ['cluster_umap_HDBSCAN1_L0', 'cluster_umap_HDBSCAN1_L1',
     'cluster_umap_HDBSCAN1_L2', 'cluster_umap_HDBSCAN1_L3',
     'cluster_umap_HDBSCAN1_L4'],#, 'cluster_umap_HDBSCAN1_L5'],
    top_percent=1,
    opacity=0.1,
    max_size=20,
    camera_eye=(0.5, 0.5, 0)
)


# +
import numpy as np
import plotly.graph_objects as go

def plot_cluster_surfaces(
    ds,
    var_names,
    n_slices=7,
    percentile_low=2,
    percentile_high=98,
    sigma_factor=None,
    opacity=0.5,
    colorscale='Viridis',
    highlight_bias_mV=None,
    camera_eye=None,
    camera_center=None,
    camera_up=None
):
    """
    Render 3D stacked surfaces for each variable in a Dataset.

    For each variable in `var_names`, this function extracts evenly spaced
    bias slices and plots them as semi-transparent surfaces in a 3D scene.
    A robust color range is computed from the specified percentiles.
    If `sigma_factor` is provided, opacity follows a Gaussian function of bias;
    otherwise, a uniform opacity (passed via the `opacity` parameter) is used.
    Additionally, if `highlight_bias_mV` is set, the slice closest to that value
    will be drawn with full opacity (1.0).

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing 3D DataArrays over dimensions (Y, X, bias_mV).
    var_names : list of str
        List of variable names in `ds` to visualize.
    n_slices : int, optional
        Number of bias levels (slices) to sample per variable (default: 7).
    percentile_low : float, optional
        Lower percentile for color scaling (default: 2).
    percentile_high : float, optional
        Upper percentile for color scaling (default: 98).
    sigma_factor : float or None, optional
        If float: defines Gaussian opacity via
            sigma = max(|bias|) / sigma_factor
        and opacity = exp(-bias^2/(2*sigma^2)).
        If None: use uniform opacity as given by the `opacity` parameter
        (default: None).
    opacity : float, optional
        Uniform opacity to apply when `sigma_factor` is None (range 0-1,
        default: 0.5).
    colorscale : str, optional
        Name of Plotly colorscale for surfacecolor mapping (default: 'Viridis').
    highlight_bias_mV : float or None, optional
        If set, the slice with bias closest to this value will be rendered
        with full opacity (1.0). Other slices use their computed opacity.
        Default is None (no highlight).
    camera_eye : tuple of float or None, optional
        Coordinates (x, y, z) of the camera 'eye' position. If None,
        the default Plotly camera eye is used (default: None).
    camera_center : dict or None, optional
        Dictionary with keys 'x', 'y', 'z' defining the camera's look-at
        center. If None, the default center is used (default: None).
    camera_up : dict or None, optional
        Dictionary with keys 'x', 'y', 'z' defining the camera's up
        direction vector. If None, the default up vector is used (default: None).

    Notes
    -----
    - The bias coordinate must be present as 'bias_mV' in each variable.
    - Color limits (cmin, cmax) are computed per-variable but shared across
      all slices of that variable.
    - The first slice will display the colorbar legend.
    - Highlighted slice takes precedence over uniform or Gaussian opacity.
    """
    # Extract bias coordinate values
    bias = ds[var_names[0]].coords['bias_mV'].values
    # Compute indices for evenly spaced slices
    slice_idxs = np.linspace(0, len(bias) - 1, n_slices, dtype=int)

    # Determine which slice to highlight, if requested
    if highlight_bias_mV is not None:
        # compute absolute difference for each slice index
        diffs = [abs(bias[idx] - highlight_bias_mV) for idx in slice_idxs]
        highlight_idx = slice_idxs[int(np.argmin(diffs))]
    else:
        highlight_idx = None

    # Iterate over each variable to plot
    for var in var_names:
        data = ds[var].values  # shape: (Y, X, bias)
        # Compute robust color scaling limits
        p_low, p_high = np.nanpercentile(data, [percentile_low, percentile_high])

        # Precompute sigma if Gaussian opacity is requested
        if sigma_factor is not None:
            max_bias = float(np.max(np.abs(bias)))
            sigma = max_bias / sigma_factor

        # Initialize Plotly figure
        fig = go.Figure()

        # Add surface for each selected bias slice
        for idx in slice_idxs:
            z0 = float(bias[idx])
            slice_img = data[:, :, idx]

            # Determine opacity based on priority:
            # 1. Highlight slice -> full opacity
            # 2. Gaussian opacity if sigma_factor given
            # 3. Uniform opacity otherwise
            if highlight_idx is not None and idx == highlight_idx:
                slice_opacity = 1.0
            elif sigma_factor is not None:
                slice_opacity = float(np.exp(-(z0 ** 2) / (2 * sigma ** 2)))
            else:
                slice_opacity = opacity

            fig.add_trace(
                go.Surface(
                    x=ds[var].coords['X'].values,
                    y=ds[var].coords['Y'].values,
                    z=np.full_like(slice_img, z0),
                    surfacecolor=slice_img,
                    cmin=p_low,
                    cmax=p_high,
                    colorscale=colorscale,
                    opacity=slice_opacity,
                    showscale=bool(idx == slice_idxs[0])
                )
            )

        # Configure layout and camera settings
        layout_update = {
            'title': f"{var} Stack",
            'scene': {
                'xaxis_title': 'X',
                'yaxis_title': 'Y',
                'zaxis_title': 'bias_mV',
                'aspectmode': 'auto'
            },
            'margin': {'l': 0, 'r': 0, 'b': 0, 't': 30}
        }

        camera_dict = {}
        if camera_eye is not None:
            camera_dict['eye'] = {'x': camera_eye[0], 'y': camera_eye[1], 'z': camera_eye[2]}
        if camera_center is not None:
            camera_dict['center'] = camera_center
        if camera_up is not None:
            camera_dict['up'] = camera_up
        if camera_dict:
            layout_update['scene']['camera'] = camera_dict

        fig.update_layout(**layout_update)
        fig.show()



# -

plot_cluster_surfaces(
    ds_opt2,
    #['LDOS', 'cluster_umap_HDBSCAN1_L1'],
    #['LDOS'],#, 'cluster_umap_HDBSCAN1_L1'],
    ['cluster_umap_HDBSCAN1_L1'],#, 'cluster_umap_HDBSCAN1_L1'],
    n_slices=7,
    sigma_factor= None,
    opacity =0.2,
    highlight_bias_mV=0, # hilight bias_mV =0 
    percentile_low=2,
    percentile_high=99.8,
    #colorscale='Gray_r',
    colorscale='viridis',
    camera_eye=(1.4, 1.4, 1.2),
    camera_center={'x':0.5, 'y':0.5, 'z':0.5},
    camera_up={'x':0, 'y':0, 'z':0.3}
)



# ## volume plot view 

# +
import numpy as np
import xarray as xr
import plotly.graph_objects as go
from plotly.graph_objs import Volume, Surface
from ipywidgets import IntSlider, FloatSlider, HTML, Label, HBox, VBox
from IPython.display import display

def plot_interactive_volume_and_three_slices(
    ds: xr.Dataset,
    var_name: str,
    colorscale: str = 'Viridis',
    init_opacity: float = 0.5
) -> None:
    """
    Render a 3D volume plus three orthogonal slice planes (X, Y, Bias)
    with full interactive controls:
      - Independent opacity sliders for volume and each slice
      - Percentile low/high sliders per slice to adjust contrast
      - Index sliders per slice, showing actual coordinate values
      - All controls laid out in a compact table at the top
    """
    # 1) Extract data array and coordinates
    data = ds[var_name].values                   # shape (Y, X, B)
    Y = ds[var_name].coords['Y'].values
    X = ds[var_name].coords['X'].values
    B = ds[var_name].coords['bias_mV'].values

    # 2) Prepare flattened values for volume rendering (replace NaN)
    flat_vals = np.nan_to_num(data.flatten(), nan=0.0)

    # 3) Helper to create a slice Surface trace
    def make_slice(axis: str, idx: int, opacity: float,
                   p_low: float, p_high: float) -> Surface:
        if axis == 'x':
            xi, vals = X[idx], data[:, idx, :].T
            xx = np.full_like(vals, xi)
            yy = np.tile(Y, (len(B), 1))
            zz = np.tile(B[:, None], (1, len(Y)))
        elif axis == 'y':
            yi, vals = Y[idx], data[idx, :, :].T
            yy = np.full_like(vals, yi)
            xx = np.tile(X, (len(B), 1))
            zz = np.tile(B[:, None], (1, len(X)))
        else:  # bias plane
            bi, vals = B[idx], data[:, :, idx]
            zz = np.full_like(vals, bi)
            xx, yy = np.meshgrid(X, Y)

        return Surface(
            x=xx, y=yy, z=zz,
            surfacecolor=np.nan_to_num(vals, nan=0.0),
            cmin=np.nanpercentile(vals, p_low),
            cmax=np.nanpercentile(vals, p_high),
            colorscale=colorscale,
            showscale=False,
            opacity=opacity
        )

    # 4) Initial parameters
    ix0, iy0, ib0 = len(X)//2, len(Y)//2, len(B)//2
    vol_op0 = 0.2            # initial volume opacity
    sl_op0 = init_opacity    # initial slice opacity
    pl0, ph0 = 2.0, 99.5     # initial percentile range

    # 5) Create figure with volume + 3 slices
    fig = go.FigureWidget(data=[
        Volume(
            x=np.repeat(X, len(Y)*len(B)),
            y=np.tile(np.repeat(Y, len(X)), len(B)),
            z=np.tile(B, len(X)*len(Y)),
            value=flat_vals,
            opacity=vol_op0,
            opacityscale=[
                [0.00, 0.00],
                [0.10, 0.02],
                [0.50, 0.10],
                [1.00, 0.30],
            ],
            isomin=np.nanpercentile(data, 2),
            isomax=np.nanpercentile(data, 98),
            caps=dict(x_show=False, y_show=False, z_show=False),
            colorscale=colorscale,
            showscale=False
        ),
        make_slice('x', ix0, sl_op0, pl0, ph0),
        make_slice('y', iy0, sl_op0, pl0, ph0),
        make_slice('bias', ib0, sl_op0, pl0, ph0),
    ])
    fig.update_layout(
        title=f"{var_name} Volume + 3 Slices",
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='bias_mV',
            aspectmode='auto'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    # 6) Create widgets
    # Volume opacity
    vol_op_slider = FloatSlider(value=vol_op0, min=0.0, max=1.0, step=0.05,
                                description='Vol', layout={'width':'150px'})

    # Slice index sliders
    ix_slider = IntSlider(value=ix0, min=0, max=len(X)-1, step=1,
                          layout={'width':'200px'})
    iy_slider = IntSlider(value=iy0, min=0, max=len(Y)-1, step=1,
                          layout={'width':'200px'})
    ib_slider = IntSlider(value=ib0, min=0, max=len(B)-1, step=1,
                          layout={'width':'200px'})

    # Coordinate labels
    ix_label = HTML(f"{X[ix0]:.3g}", layout={'width':'80px'})
    iy_label = HTML(f"{Y[iy0]:.3g}", layout={'width':'80px'})
    ib_label = HTML(f"{B[ib0]:.3g}", layout={'width':'80px'})

    # Slice opacity sliders
    opx_slider = FloatSlider(value=sl_op0, min=0.0, max=1.0, step=0.05,
                             layout={'width':'150px'})
    opy_slider = FloatSlider(value=sl_op0, min=0.0, max=1.0, step=0.05,
                             layout={'width':'150px'})
    opb_slider = FloatSlider(value=sl_op0, min=0.0, max=1.0, step=0.05,
                             layout={'width':'150px'})

    # Percentile range sliders
    plx_slider = FloatSlider(value=pl0, min=0.0, max=50.0, step=0.1,
                             layout={'width':'100px'})
    phx_slider = FloatSlider(value=ph0, min=50.0, max=100.0, step=0.1,
                             layout={'width':'100px'})
    ply_slider = FloatSlider(value=pl0, min=0.0, max=50.0, step=0.1,
                             layout={'width':'100px'})
    phy_slider = FloatSlider(value=ph0, min=50.0, max=100.0, step=0.1,
                             layout={'width':'100px'})
    plb_slider = FloatSlider(value=pl0, min=0.0, max=50.0, step=0.1,
                             layout={'width':'100px'})
    phb_slider = FloatSlider(value=ph0, min=50.0, max=100.0, step=0.1,
                             layout={'width':'100px'})

    # 7) Callback to update all traces
    def on_change(_):
        ix, iy, ib = ix_slider.value, iy_slider.value, ib_slider.value
        opv = vol_op_slider.value
        opx, opy, opb = opx_slider.value, opy_slider.value, opb_slider.value
        plx, phx = plx_slider.value, phx_slider.value
        ply, phy = ply_slider.value, phy_slider.value
        plb, phb = plb_slider.value, phb_slider.value

        # Update labels
        ix_label.value = f"{X[ix]:.3g}"
        iy_label.value = f"{Y[iy]:.3g}"
        ib_label.value = f"{B[ib]:.3g}"

        with fig.batch_update():
            # Volume
            fig.data[0].opacity = opv

            # X-slice
            sx = make_slice('x', ix, opx, plx, phx)
            fig.data[1].x = sx.x; fig.data[1].y = sx.y; fig.data[1].z = sx.z
            fig.data[1].surfacecolor = sx.surfacecolor; fig.data[1].opacity = opx

            # Y-slice
            sy = make_slice('y', iy, opy, ply, phy)
            fig.data[2].x = sy.x; fig.data[2].y = sy.y; fig.data[2].z = sy.z
            fig.data[2].surfacecolor = sy.surfacecolor; fig.data[2].opacity = opy

            # Bias-slice
            sb = make_slice('bias', ib, opb, plb, phb)
            fig.data[3].x = sb.x; fig.data[3].y = sb.y; fig.data[3].z = sb.z
            fig.data[3].surfacecolor = sb.surfacecolor; fig.data[3].opacity = opb

    # Register callback
    for w in (vol_op_slider,
              ix_slider, iy_slider, ib_slider,
              opx_slider, opy_slider, opb_slider,
              plx_slider, phx_slider,
              ply_slider, phy_slider,
              plb_slider, phb_slider):
        w.observe(on_change, names='value')

    # 8) Layout controls in table form
    header = HBox([
        Label('Axis',    layout={'width':'60px'}),
        Label('Index',   layout={'width':'200px'}),
        Label('Value',   layout={'width':'80px'}),
        Label('Opacity', layout={'width':'150px'}),
        Label('Low %',   layout={'width':'100px'}),
        Label('High %',  layout={'width':'100px'}),
    ])

    row_vol = HBox([ Label('Vol', layout={'width':'60px'}), vol_op_slider ])
    row_x   = HBox([ Label('X',   layout={'width':'60px'}), ix_slider, ix_label,
                     opx_slider, plx_slider, phx_slider ])
    row_y   = HBox([ Label('Y',   layout={'width':'60px'}), iy_slider, iy_label,
                     opy_slider, ply_slider, phy_slider ])
    row_b   = HBox([ Label('Bias',layout={'width':'60px'}), ib_slider, ib_label,
                     opb_slider, plb_slider, phb_slider ])

    controls = VBox([ header, row_vol, row_x, row_y, row_b ])

    # Display controls above the figure
    display(VBox([controls, fig]))



# +
plot_interactive_volume_and_three_slices(
    ds_opt2,
    var_name='cluster_umap_HDBSCAN1_L1',
    colorscale='Viridis',
    init_opacity=0.5

)


# +
plot_interactive_volume_and_three_slices(
    ds_opt2,
    var_name='LDOS',
    colorscale='Viridis',
    init_opacity=0.5

)


# +
import numpy as np
import xarray as xr
import plotly.graph_objects as go
from skimage import measure
from ipywidgets import FloatSlider, HBox, VBox, Label
from IPython.display import display

def plot_interactive_mesh_volume(
    ds: xr.Dataset,
    var_name: str,
    init_value: float = None,
    colorscale: str = 'Viridis'
) -> None:
    """
    Extract and display a single isosurface mesh from 3D data,
    mapping surface colors to the colormap based on vertex intensity,
    and using the true (X, Y, bias_mV) coordinates. Axes ranges
    are fixed to the full extent of the data grid.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the 3D DataArray.
    var_name : str
        Name of the DataArray (dims: Y, X, bias_mV).
    init_value : float, optional
        Initial threshold for the isosurface. Defaults to median of the data.
    colorscale : str, optional
        Plotly colorscale name for the mesh intensity.
    """
    # 1) Load data and coordinate arrays
    da = ds[var_name]
    vol = da.values
    Y = da.coords['Y'].values
    X = da.coords['X'].values
    B = da.coords['bias_mV'].values

    # 2) Prepare flattened data for range
    flat = np.nan_to_num(vol.flatten(), nan=0.0)
    if init_value is None:
        init_value = float(np.nanpercentile(flat, 50))

    # grid spacing and origin
    dy, dx, dz = Y[1] - Y[0], X[1] - X[0], B[1] - B[0]
    y0, x0, z0 = Y[0], X[0], B[0]

    # 3) Compute mesh for a threshold
    def compute_mesh(level: float):
        verts, faces, normals, values = measure.marching_cubes(
            vol, level=level, spacing=(dy, dx, dz)
        )
        # convert to real-world coordinates
        ys = y0 + verts[:, 0]
        xs = x0 + verts[:, 1]
        zs = z0 + verts[:, 2]
        i, j, k = faces.T
        return xs, ys, zs, i, j, k, values

    # 4) Initial mesh
    xs, ys, zs, i, j, k, vals = compute_mesh(init_value)

    # 5) Plotly Mesh3d trace
    mesh = go.Mesh3d(
        x=xs, y=ys, z=zs,
        i=i, j=j, k=k,
        intensity=vals,
        colorscale=colorscale,
        cmin=np.nanmin(flat), cmax=np.nanmax(flat),
        showscale=True,
        colorbar=dict(title=var_name),
        opacity=1.0,
        name='isosurface'
    )

    # 6) FigureWidget with fixed axis ranges
    fig = go.FigureWidget(data=[mesh])
    fig.update_layout(
        title=f"{var_name} Isosurface (value={init_value:.3g})",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='bias_mV',
            aspectmode='auto',
            xaxis=dict(range=[X.min(), X.max()]),
            yaxis=dict(range=[Y.min(), Y.max()]),
            zaxis=dict(range=[B.min(), B.max()]),
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # 7) Slider to pick value
    slider = FloatSlider(
        value=init_value,
        min=float(np.nanmin(flat)), max=float(np.nanmax(flat)),
        step=(np.nanmax(flat) - np.nanmin(flat)) / 200,
        description='value:',
        continuous_update=False,
        layout={'width': '500px'}
    )

    # 8) Callback updates mesh only
    def on_value_change(change):
        lvl = change['new']
        xs, ys, zs, i, j, k, vals = compute_mesh(lvl)
        with fig.batch_update():
            fig.data[0].x, fig.data[0].y, fig.data[0].z = xs, ys, zs
            fig.data[0].i, fig.data[0].j, fig.data[0].k = i, j, k
            fig.data[0].intensity = vals
            fig.layout.title.text = f"{var_name} Isosurface (value={lvl:.3g})"
            # axis ranges remain fixed

    slider.observe(on_value_change, names='value')

    # 9) Display controls + figure
    display(VBox([HBox([Label("value:"), slider]), fig]))



# -

plot_interactive_mesh_volume(ds_opt2, 'cluster_umap_HDBSCAN1_L1')

plot_interactive_mesh_volume(ds_opt2, 'LDOS')



ds_opt2

# +
import numpy as np
import xarray as xr
import plotly.graph_objects as go
from plotly.graph_objs import Volume, Surface
from ipywidgets import IntSlider, FloatSlider, Checkbox, HBox, VBox, Label
from IPython.display import display

def plot_overlay_volume_with_slices(
    ds: xr.Dataset,
    var_name: str,
    colorscale: str = 'Viridis',
    init_vol_opacity: float = 0.2,
    init_slice_opacity: float = 0.5
) -> None:
    """
    Render a 3D volume with three orthogonal slice planes overlaid,
    and provide controls to toggle each slice on/off and adjust opacity.

    Controls:
      - volume opacity slider
      - for each slice (X, Y, Bias):
          * index slider
          * visibility checkbox
          * opacity slider
    """
    # 1) Extract data and coords
    da = ds[var_name]
    data = da.values                   # shape (Y, X, B)
    Y = da.coords['Y'].values
    X = da.coords['X'].values
    B = da.coords['bias_mV'].values

    # 2) Flatten for volume
    flat = np.nan_to_num(data.flatten(), nan=0.0)

    # 3) Helper to build a slice Surface
    def make_slice(axis, idx, opacity):
        if axis == 'x':
            xi, vals = X[idx], data[:, idx, :].T
            xx = np.full_like(vals, xi)
            yy = np.tile(Y, (len(B),1))
            zz = np.tile(B[:,None], (1,len(Y)))
        elif axis == 'y':
            yi, vals = Y[idx], data[idx, :, :].T
            yy = np.full_like(vals, yi)
            xx = np.tile(X, (len(B),1))
            zz = np.tile(B[:,None], (1,len(X)))
        else:  # bias plane
            bi, vals = B[idx], data[:,:,idx]
            zz = np.full_like(vals, bi)
            xx, yy = np.meshgrid(X, Y)
        return Surface(
            x=xx, y=yy, z=zz,
            surfacecolor=np.nan_to_num(vals, nan=0.0),
            cmin=np.nanpercentile(data, 2),
            cmax=np.nanpercentile(data, 98),
            colorscale=colorscale,
            showscale=False,
            opacity=opacity
        )

    # 4) Initial indices
    ix0, iy0, ib0 = len(X)//2, len(Y)//2, len(B)//2

    # 5) Create the figure with volume + slices
    fig = go.FigureWidget(data=[
        Volume(
            x=np.repeat(X, len(Y)*len(B)),
            y=np.tile(np.repeat(Y, len(X)), len(B)),
            z=np.tile(B, len(X)*len(Y)),
            value=flat,
            opacity=init_vol_opacity,
            opacityscale=[
                [0.00, 0.00],
                [0.10, 0.02],
                [0.50, 0.10],
                [1.00, 0.30],
            ],
            isomin=np.nanpercentile(data, 2),
            isomax=np.nanpercentile(data, 98),
            caps=dict(x_show=False, y_show=False, z_show=False),
            colorscale=colorscale,
            showscale=False
        ),
        make_slice('x', ix0, init_slice_opacity),
        make_slice('y', iy0, init_slice_opacity),
        make_slice('bias', ib0, init_slice_opacity),
    ])
    fig.update_layout(
        title=f"{var_name}: Volume + Overlay Slices",
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='bias_mV',
            aspectmode='auto',
            xaxis=dict(range=[X.min(), X.max()]),
            yaxis=dict(range=[Y.min(), Y.max()]),
            zaxis=dict(range=[B.min(), B.max()]),
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    # 6) Widgets
    vol_op = FloatSlider(value=init_vol_opacity, min=0, max=1, step=0.05,
                         description='Vol Op:', layout={'width':'180px'})

    ix_idx = IntSlider(value=ix0, min=0, max=len(X)-1, step=1,
                       description='X idx:', layout={'width':'200px'})
    iy_idx = IntSlider(value=iy0, min=0, max=len(Y)-1, step=1,
                       description='Y idx:', layout={'width':'200px'})
    ib_idx = IntSlider(value=ib0, min=0, max=len(B)-1, step=1,
                       description='B idx:', layout={'width':'200px'})

    chk_x = Checkbox(value=True, description='X slice')
    chk_y = Checkbox(value=True, description='Y slice')
    chk_b = Checkbox(value=True, description='B slice')

    op_x = FloatSlider(value=init_slice_opacity, min=0, max=1, step=0.05,
                       description='X op:', layout={'width':'180px'})
    op_y = FloatSlider(value=init_slice_opacity, min=0, max=1, step=0.05,
                       description='Y op:', layout={'width':'180px'})
    op_b = FloatSlider(value=init_slice_opacity, min=0, max=1, step=0.05,
                       description='B op:', layout={'width':'180px'})

    # 7) Callback to update figure
    def update(_):
        with fig.batch_update():
            fig.data[0].opacity = vol_op.value

            # X slice
            if chk_x.value:
                sx = make_slice('x', ix_idx.value, op_x.value)
                fig.data[1].x, fig.data[1].y, fig.data[1].z = sx.x, sx.y, sx.z
                fig.data[1].surfacecolor = sx.surfacecolor
                fig.data[1].opacity = op_x.value
            else:
                fig.data[1].opacity = 0

            # Y slice
            if chk_y.value:
                sy = make_slice('y', iy_idx.value, op_y.value)
                fig.data[2].x, fig.data[2].y, fig.data[2].z = sy.x, sy.y, sy.z
                fig.data[2].surfacecolor = sy.surfacecolor
                fig.data[2].opacity = op_y.value
            else:
                fig.data[2].opacity = 0

            # Bias slice
            if chk_b.value:
                sb = make_slice('bias', ib_idx.value, op_b.value)
                fig.data[3].x, fig.data[3].y, fig.data[3].z = sb.x, sb.y, sb.z
                fig.data[3].surfacecolor = sb.surfacecolor
                fig.data[3].opacity = op_b.value
            else:
                fig.data[3].opacity = 0

    # 8) Register observers
    for w in (vol_op,
              ix_idx, iy_idx, ib_idx,
              chk_x, chk_y, chk_b,
              op_x, op_y, op_b):
        w.observe(update, names='value')

    # 9) Layout and display
    controls = VBox([
        HBox([vol_op]),
        HBox([ix_idx, chk_x, op_x]),
        HBox([iy_idx, chk_y, op_y]),
        HBox([ib_idx, chk_b, op_b])
    ])
    display(VBox([controls, fig]))



# -

plot_overlay_volume_with_slices(
    ds_opt2,
    var_name='cluster_umap_HDBSCAN1_L0',
    colorscale='Viridis',
    init_vol_opacity=0.2,
    init_slice_opacity=0.5
)



# +
import numpy as np

# Create a 3D grid of scalar values (e.g., a 3D Gaussian blob)
x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)
z = np.linspace(-2, 2, 50)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
values = np.exp(-(X**2 + Y**2 + Z**2))  # 3D Gaussian

# Save data for visualization
import plotly.io as pio
import plotly.graph_objects as go

# 3D volume rendering with Plotly
volume = go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=values.flatten(),
    isomin=0.1,
    isomax=1.0,
    opacity=0.1,  # overall opacity
    surface_count=15,  # number of isosurfaces
    colorscale='Viridis',
)

# 3D isosurface rendering
isosurface = go.Isosurface(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=values.flatten(),
    isomin=0.5,
    isomax=0.5,
    surface_count=1,
    caps=dict(x_show=False, y_show=False, z_show=False),
    colorscale='Reds',
    opacity=0.6
)

fig = go.Figure(data=[volume, isosurface])
fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
    ),
    title='3D Volume and Isosurface Visualization'
)

# -





ds_opt2

# +
ds_opt2.LDOS.sel(bias_mV=0).plot()

#ds_opt2.cluster_umap_HDBSCAN1_L0.sel(bias_mV=0).plot()

#cluster_umap_HDBSCAN1_L1

# +
import numpy as np
import matplotlib.pyplot as plt

# 1) 대상 DataArray 추출
#da = ds_opt2.LDOS.sel(bias_mV=0)#
#da = ds_opt2.cluster_umap_HDBSCAN1_L0.sel(bias_mV=0)
da = ds_opt2.cluster_umap_HDBSCAN1_L1.sel(bias_mV=0)
#da = ds_opt2.cluster_umap_HDBSCAN1_L2.sel(bias_mV=0)
#da = ds_opt2.cluster_umap_HDBSCAN1_L3.sel(bias_mV=0)




# 2) 로버스트 컬러 범위 계산 (2, 98 퍼센타일)
p0, p98 = np.nanpercentile(ds_opt2.LDOS.sel(bias_mV=0).values, [0, 100])

# 3) 플롯
fig, ax = plt.subplots(figsize=(6,5))
da.plot(
    ax=ax,
    cmap='viridis',  # 원하시는 colormap
    vmin=p0,
    vmax=p98
)
ax.set_title('L1 @ bias=0 mV')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()

# -

ds_opt2.ZB_mask.notnull().plot()

GS_LDOS_2T_003.where(ds_opt2.ZB_mask.notnull())


GS_LDOS_2T_003= xr.open_dataset('GS_LDOS_2T_003.nc')
#GS_LDOS_2T_003
#GS_LDOS_2T_003.where(ds_opt2.ZB_mask.notnull())
#GS_LDOS_2T_003.where(ds_opt2.ZB_mask.isnull())

updated_GS_LDOS_0T002_N_2T003 = xr.open_dataset('updated_GS_LDOS_0T002_N_2T003.nc')
updated_GS_LDOS_0T002_N_2T003

#updated_GS_LDOS_0T002_N_2T003_th =  threshold_isodata_xr(updated_GS_LDOS_0T002_N_2T003.sel(bias_mV=0))
updated_GS_LDOS_0T002_N_2T003_th =  threshold_isodata_xr(updated_GS_LDOS_0T002_N_2T003.sel(bias_mV=0))

#updated_GS_LDOS_0T002_N_2T003_th.LDOS.notnull().plot()
updated_GS_LDOS_0T002_N_2T003_th.LDOS.isnull().plot()

# +
import numpy as np
import pandas as pd

# 1) Boolean DataArray for non-null positions at bias=0
da_mask = updated_GS_LDOS_0T002_N_2T003_th.LDOS.notnull()  # dims (Y, X)

# 2) Find indices where mask is True
y_idx, x_idx = np.where(da_mask.values)

# 3) Retrieve real-world coordinates
x_coords = da_mask.coords['X'].values[x_idx]
y_coords = da_mask.coords['Y'].values[y_idx]

# 4) Build DataFrame
df_points = pd.DataFrame({
    'X': x_coords,
    'Y': y_coords
})

print(df_points.head())
# If you want to reset the index:
# df_points = df_points.reset_index(drop=True)

# -

df_0T_peaks_points  = df_points 
df_0T_peaks_points
## use the cropped area peaks 
# defect position from cropped area 


# +
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# — 1) df_0T_peaks_points 예시 DataFrame —
# — 2) ZB_mask의 notnull 위치 인덱스와 좌표 추출 —
mask = ds_opt2.ZB_mask.notnull()         # (Y, X) boolean
y_idx, x_idx = np.where(mask.values)     # True인 인덱스
X_coords = ds_opt2.X.values[x_idx]       # 해당 열마다 X 좌표
Y_coords = ds_opt2.Y.values[y_idx]       # 해당 행마다 Y 좌표

# — 3) KD-Tree 구성 (peak points) —
tree = cKDTree(df_0T_peaks_points[['X','Y']].values)

# — 4) 각 ZB_mask 점에서 가장 가까운 peak까지 거리 계산 —
points = np.column_stack([X_coords, Y_coords])
distances, _ = tree.query(points, k=1)

# — 5) 거리 값을 원래 그리드 형태로 되돌리기 —
dist_map = np.full(mask.shape, np.nan, dtype=float)
dist_map[y_idx, x_idx] = distances

# — 6) xarray.DataArray로 변환 —
dist_da = xr.DataArray(
    dist_map,
    coords={'Y': ds_opt2.Y, 'X': ds_opt2.X},
    dims=['Y','X'],
    name='dist_to_peaks'
)

# — 7) 거리 지도 플롯 —
plt.figure(figsize=(6,5))
im = dist_da.plot(
    cmap='viridis',
    add_colorbar=True,
    cbar_kwargs={'label': 'Distance (m)'}
)
plt.title('Distance to Nearest Peak (only ZB_mask locations)')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.tight_layout()

# ◀ 여기서 SVG로 저장
plt.savefig('distance_to_0T_preexisting_peaks.svg', format='svg', dpi=300, bbox_inches='tight')

plt.show()

# -

grid_LDOS_SnD_pks_0T002_WholeRange.nc

grid_0T_002_fit = xr.open_dataset('grid_LDOS_SnD_pks_0T002_WholeRange.nc')
grid_0T_002_fit

grid_0T_002_fit.LDOS.sel(bias_mV=0).plot()

#grid_0T_002_fit_th =  threshold_isodata_xr(grid_0T_002_fit[['LDOS']].sel(bias_mV=0))
grid_0T_002_fit_th =  threshold_otsu_xr(grid_0T_002_fit[['LDOS']].sel(bias_mV=0))

grid_0T_002_fit_th.LDOS.notnull().plot()

# +
### another peak point list

# +
import numpy as np
import pandas as pd

# 1) Boolean DataArray for non-null positions at bias=0
da_mask = grid_0T_002_fit_th.LDOS.notnull()  # dims (Y, X)

# 2) Find indices where mask is True
y_idx, x_idx = np.where(da_mask.values)

# 3) Retrieve real-world coordinates
x_coords = da_mask.coords['X'].values[x_idx]
y_coords = da_mask.coords['Y'].values[y_idx]

# 4) Build DataFrame
df_points = pd.DataFrame({
    'X': x_coords,
    'Y': y_coords
})

print(df_points.head())
# If you want to reset the index:
# df_points = df_points.reset_index(drop=True)

# -

df_0T_peaks_points_all  = df_points 

# ## use all pre exsiting peak points. 

# +
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# — 1) df_0T_peaks_points 예시 DataFrame —
# — 2) ZB_mask의 notnull 위치 인덱스와 좌표 추출 —
mask = ds_opt2.ZB_mask.notnull()         # (Y, X) boolean
y_idx, x_idx = np.where(mask.values)     # True인 인덱스
X_coords = ds_opt2.X.values[x_idx]       # 해당 열마다 X 좌표
Y_coords = ds_opt2.Y.values[y_idx]       # 해당 행마다 Y 좌표

# — 3) KD-Tree 구성 (peak points) —
tree = cKDTree(df_0T_peaks_points_all[['X','Y']].values)

# — 4) 각 ZB_mask 점에서 가장 가까운 peak까지 거리 계산 —
points = np.column_stack([X_coords, Y_coords])
distances, _ = tree.query(points, k=1)

# — 5) 거리 값을 원래 그리드 형태로 되돌리기 —
dist_map = np.full(mask.shape, np.nan, dtype=float)
dist_map[y_idx, x_idx] = distances

# — 6) xarray.DataArray로 변환 —
dist_da = xr.DataArray(
    dist_map,
    coords={'Y': ds_opt2.Y, 'X': ds_opt2.X},
    dims=['Y','X'],
    name='dist_to_peaks'
)

# — 7) 거리 지도 플롯 —
plt.figure(figsize=(6,5))
im = dist_da.plot(
    cmap='viridis',
    add_colorbar=True,
    cbar_kwargs={'label': 'Distance (m)'}
)
plt.title('Distance to Nearest Peak (only ZB_mask locations)')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.tight_layout()

# ◀ 여기서 SVG로 저장
plt.savefig('distance_to_0T_preexisting_peaks.svg', format='svg', dpi=300, bbox_inches='tight')

plt.show()

# -

ds_opt2.cluster_umap_HDBSCAN1_L1.sel(bias_mV=0).plot()


dist_da

ds_opt2.cluster_umap_HDBSCAN1_L1.sel(bias_mV=0)







# +
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1) Prepare region mask (True = superconducting, False = non-zero-gap)
mask2d = ds_opt2.ZB_mask.isnull()  # dims (Y, X)

# 2) Build long-form DataFrame with one row per (Y, X, bias_mV)
records = []
bias_vals = GS_LDOS_2T_003.bias_mV.values

for region_label, region_mask in [
    
    ('superconducting', mask2d), ('nonzero in-gap states', ~mask2d),
    
]:
    ys, xs = np.where(region_mask.values)
    for y, x in zip(ys, xs):
        # extract LDOS vs bias at this (y, x)
        ldos_curve = GS_LDOS_2T_003.LDOS.values[y, x, :]
        for b, v in zip(bias_vals, ldos_curve):
            if not np.isnan(v):  # skip nan values
                records.append({'bias_mV': b, 'LDOS': v, 'region': region_label})

df = pd.DataFrame.from_records(records)

# 3) Plot with seaborn lineplot, showing 95% CI by default
plt.figure(figsize=(4, 3))
sns.lineplot(
    data=df,
    x='bias_mV',
    y='LDOS',
    hue='region',
    estimator='mean',
    #errorbar=('ci', 95),
    errorbar='sd',
    palette='tab10'
)
plt.xlabel('Bias (mV)')
plt.ylabel('LDOS')
plt.title('')
# legend를 우하단으로 이동
plt.legend(loc='upper right', title='')

plt.tight_layout()

plt.savefig('SC_INgap_STS_plot.svg', format='svg', dpi=300, bbox_inches='tight')

plt.show()
# -

ds_opt2.best_fit.sel(bias_mV=0).plot()

ds_opt2.LDOS.where(ds_opt2.ZB_mask.notnull()).sel(bias_mV=0).plot()
#ds_opt2.ZB_mask.plot()

# +
import numpy as np
import matplotlib.pyplot as plt

# 1) Compute robust vmin/vmax from the full LDOS volume
da_ldos = ds_opt2.LDOS  # shape (Y, X, bias_mV)
p0, p99 = np.nanpercentile(da_ldos.values, [0, 99])

# 2) Prepare list of DataArrays to plot (best_fit and L1)
maps = [
    #('Best Fit', ds_opt2.best_fit),
    ('Original LDOS', ds_opt2.LDOS.where(ds_opt2.ZB_mask.notnull())),
    ('ZBP cluster', ds_opt2.cluster_umap_HDBSCAN1_L1)
]

# 3) Create 1×2 grid of subplots with equal aspect ratio
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

for ax, (label, da) in zip(axes, maps):
    # select bias = 0 slice
    img = da.sel(bias_mV=0)
    pcm = img.plot(
        ax=ax,
        cmap='viridis',
        vmin=p0,
        vmax=p99,
        add_colorbar=False
    )
    ax.set_title(f"{label} @ 0 mV")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')  # 유지: x축과 y축 동일 비율

# add a single colorbar alongside both plots
cbar = fig.colorbar(pcm, ax=axes.tolist(), orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label('Value')

plt.show()


# +
import numpy as np
import matplotlib.pyplot as plt

# 1) extract bias=0 slices
orig    = ds_opt2.LDOS.sel(bias_mV=0)
cluster = ds_opt2.cluster_umap_HDBSCAN1_L1.sel(bias_mV=0)

# 2) compute fraction map (cluster contribution / original)
#    mask out invalid positions to avoid division by zero or NaN
fraction = (cluster / orig).where(np.isfinite(cluster) & np.isfinite(orig))

# 3) robust color scales for plotting
p0, p99   = np.nanpercentile(ds_opt2.LDOS.values, [0, 99])
c0, c99   = np.nanpercentile(cluster.values, [0, 99])
fmax      = np.nanpercentile(fraction.values, 99)

# 4) prepare maps and their plotting parameters
maps = [
    ('Original LDOS',      orig,     'viridis', p0,  p99),
    ('ZBP Cluster',        cluster,  'viridis', c0,  c99),
    ('Cluster Fraction',   fraction, 'cividis',    0.0, fmax),
]

# 5) create a 1×3 grid
fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

for ax, (title, da, cmap, vmin, vmax) in zip(axes, maps):
    pcm = da.plot(
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False
    )
    ax.set_title(f"{title} @ 0 mV")
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')

# 6) colorbar for the fraction map (rightmost panel)
cbar = fig.colorbar(
    pcm, ax=axes[-1],
    orientation='vertical',
    fraction=0.046,
    pad=0.04
)
cbar.set_label('Fraction')

# 7) save as SVG
plt.savefig(
    'cluster_fraction_map.svg',
    format='svg',
    dpi=300,
    bbox_inches='tight'
)

plt.show()


# +
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# 1) Prepare the three existing maps at bias = 0 mV
orig     = ds_opt2.LDOS.where(ds_opt2.ZB_mask.notnull()).sel(bias_mV=0)
cluster  = ds_opt2.cluster_umap_HDBSCAN1_L1.sel(bias_mV=0)
fraction = (cluster / orig).where(np.isfinite(cluster) & np.isfinite(orig))

# 2) Compute robust color limits
p0,   p99 = np.nanpercentile(ds_opt2.LDOS.values,           [0,  99])
c0,   c99 = np.nanpercentile(cluster.values,                [0,  99])
fmax      = np.nanpercentile(fraction.values,               99)

# 3) Build the distance‐to‐peaks map (only at ZB_mask locations)
mask     = ds_opt2.ZB_mask.notnull()
y_idx, x_idx = np.where(mask.values)
Xc       = ds_opt2.X.values[x_idx]
Yc       = ds_opt2.Y.values[y_idx]

# your DataFrame of peak points (must exist already)
# df_0T_peaks_points = pd.DataFrame({'X': [...], 'Y': [...]})

tree     = cKDTree(df_0T_peaks_points[['X','Y']].values)
points   = np.column_stack([Xc, Yc])
distances, _ = tree.query(points, k=1)

dist_map = np.full(mask.shape, np.nan, dtype=float)
dist_map[y_idx, x_idx] = distances

dist_da  = xr.DataArray(
    dist_map,
    coords={'Y': ds_opt2.Y, 'X': ds_opt2.X},
    dims=['Y','X'],
    name='dist_to_peaks'
)
dmax     = np.nanpercentile(dist_da.values, 99)

# 4) Pack into panel list: (title, DataArray, cmap, vmin, vmax)
panels = [
    ('Original LDOS',       orig,     'viridis', p0,   p99),
    ('ZBP Cluster',         cluster,  'viridis', c0,   c99),
    ('Cluster Fraction',    fraction, 'cividis',    0.0,  1.0),
    ('Distance to Peaks',   dist_da,  'viridis', 0.0,  dmax),
]

# 5) Create 1×4 subplot and plot each
fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)

for ax, (title, da, cmap, vmin, vmax) in zip(axes, panels):
    pcm = da.plot(
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False
    )
    ax.set_title(f"{title} @ 0 mV")
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    # individual colorbar for each panel
    fig.colorbar(pcm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

# 6) Save combined figure as SVG
plt.savefig(
    'combined_4_panels.svg',
    format='svg',
    dpi=300,
    bbox_inches='tight'
)

plt.show()


# +
import numpy as np
from scipy.stats import pearsonr

# 1) 두 DataArray 준비
#    fraction: cluster 기여도 분율 (세 번째 패널)
#    dist_da : ZB_mask 위치에서 피크까지의 거리 (네 번째 패널)
fraction_da = fraction              # xarray.DataArray, dims=('Y','X')
distance_da = dist_da               # xarray.DataArray, dims=('Y','X')

# 2) NaN이 아닌 공통 마스크 생성
mask = np.isfinite(fraction_da.values) & np.isfinite(distance_da.values)

# 3) 1차원 배열로 변환
x = fraction_da.values[mask]
y = distance_da.values[mask]

# 4) Pearson 상관계수 및 p-value 계산
r, p = pearsonr(x, y)
print(f"Pearson correlation coefficient: r = {r:.4f}, p-value = {p:.3e}")
corr_da = xr.corr(fraction_da, distance_da, dim=('Y','X'))
print(f"Pearson r (xarray.corr): {float(corr_da.values):.4f}")

# -

ds_opt2.LDOS

# +
import numpy as np
from mayavi import mlab

# Assuming ds_opt2 is already loaded in the namespace
# Extract the LDOS data and coordinates
data = ds_opt2.LDOS.values  # shape (Y, X, bias)
Y = ds_opt2.Y.values        # length Y
X = ds_opt2.X.values        # length X
Z = ds_opt2.bias_mV.values  # length bias

# Create a meshgrid of coordinates matching the data shape
# indexing='xy' ensures data[y, x, z] corresponds to xx[y,x,z], yy[y,x,z], zz[y,x,z]
xx, yy, zz = np.meshgrid(X, Y, Z, indexing='xy')

# Create a Mayavi scalar field source
src = mlab.pipeline.scalar_field(xx, yy, zz, data)

# Initialize the figure with a white background
mlab.figure(bgcolor=(1, 1, 1), size=(800, 600))

# Volume render the full dataset as semi-transparent
vol = mlab.pipeline.volume(src)
# Adjust the opacity unit distance to control overall transparency
vol._volume_property.set_scalar_opacity_unit_distance(0.1)

# Add an image plane at bias = 0 mV
# Determine the slice index closest to zero bias
slice_index = int(np.argmin(np.abs(Z - 0.0)))
plane = mlab.pipeline.image_plane_widget(
    src,
    plane_orientation='z_axes',  # plane perpendicular to Z (bias) axis
    slice_index=slice_index,
    colormap='coolwarm',         # choose a diverging colormap
    opacity=1.0                  # fully opaque slice
)
# Disable picking on the plane widget
plane.ipw.point_picker.enabled = False

# Add axes labels
mlab.axes(
    xlabel='X (m)',
    ylabel='Y (m)',
    zlabel='bias (mV)',
    color=(0, 0, 0)  # axes labels in black
)

# Show the scene
mlab.show()

# -





# +
import os
import numpy as np
import xarray as xr
import hvplot.xarray            # enables .hvplot on xarray DataArrays
import holoviews as hv
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import ipywidgets as widgets

def cluster_map_2x2panel(ds: xr.Dataset):
    """
    Create an interactive 2×2 cluster map dashboard with confirm/save controls.

    Given an xarray.Dataset `ds` containing:
      - DataArrays: 'LDOS', 'best_fit', 'cluster_maps'
      - Coordinates: 'bias_mV', 'X', 'Y'
      - 'cluster_maps' has coordinate 'cluster_label'

    This function will:
      1) Add nm‐scaled coords 'X_nm','Y_nm' to the dataset.
      2) Compute global min/max for 'best_fit'.
      3) Instantiate ipywidgets:
         - bias_slider: select bias voltage
         - cluster_left, cluster_right: select two cluster labels
         - clim_type: 'Percent' or 'Absolute'
         - clim_lower_pct, clim_upper_pct: percent color limits
         - clim_lower_val, clim_upper_val: absolute color limits
         - mode_widget: 'Global' or 'Local' scaling
         - cmap_dropdown: choose colormap
         - confirm_btn: render/update 2×2 plot
         - save_btn: save current view to SVG
      4) On Confirm:
         - Slice `LDOS`, `best_fit`, and two `cluster_maps` at chosen bias and labels
         - Compute color limits based on mode & clim_type
         - Render a 2×2 HoloViews panel via `hvplot.image`
      5) On Save:
         - Build a Matplotlib 2×2 figure with `imshow(..., aspect='equal')`,
           correct extents, integer tick labels, 10 nm scale bars whose
           start is at 10% from left and 5% from bottom, with adaptive color
           on the top‐left plot only and forced black on the other three.
         - Save to `output_figures/cluster_map2x2.svg`
         - Print the saved file path

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with required variables and coords.
    """
    # 1) Prepare dataset with nm coords
    ds_local = ds.copy().assign_coords(
        X_nm = ds['X'] * 1e9,
        Y_nm = ds['Y'] * 1e9
    )

    # 2) Compute global best_fit min/max
    global_min = float(ds_local['best_fit'].min(skipna=True).values)
    global_max = float(ds_local['best_fit'].max(skipna=True).values)

    # 3) Widget definitions
    bias_opts      = sorted(ds_local['bias_mV'].values.tolist())
    cluster_labels = sorted(ds_local['cluster_maps'].cluster_label.values.tolist())

    bias_slider   = widgets.FloatSlider(
        description='Bias (mV)',
        min=bias_opts[0], max=bias_opts[-1],
        step=(bias_opts[1]-bias_opts[0]) if len(bias_opts)>1 else 0.1,
        value=bias_opts[0],
        readout_format='.2f'
    )
    cluster_left  = widgets.IntSlider(
        description='Cluster Left',
        min=int(cluster_labels[0]), max=int(cluster_labels[-1]),
        step=1, value=int(cluster_labels[0])
    )
    cluster_right = widgets.IntSlider(
        description='Cluster Right',
        min=int(cluster_labels[0]), max=int(cluster_labels[-1]),
        step=1, value=int(cluster_labels[0])
    )
    clim_type     = widgets.ToggleButtons(
        description='Clim Type',
        options=['Percent','Absolute'],
        value='Percent'
    )
    clim_lower_pct = widgets.FloatSlider(
        description='Lower (%)',
        min=0.0, max=100.0, step=0.5, value=0.0, readout_format='.1f'
    )
    clim_upper_pct = widgets.FloatSlider(
        description='Upper (%)',
        min=0.0, max=100.0, step=0.5, value=100.0, readout_format='.1f'
    )
    clim_lower_val = widgets.FloatSlider(
        description='Lower val',
        min=global_min, max=global_max,
        step=(global_max-global_min)/100, value=global_min,
        readout_format='.2e'
    )
    clim_upper_val = widgets.FloatSlider(
        description='Upper val',
        min=global_min, max=global_max,
        step=(global_max-global_min)/100, value=global_max,
        readout_format='.2e'
    )
    mode_widget   = widgets.RadioButtons(
        description='Color Mode',
        options=['Global','Local'],
        value='Global'
    )
    cmap_dropdown = widgets.Dropdown(
        description='Colormap',
        options=['bwr','viridis','plasma','inferno','magma','cividis'],
        value='bwr'
    )
    confirm_btn   = widgets.Button(description='Confirm', button_style='primary')
    save_btn      = widgets.Button(description='Save SVG', button_style='success')
    output        = widgets.Output()

    # show/hide percent vs absolute sliders
    def toggle_clim(evt=None):
        pct = (clim_type.value == 'Percent')
        clim_lower_pct.layout.display = None if pct else 'none'
        clim_upper_pct.layout.display = None if pct else 'none'
        clim_lower_val.layout.display = None if not pct else 'none'
        clim_upper_val.layout.display = None if not pct else 'none'
    clim_type.observe(toggle_clim, names='value')
    toggle_clim()

    # 4) Helper: compute color limits
    def compute_limits(da: xr.DataArray):
        if clim_type.value == 'Percent':
            lp, up = clim_lower_pct.value/100, clim_upper_pct.value/100
            if mode_widget.value == 'Global':
                return (global_min + (global_max-global_min)*lp,
                        global_min + (global_max-global_min)*up)
            else:
                mn, mx = float(da.min()), float(da.max())
                return (mn + (mx-mn)*lp, mn + (mx-mn)*up)
        else:
            return clim_lower_val.value, clim_upper_val.value

    # 5) Confirm callback: render 2×2 hvplot
    def on_confirm(_):
        with output:
            clear_output()
            bias = bias_slider.value
            cl_l = cluster_left.value
            cl_r = cluster_right.value

            # Slice DataArrays
            ldos  = ds_local['LDOS']      .sel(bias_mV=bias, method='nearest')
            best  = ds_local['best_fit']  .sel(bias_mV=bias, method='nearest')
            cmap  = ds_local['cluster_maps']
            left  = cmap.sel(cluster_label=cl_l, bias_mV=bias, method='nearest')
            right = cmap.sel(cluster_label=cl_r, bias_mV=bias, method='nearest')

            # Compute limits for each
            v1 = compute_limits(ldos)
            v2 = compute_limits(best)
            v3 = compute_limits(left)
            v4 = compute_limits(right)

            opts = dict(
                cmap       = cmap_dropdown.value,
                colorbar   = True,
                xlabel     = 'X (nm)',
                ylabel     = 'Y (nm)',
                aspect     = 'equal',
                frame_width  = 300,
                frame_height = 300
            )
            # Ticks: 5 along X and Y using nm coords
            nx, ny = ds_local.sizes['X'], ds_local.sizes['Y']
            ix = np.linspace(0, nx-1, 5).astype(int)
            iy = np.linspace(0, ny-1, 5).astype(int)
            xticks = [(float(ds_local['X'][i]), f"{(ds_local['X'][i]*1e9):.0f}") for i in ix]
            yticks = [(float(ds_local['Y'][i]), f"{(ds_local['Y'][i]*1e9):.0f}") for i in iy]
            opts.update(xticks=xticks, yticks=yticks)

            # Titles with two‐decimal bias
            t = f"{bias:.2f} mV"
            p1 = ldos .hvplot.image(title=f"LDOS @ {t}", clim=v1, **opts)
            p2 = best .hvplot.image(title=f"best_fit @ {t}\nMode={mode_widget.value}", clim=v2, **opts)
            p3 = left .hvplot.image(title=f"Cluster {cl_l}", clim=v3, **opts)
            p4 = right.hvplot.image(title=f"Cluster {cl_r}", clim=v4, **opts)

            layout = (p1 + p2 + p3 + p4).cols(2)
            display(layout)

            # Store for saving
            output.last = dict(
                bias=bias, cl_l=cl_l, cl_r=cl_r,
                limits=(v1, v2, v3, v4), cmap=cmap_dropdown.value
            )

    confirm_btn.on_click(on_confirm)

    # 6) Save callback: Matplotlib 2×2 SVG
    def on_save(_):
        with output:
            clear_output()
            if not hasattr(output, 'last'):
                print("⚠️ Please Confirm first.")
                return
            params = output.last
            bias, cl_l, cl_r = params['bias'], params['cl_l'], params['cl_r']
            (v1, v2, v3, v4) = params['limits']
            cmap = params['cmap']

            # Slice again
            da_list = [
                ds_local['LDOS']     .sel(bias_mV=bias, method='nearest'),
                ds_local['best_fit'] .sel(bias_mV=bias, method='nearest'),
                ds_local['cluster_maps'].sel(cluster_label=cl_l, bias_mV=bias, method='nearest'),
                ds_local['cluster_maps'].sel(cluster_label=cl_r, bias_mV=bias, method='nearest')
            ]
            titles = [
                f"LDOS @ {bias:.2f} mV",
                f"best_fit @ {bias:.2f} mV\nMode={mode_widget.value}",
                f"Cluster {cl_l}",
                f"Cluster {cl_r}"
            ]

            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)

            for idx, (ax, da, title, (vmin, vmax)) in enumerate(zip(
                axes.flatten(), da_list, titles, [v1, v2, v3, v4]
            )):
                X_nm = da['X_nm'].values
                Y_nm = da['Y_nm'].values
                extent = [X_nm.min(), X_nm.max(), Y_nm.min(), Y_nm.max()]
                im = ax.imshow(
                    da.values,
                    extent=extent,
                    origin='lower',
                    cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    aspect='equal'
                )
                ax.set_title(title)
                ax.set_xlabel('X (nm)')
                ax.set_ylabel('Y (nm)')

                # integer tick labels
                xt = ax.get_xticks()
                yt = ax.get_yticks()
                ax.set_xticklabels([f"{x:.0f}" for x in xt], rotation=45, ha='right')
                ax.set_yticklabels([f"{y:.0f}" for y in yt])

                # reduced‐size colorbar (half width)
                cbar = fig.colorbar(im, ax=ax, fraction=0.075, pad=0.02)
                cbar.set_label('Intensity')

                # 10 nm scale bar at 10% from left, 5% from bottom
                x_span = extent[1] - extent[0]
                y_span = extent[3] - extent[2]
                x0 = extent[0] + 0.10 * x_span   # ← moved to 10% from left
                y0 = extent[2] + 0.05 * y_span
                length = 10.0

                # adaptive color only for top-left (idx==0), else black
                if idx == 0:
                    xc = x0 + length/2
                    i_center = np.argmin(np.abs(X_nm - xc))
                    j_y      = np.argmin(np.abs(Y_nm - y0))
                    sample   = da.values[j_y, i_center]
                    bar_color = 'black' if sample > (vmin+vmax)/2 else 'white'
                else:
                    bar_color = 'black'

                ax.hlines(y=y0, xmin=x0, xmax=x0+length, colors=bar_color, linewidth=3)
                ax.text(
                    x0 + length/2,
                    y0 + 0.02*y_span,
                    '10 nm',
                    color=bar_color,
                    ha='center', va='bottom',
                    fontsize=10, weight='bold'
                )

            # save SVG
            folder = 'output_figures'
            os.makedirs(folder, exist_ok=True)
            fname = os.path.join(folder, 'cluster_map2x2.svg')
            fig.savefig(fname, format='svg')
            plt.close(fig)
            print(f"✅ Saved SVG to '{fname}'")

    save_btn.on_click(on_save)

    # 7) Display UI
    controls = widgets.VBox([
        bias_slider,
        cluster_left,
        cluster_right,
        clim_type,
        widgets.HBox([clim_lower_pct, clim_upper_pct]),
        widgets.HBox([clim_lower_val, clim_upper_val]),
        mode_widget,
        cmap_dropdown,
        widgets.HBox([confirm_btn, save_btn])
    ])
    display(widgets.VBox([controls, output]))


# Usage:
# ds = ds_opt2.copy()
# cluster_map_2x2panel(ds)

# -

#dashboard = cluster_map_2x2panel(ds_opt2)
dashboard = cluster_map_2x2panel(ds_cluster)

dashboard



# vortex1
'''ds_opt2.sel(
    Y = slice(2.0e-7, 2.15e-7),
    X = slice(0.5e-7, 0.65e-7)).LDOS.where(ds_opt2.bias_mV==0,drop = True).plot()'''
# vortex2
'''ds_opt2.sel(
    Y = slice(2.25e-7, 2.4e-7),
    X = slice(0.65e-7, 0.8e-7)).LDOS.where(ds_opt2.bias_mV==0,drop = True).plot()'''
#vortex3
'''ds_opt2.sel(
    Y = slice(2.32e-7, 2.47e-7),
    X = slice(0.95e-7, 1.1e-7)).LDOS.where(ds_opt2.bias_mV==0,drop = True).plot()'''
#vortex4
'''ds_opt2.sel(
    Y = slice(1.95e-7, 2.1e-7),
    X = slice(0.85e-7, 1.0e-7)).LDOS.where(ds_opt2.bias_mV==0,drop = True).plot()'''
#voretex5
'''ds_opt2.sel(
    Y = slice(2.05e-7, 2.2e-7),
    X = slice(1.1e-7, 1.25e-7)).LDOS.where(ds_opt2.bias_mV==0,drop = True).plot()'''
#voretex6
'''ds_opt2.sel(
    Y = slice(1.7e-7, 1.85e-7),
    X = slice(0.75e-7, 0.9e-7)).LDOS.where(ds_opt2.bias_mV==0,drop = True).plot()'''

ds_opt2.sel(
    Y = slice(1.75e-7, 1.9e-7),
    X = slice(1.12e-7, 1.27e-7)).LDOS.where(ds_opt2.bias_mV==0,drop = True).plot()

ds_zm1 = ds_opt2.sel(
    Y = slice(1.75e-7, 1.9e-7),
    X = slice(1.12e-7, 1.27e-7))

#dashboard = cluster_map_2x2panel(ds_zm1)
dashboard = cluster_map_2x2panel(ds_opt2)

dashboard



cluster_select_labels_interactive(ds) 

ds_opt2.to_netcdf('grid_2T_003_fit_cluster9_20250528.nc')

ds_cluster= ds_opt2.copy()



ds_cluster =  xr.open_dataset('grid_2T_003_fit_cluster9_20250528.nc')
ds_cluster



# # figure1  Raw, filtered, + stepbystep images
#

# ### Raw LDOS data

# * Use ds_cluster instead of ds_opt2



# +
#ds_opt2.LDOS.sel(bias_mV=0).plot(robust = True)
#isns.imshow(ds_opt2.LDOS.sel(bias_mV=0).values)

# 1) 전체 폰트 패밀리와 사이즈 설정 (Arial, 14pt)
isns.set_context(
    mode="notebook",
    fontfamily="Arial",
    rc={"font.size": 16}
)
ax = isns.imshow(ds_opt2.LDOS.sel(bias_mV=0).values, 
                 dx=1, 
                 units="nm",
                 cmap="viridis",
                 #robust = True,
                )
plt.show()
# SVG 포맷으로 저장
ax.figure.savefig(
    "ldos_map_0mV_2T003.svg",
    format="svg",
    bbox_inches="tight"
)

# Figure 객체 종료 (메모리 해제 및 중복 출력 방지)
plt.close(ax.figure)
# -

# ### Clustered LDOS data

# +
#ds_opt2.cluster_umap_HDBSCAN9_L2.sel(bias_mV=0)

isns.set_context(
    mode="notebook",
    fontfamily="Arial",
    rc={"font.size": 16}
)
ax = isns.imshow(ds_opt2.cluster_umap_HDBSCAN9_L2.sel(bias_mV=0).values, 
                 dx=1, 
                 units="nm",
                 cmap="viridis",
                 #robust = True,
                )
plt.show()
# SVG 포맷으로 저장
ax.figure.savefig(
    "ldos_map_cluster2_0mV_2T003.svg",
    format="svg",
    bbox_inches="tight"
)

# Figure 객체 종료 (메모리 해제 및 중복 출력 방지)
plt.close(ax.figure)

# +
## cluster2 multi-otsu 3 results

#ds_opt2.cluster_umap_HDBSCAN9_L2.sel(bias_mV=0)

isns.set_context(
    mode="notebook",
    fontfamily="Arial",
    rc={"font.size": 16}
)
ax = isns.imshow(threshold_multiotsu_xr(
    ds_opt2.sel(bias_mV=0)[['cluster_umap_HDBSCAN9_L2']],multiclasses=4
).cluster_umap_HDBSCAN9_L2.values, 
                 dx=1, 
                 units="nm",
                 cmap="viridis",
                 #robust = True,
                )
plt.show()
# SVG 포맷으로 저장
ax.figure.savefig(
    "ldos_map_cluster2_0mV_2T003_multiotsu3.svg",
    format="svg",
    bbox_inches="tight"
)

# Figure 객체 종료 (메모리 해제 및 중복 출력 방지)
plt.close(ax.figure)

# +
#threshold_multiotsu_xr(ds_opt2.sel(bias_mV=0)[['cluster_umap_HDBSCAN9_L2']],multiclasses=3).cluster_umap_HDBSCAN9_L2.plot()

ds_cl2_thres = threshold_multiotsu_xr(ds_opt2.sel(bias_mV=0)[['cluster_umap_HDBSCAN9_L2']],multiclasses=4).cluster_umap_HDBSCAN9_L2
# -

cluster_zbP_mask = (ds_cl2_thres==3)


ds_opt2.sel(bias_mV=0)[['cluster_umap_HDBSCAN9_L2']].where(cluster_zbP_mask,drop = False).cluster_umap_HDBSCAN9_L2.plot()

# +
## cluster2 multi-otsu 3 results

#ds_opt2.cluster_umap_HDBSCAN9_L2.sel(bias_mV=0)

isns.set_context(
    mode="notebook",
    fontfamily="Arial",
    rc={"font.size": 16}
)
ax = isns.imshow(ds_opt2.sel(bias_mV=0)[['cluster_umap_HDBSCAN9_L2']].where(cluster_zbP_mask).cluster_umap_HDBSCAN9_L2.notnull().values
                 , 
                 dx=1, 
                 units="nm",
                 cmap="bwr",
                 #robust = True,
                )
plt.show()
# SVG 포맷으로 저장
ax.figure.savefig(
    "ldos_map_cluster2_0mV_2T003_multiotsu3.svg",
    format="svg",
    bbox_inches="tight"
)

# Figure 객체 종료 (메모리 해제 및 중복 출력 방지)
plt.close(ax.figure)

# +
## cluster2 multi-otsu 3 results

#ds_opt2.cluster_umap_HDBSCAN9_L2.sel(bias_mV=0)

isns.set_context(
    mode="notebook",
    fontfamily="Arial",
    rc={"font.size": 16}
)
ax = isns.imshow(ds_opt2.sel(bias_mV=0)[['cluster_umap_HDBSCAN9_L2']].where(cluster_zbP_mask).cluster_umap_HDBSCAN9_L2.notnull().values
                 , 
                 dx=1, 
                 units="nm",
                 cmap="Greens",
                 #robust = True,
                )
plt.show()
# SVG 포맷으로 저장
ax.figure.savefig(
    "ldos_map_cluster2_0mV_2T003_multiotsu3-1.svg",
    format="svg",
    bbox_inches="tight"
)

# Figure 객체 종료 (메모리 해제 및 중복 출력 방지)
plt.close(ax.figure)

# +
#threshold_otsu_xr(ds_opt2.sel(bias_mV=0)[['cluster_umap_HDBSCAN9_L2']].fillna(0) ).cluster_umap_HDBSCAN9_L2.plot()
# -

ds_3 = ds_opt2[['LDOS']].copy()
ds_3

filter_gaussian_xr(ds_3, overwrite=True, )

# +
# ── 최상단에서 한 번만 실행 ───────────────────────────────────────────────
import panel as pn
import holoviews as hv
import hvplot.xarray  # hvplot.xarray API 활성화

pn.extension()            # Panel extension: DO NOT pass 'bokeh' here
hv.extension('bokeh')     # HoloViews  Bokeh extension

# ── 이하 최소 재현 테스트 코드 ─────────────────────────────────────────
import numpy as np
import xarray as xr

# 더미 3D 데이터셋 생성
bias = np.linspace(0, 100, 5)
X    = np.linspace(0,   1, 10)
Y    = np.linspace(0,   1, 10)
data = np.random.rand(len(bias), len(X), len(Y))
ds   = xr.DataArray(data,
                    coords=[('bias_mV', bias), ('X', X), ('Y', Y)],
                    name='LDOS').to_dataset()

# 슬라이더 위젯
bias_slider = pn.widgets.FloatSlider(
    name='Bias (mV)',
    start=float(ds.bias_mV.min()), end=float(ds.bias_mV.max()),
    step=float(ds.bias_mV.diff(dim='bias_mV').mean())
)
x_slider = pn.widgets.FloatSlider(
    name='X',
    start=float(ds.X.min()), end=float(ds.X.max()),
    step=float(ds.X.diff(dim='X').mean())
)
y_slider = pn.widgets.FloatSlider(
    name='Y',
    start=float(ds.Y.min()), end=float(ds.Y.max()),
    step=float(ds.Y.diff(dim='Y').mean())
)

# 플롯 함수 정의
def plot_bias(bias_val):
    da = ds.LDOS.sel(bias_mV=bias_val, method='nearest')
    return da.hvplot.image(
        x='X', y='Y',
        clim=(ds.LDOS.min(), ds.LDOS.max()),
        cmap='Viridis', aspect='equal'
    ).opts(title=f"Bias = {bias_val:.2f} mV")

def plot_x(x_val):
    da = ds.LDOS.sel(X=x_val, method='nearest')
    return da.hvplot.image(
        x='bias_mV', y='Y',
        clim=(ds.LDOS.min(), ds.LDOS.max()),
        cmap='Viridis'
    ).opts(title=f"X = {x_val:.2f}")

def plot_y(y_val):
    da = ds.LDOS.sel(Y=y_val, method='nearest')
    return da.hvplot.image(
        x='bias_mV', y='X',
        clim=(ds.LDOS.min(), ds.LDOS.max()),
        cmap='Viridis'
    ).opts(title=f"Y = {y_val:.2f}")

# pn.bind -> pn.panel 으로 감싸기 (error_policy 생략)
bias_pane = pn.panel(pn.bind(plot_bias, bias_val=bias_slider))
x_pane    = pn.panel(pn.bind(plot_x,    x_val=x_slider))
y_pane    = pn.panel(pn.bind(plot_y,    y_val=y_slider))

# 레이아웃 구성 및 서빙
dashboard = pn.Column(
    pn.Row(bias_slider, x_slider, y_slider),
    pn.Tabs(
        ('Bias Slice', bias_pane),
        ('X Slice',    x_pane),
        ('Y Slice',    y_pane),
    )
)

dashboard.servable()

# -



# +
#grid_data_dim_slicing(filter_gaussian_xr (ds_opt2[['LDOS']], sigma=1, overwrite= True),channel ='LDOS')
# -

'''
plot_cluster_statistics(
    ds_opt2, 
    cluster_var = None,
    clusters_to_keep = None,
    model_type = None,
    remove_background = False,
    remove_neg_amp = False,
    remove_outliers = False

)'''





ds_cluster

# +
ds_cluster.LDOS.where(ds_cluster.ZB_mask.isnull()).mean(dim =['X','Y']).plot()
ds_cluster.LDOS.where(~ds_cluster.ZB_mask.isnull()).mean(dim =['X','Y']).plot()

#ds_cluster.LDOS.where(~ds_cluster.ZB_mask.isnull())

# +
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xarray as xr

# seaborn 스타일 설정
sns.set_style("whitegrid")

# 1) X, Y 차원을 'pixel'으로 병합 (stack)
ldos_stacked = ds_cluster.LDOS.stack(pixel=("X", "Y"))    # dims: ("pixel", "bias_mV")
mask_stacked = ds_cluster.ZB_mask.stack(pixel=("X", "Y"))  # dims: ("pixel",)

# 2) pixel별로 ZBP 여부 문자열 카테고리 생성
category = xr.where(mask_stacked.notnull(), "ZBP", "no_ZBP")  # dims: ("pixel",)

# 3) pandas DataFrame으로 변환
#   (1) LDOS 데이터: to_series() → reset_index()
df_ldos = ldos_stacked.to_series().reset_index()
#      컬럼 순서: ['bias_mV', 'X', 'Y', 'LDOS']

#   (2) category 정보: to_series() → reset_index()
df_category = category.to_series().reset_index()
#      컬럼 순서: ['X', 'Y', 'ZB_mask']
df_category = df_category.rename(columns={"ZB_mask": "category"})

# 4) 두 DataFrame을 X, Y 기준으로 병합
df = pd.merge(df_ldos, df_category, on=["X", "Y"], how="left")
# df.columns: ['bias_mV', 'X', 'Y', 'LDOS', 'category']

# 5) seaborn relplot으로 "bias_mV vs LDOS" 평균곡선 및 95% CI 그리기
#    → legend=False 를 주면 hue 범례(오른쪽 범례)가 생성되지 않습니다.
g = sns.relplot(
    data=df,
    x="bias_mV",
    y="LDOS",
    hue="category",
    kind="line",
    #ci=95,
    ci= 'sd',
    height=4,
    aspect=1,
    legend=False         # 자동 생성되는 hue 범례를 끕니다.
)

g.set_axis_labels("Bias (mV)", "LDOS")
g.fig.suptitle("averaged LDOS (confidence interval: standard deviation)", y=1.02)

# 6) 축 내부에 하나의 범례만 직접 그리기
#    → 오른쪽 대신 왼쪽 위에 “SC vs In-Gap” 범례를 넣고 싶다면, 아래와 같이 수동 생성
ax = g.ax  # relplot 사용 시 단일 축이므로 .ax로 접근 가능합니다.

# hue 순서(색상)와 텍스트 레이블을 직접 정의합니다.
# 예: ZBP → "IGS", no_ZBP → "SC"
hue_order = ["no_ZBP", "ZBP"]
labels    = ["SC", "IGS"]  # 축에 표시할 텍스트
colors    = [sns.color_palette()[0], sns.color_palette()[1]]  

# 빈 핸들을 만들어서, 실제 색상만 맞춰줄 겁니다.
handles = [plt.Line2D([], [], color=colors[i], lw=2) for i in range(len(labels))]

# 범례 위치를 축 내부로 설정 (예: upper left)
ax.legend(
    handles    = handles,
    labels     = labels,
    title      = "SC vs In-Gap",
    loc        = "upper left",
    frameon    = True,
    fontsize   = 10,
)

plt.show()

# -





# # Figure 2 

# +
## single point peak fitting results 
# -

ds_opt2.ZB_mask.notnull().plot()

ds_opt2.sel(bias_mV=0).ZB_mask.plot()

fig,dfY69X148 = plot_region_fitting_result_from_dsfit(ds_opt2,
                                              use_zb_mask=True,
                                              zb_mask_key='ZB_mask',
                                              allowed_models=['Lorentzian', 'Gaussian', 'Voigt'],
                                              show_shade=False,
                                              y_idx=69,
                                              x_idx=148,
                                              return_fig=True)
#fig

# +
#  dfY69X148


ds_opt2.isel(Y=69,X=148)

# +
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1) Prepare the DataFrame: move bias_mV from index to a column
df = dfY69X148.copy().reset_index()

# 2) Set the figure size
plt.figure(figsize=(6, 5))

# 3) Plot LDOS as a solid line
sns.lineplot(
    x='bias_mV', y='LDOS', data=df,
    linestyle='-', linewidth=2,
    label='LDOS'
)

# 4) Plot best_fit as a thick, semi-transparent solid line
sns.lineplot(
    x='bias_mV', y='best_fit', data=df,
    linestyle='-', linewidth=4, alpha=0.5,
    label='best_fit'
)

# 5) Plot peaks 0–5 as dashed lines without markers
for i in range(6):
    sns.lineplot(
        x='bias_mV', y=f'peak{i}', data=df,
        linestyle='--', legend=False
    )

# 6) Set axis labels and title
plt.xlabel('Bias (mV)')
plt.ylabel('LDOS (A/V)')
plt.title('LDOS @ (204 nm, 124 nm)')

# 7) Adjust layout to minimize whitespace
plt.tight_layout()

# 8) Save the figure as SVG with tight bounding box
plt.savefig('ldos_dfY69X148_plot.svg', format='svg', bbox_inches='tight')

# 9) Display the plot
plt.show()

# -

ds_opt2[['LDOS

# +

# 1) NumPy 배열 및 좌표 추출
data = ds_opt2.LDOS.values
ys   = ds_opt2.Y.values
xs   = ds_opt2.X.values
zs   = ds_opt2.bias_mV.values

# 2) 10개의 bias 슬라이스 인덱스 (0 포함, 등간격)
slice_idxs = np.linspace(0, len(zs)-1, 7, dtype=int)

# 3) 전역 robust 범위 계산 (2, 98 퍼센타일)
p2, p98 = np.nanpercentile(data, [2, 98])
p2, p98 = float(p2), float(p98)

# 4) bias 범위에서 sigma 설정 (가우시안 opacity)
max_bias   = float(np.max(np.abs(zs)))
sigma_bias = max_bias / 3

# 5) Plotly Figure 생성
fig = go.Figure()

# 6) 각 슬라이스를 평면(surface)으로 추가하며 Gaussian opacity & robust 색상 범위 적용
for i, idx in enumerate(slice_idxs):
    z0        = float(zs[idx])
    slice_img = data[:, :, idx]

    opacity = np.exp(-(z0**2) / (2 * sigma_bias**2))
    opacity = float(opacity)

    fig.add_trace(go.Surface(
        x=xs,
        y=ys,
        z=np.full_like(slice_img, z0),
        surfacecolor=slice_img,
        cmin=p2,
        cmax=p98,
        colorscale='Viridis',
        opacity=opacity,
        showscale=(i == 0)
    ))

# 7) 레이아웃 세부 조정
fig.update_layout(
    title="LDOS 3D Stack with Gaussian Opacity & Robust Contrast",
    scene=dict(
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        zaxis_title='bias (mV)',
        aspectmode='auto'
    ),
    margin=dict(l=0, r=0, b=0, t=30)
)


# 8) Display figure
fig.show()

# 9) Save figure as SVG
# Requires the 'kaleido' package: pip install kaleido
#fig.write_image("ldos_3d_stack_plotly.svg")

# -

import importlib
spec = importlib.util.find_spec("kaleido")
if spec is not None:
    print("kaleido is installed")
else:
    print("kaleido is not installed")



# +
# Mayavi를 이용한 3D slice 스태킹 예제 코드
import numpy as np
from mayavi import mlab

# 1) xarray 데이터에서 NumPy 배열 및 좌표 추출
data = ds_opt2.LDOS.values            # shape: (Y, X, bias_mV)
ys   = ds_opt2.Y.values
xs   = ds_opt2.X.values
zs   = ds_opt2.bias_mV.values

# 2) 10개의 bias 슬라이스 인덱스 (0 포함, 등간격)
slice_idxs = np.linspace(0, len(zs)-1, 10, dtype=int)

# 3) Mayavi Figure 생성
mlab.figure(bgcolor=(1,1,1), size=(800, 600))

# 4) 각 슬라이스를 XY 평면에 스태킹
for idx in slice_idxs:
    z0    = zs[idx]
    slice_img = data[:, :, idx]       # Y×X
    Xg, Yg = np.meshgrid(xs, ys)      # 그리드 생성
    # surf: (x, y, z) 평면 위에 scalars 로 컬러맵 적용
    mlab.surf(
        Xg, Yg,
        np.full_like(Xg, z0),         # bias 위치를 높이(z)에 사용
        scalars=slice_img,
        colormap='viridis',
        opacity=0.7
    )

# 5) 축 및 컬러바 옵션
mlab.axes(
    xlabel='X (m)', ylabel='Y (m)', zlabel='bias (mV)',
    ranges=[xs.min(), xs.max(), ys.min(), ys.max(), zs.min(), zs.max()]
)
mlab.colorbar(title='LDOS', orientation='vertical')
mlab.view(azimuth=45, elevation=60)  # 시점 설정
mlab.show()

# -

# ## export graph & data fro Dr. shin

grid_LDOS_SnD_pks=ds_opt2.copy()

# +
import matplotlib.pyplot as plt

# 1) 좌표 리스트
# for 2T 003
select_coords = [
    (24, 50),
    (29, 152),
    (53, 97),
    (81, 16),
    (104, 149),
    (125, 32),
    (131, 118),
    (145, 99)
]
# 2) Y값 기준으로 위→아래 정렬 및 번호 매기기
sorted_coords = sorted(select_coords, key=lambda yx: yx[0])
numbers = list(range(1, len(sorted_coords) + 1))

# 3) 그림 생성 및 지도 플롯
fig, ax = plt.subplots(figsize=(6,6))
grid_LDOS_SnD_pks.LDOS.sel(bias_mV=0).plot(ax=ax)

# 4) 좌표별로 점 찍고 번호 텍스트는 점 위쪽에 배치
for (y, x), num in zip(sorted_coords, numbers):
    # 실제 축 좌표값
    x_val = grid_LDOS_SnD_pks['X'].isel(X=x).values
    y_val = grid_LDOS_SnD_pks['Y'].isel(Y=y).values

    # 반투명 붉은 점
    ax.scatter(
        x_val, y_val,
        color='red', alpha=0.5, s=50,
        edgecolors='none', zorder=10
    )
    ax.set_aspect('equal', adjustable='box')
    
    # 번호 텍스트를 점의 위쪽으로 오프셋
    ax.annotate(
        str(num),
        xy=(x_val, y_val),
        xytext=(0, 5),            # y축으로 5포인트만큼 위쪽으로 이동
        textcoords='offset points',
        ha='center', va='bottom', # 중앙 정렬, 텍스트의 하단이 기준점
        color='white',
        fontsize=12,
        fontweight='bold',
        zorder=11
    )

# 5) 최종 레이아웃
ax.set_title('Bias=0 LDOS Map with Selected Points')
plt.tight_layout()
plt.show()


# +
import matplotlib.pyplot as plt

# 1) 저장할 좌표 리스트
# for 2T 003
select_coords = [
    (24, 50),
    (29, 152),
    (53, 97),
    (81, 16),
    (104, 149),
    (125, 32),
    (131, 118),
    (145, 99)
]


# 2) 순회하면서 그림과 DataFrame 생성·저장
for idx, (y, x) in enumerate(select_coords, start=1):
    label = f"{idx}_Y{y}X{x}"
    
    # a) 함수 호출: fig, df 반환
    fig, df = plot_region_fitting_result_from_dsfit(
        grid_LDOS_SnD_pks,
        weight_function_show=False,
        use_zb_mask=True,
        y_idx=y,
        x_idx=x,
        return_fig=True
    )
    
    # b) SVG로 그림 저장
    fig.savefig(f"{label}.svg", format='svg', bbox_inches='tight')
    
    # **c) 화면에 그림 출력**
    plt.show()
    
    # d) CSV로 DataFrame 저장
    df.to_csv(f"df_{label}.csv", index=True)
    
    # e) 메모리 해제
    plt.close(fig)


# +
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from lmfit.models import LorentzianModel, GaussianModel, VoigtModel, ConstantModel
from functools import reduce

def plot_fitting_result_facet_grid(
        ds_out,
        model_type=['Lorentzian','Gaussian','Voigt'],
        point_list=20,
        ncol=4,
        weight_function_show=False,
        ZB_masking=True,
        show_legend=True,
        share_xy=True,
        show_shapes=True):
    """
    Version v1.9.1:
    - Default model_type is a list of allowed models; per-pixel best model taken from ds_out['model_type'].

    Parameters
    ----------
    ds_out : xarray.Dataset
        Must include dims Y,X,peak and variables:
            - 'LDOS', 'bias_mV', 'peak_center', 'peak_amplitude', 'peak_sigma'
            - 'redchi', 'level_proximity', 'model_type'
            - Optional: 'background_value', 'best_fit', 'ZB_mask'
    model_type : list of str or str
        If list: allowed models and per-pixel model chosen from ds_out['model_type'].
        If str: override and use this fixed model for all pixels.
    point_list : int or list of (Y, X)
        Pixels to plot.
    ncol : int
        Number of columns in grid.
    weight_function_show : bool
        Show convolution and weight function.
    ZB_masking : bool
        Restrict pixel selection by 'ZB_mask'.
    show_legend : bool
        Toggle legend in subplots.
    share_xy : bool
        Share axes among subplots.
    show_shapes : bool
        Toggle filled shapes for CdGM peaks.
    """
    Ny, Nx = ds_out.dims['Y'], ds_out.dims['X']
    # select points
    if isinstance(point_list, int):
        if ZB_masking and 'ZB_mask' in ds_out:
            mask = ~np.isnan(ds_out['ZB_mask'].values)
            ys, xs = np.where(mask)
            coords = list(zip(ys,xs))
        else:
            coords = [(y,x) for y in range(Ny) for x in range(Nx)]
        idxs = np.random.choice(len(coords), min(point_list,len(coords)), replace=False)
        selected = [coords[i] for i in idxs]
    else:
        selected = point_list
    # setup grid
    n_points = len(selected)
    nrow = int(np.ceil(n_points/ncol))
    fig, axes = plt.subplots(nrow,ncol,figsize=(5*ncol,5*nrow), sharex=share_xy, sharey=share_xy)
    axes = axes.flatten()
    # constants
    bias = ds_out['bias_mV'].values
    w_sigma = ds_out.attrs.get('weight_sigma',1.0)
    Ef = ds_out.attrs.get('Ef',4.4)
    SCgap = ds_out.attrs.get('SCgap',1.8)
    E_mu = SCgap**2/Ef
    for i,(y,x) in enumerate(selected):
        ax = axes[i]
        ldos = ds_out['LDOS'].isel(Y=y,X=x).values
        n_peaks = ds_out.dims['peak']
        centers = ds_out['peak_center'].isel(Y=y,X=x).values
        amps = ds_out['peak_amplitude'].isel(Y=y,X=x).values
        sigmas = ds_out['peak_sigma'].isel(Y=y,X=x).values
        level_prox = ds_out['level_proximity'].isel(Y=y,X=x).values
        redchi = ds_out['redchi'].isel(Y=y,X=x).item()
        # determine per-pixel model
        if isinstance(model_type, list):
            per = ds_out['model_type'].isel(Y=y,X=x).item().capitalize()
        else:
            per = model_type.capitalize()
        if per not in (model_type if isinstance(model_type,list) else [model_type]):
            raise ValueError(f"Model '{per}' not in allowed model_type list.")
        # map to class and color
        if per=='Lorentzian': MC, col = LorentzianModel, 'r'
        elif per=='Gaussian': MC, col = GaussianModel, 'b'
        else:               MC, col = VoigtModel,      'g'
        # plot LDOS
        ax.plot(bias,ldos,'k-',alpha=0.6,label='LDOS')
        # background
        if 'background_value' in ds_out:
            bgv = ds_out['background_value'].isel(Y=y,X=x).item()
            if not np.isnan(bgv):
                ax.plot(bias, np.full_like(bias,bgv),'--',color='gray',label='Background')
        # build and eval composite
        models=[]
        if 'background_value' in ds_out: models.append(ConstantModel(prefix='bkg_'))
        for j in range(n_peaks):
            if not np.isnan(centers[j]): models.append(MC(prefix=f'peak{j}_'))
        comp = reduce(lambda a,b: a+b, models)
        params=comp.make_params()
        if 'background_value' in ds_out: params['bkg_c'].set(value=bgv)
        for j in range(n_peaks):
            if not np.isnan(centers[j]):
                params[f'peak{j}_center'].set(value=centers[j])
                params[f'peak{j}_amplitude'].set(value=amps[j])
                params[f'peak{j}_sigma'].set(value=sigmas[j])
        best = comp.eval(params=params,x=bias)
        ax.plot(bias,best,'-',linewidth=4,color=col,alpha=0.5,label=f'{per} Fit (χ²={redchi:.2e})')
        # convolution
        if weight_function_show:
            wfunc = np.exp(-0.5*(bias/w_sigma)**2)
            conv = np.convolve(best,wfunc,mode='same')/np.sum(wfunc)
            ax.plot(bias,conv,'-',linewidth=3,color=col,alpha=0.8,label='Convoluted')
        # peaks and shapes
        cmap=get_cmap('tab20')(np.linspace(0,1,max(n_peaks,10)))
        alpha_p=0.5 if show_shapes else 1.0
        for j in range(n_peaks):
            key=f'peak{j}_'
            if key in comp.eval_components(params=params,x=bias):
                pk=comp.eval_components(params=params,x=bias)[key]
                ax.plot(bias,pk,'--',color=cmap[j],alpha=alpha_p,label=f'Peak {j}')
                if show_shapes and not np.isnan(level_prox[j]) and abs(level_prox[j])<SCgap:
                    lp=level_prox[j]; frac=abs((lp/E_mu)%1)
                    if frac>0.5: frac=1-frac
                    if lp==0: c,ls,fill='gray','-',True
                    elif np.isclose(frac,0): c,ls,fill='cyan','--',True
                    elif np.isclose(frac,0.5): c,ls,fill='magenta','--',True
                    else: c,ls,fill='gray','--',False
                    ax.axvline(lp,color=c,linestyle=ls)
                    if fill: ax.fill_between(bias,pk,color=c,alpha=0.3)
        ax.axvline(0,color='gray',linestyle=':')
        #ax.set_title(f'(Y={y},X={x},model={per})',fontsize=10)
        y_phys = ds_out['Y'].isel(Y=y).item()*1E9
        x_phys = ds_out['X'].isel(X=x).item()*1E9
        ax.set_title(
            f'Y_idx={y}, X_idx={x}, model={per}\n Y={y_phys:.2f} nm, X={x_phys:.2f} nm',
            fontsize=10
        )       
        if show_legend: ax.legend(fontsize=8,loc='upper right')
        else:
            v=~np.isnan(ldos)
            ax.set_xlim(bias[v].min(),bias[v].max()); ax.set_ylim(ldos[v].min(),ldos[v].max())
        ax.grid(True)
    # disable extras
    for k in range(n_points,nrow*ncol): axes[k].axis('off')
    plt.suptitle('Fit Result per Pixel',fontsize=16)
    plt.tight_layout()
    plt.show()



# +
#grid_LDOS_SnD_pks_results.sel( X = slice(0.5E-7,0.7E-7), Y = slice(2.0E-7,2.2E-7))
# -

plot_fitting_result_facet_grid(ds_opt2,#.sel( X = slice(0.5E-7,0.7E-7), Y = slice(2.0E-7,2.2E-7)), 
                              model_type=['Lorentzian', 'Gaussian', 'Voigt'], 
                                   point_list=60, 
                                   ncol=6, 
                                   weight_function_show=False,
                                   ZB_masking=True,
                                   show_legend=True,
                                   share_xy=False,show_shapes=False)





# # Figure 3 3x3 zoomed in pixel map --> clustering feature mapping 

# +
### choose selection area  with ZBM marker 
### c map, A map, W map for the area 
# -

ds_cluster

ds_cluster.LDOS.sel(bias_mV=0).plot()

ds_cluster.LDOS.sel(bias_mV=0).sel(X= slice(0.55E-7,0.6E-7), Y= slice (2.08E-7,2.13E-7)).plot()
# 10x 10 pixel 



ds_crop0= ds_cluster.sel(X= slice(0.55E-7,0.6E-7), Y= slice (2.08E-7,2.13E-7))

# +
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 1) 크롭 영역 경계 (m 단위)
x0, x1 = 0.55e-7, 0.6e-7
y0, y1 = 2.08e-7, 2.13e-7

# 2) 데이터 준비
full = ds_cluster.LDOS.sel(bias_mV=0)
crop = full.sel(X=slice(x0, x1), Y=slice(y0, y1))

# 3) 좌표 배열
X = full['X'].values
Y = full['Y'].values

# 4) Figure / Axes
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

# 5) 전체 영역 imshow (raster)
im0 = ax0.imshow(
    full.values,
    extent=(X.min(), X.max(), Y.min(), Y.max()),
    origin='lower',
    cmap='viridis',
    aspect='equal',
    interpolation='nearest',
    zorder=0
)
# 축 고정
ax0.set_xlim(X.min(), X.max())
ax0.set_ylim(Y.min(), Y.max())

# 6) 붉은 크롭 박스 (zorder=1)
rect = Rectangle(
    (x0, y0), x1 - x0, y1 - y0,
    fill=False, edgecolor='red', linewidth=2, zorder=1
)
ax0.add_patch(rect)

ax0.set_title('Full LDOS at bias=0 mV')
ax0.set_xlabel('X [nm]')
ax0.set_ylabel('Y [nm]')

# 7) 크롭 영역 imshow
im1 = ax1.imshow(
    crop.values,
    extent=(x0, x1, y0, y1),
    origin='lower',
    cmap='viridis',
    aspect='equal',
    interpolation='nearest'
)
ax1.set_title('Cropped Region')
ax1.set_xlabel('X [nm]')
ax1.set_ylabel('Y [nm]')

# 8) 컬러바 (공유)
cbar = fig.colorbar(im1, ax=(ax0, ax1),
                    orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label('LDOS')

# 9) SVG 저장
fig.savefig('ldos_full_and_cropped_final.svg', format='svg')

plt.show()

# -

ds_crop0



# ## 각 pixel별 peak들의 정보를  c,a,w 의 형태로 다시 mapping 

# +
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

def plot_peak_subcell(
    ds: xr.Dataset,
    mode: str = 'center',
    cmap_name: str = 'viridis',
    figsize: tuple = (8, 8)
):
    """
    ds의 peak 데이터를 subcell로 분할해 시각화합니다.
    mode='center'일 때는 0으로부터의 절대값 거리를 색상으로 표현하며,
    이 경우 컬러바 레이블에 절대값임을 표시합니다.
    subcell 전체 영역에는 굵은 테두리를 추가합니다.

    Returns
    -------
    (fig, ax), (demo_fig, demo_ax)
    """
    # 1) 변수 선택
    mode_map = {
        'center':    'peak_center',
        'width':     'peak_sigma',
        'amplitude': 'peak_amplitude'
    }
    if mode not in mode_map:
        raise ValueError(f"mode는 {list(mode_map)} 중 하나여야 합니다.")
    varname = mode_map[mode]
    raw = ds[varname].values          # (ny, nx, n_peaks)

    # 2) colormap 준비 (기존 로직)
    if cmap_name == 'berlin':
        if cmap_name in plt.colormaps():
            cmap = plt.get_cmap(cmap_name)
        else:
            try:
                from matplotlib_colormaps import cm as extra_cm
                plt.register_cmap(name='berlin', cmap=extra_cm.berlin)
                cmap = plt.get_cmap('berlin')
            except ImportError:
                cmap = plt.get_cmap('viridis')
    elif cmap_name == 'white_blue_purple_red_white':
        colors = [
            (1,1,1), (0,0,1),
            (0.4,0,0.6), (1,0,0),
            (1,1,1)
        ]
        cmap = LinearSegmentedColormap.from_list(cmap_name, colors)
    else:
        cmap = plt.get_cmap(cmap_name)

    # 3) subcell grid 크기 계산
    ny, nx, _ = raw.shape
    counts = np.sum(~np.isnan(raw), axis=2)
    m = int(np.nanmax(counts))
    s = int(np.ceil(np.sqrt(m)))
    row_sub = col_sub = s
    total_slots = s * s

    # 4) 실제 좌표 & 픽셀 크기
    x_coords = ds['X'].values
    y_coords = ds['Y'].values
    dx = np.mean(np.diff(x_coords))
    dy = np.mean(np.diff(y_coords))

    # 5) 'center' 모드 처리: 절대값, 컬러바 0 기준
    if mode == 'center':
        data = np.abs(raw)
        var_label = f"|{varname}|"
        norm = plt.Normalize(0, np.nanmax(data))
    else:
        data = raw
        var_label = varname
        norm = plt.Normalize(np.nanmin(data), np.nanmax(data))

    # 6) 메인 플롯
    fig, ax = plt.subplots(figsize=figsize)
    for yi in range(ny):
        for xi in range(nx):
            active = np.where(~np.isnan(data[yi, xi, :]))[0]
            n_act = active.size
            x0_pix = x_coords[xi] - dx/2
            y0_pix = y_coords[yi] - dy/2
            for j in range(total_slots):
                r = j // col_sub
                c = j % col_sub
                x0 = x0_pix + c*(dx/col_sub)
                y0 = y0_pix + (row_sub-1 - r)*(dy/row_sub)
                w = dx/col_sub; h = dy/col_sub
                if j < n_act:
                    val = data[yi, xi, active[j]]
                    rect = Rectangle((x0, y0), w, h,
                                     facecolor=cmap(norm(val)),
                                     edgecolor=None)
                    ax.add_patch(rect)
                else:
                    # ▽ 여기를 hatch 패턴으로 변경
                    rect = Rectangle(
                        (x0, y0), w, h,
                        facecolor='none',
                        edgecolor='black',
                        hatch='xx',    # 여기서 ‘xx’ 대신 '///', 'OO', ‘\\\\’ 등 원하는 패턴
                        linewidth=0.5
                    )
                    ax.add_patch(rect)

    # 7) 픽셀 경계선
    for xi in range(nx+1):
        ax.axvline(x_coords[0]-dx/2 + xi*dx, color='black', lw=0.8)
    for yi in range(ny+1):
        ax.axhline(y_coords[0]-dy/2 + yi*dy, color='black', lw=0.8)

    # → 전체 subcell 영역 테두리
    border = Rectangle(
        (x_coords[0]-dx/2, y_coords[0]-dy/2),
        nx*dx, ny*dy,
        fill=False, edgecolor='black',
        linewidth=1.5, zorder=5
    )
    ax.add_patch(border)

    # 8) 축 레이블 & 틱 포맷터 (nm 단위)
    def to_nm(x, pos):
        return f"{x*1e9:.1f}"
    ax.xaxis.set_major_formatter(FuncFormatter(to_nm))
    ax.yaxis.set_major_formatter(FuncFormatter(to_nm))

    ax.set_xlim(x_coords[0]-dx/2, x_coords[-1]+dx/2)
    ax.set_ylim(y_coords[0]-dy/2, y_coords[-1]+dy/2)
    ax.set_aspect('equal')
    ax.set_xlabel('X [nm]')
    ax.set_ylabel('Y [nm]')
    ax.set_title(f"{var_label} per Subcell")

    # 9) 컬러바
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical',
                        fraction=0.046, pad=0.04)
    if mode == 'center':
        cbar.set_label(f"{var_label} (distance from 0)")
    else:
        cbar.set_label(var_label)

    plt.tight_layout()

    # 10) 데모 플롯: subcell 인덱스 시각화 (변경 없음)
    demo_fig, demo_ax = plt.subplots(figsize=(2, 2))
    for j in range(total_slots):
        r = j // col_sub
        c = j % col_sub
        x0 = c*(1/s)
        y0 = (s-1-r)*(1/s)
        demo_ax.add_patch(Rectangle((x0, y0), 1/s, 1/s,
                                   fill=False, edgecolor='black'))
        if j < m:
            demo_ax.text(x0+0.5/s, y0+0.5/s, str(j),
                         ha='center', va='center', fontsize=12)
        else:
            # 굵은 hatch demo
            demo_ax.add_patch(
                Rectangle((x0, y0), 1/s, 1/s,
                          facecolor='none',
                          edgecolor='black',
                          hatch='xx',
                          linewidth=1.0)
            )

    demo_ax.set_xlim(0, 1)
    demo_ax.set_ylim(0, 1)
    demo_ax.set_aspect('equal')
    demo_ax.axis('off')
    demo_ax.set_title(f"Subcell demo ({s}×{s}, m={m})")

    plt.tight_layout()
    return (fig, ax), (demo_fig, demo_ax)



# -

# ## 'center'|'width'|'amplitude' 중하나를 골라서 plot, 색깔은 그에 맞춰서 변화 

# 기본 호출: peak_center, custom cmap, 8″×8″
(main_fig, main_ax), (demo_fig, demo_ax) = plot_peak_subcell(
    ds_crop0,
    mode='center',                         # 'center'|'width'|'amplitude'
    cmap_name='Purples_r',
    #mode='width',                         # 'center'|'width'|'amplitude'
    #cmap_name='Greens_r',
    #mode='amplitude',                         # 'center'|'width'|'amplitude'
    #cmap_name='Blues_r',
    #cmap_name='white_blue_purple_red_white',
    #cmap_name='Greens_r',
    figsize=(8, 8)
)
plt.show()  # 두 개의 Figure 가 차례로 나타납니다
# 2) SVG로 저장
main_fig.savefig('peak_center_subcell.svg', format='svg')
demo_fig.savefig('subcell_demo.svg', format='svg')



# +
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import ipywidgets as widgets
from IPython.display import display, clear_output
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter

# ── Globals to hold user's selection ──
selected_cluster_var = None
selected_cluster_labels = None

def select_cluster_data_var_interactive(ds, callback):
    """
    Let the user pick which ds.data_vars contains per-peak cluster labels.
    On confirm, sets global `selected_cluster_var` and calls callback(var).
    """
    global selected_cluster_var
    selector = widgets.RadioButtons(
        options=[k for k in ds.data_vars if 'cluster' in k],
        description='Cluster var:'
    )
    btn = widgets.Button(description='Confirm', button_style='primary')
    out = widgets.Output()
    def on_confirm(_):
        global selected_cluster_var
        with out:
            clear_output()
            selected_cluster_var = selector.value
            print(f"✅ Selected cluster var: {selected_cluster_var}")
            callback(selected_cluster_var)
    btn.on_click(on_confirm)
    display(widgets.VBox([selector, btn, out]))

def select_labels_in_cluster_interactive(ds, callback):
    """
    After cluster var is chosen, let user pick one or more integer labels.
    Excludes noise label (-1). On confirm, calls callback(labels).
    """
    global selected_cluster_var, selected_cluster_labels
    if selected_cluster_var is None:
        raise RuntimeError("먼저 클러스터 변수를 선택하세요.")
    da = ds[selected_cluster_var]
    labels = np.unique(da.values[~np.isnan(da.values)])
    labels = sorted(int(l) for l in labels if l >= 0)
    selector = widgets.SelectMultiple(options=labels, description='Highlight labels:')
    btn = widgets.Button(description='Confirm', button_style='success')
    out = widgets.Output()
    def on_confirm(_):
        global selected_cluster_labels
        with out:
            clear_output()
            selected_cluster_labels = list(selector.value)
            print(f"✅ Highlight labels: {selected_cluster_labels}")
            callback(selected_cluster_labels)
    btn.on_click(on_confirm)
    display(widgets.VBox([selector, btn, out]))

def interactive_plot_peak_subcell_clusters(ds: xr.Dataset, figsize=(8,8)):
    """
    1) Select cluster var → 2) Select labels → 
    3) Plot fixed 3×3 subcell grid, mark unused/noise as 'X',
       remove outer margins, show legend + Save SVG button.
    """
    def on_var(var):
        # --- 반드시 전역변수에 저장해야 이후 on_labels 에서 사용됩니다 ---
        global selected_cluster_var
        selected_cluster_var = var
        select_labels_in_cluster_interactive(ds, callback=on_labels)

    def on_labels(chosen_labels):
        global selected_cluster_var
        clust = ds[selected_cluster_var].values  # shape (Y, X, peak)
        ny, nx, npk = clust.shape
        X = ds['X'].values; Y = ds['Y'].values
        dx = np.mean(np.diff(X)); dy = np.mean(np.diff(Y))

        # fixed 3x3
        s = 3
        total = s * s

        # prepare colormap
        unique = sorted(int(l) for l in np.unique(clust[np.isfinite(clust)]) if l>=0)
        base = plt.get_cmap('tab10')
        palette = base.colors if hasattr(base, 'colors') else base(np.arange(10))
        cmap = ListedColormap(palette[:len(unique)])
        lab2idx = {lab:i for i,lab in enumerate(unique)}

        # --- draw ---
        fig, ax = plt.subplots(figsize=figsize)
        # remove outer margins
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        for yi in range(ny):
            for xi in range(nx):
                act = np.where(~np.isnan(clust[yi,xi,:]))[0]
                x0pix = X[xi] - dx/2
                y0pix = Y[yi] - dy/2
                for j in range(total):
                    r, c = divmod(j, s)
                    x0 = x0pix + c*(dx/s)
                    y0 = y0pix + (s-1-r)*(dy/s)
                    w, h = dx/s, dy/s

                    if j < len(act):
                        # 활성된 peak
                        pk = act[j]
                        lab = int(clust[yi,xi,pk])
                        if lab in chosen_labels:
                            # 선택된 클러스터 → 컬러 채움
                            face = cmap(lab2idx[lab])
                            rect = Rectangle((x0,y0), w, h,
                                             facecolor=face, edgecolor=None)
                        else:
                            # 선택되지 않은 또는 noise(-1) → hatch 처리
                            rect = Rectangle((x0,y0), w, h,
                                             facecolor='none',
                                             edgecolor='black',
                                             hatch='XX',
                                             linewidth=0.5)
                        ax.add_patch(rect)

                    else:
                        # 빈 슬롯 → hatch 처리
                        rect = Rectangle((x0,y0), w, h,
                                         facecolor='none',
                                         edgecolor='black',
                                         hatch='XX',
                                         linewidth=0.5)
                        ax.add_patch(rect)

        # pixel grid + border
        for xi in range(nx+1):
            ax.axvline(X[0]-dx/2 + xi*dx, color='black', lw=0.8)
        for yi in range(ny+1):
            ax.axhline(Y[0]-dy/2 + yi*dy, color='black', lw=0.8)
        ax.add_patch(Rectangle((X[0]-dx/2, Y[0]-dy/2),
                               nx*dx, ny*dy,
                               fill=False, edgecolor='black', linewidth=2))

        # fix limits
        ax.set_xlim(X[0]-dx/2, X[-1]+dx/2)
        ax.set_ylim(Y[0]-dy/2, Y[-1]+dy/2)

        # axis formatting
        def to_nm(x, pos): return f"{x*1e9:.1f}"
        ax.xaxis.set_major_formatter(FuncFormatter(to_nm))
        ax.yaxis.set_major_formatter(FuncFormatter(to_nm))
        ax.set_xlabel('X [nm]'); ax.set_ylabel('Y [nm]')
        ax.set_aspect('equal')
        ax.set_title(f"Peaks colored by {selected_cluster_var}")

        # legend
        handles = [Rectangle((0,0),1,1, facecolor=cmap(lab2idx[l]), edgecolor='none')
                   for l in unique]
        ax.legend(handles, [str(l) for l in unique],
                  title=selected_cluster_var,
                  bbox_to_anchor=(1,1))

        plt.show()

        # ── Save SVG button ──
        save_btn = widgets.Button(description='Save SVG', button_style='info')
        save_out = widgets.Output()
        def on_save(_):
            fname = f"peaks_{selected_cluster_var}.svg"
            fig.savefig(fname, format='svg', bbox_inches='tight')
            with save_out:
                clear_output()
                print(f"✅ Saved SVG as '{fname}'")
        save_btn.on_click(on_save)
        display(widgets.HBox([save_btn]), save_out)

    # Step 1: select cluster var
    select_cluster_data_var_interactive(ds, callback=on_var)



# -

# 한 번만 호출하면, 차례로 대화형 위젯이 떠서
# 1) cluster 변수 선택 → 2) highlight할 레이블 선택 → 3) 결과 시각화
interactive_plot_peak_subcell_clusters(ds_crop0)





# ### plot_region_fitting_result_from_dsfit 함수 테스트 

fig,_ = plot_region_fitting_result_from_dsfit(ds_opt2,
                                              use_zb_mask=True,
                                              zb_mask_key='ZB_mask',
                                              allowed_models=['Lorentzian', 'Gaussian', 'Voigt'],
                                              show_shade=False,
                                              y_idx=69,
                                              x_idx=148,
                                              return_fig=True)
#fig

# ##  peak feature extraction map for crop

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from lmfit.models import LorentzianModel, GaussianModel, VoigtModel, ConstantModel
from functools import reduce

def plot_region_fitting_result_from_dsfit(
    ds_fit: xr.Dataset,
    model_type: str = None,
    allowed_models: list = ['Lorentzian','Gaussian','Voigt'],
    y_idx: int = None,
    x_idx: int = None,
    weight_function_show: bool = False,
    use_zb_mask: bool = False,
    zb_mask_key: str = 'ZB_mask',
    show_shade: bool = True,
    show_legend: bool = True,
    return_fig: bool = False
):
    """
    Reconstruct and plot the best-fit LDOS curve for a single pixel.

    Parameters
    ----------
    ds_fit : xarray.Dataset
        Combined dataset with dims ('Y','X','bias_mV','peak') and variables
        'LDOS','bias_mV','peak_center','peak_amplitude','peak_sigma','redchi'.
        Optional: 'background_value', 'model_type', zb_mask_key mask.
        Must also have attrs 'weight_sigma','Ef','SCgap' if weight or shading used.
    model_type : str or None
        If None, uses ds_fit['model_type'] at (y_idx,x_idx); otherwise forces this model.
    allowed_models : list of str
        Valid model names when model_type is None.
    y_idx, x_idx : int or None
        Pixel indices. If both None and use_zb_mask=True, picks a valid pixel from mask.
    weight_function_show : bool
        If True, overlay the convolution of the fit with the weight function.
    use_zb_mask : bool
        If True, mask out invalid bias points via ds_fit[zb_mask_key].
    zb_mask_key : str
        Name of the zero-bias mask variable.
    show_shade : bool
        If True, shade under individual peaks based on CdGM proximity.
    show_legend : bool
        If True, display legend.
    return_fig : bool
        If True, return (fig, df); otherwise show plot and return df.

    Returns
    -------
    fig : matplotlib.figure.Figure (optional)
    df : pandas.DataFrame
        Indexed by bias_mV with columns 'LDOS','best_fit','peak0',…,
        and optional 'convoluted_fit','weight_function'.
    """
    # 1) determine pixel indices
    Ny, Nx = ds_fit.dims['Y'], ds_fit.dims['X']
    if use_zb_mask and y_idx is None and x_idx is None and zb_mask_key in ds_fit:
        raw = ds_fit[zb_mask_key].values
        valid = np.any(~np.isnan(raw), axis=-1) if raw.ndim==3 else raw.astype(bool)
        redchi = ds_fit['redchi'].values
        ys, xs = np.where(valid & ~np.isnan(redchi))
        if ys.size > 0:
            sel = np.random.randint(len(ys))
            y_idx, x_idx = int(ys[sel]), int(xs[sel])
    if y_idx is None:
        y_idx = np.random.randint(Ny)
    if x_idx is None:
        x_idx = np.random.randint(Nx)

    # 2) extract bias & LDOS
    bias = ds_fit['bias_mV'].values
    ldos = ds_fit['LDOS'].isel(Y=y_idx, X=x_idx).values

    # 3) select model class
    if model_type is None:
        chosen = ds_fit['model_type'].isel(Y=y_idx, X=x_idx).item().capitalize()
    else:
        chosen = model_type.capitalize()
    if chosen not in allowed_models:
        raise ValueError(f"Model '{chosen}' not in {allowed_models}")
    ModelClass = {
        'Lorentzian': LorentzianModel,
        'Gaussian':   GaussianModel,
        'Voigt':      VoigtModel
    }[chosen]

    # 4) gather valid peak parameters
    centers = ds_fit['peak_center'].isel(Y=y_idx, X=x_idx).values
    amps    = ds_fit['peak_amplitude'].isel(Y=y_idx, X=x_idx).values
    sigmas  = ds_fit['peak_sigma'].isel(Y=y_idx, X=x_idx).values
    peak_idxs = [
        i for i, (c,a,s) in enumerate(zip(centers,amps,sigmas))
        if not (np.isnan(c) or np.isnan(a) or np.isnan(s))
    ]

    # 5) build composite model & set params
    models = []
    if 'background_value' in ds_fit:
        models.append(ConstantModel(prefix='bkg_'))
        bgv = ds_fit['background_value'].isel(Y=y_idx, X=x_idx).item()
    for i in peak_idxs:
        models.append(ModelClass(prefix=f'peak{i}_'))
    comp = reduce(lambda a,b: a + b, models)
    params = comp.make_params()
    if 'background_value' in ds_fit:
        params['bkg_c'].set(value=bgv)
    for i in peak_idxs:
        params[f'peak{i}_center'].set(value=centers[i])
        params[f'peak{i}_amplitude'].set(value=amps[i])
        params[f'peak{i}_sigma'].set(value=sigmas[i])

    # 6) evaluate best-fit & components
    best_fit  = comp.eval(params=params, x=bias)
    comps_vals = comp.eval_components(params=params, x=bias)

    # 7) optional weight-function convolution
    if weight_function_show:
        wsig = ds_fit.attrs.get('weight_sigma',1.0)
        wfunc = np.exp(-bias**2/(2*wsig**2))
        conv  = np.convolve(best_fit, wfunc, mode='same')/np.sum(wfunc)

    # 8) plot
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(bias, ldos,      'k-', lw=1.5, alpha=0.8, label='LDOS', zorder=1)
    ax.plot(bias, best_fit,  '-', color='C1', lw=3, label=f'{chosen} Fit', zorder=2)
    if weight_function_show:
        ax.plot(bias, conv, '--', color='C1', lw=2, label='Convoluted Fit', zorder=1)
    for i in peak_idxs:
        ax.plot(bias, comps_vals[f'peak{i}_'], '--', lw=1, alpha=0.6, label=f'Peak {i}', zorder=1)

    # 9) optional CdGM shading
    if show_shade and 'level_proximity' in ds_fit:
        lvl = ds_fit['level_proximity'].isel(Y=y_idx, X=x_idx).values
        Ef  = ds_fit.attrs.get('Ef',1.0)
        SCg = ds_fit.attrs.get('SCgap',1.0)
        E_mu = SCg**2/Ef
        for i in peak_idxs:
            lp = lvl[i]
            if not np.isnan(lp) and abs(lp)<SCg:
                ax.fill_between(bias, 0, comps_vals[f'peak{i}_'], color='C2', alpha=0.3)

    ax.axvline(0, color='gray', ls=':', lw=1)
    ax.set_xlabel('Bias (mV)')
    ax.set_ylabel('LDOS')
    if show_legend:
        ax.legend(fontsize=8, loc='best')
    plt.tight_layout()

    # 10) assemble DataFrame
    data = {'LDOS': ldos, 'best_fit': best_fit}
    for name, arr in comps_vals.items():
        data[name.rstrip('_')] = arr
    if weight_function_show:
        data['convoluted_fit']  = conv
        data['weight_function'] = wfunc
    df = pd.DataFrame(data, index=bias)
    df.index.name = 'bias_mV'

    return (fig, df) if return_fig else df


def plot_fitting_results_grid(
    ds_cluster: xr.Dataset,
    figsize_per_cell: tuple = (3, 3),
    remove_subtitles: bool = False,
    remove_labels: bool = False,
    **fit_kwargs
):
    """
    Arrange per-pixel fits in an Ny×Nx grid by embedding the rasterized
    output of plot_region_fitting_result_from_dsfit into each cell.

    Parameters
    ----------
    ds_cluster : xarray.Dataset
        Dataset with dims ('Y','X',...) to iterate pixels over.
    figsize_per_cell : tuple
        Width and height (inches) for each subplot cell.
    remove_subtitles : bool
        If True, clear the inner plot titles.
    remove_labels : bool
        If True, clear the inner x/y axis labels and ticks.
    **fit_kwargs :
        Keyword arguments passed directly to
        plot_region_fitting_result_from_dsfit.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray of Axes, shape (Ny, Nx)
    fit_dfs : dict[(y_idx, x_idx) -> pandas.DataFrame]
    """
    Ny, Nx = ds_cluster.dims['Y'], ds_cluster.dims['X']

    # create tightly spaced grid
    fig, axes = plt.subplots(
        Ny, Nx,
        figsize=(figsize_per_cell[0]*Nx, figsize_per_cell[1]*Ny),
        gridspec_kw={'wspace':0, 'hspace':0, 'left':0, 'right':1, 'top':1, 'bottom':0}
    )
    axes = np.atleast_2d(axes).reshape(Ny, Nx)

    fit_dfs = {}

    for yi in range(Ny):
        for xi in range(Nx):
            ax = axes[yi, xi]

            # generate the per-pixel figure
            fig_fit, df_fit = plot_region_fitting_result_from_dsfit(
                ds_cluster,
                y_idx=yi,
                x_idx=xi,
                return_fig=True,
                **fit_kwargs
            )

            # optionally strip title/labels
            src_ax = fig_fit.axes[0]
            if remove_subtitles:
                src_ax.set_title('')
            if remove_labels:
                src_ax.set_xlabel(''); src_ax.set_ylabel('')
                src_ax.set_xticks([]); src_ax.set_yticks([])

            # render to buffer
            fig_fit.canvas.draw()
            w, h = fig_fit.canvas.get_width_height()
            buf = np.frombuffer(fig_fit.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(h, w, 3)

            # embed with correct orientation
            ax.imshow(buf, origin='upper', aspect='auto')
            ax.axis('off')

            plt.close(fig_fit)
            fit_dfs[(yi, xi)] = df_fit

            # draw a thin border around each cell
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)

    fig.suptitle("Per-Pixel Fitting Results", fontsize=16)
    return fig, axes, fit_dfs



# -

fig, axes, fit_dfs = plot_fitting_results_grid(
    ds_crop0,
    figsize_per_cell=(2.5, 2.5),
    remove_subtitles=False,
    remove_labels=False,
    use_zb_mask=True,
    zb_mask_key='ZB_mask',
    allowed_models=['Lorentzian','Gaussian','Voigt'],
    show_shade=False,
    show_legend = False,
    weight_function_show=False,
)
plt.show()

# # cropped 영역에 대해서 subgrid curve plot 새로 

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from lmfit.models import LorentzianModel, GaussianModel, VoigtModel, ConstantModel
from functools import reduce

def plot_region_fitting_result_from_dsfit_no_label(
    ds_fit: xr.Dataset,
    model_type: str = None,
    allowed_models: list = ['Lorentzian','Gaussian','Voigt'],
    y_idx: int = None,
    x_idx: int = None,
    weight_function_show: bool = False,
    use_zb_mask: bool = False,
    zb_mask_key: str = 'ZB_mask',
    show_shade: bool = True,
    return_fig: bool = False
):
    """
    Reconstruct and plot the best‐fit LDOS for a single pixel without any
    titles, axis labels, tick labels, background offset line, or vertical guide lines.
    All fit curves are drawn in gray.

    Parameters
    ----------
    ds_fit : xarray.Dataset
        Must contain dimensions ('Y','X','bias_mV','peak') and variables:
        'LDOS', 'bias_mV', 'peak_center', 'peak_amplitude', 'peak_sigma', 'redchi'.
        Optionally 'background_value', 'model_type', and a zero‐bias mask under zb_mask_key.
        Attributes 'weight_sigma', 'Ef', 'SCgap' control convolution and shading.
    model_type : str or None
        If None, use ds_fit['model_type'] at (y_idx,x_idx); otherwise force this model.
    allowed_models : list of str
        Valid model names if model_type is None.
    y_idx, x_idx : int or None
        Pixel indices. If both None and use_zb_mask=True, selects a valid pixel by mask.
    weight_function_show : bool
        If True, overlay the convolution of the fit with the weight function.
    use_zb_mask : bool
        If True, mask out invalid bias points using ds_fit[zb_mask_key].
    zb_mask_key : str
        Name of zero‐bias mask variable in ds_fit.
    show_shade : bool
        If True, compute CdGM proximity but do not draw vertical lines.
    return_fig : bool
        If True, return (fig, df); otherwise return df only.

    Returns
    -------
    fig : matplotlib.figure.Figure (optional)
    df : pandas.DataFrame
        Indexed by bias_mV, columns: 'LDOS','best_fit','peak0',…,'convoluted_fit','weight_function'.
    """
    # 1) Determine pixel indices
    ny, nx = ds_fit.dims['Y'], ds_fit.dims['X']
    if use_zb_mask and y_idx is None and x_idx is None and zb_mask_key in ds_fit:
        raw_mask = ds_fit[zb_mask_key].values
        valid_mask = np.any(~np.isnan(raw_mask), axis=2) if raw_mask.ndim==3 else raw_mask.astype(bool)
        rc = ds_fit['redchi'].values
        ys, xs = np.where(valid_mask & ~np.isnan(rc))
        if ys.size:
            sel = np.random.randint(len(ys))
            y_idx, x_idx = int(ys[sel]), int(xs[sel])
    if y_idx is None:
        y_idx = np.random.randint(ny)
    if x_idx is None:
        x_idx = np.random.randint(nx)

    # 2) Extract bias and LDOS
    bias = ds_fit['bias_mV'].values
    ldos = ds_fit['LDOS'].isel(Y=y_idx, X=x_idx).values

    # 3) Select fitting model
    if model_type is None:
        chosen = ds_fit['model_type'].isel(Y=y_idx, X=x_idx).item().capitalize()
    else:
        chosen = model_type.capitalize()
    if chosen not in allowed_models:
        raise ValueError(f"Model '{chosen}' not in {allowed_models}")
    ModelClass = {'Lorentzian':LorentzianModel, 'Gaussian':GaussianModel, 'Voigt':VoigtModel}[chosen]

    # 4) Gather valid peak parameters
    centers = ds_fit['peak_center'].isel(Y=y_idx, X=x_idx).values
    amps    = ds_fit['peak_amplitude'].isel(Y=y_idx, X=x_idx).values
    sigmas  = ds_fit['peak_sigma'].isel(Y=y_idx, X=x_idx).values
    peak_idxs = [i for i,(c,a,s) in enumerate(zip(centers,amps,sigmas))
                 if not (np.isnan(c) or np.isnan(a) or np.isnan(s))]

    # 5) Build composite model and set params
    models = []
    if 'background_value' in ds_fit:
        models.append(ConstantModel(prefix='bkg_'))
        bgv = ds_fit['background_value'].isel(Y=y_idx, X=x_idx).item()
    for i in peak_idxs:
        models.append(ModelClass(prefix=f'peak{i}_'))
    comp = reduce(lambda a,b: a+b, models)
    params = comp.make_params()
    if 'background_value' in ds_fit:
        params['bkg_c'].set(value=bgv)
    for i in peak_idxs:
        params[f'peak{i}_center'].set(value=centers[i])
        params[f'peak{i}_amplitude'].set(value=amps[i])
        params[f'peak{i}_sigma'].set(value=sigmas[i])

    # 6) Evaluate best-fit and individual components
    best_fit  = comp.eval(params=params, x=bias)
    comps_vals = comp.eval_components(params=params, x=bias)

    # 7) Optional weight-function convolution
    if weight_function_show:
        wsig = ds_fit.attrs.get('weight_sigma',1.0)
        wfunc = np.exp(-bias**2/(2*wsig**2))
        conv  = np.convolve(best_fit, wfunc, mode='same')/np.sum(wfunc)

    # 8) Plot all curves in gray, remove background offset line and vertical guides
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(bias, ldos,      color='lightgray', linestyle=':', linewidth=1, zorder=1)
    ax.plot(bias, best_fit,  color='gray',       linestyle='-', linewidth=3, zorder=2)
    for i in peak_idxs:
        ax.plot(bias, comps_vals[f'peak{i}_'], color='gray', linestyle='--', linewidth=1, zorder=1)
    if weight_function_show:
        ax.plot(bias, conv, color='gray', linestyle='-.', linewidth=2, zorder=1)

    # 9) CdGM shading logic without drawing vertical lines
    if show_shade and 'level_proximity' in ds_fit:
        lvl = ds_fit['level_proximity'].isel(Y=y_idx, X=x_idx).values
        Ef  = ds_fit.attrs.get('Ef',1.0)
        SCg = ds_fit.attrs.get('SCgap',1.0)
        E_mu = SCg**2/Ef
        # intentionally do NOT call ax.axvline here

    # 10) Remove titles, axis labels, tick labels; keep only spines
    ax.set_title('')
    ax.set_xlabel(''); ax.set_ylabel('')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)

    plt.tight_layout()

    # 11) Assemble DataFrame
    data = {'LDOS': ldos, 'best_fit': best_fit}
    for name, arr in comps_vals.items():
        if name != 'bkg_':
            data[name.rstrip('_')] = arr
    if weight_function_show:
        data['convoluted_fit']  = conv
        data['weight_function'] = wfunc
    df = pd.DataFrame(data, index=bias)
    df.index.name = 'bias_mV'

    return (fig, df) if return_fig else df


def plot_fitting_results_grid(
    ds_cluster: xr.Dataset,
    figsize_per_cell: tuple = (3, 3),
    **fit_kwargs
):
    """
    Arrange per-pixel fitting plots in an Ny×Nx grid with no spacing,
    embedding the rasterized output of
    plot_region_fitting_result_from_dsfit_no_label into each cell
    and drawing a thin border around each.

    Parameters
    ----------
    ds_cluster : xarray.Dataset
        Dataset with dims ('Y','X',...) to iterate over pixels.
    figsize_per_cell : tuple (width, height) in inches
        Size of each small subplot.
    **fit_kwargs :
        All keyword arguments passed on to
        plot_region_fitting_result_from_dsfit_no_label.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray of Axes, shape (Ny, Nx)
    fit_dfs : dict[(y_idx, x_idx) -> pandas.DataFrame]
    """
    Ny, Nx = ds_cluster.dims['Y'], ds_cluster.dims['X']

    fig, axes = plt.subplots(
        Ny, Nx,
        figsize=(figsize_per_cell[0] * Nx, figsize_per_cell[1] * Ny),
        gridspec_kw={'wspace':0, 'hspace':0, 'left':0, 'right':1, 'top':1, 'bottom':0}
    )
    axes = np.atleast_2d(axes).reshape(Ny, Nx)

    fit_dfs = {}

    for yi in range(Ny):
        for xi in range(Nx):
            ax = axes[yi, xi]

            fig_fit, df_fit = plot_region_fitting_result_from_dsfit_no_label(
                ds_cluster,
                y_idx=yi,
                x_idx=xi,
                return_fig=True,
                **fit_kwargs
            )

            fig_fit.canvas.draw()
            w, h = fig_fit.canvas.get_width_height()
            buf = np.frombuffer(fig_fit.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(h, w, 3)  # no vertical flip

            ax.imshow(buf, origin='upper', aspect='auto')  # use upper
            ax.set_xticks([]); ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
                spine.set_color('black')

            plt.close(fig_fit)
            fit_dfs[(yi, xi)] = df_fit

    fig.suptitle("Per-Pixel Fitting Results (No Labels)", fontsize=16)
    return fig, axes, fit_dfs



# +
import matplotlib.pyplot as plt

# 1) 필요한 함수들(위에서 정의한 두 함수)를 이미 불러왔다고 가정합니다.

# 2) 그리드 전체에 “no‐label” 버전 피팅 결과를 임베드
fig, axes, fit_dfs = plot_fitting_results_grid(
    ds_cluster=ds_crop0,                  # 사용하실 Dataset
    figsize_per_cell=(2.5, 2.5),          # 셀 하나당 크기 (inch)
    # plot_region_fitting_result_from_dsfit_no_label 에 전달할 옵션들
    weight_function_show=False,
    use_zb_mask=True,
    zb_mask_key='ZB_mask',
    show_shade=True,
    model_type=None,                      # None 이면 ds_fit['model_type'] 사용
    allowed_models=['Lorentzian','Gaussian','Voigt']
)

# 3) 화면에 표시
plt.show()

# 4) (선택) SVG 로 저장
fig.savefig('fits_no_labels_grid.svg', format='svg')

# 5) (선택) 특정 픽셀의 DataFrame 확인
# 예: Y=3, X=4 픽셀
#df_3_4 = fit_dfs[(3,4)]
#print(df_3_4.head())

# -

ds_cluster.sel(cluster_label=2)



# ###  cluster label만 따로 뽑아서 coloring 
#
#

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import ipywidgets as widgets
from IPython.display import display, clear_output
from lmfit.models import LorentzianModel, GaussianModel, VoigtModel, ConstantModel
from functools import reduce
from matplotlib.cm import get_cmap

# ── 인터랙티브 헬퍼 1: 어떤 data_var가 클러스터 레이블인지 선택 ──
selected_cluster_var_global = None
def select_cluster_data_var_interactive(ds, callback=None):
    """
    Let the user pick which ds.data_vars contains the per-peak cluster labels.
    """
    global selected_cluster_var_global
    selector = widgets.RadioButtons(
        options=[v for v in ds.data_vars if 'cluster' in v],
        description='Cluster var:'
    )
    btn = widgets.Button(description='Confirm', button_style='primary')
    out = widgets.Output()

    def on_confirm(_):
        global selected_cluster_var_global
        with out:
            clear_output()
            selected_cluster_var_global = selector.value
            print(f"✅ cluster var: {selected_cluster_var_global}")
            if callback:
                callback(selected_cluster_var_global)

    btn.on_click(on_confirm)
    display(widgets.VBox([selector, btn, out]))


# ── 인터랙티브 헬퍼 2: 어떤 레이블(들)을 하이라이트할지 선택 ──
selected_cluster_labels_global = None
def select_labels_in_cluster_interactive(ds, callback=None):
    """
    After cluster var is chosen, pick one or more integer labels to highlight.
    """
    global selected_cluster_var_global, selected_cluster_labels_global
    if not selected_cluster_var_global:
        raise ValueError("먼저 select_cluster_data_var_interactive를 호출하세요.")
    da = ds[selected_cluster_var_global]
    labels = sorted(int(x) for x in np.unique(da.values) if not np.isnan(x))
    selector = widgets.SelectMultiple(options=labels, description='Labels:')
    btn = widgets.Button(description='Confirm', button_style='success')
    out = widgets.Output()

    def on_confirm(_):
        global selected_cluster_labels_global
        with out:
            clear_output()
            selected_cluster_labels_global = list(selector.value)
            print(f"✅ highlight labels: {selected_cluster_labels_global}")
            if callback:
                callback(selected_cluster_labels_global)

    btn.on_click(on_confirm)
    display(widgets.VBox([selector, btn, out]))


# ── 픽셀 단위 fitting plot (레이블·틱 없이) ──
def plot_region_fitting_result_from_dsfit_no_label(
    ds_fit: xr.Dataset,
    y_idx:int, x_idx:int,
    model_type: str=None,
    allowed_models:list=['Lorentzian','Gaussian','Voigt'],
    weight_function_show:bool=False,
    use_zb_mask:bool=False,
    zb_mask_key:str='ZB_mask',
    show_shade:bool=True,
    highlight_labels:list=None,
    return_fig:bool=False
):
    """
    Single‐pixel fit plot without titles/axes/ticks.
    Only fill under those peak‐components whose cluster label ∈ highlight_labels.
    """
    # 1) 데이터 꺼내기
    bias = ds_fit['bias_mV'].values
    ldos = ds_fit['LDOS'].isel(Y=y_idx,X=x_idx).values

    # 2) 모델 클래스 선택
    if model_type is None:
        chosen = ds_fit['model_type'].isel(Y=y_idx,X=x_idx).item().capitalize()
    else:
        chosen = model_type.capitalize()
    if chosen not in allowed_models:
        raise ValueError(f"Model '{chosen}' not in {allowed_models}")
    ModelClass = {'Lorentzian':LorentzianModel,
                  'Gaussian':  GaussianModel,
                  'Voigt':      VoigtModel}[chosen]

    # 3) peak 파라미터 가져오기
    centers = ds_fit['peak_center'].isel(Y=y_idx,X=x_idx).values
    amps    = ds_fit['peak_amplitude'].isel(Y=y_idx,X=x_idx).values
    sigmas  = ds_fit['peak_sigma'].isel(Y=y_idx,X=x_idx).values
    peak_idxs = [
        i for i,(c,a,s) in enumerate(zip(centers,amps,sigmas))
        if not (np.isnan(c) or np.isnan(a) or np.isnan(s))
    ]

    # 4) composite 모델 구성
    models = []
    if 'background_value' in ds_fit:
        models.append(ConstantModel(prefix='bkg_'))
        bgv = ds_fit['background_value'].isel(Y=y_idx,X=x_idx).item()
    for i in peak_idxs:
        models.append(ModelClass(prefix=f'peak{i}_'))
    comp   = reduce(lambda a,b: a+b, models)
    params = comp.make_params()
    if 'background_value' in ds_fit:
        params['bkg_c'].set(value=bgv)
    for i in peak_idxs:
        params[f'peak{i}_center'].set(value=centers[i])
        params[f'peak{i}_amplitude'].set(value=amps[i])
        params[f'peak{i}_sigma'].set(value=sigmas[i])

    # 5) fitting 결과 계산
    best_fit   = comp.eval(params=params, x=bias)
    comps_vals = comp.eval_components(params=params, x=bias)

    # 6) (선택) weight-function convolution
    if weight_function_show:
        wsig = ds_fit.attrs.get('weight_sigma',1.0)
        wfunc = np.exp(-bias**2/(2*wsig**2))
        conv  = np.convolve(best_fit, wfunc, mode='same')/np.sum(wfunc)

    # 7) 기본 회색 곡선만 그리기
    fig, ax = plt.subplots(figsize=(4,3))
    ax.plot(bias, ldos,     color='lightgray', linestyle=':', linewidth=1, zorder=1)
    ax.plot(bias, best_fit, color='gray',       linestyle='-', linewidth=3, zorder=2)
    for i in peak_idxs:
        ax.plot(bias, comps_vals[f'peak{i}_'], color='gray',
                linestyle='--', linewidth=1, zorder=1)
    if weight_function_show:
        ax.plot(bias, conv, color='gray', linestyle='-.', linewidth=2, zorder=1)

    # 8) highlight
    if highlight_labels and selected_cluster_var_global:
        cluster_arr = ds_fit[selected_cluster_var_global] \
                         .isel(Y=y_idx,X=x_idx).values
        cmap = get_cmap('Greens')
        for i in peak_idxs:
            if cluster_arr[i] in highlight_labels:
                ax.fill_between(
                    bias,
                    comps_vals[f'peak{i}_'],
                    0,
                    facecolor=cmap(0.6),
                    alpha=0.3,
                    zorder=0
                )

    # 9) 제목·레이블·틱 제거, spine만 남기기
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)

    plt.tight_layout()

    # 10) DataFrame 구성
    data = {'LDOS': ldos, 'best_fit': best_fit}
    for name, arr in comps_vals.items():
        if name!='bkg_':
            data[name.rstrip('_')] = arr
    if weight_function_show:
        data['convoluted_fit']  = conv
        data['weight_function'] = wfunc
    df = pd.DataFrame(data, index=bias)
    df.index.name = 'bias_mV'

    return (fig, df) if return_fig else df


# ── 픽셀별 그림을 격자에 임베드 ──
def plot_fitting_results_grid(
    ds_cluster: xr.Dataset,
    figsize_per_cell=(2.5,2.5),
    highlight_labels=None,
    **fit_kwargs
):
    """
    Tile per-pixel no-label fits in an Ny×Nx grid with no spacing,
    draw thin border, and forward highlight_labels.
    """
    Ny, Nx = ds_cluster.dims['Y'], ds_cluster.dims['X']
    fig, axes = plt.subplots(
        Ny, Nx,
        figsize=(figsize_per_cell[0]*Nx,
                 figsize_per_cell[1]*Ny),
        gridspec_kw={'wspace':0,'hspace':0,'left':0,'right':1,'top':1,'bottom':0}
    )
    axes = np.atleast_2d(axes).reshape(Ny,Nx)
    fit_dfs = {}

    for y in range(Ny):
        for x in range(Nx):
            ax = axes[y,x]
            fig_fit, df = plot_region_fitting_result_from_dsfit_no_label(
                ds_cluster,
                y_idx=y, x_idx=x,
                highlight_labels=highlight_labels,
                return_fig=True,
                **fit_kwargs
            )
            # buffer → image
            fig_fit.canvas.draw()
            w,h = fig_fit.canvas.get_width_height()
            buf = np.frombuffer(fig_fit.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(h,w,3)[::-1,:,:]
            ax.imshow(buf, origin='lower', aspect='auto')
            ax.axis('off')
            plt.close(fig_fit)
            fit_dfs[(y,x)] = df
            # thin border
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_linewidth(0.5)

    fig.suptitle("Per‐Pixel Fitting Results (No Labels)", fontsize=16)
    return fig, axes, fit_dfs


# ── 실행 오케스트레이터 ──
def interactive_cluster_plot_and_grid(
    ds_cluster, figsize_per_cell=(2.5,2.5), **plot_kwargs
):
    """
    1) 클러스터 변수 선택 → 2) 레이블 선택 → 3) 그리기 → 4) SVG 저장 버튼
    """
    def on_labels(lbls):
        fig, axes, dfs = plot_fitting_results_grid(
            ds_cluster,
            figsize_per_cell=figsize_per_cell,
            highlight_labels=lbls,
            **plot_kwargs
        )
        plt.show()

        # ── 여기서 SVG 저장 버튼 추가 ──
        save_btn = widgets.Button(
            description='Save as SVG',
            button_style='success'
        )
        out = widgets.Output()

        def on_save(_):
            with out:
                clear_output()
                fig.savefig('per_pixel_fitting_grid.svg', format='svg')
                print("✅ Saved figure as 'per_pixel_fitting_grid.svg'")

        save_btn.on_click(on_save)
        display(widgets.HBox([save_btn, out]))


    def on_cluster(var):
        select_labels_in_cluster_interactive(
            ds_cluster, callback=on_labels)

    select_cluster_data_var_interactive(
        ds_cluster, callback=on_cluster)


# ── 사용 예시 ──
interactive_cluster_plot_and_grid(
    ds_crop0,
    figsize_per_cell=(2.5,2.5),
    weight_function_show=False,
    use_zb_mask=True,
    zb_mask_key='ZB_mask',
    show_shade=False
)

# -

ds_crop0.where(ds_crop0.cluster_label==2,drop= True).peak_center







# +
# move on to the Figure 4 2T 005 case
# -















# ## check Pytorch & scikit learn  installation in the system 
#

# #### select several points(interactively) 

# Enable Bokeh and Panel output
output_notebook()
pn.extension('tabulator')  # Ensure Tabulator is loaded


# +
import hvplot.xarray
import holoviews as hv
from bokeh.models import ColumnDataSource, PointDrawTool
from bokeh.io import output_notebook
import pandas as pd
import panel as pn

# Enable Bokeh and Panel output
output_notebook()
pn.extension('tabulator')  # Ensure Tabulator is loaded

def interactive_points_selection_w_ldos_plot(grid_LDOS_SnD_pks, bias_mV_ref=0.0):
    """
    Creates an interactive LDOS plot with point selection functionality.
    Returns a DataFrame (points_df) with selected points' coordinates.

    Parameters:
    grid_LDOS_SnD_pks (xarray.Dataset): The input dataset containing LDOS data.
    bias_mV_ref (float): Reference bias voltage for the initial plot.

    Returns:
    points_df (pd.DataFrame): DataFrame containing X and Y coordinates of selected points.
    """

    # Step 1: Set up the interactive plot with the LDOS data at the specified bias voltage
    ldos_plot = grid_LDOS_SnD_pks.LDOS.sel(bias_mV=bias_mV_ref, method='nearest').hvplot.image(
        cmap='viridis', 
        title=f'LDOS Interactive Plot (bias_mV={bias_mV_ref})', 
        frame_height=400,  
        frame_width=500,   
        aspect='equal',
        xlabel='X', ylabel='Y',  # Label the axes correctly
        x='X', y='Y',            # Ensure the dimensions are correctly assigned
        tools=["pan", "wheel_zoom", "box_zoom", "tap", "reset"]  # Include interaction tools
    )

    # Convert hvplot to Bokeh figure using the hvplot `bokeh` backend
    background_fig = hv.render(ldos_plot)

    # Step 2: Set up a scatter plot for dynamic point updates
    source = ColumnDataSource(data=dict(x=[], y=[]))
    background_fig.scatter(x="x", y="y", source=source, size=10, color="red")

    # Step 3: Add PointDrawTool for interactive point selection
    draw_tool = PointDrawTool(renderers=[background_fig.renderers[-1]], empty_value='black')
    background_fig.add_tools(draw_tool)
    background_fig.toolbar.active_tap = draw_tool

    # Step 4: Prepare a DataFrame to store the clicked points and show it as a table
    points_df = pd.DataFrame(columns=['X', 'Y'])
    points_table = pn.widgets.Tabulator(points_df, width=300, height=200)

    # Step 5: Function to update the DataFrame dynamically and reflect it in the table
    def update_points(attr, old, new):
        new_points = [(x, y) for x, y in zip(source.data['x'], source.data['y'])]  # Swap X and Y
        nonlocal points_df
        points_df = pd.DataFrame(new_points, columns=['X', 'Y'])  # Update DataFrame with swapped X and Y
        points_table.value = points_df  # Reflect changes in the table

    # Link changes in the ColumnDataSource to the update_points function
    source.on_change('data', update_points)

    # Step 6: Display the layout with the interactive plot and the table using Panel
    layout = pn.Row(pn.panel(background_fig), points_table)  # Ensure the image is displayed with Panel's layout

    # Return the layout to display inline
    return layout

# Usage:
# layout  = interactive_points_selection_w_ldos_plot(grid_LDOS_SnD_pks, bias_mV_ref=0.0)
# layout # it will show the interactive plots. select some points (red dots)
# points_df  = layout[1].value  # ** point_df will be obtained from the points_table
# points_df

# -



layout  = interactive_points_selection_w_ldos_plot(grid_LDOS_SnD_pks, bias_mV_ref=0.0)

layout

points_df  = layout[1].value
points_df

# +
#grid_LDOS_SnD_pks

# +
import xarray as xr
import numpy as np
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt

def fit_multiple_lorentzian_LDOSsingleXY(ldos_pks_data, x_target_position=None, y_target_position=None, 
                                         initial_peak_guess='LDOS', margin2zerobias=0.1, 
                                         margin2levels=0.1, Ef=4.4, SCgap=1.8, maxfev=20000, 
                                         initial_amp_ratio=0.5, initial_width_ratio=1.0,
                                         bounds_amp=(0, 1e6), bounds_wid=(0, 1e3),
                                         visualization=False):
    """
    Fit Lorentzian curves to LDOS data at a single (X, Y) position in a given dataset.
    """

    def lorentzian(x, amp, cen, wid):
        """Lorentzian function definition."""
        return np.abs(amp) / (1 + ((x - cen) / wid)**2)
    
    def multiple_lorentzians(x, *params):
        """Model with multiple Lorentzian peaks."""
        y = np.zeros_like(x)
        for i in range(0, len(params), 3):
            amp, cen, wid = params[i:i+3]
            y += lorentzian(x, amp, cen, wid)
        return y
    
    def objective_function(params, x_data, ldos_smoothed):
        """Objective function to minimize during optimization."""
        return np.sum((multiple_lorentzians(x_data, *params) - ldos_smoothed) ** 2)
    
    def calculate_r_squared(ldos_smoothed, fit_result):
        """Calculate the R-squared value."""
        ss_res = np.sum((ldos_smoothed - fit_result) ** 2)
        ss_tot = np.sum((ldos_smoothed - np.mean(ldos_smoothed)) ** 2)
        return 1 - (ss_res / ss_tot)

    if x_target_position is None:
        x_target_position = np.random.choice(ldos_pks_data.X.values)
    if y_target_position is None:
        y_target_position = np.random.choice(ldos_pks_data.Y.values)

    ldos = ldos_pks_data['LDOS'].sel(X=x_target_position, Y=y_target_position, method='nearest').values
    ldos_smoothed = ldos_pks_data['LDOS_smoothed'].sel(X=x_target_position, Y=y_target_position, method='nearest').values
    x_data = ldos_pks_data.bias_mV.values

    if initial_peak_guess == 'LDOS':
        peak_bias = ldos_pks_data['LDOS_smoothed_peak_bias'].sel(X=x_target_position, Y=y_target_position, method='nearest').values
        peak_heights = ldos_pks_data['LDOS_smoothed_peak_heights'].sel(X=x_target_position, Y=y_target_position, method='nearest').values
        peak_widths = ldos_pks_data['LDOS_smoothed_peak_widths_mV'].sel(X=x_target_position, Y=y_target_position, method='nearest').values
    elif initial_peak_guess == '2deriv':
        peak_bias = ldos_pks_data['LDOS_peak_bias_2nd'].sel(X=x_target_position, Y=y_target_position, method='nearest').values
        peak_heights = ldos_pks_data['LDOS_peak_height_2nd'].sel(X=x_target_position, Y=y_target_position, method='nearest').values
        peak_widths = ldos_pks_data['LDOS_2deriv_smoothed_peak_widths_mV'].sel(X=x_target_position, Y=y_target_position, method='nearest').values

    peak_dim = len(peak_bias)
    amplitude = np.full((peak_dim,), np.nan)
    center = np.full((peak_dim,), np.nan)
    width = np.full((peak_dim,), np.nan)
    aspect_ratio_map = np.full((peak_dim,), np.nan)
    levels_proximity_map = np.full((peak_dim,), np.nan)
    fitted_ldos = np.full((len(x_data),), np.nan)

    E_mu = (SCgap ** 2) / Ef
    n_min = int(np.floor(x_data.min() / E_mu))
    n_max = int(np.ceil(x_data.max() / E_mu))

    levels = [n * E_mu for n in range(n_min, n_max + 1)]
    half_levels = [(n + 0.5) * E_mu for n in range(n_min, n_max)]

    initial_guesses = []
    valid_indices = ~np.isnan(peak_bias) & ~np.isnan(peak_heights)
    for pb, ph, pw in zip(peak_bias[valid_indices], peak_heights[valid_indices], peak_widths[valid_indices]):
        initial_guesses.extend([ph * initial_amp_ratio, pb, pw * initial_width_ratio])

    fitting_results = {}
    try:
        popt_cf, _ = curve_fit(
            multiple_lorentzians, 
            x_data, 
            ldos_smoothed, 
            p0=initial_guesses, 
            bounds=([0, -np.inf, 0] * peak_dim, [np.inf, np.inf, np.inf] * peak_dim),
            maxfev=maxfev
        )
        fit_result_cf = multiple_lorentzians(x_data, *popt_cf)
        r_squared_cf = calculate_r_squared(ldos_smoothed, fit_result_cf)
        if r_squared_cf > 0:
            fitting_results['curve_fit'] = (popt_cf, fit_result_cf, r_squared_cf)
        else:
            print(f"Default curve_fit resulted in negative R-squared: {r_squared_cf:.4f}. Trying other methods.")
            raise ValueError("Negative R-squared")
    except (RuntimeError, ValueError):
        print("Default curve_fit failed or provided negative R-squared.")

    try:
        popt_lm, _ = curve_fit(
            multiple_lorentzians, 
            x_data, 
            ldos_smoothed, 
            p0=initial_guesses, 
            method='lm', 
            maxfev=maxfev
        )
        fit_result_lm = multiple_lorentzians(x_data, *popt_lm)
        r_squared_lm = calculate_r_squared(ldos_smoothed, fit_result_lm)
        fitting_results['Levenberg-Marquardt'] = (popt_lm, fit_result_lm, r_squared_lm)
    except RuntimeError:
        print("Levenberg-Marquardt fitting failed: This method does not support bounds.")

    try:
        result_nm = minimize(objective_function, initial_guesses, args=(x_data, ldos_smoothed), method='nelder-mead', options={'maxiter': 1000})
        fit_result_nm = multiple_lorentzians(x_data, *result_nm.x)
        r_squared_nm = calculate_r_squared(ldos_smoothed, fit_result_nm)
        fitting_results['Nelder-Mead'] = (result_nm.x, fit_result_nm, r_squared_nm)
    except RuntimeError:
        print("Nelder-Mead fitting failed: This method does not support bounds.")

    try:
        result_bfgs = minimize(objective_function, initial_guesses, args=(x_data, ldos_smoothed), method='BFGS', options={'maxiter': 1000})
        fit_result_bfgs = multiple_lorentzians(x_data, *result_bfgs.x)
        r_squared_bfgs = calculate_r_squared(ldos_smoothed, fit_result_bfgs)
        fitting_results['BFGS'] = (result_bfgs.x, fit_result_bfgs, r_squared_bfgs)
    except RuntimeError:
        print("BFGS fitting failed: This method does not support bounds.")

    if fitting_results:
        best_method = max(fitting_results, key=lambda k: fitting_results[k][2])
        popt, fit_result, r_squared = fitting_results[best_method]
        print(f"Best fitting method: {best_method} with R-squared = {r_squared:.4f}")
        if best_method != 'curve_fit' and any(popt[::3] < 0):
            print(f"Warning: {best_method} does not support bounds, and some amplitude values are negative.")
    else:
        print("All fitting methods failed.")
        popt = []
        fit_result = np.zeros_like(x_data)
        r_squared = np.nan

    for i in range(0, len(popt), 3):
        amp, cen, wid = popt[i:i+3]
        amplitude[i // 3] = amp
        center[i // 3] = cen
        width[i // 3] = wid
        aspect_ratio_map[i // 3] = amp / wid
        closest_level = min(levels + half_levels, key=lambda level: np.abs(cen - level))
        if np.abs(cen - closest_level) <= margin2levels:
            levels_proximity_map[i // 3] = closest_level
        else:
            levels_proximity_map[i // 3] = np.nan

    fitted_ldos = fit_result


    if visualization:
        fig, axs = plt.subplots(2, 2, figsize=(14, 12))

        axs[0, 0].plot(ldos_pks_data.bias_mV.values, ldos, color='black', label='LDOS')
        axs[0, 0].plot(ldos_pks_data.bias_mV.values, ldos_smoothed, color='skyblue', linewidth=2, alpha=0.7, label='LDOS Smoothed')
        for i, (pb, ph, pw) in enumerate(zip(peak_bias[valid_indices], peak_heights[valid_indices], peak_widths[valid_indices])):
            color = f'C{i}'
            axs[0, 0].scatter(pb, ph, color=color, marker='o', s=50, label=f'Peak {i}')
            axs[0, 0].hlines(ph / 2, pb - pw / 2, pb + pw / 2, color=color, linestyle='--', alpha=0.7)
            axs[0, 0].text(pb, ph + 0.05 * max(ldos_smoothed), f'{i}', color=color, fontsize=12, ha='center')
        axs[0, 0].set_title(f'LDOS with Peaks (X={x_target_position:.2e}, Y={y_target_position:.2e})')
        axs[0, 0].set_xlabel('Bias (mV)')
        axs[0, 0].set_ylabel('LDOS (A/V)')
        axs[0, 0].legend()

        axs[0, 1].plot(x_data, ldos_smoothed, color='black', label='LDOS Smoothed')
        axs[0, 1].plot(x_data, fit_result, color='red', label=f'{best_method} Fit', linewidth=2)
        for i in range(0, len(popt), 3):
            amp, cen, wid = popt[i:i+3]
            axs[0, 1].plot(x_data, lorentzian(x_data, amp, cen, wid), linestyle='--', alpha=0.7)
            axs[0, 1].scatter(cen, amp, s=50)
        axs[0, 1].set_title('Lorentzian Fit Results')
        axs[0, 1].set_xlabel('Bias (mV)')
        axs[0, 1].set_ylabel('LDOS (A/V)')
        axs[0, 1].legend()

        ldos_map = ldos_pks_data.LDOS.sel(bias_mV=0, method='nearest')
        ldos_map.plot(ax=axs[1, 0], cmap='viridis', robust=True)
        axs[1, 0].scatter([x_target_position], [y_target_position], color='red', s=100, marker='x')
        axs[1, 0].set_title('LDOS Map at bias_mV = 0')

        axs[1, 1].plot(x_data, ldos_smoothed, color='black', label='LDOS Smoothed')
        axs[1, 1].plot(x_data, fit_result, color='red', label=f'{best_method} Fit', linewidth=2)
        for i in range(0, len(popt), 3):
            amp, cen, wid = popt[i:i+3]
            axs[1, 1].plot(x_data, lorentzian(x_data, amp, cen, wid), linestyle='--', alpha=0.7)
            axs[1, 1].scatter(cen, amp, s=50)
            if np.abs(cen) <= margin2zerobias:
                axs[1, 1].fill_between(x_data, 0, lorentzian(x_data, amp, cen, wid), color='black', alpha=0.2)
            else:
                closest_level = min(levels + half_levels, key=lambda level: np.abs(cen - level))
                if np.abs(cen - closest_level) <= margin2levels:
                    color = 'blue' if closest_level in levels else 'red'
                    axs[1, 1].fill_between(x_data, 0, lorentzian(x_data, amp, cen, wid), color=color, alpha=0.2)
        xlim = (x_data.min(), x_data.max())
        ylim = axs[1, 1].get_ylim()  
        for level in levels:
            if xlim[0] <= level <= xlim[1]:  
                axs[1, 1].axvline(level, color='blue', linestyle=':', alpha=0.4)
        for level in half_levels:
            if xlim[0] <= level <= xlim[1]:  
                axs[1, 1].axvline(level, color='red', linestyle=':', alpha=0.4)
        axs[1, 1].axvline(0, color='black', linestyle=':', linewidth=1, alpha=0.4)
        axs[1, 1].set_xlim(xlim)
        axs[1, 1].set_ylim(ylim)
        axs[1, 1].set_title('Lorentzian Fit Results with Levels')
        axs[1, 1].set_xlabel('Bias (mV)')
        axs[1, 1].set_ylabel('LDOS (A/V)')
        axs[1, 1].legend()

        plt.tight_layout()
        plt.show()

    fitted_dataset = xr.Dataset(
        {
            "amplitude": (["peak"], amplitude),
            "center": (["peak"], center),
            "width": (["peak"], width),
            "aspect_ratio": (["peak"], aspect_ratio_map),
            "level_proximity": (["peak"], levels_proximity_map),
            "r_squared": ([], r_squared),
            "fit_result": (["bias_mV"], fitted_ldos),
        },
        coords={
            "peak": np.arange(peak_dim),
            "bias_mV": x_data
        },
        attrs={
            **ldos_pks_data.attrs,
            "Ef": Ef,
            "margin2zerobias": margin2zerobias,
            "margin2levels": margin2levels
        }
    )

    return fitted_dataset

# -

###### ldos_pks_data_cropped
x_target_position = points_df['X'].iloc[-2]  # Last X value
y_target_position = points_df['Y'].iloc[-2]  # Last Y value
ldos_pks_data_fitted = fit_multiple_lorentzian_LDOSsingleXY(ldos_pks_data = grid_LDOS_SnD_pks,
                                                           x_target_position = x_target_position,
                                                           y_target_position = y_target_position,
                                                           margin2zerobias=0.15,
                                                           margin2levels=0.15,
                                                            maxfev=100000,
                                                            bounds_amp=(0, 1e6),
                                                            bounds_wid=(0, 1e3),
                                                           initial_peak_guess='2deriv',visualization=True)
# +
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import xarray as xr
import hvplot.xarray
import pandas as pd
import matplotlib.cm as cm


def fit_multiple_lorentzian_LDOS_points_N_plot(ldos_pks_data, points_df, bias_mV_ref=0.0, margin2zerobias=0.15, margin2levels=0.15, 
                                                maxfev=100000, initial_peak_guess='2deriv', wrap_col=4, Ef=4.4, SCgap=1.8):
    """
    Perform Lorentzian fitting on multiple (X, Y) points provided by the user.
    Show both the LDOS map with selected points and the Lorentzian fitting results.

    Parameters:
    - ldos_pks_data: xarray.Dataset
        LDOS data containing the `LDOS` variable.
    - points_df: pandas.DataFrame
        DataFrame containing the selected points with columns ['X', 'Y'].
    - bias_mV_ref: float, optional
        Reference bias voltage for initial LDOS map plot. Default is 0.0 mV.
    - margin2zerobias: float, optional
        Margin for fitting close to zero bias. Default is 0.15.
    - margin2levels: float, optional
        Margin for fitting close to levels. Default is 0.15.
    - maxfev: int, optional
        Maximum function evaluations for fitting. Default is 100000.
    - initial_peak_guess: str, optional
        Method for initial peak guess. Default is '2deriv'.
    - wrap_col: int, optional
        Number of columns to wrap the Lorentzian plots. Default is 4.
    - Ef: float, optional
        Fermi energy for calculating energy levels. Default is 4.4.
    - SCgap: float, optional
        Superconducting gap for calculating energy levels. Default is 1.8.

    Returns:
    - ldos_pks_data_fitted_points: xarray.Dataset
        Dataset containing Lorentzian fitting results for each selected point.
    """

    # Step 1: Plot LDOS Map with Selected Points
    fitted_datasets = []
    ldos_map = ldos_pks_data.LDOS.sel(bias_mV=bias_mV_ref, method='nearest')

    fig, ax = plt.subplots(figsize=(6, 6))
    ldos_map.plot(ax=ax, cmap='viridis')
    ax.set_aspect('equal')

    cmap = plt.get_cmap('tab10')
    for i, (x_target_position, y_target_position) in points_df.iterrows():
        color = cmap(i % cmap.N)  # Use discrete colormap for consistent colors
        ax.scatter(x_target_position, y_target_position, color=color, s=150, marker='o', alpha=0.5, linewidths=3)
        text = ax.text(x_target_position, y_target_position, f'{i+1}',
                       color=color, fontsize='large', fontweight='bold', ha='left', va='bottom')
        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()])

    plt.title('LDOS Map with Selected Points')
    plt.xlabel('X (position)')
    plt.ylabel('Y (position)')
    plt.show()

    # Step 2: Perform Lorentzian Fitting for Each Point
    for i, (x_target_position, y_target_position) in points_df.iterrows():
        print(f"Fitting point {i + 1}: X={x_target_position:.2e}, Y={y_target_position:.2e}")

        # Call the single-point fitting function for each point
        fitted_dataset = fit_multiple_lorentzian_LDOSsingleXY(
            ldos_pks_data=ldos_pks_data,
            x_target_position=x_target_position,
            y_target_position=y_target_position,
            margin2zerobias=margin2zerobias,
            margin2levels=margin2levels,
            maxfev=maxfev,
            initial_peak_guess=initial_peak_guess,
            visualization=False
        )

        fitted_dataset = fitted_dataset.assign_coords({'X': x_target_position, 'Y': y_target_position})

        if fitted_dataset is not None:
            fitted_datasets.append(fitted_dataset)
        else:
            print(f"Fitting failed for point {i + 1}")

    if fitted_datasets:
        ldos_pks_data_fitted_points = xr.concat(fitted_datasets, dim='point')
    else:
        raise ValueError("No fitted datasets were generated.")

    # Step 3: Plot Lorentzian Fitting Results
    def lorentzian(x, amp, cen, wid):
        return np.abs(amp) / (1 + ((x - cen) / wid)**2)

    num_points = ldos_pks_data_fitted_points.dims['point']
    rows = (num_points + wrap_col - 1) // wrap_col
    fig, axes = plt.subplots(rows, wrap_col, figsize=(wrap_col * 5, rows * 5))
    axes = axes.flatten()

    E_mu = (SCgap ** 2) / Ef
    xlim = ldos_pks_data_fitted_points.bias_mV.min().item(), ldos_pks_data_fitted_points.bias_mV.max().item()
    n_min = int(np.floor(xlim[0] / E_mu))
    n_max = int(np.ceil(xlim[1] / E_mu))
    levels = [n * E_mu for n in range(n_min, n_max + 1)]
    half_levels = [(n + 0.5) * E_mu for n in range(n_min, n_max)]

    for i in range(num_points):
        ax = axes[i]
        point_data = ldos_pks_data_fitted_points.isel(point=i)
        x_data = point_data.bias_mV.values
        fit_result = point_data.fit_result.values
        amplitude = point_data.amplitude.values
        center = point_data.center.values
        width = point_data.width.values
        r_squared = point_data.r_squared.item()

        if ldos_pks_data is not None:
            x_pos, y_pos = point_data.X.item(), point_data.Y.item()
            ldos_curve = ldos_pks_data['LDOS'].sel(X=x_pos, Y=y_pos, method='nearest').values
            color = cmap(i % cmap.N)  # Use the same color as the point marker for consistency
            ax.plot(x_data, ldos_curve, color=color, label='Original LDOS', alpha=0.8)

        ax.plot(x_data, fit_result, color='red', label='Fit', linewidth=3)
        ax.scatter(center, amplitude, color='blue', marker='o')

        for j in range(len(amplitude)):
            peak_curve = lorentzian(x_data, amplitude[j], center[j], width[j])
            ax.plot(x_data, peak_curve, linestyle='--', alpha=0.6)
            if np.abs(center[j]) <= 0.1:  # Highlight near zero with gray
                ax.fill_between(x_data, 0, peak_curve, color='gray', alpha=0.5)
            elif center[j] > 0 and center[j] == int(center[j]):  # Highlight positive integer centers with blue
                ax.fill_between(x_data, 0, peak_curve, color='blue', alpha=0.5)

        for level in levels:
            if xlim[0] <= level <= xlim[1]:
                ax.axvline(level, color='blue', linestyle=':', alpha=0.4)
        for level in half_levels:
            if xlim[0] <= level <= xlim[1]:
                ax.axvline(level, color='red', linestyle=':', alpha=0.4)

        ax.set_xlim(x_data.min(), x_data.max())
        ax.set_title(f'Point {i + 1}: X={point_data.X.values:.2e}, Y={point_data.Y.values:.2e} \nR^2 = {r_squared:.2f}')
        ax.set_xlabel('Bias (mV)')
        ax.set_ylabel('LDOS (A/V)')
        ax.legend()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    return ldos_pks_data_fitted_points

# Suggested function name: fit_multiple_lorentzian_LDOS_points_N_plot

# Example use
# grid_LDOS_SnD_pks_points =  fit_multiple_lorentzian_LDOS_points_N_plot(grid_LDOS_SnD_pks, points_df=points_df, bias_mV_ref=0.0, margin2zerobias=0.15, margin2levels=0.15,
#                                            maxfev=100000, initial_peak_guess='2deriv', wrap_col=4, Ef=4.4, SCgap=1.8)



# -

grid_LDOS_SnD_pks_points = fit_multiple_lorentzian_LDOS_points_N_plot(grid_LDOS_SnD_pks,
                                                                      points_df=points_df,
                                                                      bias_mV_ref=0.0, 
                                                                      margin2zerobias=0.15,
                                                                      margin2levels=0.15, 
                                                                      maxfev=100000,
                                                                      initial_peak_guess='2deriv',
                                                                      wrap_col=4, Ef=4.4, SCgap=1.8)

grid_LDOS_SnD_pks


grid_LDOS_SnD_pks_points

# +
#grid_LDOS_SnD_pks_points
#grid_LDOS_SnD


# -



# %%time
grid_LDOS_SnD_pks_Lorentzian_fit = fit_lorentzian_LDOS_region(grid_LDOS_SnD_pks, initial_amp_ratio=0.5, margin2zerobias=0.15,
    margin2levels=0.15,initial_wid_ratio=1.0)
grid_LDOS_SnD_pks_Lorentzian_fit

grid_LDOS_SnD_pks_Lorentzian_fit.to_netcdf('grid_LDOS_SnD_pks_0T_002Lorentzian_fit.nc')

# +
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

#  fit_converged (cmap: viridis)
grid_LDOS_SnD_pks_Lorentzian_fit.fit_converged.plot(ax=axes[0], cmap='viridis')

#  r_squared (cmap: plasma)
grid_LDOS_SnD_pks_Lorentzian_fit.r_squared.plot(ax=axes[1], cmap='plasma')

#  bias_mV = 0에서의 fit_result (cmap: inferno)
bias_mV_index = (grid_LDOS_SnD_pks_Lorentzian_fit.bias_mV == 0).argmax().item()
grid_LDOS_SnD_pks_Lorentzian_fit.fit_result.isel(bias_mV=bias_mV_index).plot(ax=axes[2], cmap='inferno')

# 
titles = ['Fit Converged', 'R-squared', 'Fit Result at bias_mV = 0']
for ax, title in zip(axes.ravel(), titles):
    ax.set_title(title, fontsize='large')
    ax.set_aspect('equal')

plt.tight_layout()
plt.show()
# -


grid_LDOS_SnD_pks_Lorentzian_fit#.amplitude.where(center_near_zero_mask)



# +
import numpy as np
import matplotlib.pyplot as plt

# Near Center = 0 
tolerance = 1
center_near_zero_mask = np.abs(grid_LDOS_SnD_pks_Lorentzian_fit.center) < tolerance

# Select only the entries where center is near zero and calculate the mean along the 'peak' dimension to create 2D data
amp_filtered = grid_LDOS_SnD_pks_Lorentzian_fit.amplitude.where(center_near_zero_mask).mean(dim='peak')
width_filtered = grid_LDOS_SnD_pks_Lorentzian_fit.width.where(center_near_zero_mask).mean(dim='peak')
center_filtered = grid_LDOS_SnD_pks_Lorentzian_fit.center.where(center_near_zero_mask).mean(dim='peak')
level_proximity_filtered = grid_LDOS_SnD_pks_Lorentzian_fit.level_proximity.where(center_near_zero_mask).mean(dim='peak')

# Visualize the 4 filtered results
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# First plot: Amplitude (cmap: viridis)
amp_filtered.plot(ax=axes[0], cmap='viridis')
axes[0].set_title('Amplitude (Center ~ 0)', fontsize='large')
axes[0].set_aspect('equal')

# Second plot: Width (cmap: plasma)
width_filtered.plot(ax=axes[1], cmap='plasma')
axes[1].set_title('Width (Center ~ 0)', fontsize='large')
axes[1].set_aspect('equal')

# Third plot: Center (cmap: inferno)
center_filtered.plot(ax=axes[2], cmap='inferno')
axes[2].set_title('Center (Center ~ 0)', fontsize='large')
axes[2].set_aspect('equal')

# Fourth plot: Level Proximity (cmap: cividis)
level_proximity_filtered.plot(ax=axes[3], cmap='cividis')
axes[3].set_title('Level Proximity (Center ~ 0)', fontsize='large')
axes[3].set_aspect('equal')

# Adjust layout
plt.tight_layout()
plt.show()


# +
#grid_LDOS_SnD_pks
# -

#



# +
# Define tolerance for bias_mV near zero
tolerance = 0.01
bias_near_zero_mask = np.abs(grid_LDOS_SnD_pks.bias_mV) <= tolerance

# Background: LDOS data filtered within the tolerance range of bias_mV near zero
ldos_filtered = grid_LDOS_SnD_pks.LDOS.where(
    (grid_LDOS_SnD_pks.bias_mV < 0 + tolerance) & (grid_LDOS_SnD_pks.bias_mV > 0 - tolerance)
).mean(dim='bias_mV')

# Create a 2x2 figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))  # 2x2 grid

# First plot: Amplitude with grey background
ldos_filtered.plot(ax=axes[0, 0], cmap='Greys', alpha=1, robust=True)
amp_filtered.plot(ax=axes[0, 0], cmap='viridis', alpha=0.5)  # Overlay points with transparency
axes[0, 0].set_title('Amplitude (Center ~ 0)')

# Second plot: Width with grey background
ldos_filtered.plot(ax=axes[0, 1], cmap='Greys', alpha=1, robust=True)
width_filtered.plot(ax=axes[0, 1], cmap='plasma', alpha=0.5)  # Overlay points with transparency
axes[0, 1].set_title('Width (Center ~ 0)')

# Third plot: Center with grey background
ldos_filtered.plot(ax=axes[1, 0], cmap='Greys', alpha=1, robust=True)
center_filtered.plot(ax=axes[1, 0], cmap='inferno', alpha=0.5)  # Overlay points with transparency
axes[1, 0].set_title('Center (Center ~ 0)')

# Fourth plot: LDOS background with viridis and Level Proximity as scatter
ldos_filtered.plot(ax=axes[1, 1], cmap='viridis', alpha=1, robust=True)  # Viridis background for this plot
Y, X = np.meshgrid(grid_LDOS_SnD_pks_Lorentzian_fit.Y, grid_LDOS_SnD_pks_Lorentzian_fit.X, indexing='ij')
sc = axes[1, 1].scatter(X, Y, c=level_proximity_filtered, cmap='cividis', edgecolor='black', marker='o', alpha=0.5)  # Overlay points with transparency
axes[1, 1].set_title('LDOS Background with Level Proximity')

# Add color bar for the scatter plot in the fourth subplot
plt.colorbar(sc, ax=axes[1, 1], label='Level Proximity')

# Use a loop to set aspect and font size for all subplots
for ax in axes.ravel():
    ax.set_aspect('equal')
    ax.title.set_fontsize('large')

# Adjust layout
plt.tight_layout()
plt.show()

# -








# # 4T 

# +
grid_LDOS = GS_LDOS_4T_008.copy()

grid_LDOS.sel(bias_mV = slice(1.2,-1.2))
# -




















# ### crop grid_LDOS_pks_prprt XY range 
#

grid_LDOS.where(
    (grid_LDOS.X>3.2E-8)&
    (grid_LDOS.X<5.0E-8)& 
    (grid_LDOS.Y>2.2E-8)&
    (grid_LDOS.Y<4.0E-8)).LDOS.sel(bias_mV=0, method = 'nearest').plot(robust  = True)



grid_LDOS_pks_prprt_crop = grid_LDOS_pks_prprt.where(
    (grid_LDOS.X>3.2E-8)&
    (grid_LDOS.X<5.0E-8)& 
    (grid_LDOS.Y>2.2E-8)&
    (grid_LDOS.Y<4.0E-8), drop = True)

# #### dataframe preparation 
# *  whole area or cropped area?

# ##  crop grid_LDOS_SnD_pks 

grid_LDOS_SnD_pks

grid_LDOS_SnD_pks_crop = grid_LDOS_SnD_pks.copy()#.where( (grid_LDOS.X<-3.9E-7)&(grid_LDOS.Y<2.8E-7), drop= True).copy()
grid_LDOS_SnD_pks_crop= grid_LDOS_SnD_pks_crop[['LDOS_peak_bias_2nd','LDOS_peak_height_2nd','LDOS_2deriv_smoothed_peak_widths_mV','LDOS_2deriv_smoothed_peak_heights']]
grid_LDOS_SnD_pks_crop


# ### conver to dataframe

grid_LDOS_pks_df = grid_LDOS_SnD_pks_crop.to_dataframe().reset_index()
grid_LDOS_pks_df = grid_LDOS_pks_df[grid_LDOS_pks_df.LDOS_peak_bias_2nd.notna()]
grid_LDOS_pks_df= grid_LDOS_pks_df.rename(
    columns = {'LDOS_peak_bias_2nd' : 'bias_mV',
               'LDOS_peak_height_2nd' : 'peak_heights',
               'LDOS_2deriv_smoothed_peak_widths_mV' : 'peak_widths',
               'LDOS_2deriv_smoothed_peak_heights' : 'peak_prominences'
              })
grid_LDOS_pks_df

# ##### in case of cropping bias_mV range 

# +

# filtering bias_mV range 
#grid_LDOS_pks_df = grid_LDOS_pks_df[(grid_LDOS_pks_df.bias_mV>-0.7)&(grid_LDOS_pks_df.bias_mV<0.7)]
grid_LDOS_pks_df
# -

grid_LDOS_pks_df.to_pickle('grid_LDOS_pks_df.pkl') 

# # 2T 

grid_LDOS = GS_LDOS_0T_002.copy()

grid_LDOS_SnD = smoothing_and_deriv_LDOS(grid_LDOS,window_length_ratio=0.05 )
#find peaks in grid_LDOS_SnD
# find peaks 
grid_LDOS_SnD_pks = find_pks_grid_LDOS(grid_LDOS_SnD)
#grid_LDOS_SnD_pks
plot_ldos_with_pks(grid_LDOS_SnD_pks, wrap_ncols=4, num_random_points=16)

grid_LDOS_SnD_pks_crop = grid_LDOS_SnD_pks.copy()#.where( (grid_LDOS.X<-3.9E-7)&(grid_LDOS.Y<2.8E-7), drop= True).copy()
grid_LDOS_SnD_pks_crop= grid_LDOS_SnD_pks_crop[['LDOS_peak_bias_2nd','LDOS_peak_height_2nd','LDOS_2deriv_smoothed_peak_widths_mV','LDOS_2deriv_smoothed_peak_heights']]
grid_LDOS_SnD_pks_crop


# ### conver to dataframe

grid_LDOS_pks_df = grid_LDOS_SnD_pks_crop.to_dataframe().reset_index()
grid_LDOS_pks_df = grid_LDOS_pks_df[grid_LDOS_pks_df.LDOS_peak_bias_2nd.notna()]
grid_LDOS_pks_df= grid_LDOS_pks_df.rename(
    columns = {'LDOS_peak_bias_2nd' : 'bias_mV',
               'LDOS_peak_height_2nd' : 'peak_heights',
               'LDOS_2deriv_smoothed_peak_widths_mV' : 'peak_widths',
               'LDOS_2deriv_smoothed_peak_heights' : 'peak_prominences'
              })
grid_LDOS_pks_df



grid_LDOS_pks_df.to_pickle('grid_LDOS_pks_2T_df.pkl') 

# ##  load pickle data 

# +
import pickle

# Open the file in binary mode and load the content
with open('grid_LDOS_pks_2T_df.pkl', 'rb') as file:
    grid_LDOS_pks_df = pickle.load(file)
# -


grid_LDOS_pks_df

# ### Standardization 

# +
from sklearn.metrics import silhouette_score
import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from scipy.spatial import distance_matrix


def find_best_umap_params(dataframe, features, n_neighbors_list, min_dist_list, n_clusters=4, random_state=21, sampling_fraction=0.1):
    """
    Function to find the best combination of n_neighbors and min_dist using silhouette score.

    Parameters:
    - dataframe: pd.DataFrame, input dataframe with features.
    - features: list, list of features to use for standardization and UMAP.
    - n_neighbors_list: list, list of values for n_neighbors to test.
    - min_dist_list: list, list of values for min_dist to test.
    - n_clusters: int, number of clusters for KMeans clustering (default is 4).
    - random_state: int, random seed for reproducibility (default is 21).
    - sampling_fraction: float, fraction of data to sample for faster fitting (default is 0.1).

    Returns:
    - best_params: dict, contains the best combination of n_neighbors and min_dist.
    - best_score: float, best silhouette score achieved.
    """
    
    best_score = -1  # Initialize with a low value
    best_params = None

    # Standardize the selected features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataframe[features])

    # Sample the data to reduce computation time
    sample_data, _ = train_test_split(scaled_data, train_size=sampling_fraction, random_state=random_state)

    # Precompute distance matrix (optional for optimization)
    distance_matrix_data = distance_matrix(sample_data, sample_data)

    # Iterate over each combination of n_neighbors and min_dist
    for n_neighbors in n_neighbors_list:
        for min_dist in min_dist_list:
            print(f"Testing n_neighbors={n_neighbors}, min_dist={min_dist}")
            
            # Perform UMAP dimensionality reduction with optimizations
            umap_reducer = umap.UMAP(n_neighbors=n_neighbors,
                                     min_dist=min_dist, 
                                     n_components=2, 
                                     random_state=random_state,
                                     init='spectral',  # Use spectral initialization for faster convergence
                                     metric='precomputed',  # Use precomputed distance matrix
                                     n_epochs=50,  # Further reduce number of epochs
                                     low_memory=True,  # Reduce memory usage
                                     target_n_neighbors=5,  # Optimize embedding by reducing target neighbors
                                     transform_seed=random_state,  # Ensure transform reproducibility
                                     verbose=True)  # Display progress during the UMAP fitting
            
            umap_data = umap_reducer.fit_transform(distance_matrix_data)

            # Apply KMeans clustering on the UMAP-reduced data
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
            cluster_labels = kmeans.fit_predict(umap_data)

            # Calculate silhouette score to evaluate the quality of clustering
            score = silhouette_score(umap_data, cluster_labels)

            print(f"Silhouette Score: {score:.4f}")

            # Update best score and parameters if current score is better
            if score > best_score:
                best_score = score
                best_params = {'n_neighbors': n_neighbors, 'min_dist': min_dist}

    print(f"Best Parameters: {best_params}, Best Silhouette Score: {best_score:.4f}")
    
    return best_params, best_score

# Example usage:
n_neighbors_list = [10, 30, 50]  # List of n_neighbors values to try
min_dist_list = [0.1, 0.3, 0.5]  # List of min_dist values to try

best_params, best_score = find_best_umap_params(ldos_pks_df, 
                                                features=['X', 'Y', 'bias_mV', 'peak_heights', 'peak_widths', 'peak_prominences'],
                                                n_neighbors_list=n_neighbors_list, 
                                                min_dist_list=min_dist_list)
# -

ldos_pks_df

# +
from sklearn.metrics import silhouette_score
import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from scipy.spatial import distance_matrix


def find_best_umap_params(dataframe, features, n_neighbors_list, min_dist_list, n_clusters=4, random_state=21, sampling_fraction=0.1):
    """
    Function to find the best combination of n_neighbors and min_dist using silhouette score.

    Parameters:
    - dataframe: pd.DataFrame, input dataframe with features.
    - features: list, list of features to use for standardization and UMAP.
    - n_neighbors_list: list, list of values for n_neighbors to test.
    - min_dist_list: list, list of values for min_dist to test.
    - n_clusters: int, number of clusters for KMeans clustering (default is 4).
    - random_state: int, random seed for reproducibility (default is 21).
    - sampling_fraction: float, fraction of data to sample for faster fitting (default is 0.1).

    Returns:
    - best_params: dict, contains the best combination of n_neighbors and min_dist.
    - best_score: float, best silhouette score achieved.
    """
    
    best_score = -1  # Initialize with a low value
    best_params = None

    # Standardize the selected features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataframe[features])

    # Sample the data to reduce computation time
    sample_data, _ = train_test_split(scaled_data, train_size=sampling_fraction, random_state=random_state)

    # Precompute distance matrix (optional for optimization)
    distance_matrix_data = distance_matrix(sample_data, sample_data)

    # Iterate over each combination of n_neighbors and min_dist
    for n_neighbors in n_neighbors_list:
        for min_dist in min_dist_list:
            print(f"Testing n_neighbors={n_neighbors}, min_dist={min_dist}")
            
            # Perform UMAP dimensionality reduction with verbose=True to display progress
            umap_reducer = umap.UMAP(n_neighbors=n_neighbors,
                                     min_dist=min_dist, 
                                     n_components=2, 
                                     random_state=random_state,
                                     init='spectral',  # Use spectral initialization for faster convergence
                                     metric='precomputed',  # Use precomputed distance matrix
                                     n_epochs=100,  # Reduce number of epochs
                                     verbose=True)  # Display progress during the UMAP fitting
            
            umap_data = umap_reducer.fit_transform(distance_matrix_data)

            # Apply KMeans clustering on the UMAP-reduced data
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
            cluster_labels = kmeans.fit_predict(umap_data)

            # Calculate silhouette score to evaluate the quality of clustering
            score = silhouette_score(umap_data, cluster_labels)

            print(f"Silhouette Score: {score:.4f}")

            # Update best score and parameters if current score is better
            if score > best_score:
                best_score = score
                best_params = {'n_neighbors': n_neighbors, 'min_dist': min_dist}

    print(f"Best Parameters: {best_params}, Best Silhouette Score: {best_score:.4f}")
    
    return best_params, best_score

# Example usage:
n_neighbors_list = [10, 30, 50]  # List of n_neighbors values to try
min_dist_list = [0.1, 0.3, 0.5]  # List of min_dist values to try

best_params, best_score = find_best_umap_params(ldos_pks_df, 
                                                features=['X', 'Y', 'bias_mV', 'peak_heights', 'peak_widths', 'peak_prominences'],
                                                n_neighbors_list=n_neighbors_list, 
                                                min_dist_list=min_dist_list)
# -










# # Selections with BBOX & Lasso for (grid_LDOS)

# ## Select bounding box with holoview 
#  





# +
def bbox_select_plot(xr_data, ch_name='LDOS', number_of_bbox=3, frame_width=400):
    """
    A function to interactively select bounding boxes on an LDOS map and plot the 
    average LDOS curve for each selected region.

    Parameters:
    -----------
    xr_data : xarray.Dataset
        The dataset containing the LDOS data. It must have 'X', 'Y', and 'bias_mV' dimensions.
    
    ch_name : str, optional
        The channel name to plot and analyze. Default is 'LDOS'.
    
    number_of_bbox : int, optional
        The number of bounding boxes to select before stopping. Default is 3.
    
    frame_width : int, optional
        The width of the frame for the holoviews plot. Default is 400.

    Returns:
    --------
    bbox_list : list of tuples
        A list of bounding box coordinates selected by the user.
    """

    # Import required libraries within the function
    import holoviews as hv
    from holoviews import streams
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    from IPython.display import display

    # Activate Holoviews bokeh extension
    hv.extension('bokeh')

    # Prepare the dataset for holoviews
    xr_data_channel_hv = hv.Dataset(xr_data[ch_name])
    dmap_plane = ["X", "Y"]  # Dimensions to plot for the map (X-Y plane)

    # Create a DynamicMap to visualize the LDOS in X-Y plane
    dmap = xr_data_channel_hv.to(hv.Image, kdims=dmap_plane, dynamic=True)
    dmap.opts(colorbar=True, cmap='bwr', frame_width=frame_width, aspect='equal')

    # Initialize a list to store bounding box coordinates
    bbox_list = []

    # Prepare an image for bounding box selection at bias_mV = 0
    xr_data_channel_hv_image = hv.Dataset(xr_data[ch_name].isel(bias_mV=0)).relabel('for BBox selection : ')
    
    # Create a holoviews Points object with box select tool
    bbox_points = hv.Points(xr_data_channel_hv_image).opts(
        frame_width=frame_width, color='k', aspect='equal', alpha=0.1, tools=['box_select']
    )

    # Initialize the stream for capturing bounding box coordinates
    bound_box = streams.BoundsXY(source=bbox_points, bounds=(0, 0, 0, 0))

    def extract_bbox_data(bbox, xr_data, ch_name):
        """
        Extract the average LDOS data from a bounding box region.
        
        Parameters:
        -----------
        bbox : tuple
            A tuple of four values representing the (x0, y0, x1, y1) bounding box coordinates.
        
        xr_data : xarray.Dataset
            The dataset containing the LDOS data.
        
        ch_name : str
            The name of the channel to extract the data from.
        
        Returns:
        --------
        avg_data : numpy.ndarray
            The average LDOS data over the bounding box region.
        """
        # Extract X and Y coordinates for the bounding box
        (x0, y0, x1, y1) = bbox
        # Subset the dataset to the bounding box region
        subset = xr_data.sel(X=slice(x0, x1), Y=slice(y0, y1))
        # Compute the mean LDOS over the selected X and Y region
        avg_data = subset[ch_name].mean(dim=['X', 'Y']).values
        return avg_data

    def bbox_callback(event):
        """
        Callback function that gets triggered when a bounding box is selected.
        
        It stores the bounding box coordinates and plots the average LDOS curve 
        for each bounding box.
        """
        # Store the selected bounding box
        bbox_list.append(bound_box.bounds)

        # Extract the average LDOS data for the selected bounding box
        avg_data = extract_bbox_data(bound_box.bounds, xr_data, ch_name)

        # Plot the average LDOS curve using seaborn
        sns.lineplot(x=xr_data['bias_mV'], y=avg_data, label=f'Bounding Box {len(bbox_list)}')
        # Add shaded region for standard deviation
        plt.fill_between(xr_data['bias_mV'], avg_data - np.std(avg_data), avg_data + np.std(avg_data), alpha=0.3)

        # If the number of selected boxes reaches the limit, stop the stream
        if len(bbox_list) >= number_of_bbox:
            plt.xlabel('Bias (mV)')
            plt.ylabel('Average LDOS')
            plt.legend()
            plt.title('Average LDOS across selected bounding boxes')
            plt.show()
            return hv.streams.Streaming.stop()  # Stop after the set number of boxes is reached

    # Register the callback function to be triggered when a bounding box is selected
    bound_box.param.watch(bbox_callback, 'bounds')

    # Display the dynamic map and points for bounding box selection
    display(dmap * bbox_points)

    return bbox_list  # Return the list of selected bounding boxes

# Example usage:
# bbox_list = bbox_select_plot(grid_LDOS, ch_name='LDOS', number_of_bbox=3)



# -

bbox_list = bbox_select_plot(grid_LDOS, ch_name='LDOS', number_of_bbox=4)

bbox_list 





hv_bias_mV_slicing_with_bbox(grid_LDOS, frame_width=400, bbox_list=bbox_list)

# +
import seaborn as sns
import matplotlib.pyplot as plt
import xarray as xr

def plot_ldos_within_bbox(xr_data, bbox_list=[], ch='LDOS', bbox_color_palette=None):
    '''
    input : xarray dataset, list of bounding boxes
    output : line plot of LDOS values within each bounding box using sns.lineplot with error bars,
             returns list of datasets with full X, Y, bias_mV dimensions for each bbox
    
    * Extracts full X, Y, bias_mV data within the bounding boxes and plots the LDOS with error bars.
    * Saves the extracted datasets for each bounding box in a list.
    
    Parameters:
    - xr_data : xarray.Dataset
        The input dataset containing the LDOS data.
    - bbox_list : list of tuples
        A list of bounding boxes, where each box is represented as (x0, y0, x1, y1).
    - ch : str, optional
        The channel to be visualized. Default is 'LDOS'.
    - bbox_color_palette : list, optional
        A list of colors to use for the line plots. Defaults to seaborn's color palette.
    
    Returns:
    - list of xarray.Dataset
        A list of datasets containing the full X, Y, bias_mV dimensions for each bounding box.
    '''
    # Set default color palette if none is provided
    if bbox_color_palette is None:
        bbox_color_palette = sns.color_palette(n_colors=len(bbox_list))

    # List to store the full LDOS datasets for each bounding box
    bbox_ldos_datasets = []

    plt.figure(figsize=(6, 4))  # Adjusted figure size

    # Create a plot for each bounding box
    for i, (bbox, color) in enumerate(zip(bbox_list, bbox_color_palette)):
        x0, y0, x1, y1 = bbox
        
        # Select the LDOS data within the bounding box (full X, Y, bias_mV dimensions)
        ldos_bbox = xr_data[ch].sel(X=slice(x0, x1), Y=slice(y0, y1))

        # Add the full dataset to the list
        bbox_ldos_datasets.append(ldos_bbox)

        # Convert the full dataset to a DataFrame for seaborn plotting
        df = ldos_bbox.to_dataframe().reset_index()

        # Plotting with sns.lineplot directly using the full data, errorbar computed by seaborn
        sns.lineplot(x='bias_mV', y=ch, data=df, color=color, label=f"BBox {i+1}", errorbar='sd')

    plt.title("LDOS within Bounding Boxes with Error Bars", fontsize='large')
    plt.xlabel("Bias (mV)", fontsize='large')
    plt.ylabel("LDOS", fontsize='large')
    plt.legend()
    plt.show()

    return bbox_ldos_datasets

# Example usage
# bbox_list = [(-2.027e-07, -9.38e-08, -1.745e-07, -6.02e-08), (-1.272e-07, -4.94e-08, -9.54e-08, -2.21e-08)]
# ldos_datasets = plot_ldos_within_bbox(grid_LDOS, bbox_list=bbox_list)



# +

ldos_list = plot_ldos_within_bbox(grid_LDOS, bbox_list=bbox_list)
# -

ldos_list[0]



# ## Select lasso  with holoview 
#

# #### single area selection 


# +
import xarray as xr
import holoviews as hv
from holoviews import streams
import numpy as np
import matplotlib.pyplot as plt

hv.extension('bokeh')

# Function to create a masked xarray based on selected points
def create_masked_xarray(points, selection_stream, original_dataset):
    # Get the selected indices
    selected_indices = selection_stream.index
    if len(selected_indices) > 0:
        # Get the shape of the original data
        data_shape = original_dataset.LDOS.isel(bias_mV=0).shape
        
        # Convert 1D indices to 2D (y, x) coordinates
        selected_points_2d = np.unravel_index(selected_indices, data_shape)
        
        # Create a mask with the same shape as the original data
        mask = np.zeros(data_shape, dtype=bool)
        
        # Set the mask values to True at the selected coordinates
        mask[selected_points_2d] = True
        
        # Create a masked xarray.Dataset
        masked_data = original_dataset.LDOS.isel(bias_mV=0).where(mask, drop=False)
        
        # Return the masked dataset and the mask
        return xr.Dataset({'LDOS_masked': masked_data}), mask
    else:
        return xr.Dataset(), None

# Function to create the interactive plot with lasso/box selection
def create_interactive_plot_with_mask(dataset):
    grid_channel_hv = hv.Dataset(dataset.LDOS)
    grid_channel_hv_image = hv.Dataset(dataset.LDOS.isel(bias_mV=0))
    
    dmap = grid_channel_hv.to(hv.Image, kdims=["X", "Y"], dynamic=True).opts(
        colorbar=True, cmap='bwr', frame_width=400, aspect='equal', title='XY plane slicing:'
    ).opt()
    
    points = hv.Points(grid_channel_hv_image).opts(
        frame_width=400, aspect='equal', alpha=0.2, tools=['box_select', 'lasso_select']
    )
    
    selection = streams.Selection1D(source=points)

    # Highlight the selected points dynamically
    def highlight_selected_area(index):
        selected_points = points.iloc[index]
        if len(selected_points) > 0:
            return hv.Points(selected_points).opts(color='lime', size=10, line_color='black', alpha=0.2)
        return hv.Points([])

    dynamic_selected_points = hv.DynamicMap(lambda index: highlight_selected_area(selection.index), streams=[selection])
    
    # Overlay the image, points, and dynamic selection
    overlay = dmap * points * dynamic_selected_points
    
    return overlay, selection

# Function to save the selected area as a masked xarray and visualize the mask
def save_selected_area_as_xarray(dataset, selection_stream):
    # Create the masked dataset and mask
    masked_dataset, mask = create_masked_xarray(hv.Points(dataset.LDOS.isel(bias_mV=0)), selection_stream, dataset)
    
    # Visualize the mask if it exists
    if mask is not None:
        plt.figure(figsize=(6,6))
        plt.imshow(mask, cmap='gray')
        plt.title("Selected Area Mask")
        plt.colorbar(label='Selected (True) / Not Selected (False)')
        plt.show()

    # Return the masked dataset
    return masked_dataset


# Step 1: Create the interactive plot for selecting areas
#interactive_plot, selection_stream = create_interactive_plot_with_mask(grid_LDOS)

# Display the interactive plot (in Jupyter Notebook, for example)
#interactive_plot

# Step 2: Save the selected area after making a selection
#selected_area_xarray = save_selected_area_as_xarray(grid_LDOS, selection_stream)

# Print the resulting masked xarray dataset
#print(selected_area_xarray)

# +
# Step 1: Create the interactive plot for selecting area
interactive_plot, selection_stream = create_interactive_plot_with_mask(grid_LDOS)

# Display the interactive plot (in Jupyter Notebook, for example)
interactive_plot
# -

# Step 2: Save the selected area after making a selection
selected_area_xarray = save_selected_area_as_xarray(grid_LDOS, selection_stream)
## XY is changed.. Y=X symmetric.... 
## (Y,X)  <==> (X,Y)
# Print the resulting masked xarray dataset
print(selected_area_xarray)



# ## Multiple area selection  (Ta rich )






# #### n_components=3 
#

# +
import os
import multiprocessing

# This block checks the number of CPU cores available on the system and sets the environment variable 
# OMP_NUM_THREADS accordingly, which controls the number of threads for parallel operations.
# If you want to include logical cores (hyper-threading), you can use os.cpu_count().
# Get the total number of cores (both physical and logical if applicable)
num_cores = multiprocessing.cpu_count()
print('num_cores: ', num_cores)

# Set the number of threads for parallel operations to the total number of cores.
# You can adjust this value based on your experiments.
os.environ["OMP_NUM_THREADS"] = str(num_cores)

# Perform tasks such as KMeans clustering
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# +
## update clustering with DBSCAN to consider spatial distance between peaks
# -



# +
from sklearn.cluster import DBSCAN
import numpy as np

# X, Y와 피크 특성 간 거리를 결합한 커스텀 거리 계산 함수
def custom_distance(a, b):
    spatial_dist = np.linalg.norm(a[:2] - b[:2])  # X, Y 좌표 거리
    feature_dist = np.linalg.norm(a[2:] - b[2:])  # 피크 특성 거리
    return spatial_dist + feature_dist  # 두 거리를 결합

# DBSCAN을 커스텀 거리로 적용
clustering = DBSCAN(metric=custom_distance, eps=0.5, min_samples=5)
df['cluster'] = clustering.fit_predict(features)

# -

df = grid_LDOS_pks_df.copy()

# +
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.graph_objects as go

# Explicitly copy a slice of the DataFrame where peak_prominences are greater than 1E-12 and peak_heights are positive
df = grid_LDOS_pks_df[(grid_LDOS_pks_df.peak_prominences > 1E-12) & (grid_LDOS_pks_df.peak_heights > 0)].copy()

# Selecting relevant features for PCA analysis
features = df[['bias_mV', 'peak_heights', 'peak_prominences', 'peak_widths']].values

# Perform Principal Component Analysis (PCA) and reduce the dimensionality to 3 components
pca = PCA(n_components=3)
pca_result = pca.fit_transform(features)

# Add the PCA results as new columns to the DataFrame
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]
df['PCA3'] = pca_result[:, 2]

# Perform KMeans clustering on the PCA results, dividing the data into 7 clusters
num_clusters = 7
kmeans = KMeans(n_clusters=num_clusters, n_init=10)
df['cluster'] = kmeans.fit_predict(pca_result)

# Create a 3D scatter plot using Plotly for interactive visualization
fig = go.Figure()

# Define a mapping between cluster numbers and marker symbols for better visualization
marker_symbols = ['circle', 'circle-open', 'cross', 'diamond', 'diamond-open', 'square', 'square-open', 'x']
cluster_to_symbol = {i: marker_symbols[i % len(marker_symbols)] for i in range(num_clusters)}

# Add a trace to the plot for each cluster, assigning unique marker symbols and colors to each cluster
for cluster in range(num_clusters):
    cluster_data = df[df['cluster'] == cluster]
    fig.add_trace(go.Scatter3d(
        x=cluster_data['X'],  # X-axis data
        y=cluster_data['Y'],  # Y-axis data
        z=cluster_data['bias_mV'],  # Z-axis data
        mode='markers',
        marker=dict(size=cluster_data.peak_widths * 50,  # Adjust marker size based on peak_widths
                    color=cluster_data.peak_heights,  # Color markers based on peak_heights
                    symbol=cluster_to_symbol[cluster],  # Assign cluster-specific marker symbol
                    opacity=0.8),  # Set marker opacity
        name=f'Cluster {cluster}'  # Add cluster name to the plot legend
    ))

# Set layout options for the plot
fig.update_layout(
    scene=dict(
        xaxis=dict(title='X-axis'),  # X-axis label
        yaxis=dict(title='Y-axis'),  # Y-axis label
        zaxis=dict(title='bias_mV'),  # Z-axis label
    ),
    title='3D Scatter Plot with Clustering',  # Plot title
    margin=dict(l=10, r=10, b=10, t=10),  # Set plot margins
    width=600,  # Width of the plot
    height=600  # Height of the plot
)

# Show the interactive 3D plot
fig.show()

# Create a copy of the DataFrame for further use
df_0 = df.copy()

# -


# #### Use elbow model for efficient calculation 

# +
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Explicitly copy a slice of the DataFrame where peak_prominences are greater than 1E-12 and peak_heights are positive
df = grid_LDOS_pks_df[(grid_LDOS_pks_df.peak_prominences > 1E-12) & (grid_LDOS_pks_df.peak_heights > 0)].copy()

# Selecting relevant features for PCA analysis
features = df[['bias_mV', 'peak_heights', 'peak_prominences', 'peak_widths']].values

# Perform Principal Component Analysis (PCA) and reduce the dimensionality to 3 components
pca = PCA(n_components=3)
pca_result = pca.fit_transform(features)

# Add the PCA results as new columns to the DataFrame
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]
df['PCA3'] = pca_result[:, 2]

# Elbow Method to find the optimal number of clusters
inertia = []
cluster_range = range(1, 11)  # Test with cluster numbers from 1 to 10
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    kmeans.fit(pca_result)
    inertia.append(kmeans.inertia_)

# Plot the Elbow curve
plt.figure(figsize=(8, 5))
plt.plot(cluster_range, inertia, 'bo-', markersize=8)
plt.xlabel('Number of clusters')
plt.ylabel('Inertia (within-cluster sum of squares)')
plt.title('Elbow Method to Determine Optimal Number of Clusters')
plt.grid(True)
plt.show()

# Perform KMeans clustering with the optimal number of clusters
# Here, assume that 7 clusters were found to be optimal based on the elbow plot.
num_clusters = 7
kmeans = KMeans(n_clusters=num_clusters, n_init=10)
df['cluster'] = kmeans.fit_predict(pca_result)

# Create a 3D scatter plot using Plotly for interactive visualization
fig = go.Figure()

# Define a mapping between cluster numbers and marker symbols for better visualization
marker_symbols = ['circle', 'circle-open', 'cross', 'diamond', 'diamond-open', 'square', 'square-open', 'x']
cluster_to_symbol = {i: marker_symbols[i % len(marker_symbols)] for i in range(num_clusters)}

# Add a trace to the plot for each cluster, assigning unique marker symbols and colors to each cluster
for cluster in range(num_clusters):
    cluster_data = df[df['cluster'] == cluster]
    fig.add_trace(go.Scatter3d(
        x=cluster_data['X'],  # X-axis data
        y=cluster_data['Y'],  # Y-axis data
        z=cluster_data['bias_mV'],  # Z-axis data
        mode='markers',
        marker=dict(size=cluster_data.peak_widths * 50,  # Adjust marker size based on peak_widths
                    color=cluster_data.peak_heights,  # Color markers based on peak_heights
                    symbol=cluster_to_symbol[cluster],  # Assign cluster-specific marker symbol
                    opacity=0.8),  # Set marker opacity
        name=f'Cluster {cluster}'  # Add cluster name to the plot legend
    ))

# Set layout options for the plot
fig.update_layout(
    scene=dict(
        xaxis=dict(title='X-axis'),  # X-axis label
        yaxis=dict(title='Y-axis'),  # Y-axis label
        zaxis=dict(title='bias_mV'),  # Z-axis label
    ),
    title='3D Scatter Plot with Clustering',  # Plot title
    margin=dict(l=10, r=10, b=10, t=10),  # Set plot margins
    width=600,  # Width of the plot
    height=600  # Height of the plot
)

# Show the interactive 3D plot
fig.show()

# Create a copy of the DataFrame for further use
df_0 = df.copy()
# -


# ### Use the Silhouette Analysis for better clustering quality 
#

# +
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Explicitly copy a slice of the DataFrame where peak_prominences are greater than 1E-12 and peak_heights are positive
df = grid_LDOS_pks_df[(grid_LDOS_pks_df.peak_prominences > 1E-12) & (grid_LDOS_pks_df.peak_heights > 0)].copy()

# Selecting relevant features for PCA analysis
features = df[['bias_mV', 'peak_heights', 'peak_prominences', 'peak_widths']].values

# Perform Principal Component Analysis (PCA) and reduce the dimensionality to 3 components
pca = PCA(n_components=3)
pca_result = pca.fit_transform(features)

# Add the PCA results as new columns to the DataFrame
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]
df['PCA3'] = pca_result[:, 2]

# Silhouette Analysis to find the optimal number of clusters
silhouette_scores = []
cluster_range = range(2, 11)  # Test with cluster numbers from 2 to 10 (min 2 clusters required for silhouette score)
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    cluster_labels = kmeans.fit_predict(pca_result)
    silhouette_avg = silhouette_score(pca_result, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plot the silhouette scores to visualize the best number of clusters
plt.figure(figsize=(8, 5))
plt.plot(cluster_range, silhouette_scores, 'bo-', markersize=8)
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis to Determine Optimal Number of Clusters')
plt.grid(True)
plt.show()

# Use the number of clusters with the highest silhouette score
optimal_clusters = cluster_range[silhouette_scores.index(max(silhouette_scores))]
print(f"Optimal number of clusters based on silhouette score: {optimal_clusters}")

# Perform KMeans clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, n_init=10)
df['cluster'] = kmeans.fit_predict(pca_result)

# Create a 3D scatter plot using Plotly for interactive visualization
fig = go.Figure()

# Define a mapping between cluster numbers and marker symbols for better visualization
marker_symbols = ['circle', 'circle-open', 'cross', 'diamond', 'diamond-open', 'square', 'square-open', 'x']
cluster_to_symbol = {i: marker_symbols[i % len(marker_symbols)] for i in range(optimal_clusters)}

# Add a trace to the plot for each cluster, assigning unique marker symbols and colors to each cluster
for cluster in range(optimal_clusters):
    cluster_data = df[df['cluster'] == cluster]
    fig.add_trace(go.Scatter3d(
        x=cluster_data['X'],  # X-axis data
        y=cluster_data['Y'],  # Y-axis data
        z=cluster_data['bias_mV'],  # Z-axis data
        mode='markers',
        marker=dict(size=cluster_data.peak_widths * 50,  # Adjust marker size based on peak_widths
                    color=cluster_data.peak_heights,  # Color markers based on peak_heights
                    symbol=cluster_to_symbol[cluster],  # Assign cluster-specific marker symbol
                    opacity=0.8),  # Set marker opacity
        name=f'Cluster {cluster}'  # Add cluster name to the plot legend
    ))

# Set layout options for the plot
fig.update_layout(
    scene=dict(
        xaxis=dict(title='X-axis'),  # X-axis label
        yaxis=dict(title='Y-axis'),  # Y-axis label
        zaxis=dict(title='bias_mV'),  # Z-axis label
    ),
    title='3D Scatter Plot with Clustering',  # Plot title
    margin=dict(l=10, r=10, b=10, t=10),  # Set plot margins
    width=600,  # Width of the plot
    height=600  # Height of the plot
)

# Show the interactive 3D plot
fig.show()

# Create a copy of the DataFrame for further use
df_0 = df.copy()

# -



# Define the min and max bias_mV range
min_val = -2.0
max_val = 2.0
# Filter the dataframe based on the bias_mV range
filtered_df = df[(grid_LDOS_pks_df.bias_mV < max_val) & (grid_LDOS_pks_df.bias_mV > min_val)]
# Create a figure
fig, ax = plt.subplots(figsize = (4,3))
# Plot the histogram
filtered_df.cluster.hist(bins=9, ax=ax)
# Set the title with dynamic min and max values
ax.set_title(f'Cluster Histogram {min_val}<bias_mV<{max_val}')
ax.set_xlabel('Cluster')
ax.set_ylabel('Frequency')
plt.show()

# +
## new scatter with selective clusters 
# +
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.markers as mmarkers


# peak_prominences의 최소값과 최대값 계산
min_prominence = df['peak_prominences'].min()
max_prominence = df['peak_prominences'].max()

# peak_widths의 최소값과 최대값 계산
min_width = df['peak_widths'].min()
max_width = df['peak_widths'].max()

# Normalize 객체 생성
norm_color = Normalize(vmin=min_prominence, vmax=max_prominence)
norm_size = Normalize(vmin=min_width, vmax=max_width)

# Define a mapping from cluster number to marker symbol
marker_symbols = ['o', 'o', '+', 'D', 'd', 's', '8', 'x']
num_clusters = df['cluster'].nunique()
cluster_to_symbol = {i: marker_symbols[i % len(marker_symbols)] for i in range(num_clusters)}

# scatterplot 그리기
def scatter_with_colorbar(x, y, peak_prominences, peak_widths, cluster, **kwargs):
    ax = plt.gca()
    sizes = norm_size(peak_widths) * 300  # 크기 조절 (300은 최대 크기, 필요에 따라 조정)
    marker = cluster_to_symbol[cluster.iloc[0]]  # 각 subplot은 하나의 cluster만 포함하므로 첫 번째 값 사용
    scatter = ax.scatter(x, y, c=peak_prominences, s=sizes, norm=norm_color, cmap='viridis', marker=marker, alpha =0.2)
    ax.grid('on')
    return scatter


# FacetGrid 생성 및 scatterplot 그리기
g = sns.FacetGrid(data=df, col='cluster', height=3, col_wrap=4)
g.map(scatter_with_colorbar, 'bias_mV', 'peak_heights', 'peak_prominences', 'peak_widths', 'cluster')
g.set_titles(col_template='Cluster {col_name}')

# 그래프의 레이아웃 조정
plt.tight_layout()

# 컬러바와 범례를 위한 공간 확보
g.fig.subplots_adjust(bottom=0.2, right=0.85)

# 컬러바 추가 (가로 방향)
cbar_ax = g.fig.add_axes([0.10, 0.05, 0.50, 0.02])  # [left, bottom, width, height]
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm_color, cmap='viridis'), 
                    cax=cbar_ax, orientation='horizontal', label='Peak Prominences')

# 마커 심볼에 대한 범례 추가 (가로 방향)
legend_elements = [plt.Line2D([0], [0], marker=marker, color='w', label=f'Cluster {cluster}',
                              markerfacecolor='gray', markersize=10)
                   for cluster, marker in cluster_to_symbol.items()]

# 범례를 위한 새로운 axes 생성
legend_ax = g.fig.add_axes([0.64, -0.08, 0.20, 0.02])  # [left, bottom, width, height]
legend_ax.axis('off')  # 축 숨기기

# 범례 추가
legend = legend_ax.legend(handles=legend_elements, loc='center', ncol=2, mode='expand', 
                          borderaxespad=0., title='Cluster Markers')
legend._loc = 3  # 범례 위치 조정 (하단 가운데)

plt.show()

# +
# in case of peak_prominence y axis, height colorbar 
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.markers as mmarkers


# peak_prominences의 최소값과 최대값 계산
min_prominence = df['peak_prominences'].min()
max_prominence = df['peak_prominences'].max()

# peak_height의 최소값과 최대값 계산
min_height = df['peak_heights'].min()
max_height = df['peak_heights'].max()


# peak_widths의 최소값과 최대값 계산
min_width = df['peak_widths'].min()
max_width = df['peak_widths'].max()

# Normalize 객체 생성
#norm_color = Normalize(vmin=min_prominence, vmax=max_prominence)
norm_color = Normalize(vmin=min_height, vmax=max_height)
norm_size = Normalize(vmin=min_width, vmax=max_width)

# Define a mapping from cluster number to marker symbol
marker_symbols = ['o', 'o', '+', 'D', 'd', 's', '8', 'x']
num_clusters = df['cluster'].nunique()
cluster_to_symbol = {i: marker_symbols[i % len(marker_symbols)] for i in range(num_clusters)}

# scatterplot 그리기
def scatter_with_colorbar(x, y, peak_heights, peak_widths, cluster, **kwargs):
    ax = plt.gca()
    sizes = norm_size(peak_widths) * 300  # 크기 조절 (300은 최대 크기, 필요에 따라 조정)
    marker = cluster_to_symbol[cluster.iloc[0]]  # 각 subplot은 하나의 cluster만 포함하므로 첫 번째 값 사용
    scatter = ax.scatter(x, y, c=peak_heights, s=sizes, norm=norm_color, cmap='viridis', marker=marker, alpha =0.1)
    ax.grid('on')
    return scatter


# FacetGrid 생성 및 scatterplot 그리기
g = sns.FacetGrid(data=df, col='cluster', height=3, col_wrap=4)
g.map(scatter_with_colorbar, 'bias_mV', 'peak_prominences', 'peak_heights','peak_widths', 'cluster')
g.set_titles(col_template='Cluster {col_name}')

# 그래프의 레이아웃 조정
plt.tight_layout()

# 컬러바와 범례를 위한 공간 확보
g.fig.subplots_adjust(bottom=0.2, right=0.85)

# 컬러바 추가 (가로 방향)
cbar_ax = g.fig.add_axes([0.10, 0.05, 0.50, 0.02])  # [left, bottom, width, height]
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm_color, cmap='viridis'), 
                    cax=cbar_ax, orientation='horizontal', label='Peak heights')

# 마커 심볼에 대한 범례 추가 (가로 방향)
legend_elements = [plt.Line2D([0], [0], marker=marker, color='w', label=f'Cluster {cluster}',
                              markerfacecolor='gray', markersize=10)
                   for cluster, marker in cluster_to_symbol.items()]

# 범례를 위한 새로운 axes 생성
legend_ax = g.fig.add_axes([0.64, -0.08, 0.20, 0.02])  # [left, bottom, width, height]
legend_ax.axis('off')  # 축 숨기기

# 범례 추가
legend = legend_ax.legend(handles=legend_elements, loc='center', ncol=2, mode='expand', 
                          borderaxespad=0., title='Cluster Markers')
legend._loc = 3  # 범례 위치 조정 (하단 가운데)

plt.show()


# +
#selected_clusters = [3,4]
#selected_clusters = [2]
selected_clusters = [1,2,3,4,5]
# Create 3D scatter plot

fig = go.Figure()
filtered_peak_heights_df = filtered_df[filtered_df.peak_prominences > 1E-13][filtered_df.peak_heights > 1E-12]
#filtered_peak_heights_df
filtered_scatter_plot_df = filtered_peak_heights_df.copy()
# Add traces for each cluster

# Define a mapping from cluster number to marker symbol
marker_symbols = ['circle', 'circle-open', 'cross', 'diamond',
            'diamond-open', 'square', 'square-open', 'x']
cluster_to_symbol = {i: marker_symbols[i % len(marker_symbols)] for i in range(num_clusters)}



for cluster in selected_clusters:
    cluster_data = filtered_scatter_plot_df[filtered_scatter_plot_df['cluster'] == cluster]
    fig.add_trace(go.Scatter3d(
        x=cluster_data['X'],
        y=cluster_data['Y'],
        z=cluster_data['bias_mV'],
        mode='markers',
        marker=dict(size=cluster_data.peak_widths*50,
                    color=cluster_data.peak_heights, symbol=cluster_to_symbol[cluster],
                    opacity=0.9,colorbar=dict (title='Peak Heights') ),
        name=f'Cluster {cluster}'
    ))

# Layout settings
fig.update_layout(
    scene=dict(
        xaxis=dict(title='X-axis'),
        yaxis=dict(title='Y-axis'),
        zaxis=dict(title='bias_mV'),
    ),
    #title='3D Scatter Plot with Clustering',
     title={
        'text': f"3D Scatter Plot with Clustering<br><sup>Selected Clusters: {', '.join(map(str, selected_clusters))}</sup>",
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    margin=dict(l=10, r=10, b=10, t=10),
    
    width=750,  # Desired width (in pixels)
    height=750  # Desired height (in pixels)
    
  
)

# Show interactive plot
fig.show()
# -

# #### check statatistic 
# * between -1 ~ 1 mV,  histogram of clusters 


# +
#grid_LDOS_pks_df_pks = df[(df.cluster ==1)|(df.cluster ==2)|(df.cluster ==3)|(df.cluster ==4)|(df.cluster ==5)]
grid_LDOS_pks_df_pks = df[df['cluster'].isin([0, 5])]
#grid_LDOS_pks_df_zbp = df[(df.cluster ==3)|(df.cluster ==6)|(df.cluster ==7)|(df.cluster ==6)]
grid_LDOS_pks_df_pks




# +
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

# 데이터 불러오기 (이미 df로 로드되어 있다고 가정)
df = grid_LDOS_pks_df_pks

# 사용할 열 선택
columns_to_use = ['Y','X', 'peak','bias_mV', 'peak_heights', 'peak_prominences', 'peak_widths']

# 선택한 열만 추출
X = df[columns_to_use]

# 데이터 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans 클러스터링 수행
n_clusters = 6
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# 결과를 원본 데이터프레임에 추가
df['new_cluster'] = cluster_labels


# 클러스터별 샘플 수 확인
print("\n클러스터별 샘플 수:")
print(df['new_cluster'].value_counts().sort_index())

# (선택적) KNN을 사용한 클러스터 중심 찾기
nn = NearestNeighbors(n_neighbors=1)
nn.fit(X_scaled)
cluster_centers = kmeans.cluster_centers_
closest_points = nn.kneighbors(cluster_centers, return_distance=False)

print("\n각 클러스터의 중심에 가장 가까운 데이터 포인트:")
for i, point_idx in enumerate(closest_points):
    print(f"Cluster {i}: {df.iloc[point_idx[0]]['Y']}, {df.iloc[point_idx[0]]['X']}")

# +
import plotly.express as px

# Symbol mapping (adjust according to the number of clusters)
symbol_map = {0: 'circle', 1: 'circle-open', 2: 'cross', 3: 'diamond', 
              4: 'diamond-open', 5: 'square', 6: 'square-open', 7: 'x'}

# Create 3D scatter plot
fig = px.scatter_3d(df, 
                    x='X', y='Y', z='bias_mV',
                    color='peak_heights',
                    size='peak_widths',
                    symbol='new_cluster',
                    symbol_map=symbol_map,
                    opacity=0.3,  # Set opacity for all points (0.0 to 1.0)
                    labels={'X': 'X ', 
                            'Y': 'Y ', 
                            'bias_mV': 'bias_mV',
                            'peak_heights': 'peak_heights',
                            'peak_widths': 'peak_widths'},
                    title='LDOS pks')

# Set color scale
fig.update_traces(marker=dict(colorscale='Viridis'))

# Adjust layout
fig.update_layout(scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='bias_mV'),
                  width=800,
                  height=800)

# Display the plot
fig.show()
# -

df

# +

# x, y 값이 같은 그룹 내에서 peak bias 간격 계산
df = df.sort_values(['X', 'Y', 'bias_mV'])
df['pk_distance'] = df.groupby(['X', 'Y'])['bias_mV'].diff()

# 각 (X, Y) 그룹의 마지막 행에 np.nan 할당
df.loc[df.groupby(['X', 'Y'])['bias_mV'].idxmax(), 'pk_distance'] = np.nan

# cluster별로 kde plot 생성
g = sns.FacetGrid(df, col="cluster", col_wrap=4, height=3)
g.map(sns.kdeplot, "pk_distance", shade=True)
# 각 서브플롯에 그리드 추가
for ax in g.axes.flat:
    ax.grid(True)

g.set_axis_labels("Peak Bias Gap", "Density")
g.fig.suptitle("Distribution of Peak Bias Gaps by Cluster", y=1.02)
plt.tight_layout()
plt.show()

# +

sns.kdeplot (df.bias_mV)
# -

# ### to check peak energy position distribution to check CdGM states 

# +
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def kde_and_peaks(data, bw_method=0.2):
    """
    Perform Kernel Density Estimation (KDE) on the input data and find peaks.

    Parameters:
    data (array-like): Input data to compute KDE and detect peaks.
    bw_method (float): The bandwidth method for KDE. Default is 0.2.

    Returns:
    x_range (ndarray): The range of x values.
    y (ndarray): KDE values for the corresponding x values.
    peak_indices (ndarray): Indices of the peaks found in the KDE curve.
    """
    if len(data) > 1:  # Ensure there are at least 2 data points
        kde = gaussian_kde(data, bw_method=bw_method)
        x_range = np.linspace(data.min(), data.max(), 1000)  # Create a range for KDE evaluation
        y = kde(x_range)  # Compute KDE
        
        # Detect peaks (local maxima) in the KDE curve
        peaks = np.diff(np.sign(np.diff(y))) < 0
        peak_indices = np.where(peaks)[0] + 1  # Adjust peak indices to match the original data length
        return x_range, y, peak_indices
    else:
        return np.array([]), np.array([]), np.array([])  # Return empty arrays if data is insufficient

def plot_kde_peaks(df, col_wrap=4, height=3):
    """
    Plot KDE curves and detected peaks for each cluster in the DataFrame.

    Parameters:
    df (DataFrame): Input DataFrame containing cluster information and peak distances.
    col_wrap (int): Number of columns in the subplot grid.
    height (int): Height of each subplot.

    Returns:
    summary_df (DataFrame): A DataFrame summarizing the number of peaks per cluster.
    peak_distances_dict (dict): A dictionary containing peak distances for each cluster.
    """
    n_clusters = df['cluster'].nunique()  # Count unique clusters
    # Set up the figure and axes for subplots
    fig, axes = plt.subplots(nrows=(n_clusters + col_wrap - 1) // col_wrap, ncols=col_wrap, figsize=(col_wrap*height, height*(n_clusters // col_wrap + 1)))
    axes = axes.flatten()  # Flatten axes for easier access
    
    peak_distances_dict = {}  # Dictionary to store peak distances per cluster
    summary_data = []  # List to store peak summary information
    
    for i in range(n_clusters):
        # Select data for the current cluster
        cluster_data = df[df['cluster'] == i]['pk_distance'].dropna()
        
        if len(cluster_data) > 1:  # Check if there are enough data points
            x, y, peak_indices = kde_and_peaks(cluster_data)  # Perform KDE and find peaks
            
            # Plot histogram and KDE curve
            axes[i].hist(cluster_data, bins=50, density=True, alpha=0.5, edgecolor='black')
            axes[i].plot(x, y)
            
            # Store summary and peak distances for the current cluster
            summary_data.append({'cluster': i, 'peak_count': len(peak_indices)})
            peak_distances_dict[i] = x[peak_indices]  # Store peak distances
        else:
            # Display a message for clusters with insufficient data
            axes[i].text(0.5, 0.5, "Insufficient data", horizontalalignment='center', verticalalignment='center', transform=axes[i].transAxes)
            summary_data.append({'cluster': i, 'peak_count': 0})
    
    plt.tight_layout()  # Adjust the layout to prevent overlapping plots
    plt.show()  # Display the plot
    
    # Return summary and peak distances
    return pd.DataFrame(summary_data), peak_distances_dict



# +

# Execute the function
summary_df, peak_distances_dict = plot_kde_peaks(df, col_wrap=4, height=3)
print(summary_df)
# Concatenate all peak distances into a single array
all_distances = np.concatenate(list(peak_distances_dict.values()))

sns.scatterplot(all_distances)
# -

peak_distances_dict

# ## How to distinguish SC peaks 
# ### peak assignment comparision with 2nd derivative EDC-like results

# ## additional clustering after MZBS collecting 
#



# ## Filtering, Thresholds, and Segmentation
#
# * back to [**Preparation**](#Preparation)

grid_topo.topography.plot()



# ##### grid_topo rotation 

def rotate_2D_xr(xrdata, rotation_angle):
    """
    Rotate a 2D xarray dataset while maintaining its original center position.

    This function performs the following steps:
    1. Calculates the original center of the xarray.
    2. Pads the xarray to accommodate rotation.
    3. Rotates the padded xarray.
    4. Adjusts the coordinates to maintain the original center position.

    Parameters:
    -----------
    xrdata : xarray.Dataset
        The input 2D xarray dataset to be rotated.
    rotation_angle : float
        The angle of rotation in degrees.

    Returns:
    --------
    xarray.Dataset
        The rotated xarray dataset with preserved center position.
    """

    # Calculate original center
    original_center_x = xrdata.X.mean().item()
    original_center_y = xrdata.Y.mean().item()

    # Calculate padding size
    for ch_i, ch_name in enumerate(xrdata):
        if ch_i == 0:  # use only the first channel to calculate padding size 
            padding_shape = skimage.transform.rotate(xrdata[ch_name].values.astype('float64'),
                                                     rotation_angle,
                                                     resize=True).shape          
            padding_xy = (np.array(padding_shape) - np.array(xrdata[ch_name].shape) + 1) / 2
            padding_xy = padding_xy.astype(int)
    
    # Pad the xarray
    xrdata_pad = xrdata.pad(X=(padding_xy[0], padding_xy[0]), 
                            Y=(padding_xy[1], padding_xy[1]),
                            mode='constant',
                            cval=xrdata.min())

    # Handle cases where padding shape might differ
    if np.array(xrdata_pad[ch_name]).shape != padding_shape:
        # Case: padding shape is +1 larger than rotation shape
        x_spacing = np.diff(xrdata.X).mean()
        y_spacing = np.diff(xrdata.Y).mean()
        x_pad_dim = padding_shape[0]
        y_pad_dim = padding_shape[1]
        x_pad_arr = np.linspace(-1*padding_xy[0]*x_spacing, x_spacing*x_pad_dim, x_pad_dim+1)
        y_pad_arr = np.linspace(-1*padding_xy[1]*y_spacing, y_spacing*y_pad_dim, y_pad_dim+1)
        xrdata_pad = xrdata_pad.assign_coords({"X": x_pad_arr}).assign_coords({"Y": y_pad_arr})
        xrdata_rot = xrdata_pad.sel(X=xrdata_pad.X[:-1].values, Y=xrdata_pad.Y[:-1].values)
        print('padding size != rot_size')
    else:
        # Case: padding shape matches rotation shape
        x_spacing = np.diff(xrdata.X).mean()
        y_spacing = np.diff(xrdata.Y).mean()
        x_pad_dim = padding_shape[0]
        y_pad_dim = padding_shape[1]
        x_pad_arr = np.linspace(-1*padding_xy[0]*x_spacing, x_spacing*x_pad_dim, x_pad_dim)
        y_pad_arr = np.linspace(-1*padding_xy[1]*y_spacing, y_spacing*y_pad_dim, y_pad_dim)
        xrdata_pad = xrdata_pad.assign_coords({"X": x_pad_arr}).assign_coords({"Y": y_pad_arr})
        xrdata_rot = xrdata_pad.copy()
        print('padding size == rot_size')

    # Perform rotation on each channel
    for ch in xrdata:
        xrdata_rot[ch].values = skimage.transform.rotate(xrdata[ch].values.astype('float64'),
                                                         rotation_angle,
                                                         resize=True,
                                                         cval=xrdata[ch].values.astype('float64').min())

    # Adjust coordinates to maintain original center
    new_center_x = xrdata_rot.X.mean().item()
    new_center_y = xrdata_rot.Y.mean().item()

    x_offset = original_center_x - new_center_x
    y_offset = original_center_y - new_center_y

    # Apply offset to maintain original center
    xrdata_rot = xrdata_rot.assign_coords(X=xrdata_rot.X + x_offset, Y=xrdata_rot.Y + y_offset)

    return xrdata_rot


grid_topo_r32 = rotate_2D_xr(grid_topo, rotation_angle= 32)
grid_topo_r32.topography.plot()
plt.show()

# ##### grid_LDOS rotation  

grid_topo_r32

grid_LDOS_r32 = rotate_3D_xr(grid_LDOS, rotation_angle= 32)
hv_bias_mV_slicing(grid_LDOS_r32,ch='LDOS')

# ###### grid_LDOS rotation & slicing_bias_mV

hv_bias_mV_slicing(grid_LDOS_r32,ch='LDOS')

# ###### grid_LDOS rotation & crop near step edge 
#

grid_LDOS_r32_crop = grid_LDOS_r32.where((grid_LDOS_r32.X >1E-7)&
                                         (grid_LDOS_r32.X < 2.5E-7)&
                                         (grid_LDOS_r32.Y >2E-7)&
                                         (grid_LDOS_r32.Y < 3E-7), drop = True)
hv_bias_mV_slicing(grid_LDOS_r32_crop,ch='LDOS')

# ###### avg Y direction 

#grid_LDOS_r32_crop.LDOS.to_dataframe().groupby('X').mean().plot()#.groups#..mean()#.plot()
sns.lineplot(grid_LDOS_r32_crop.LDOS.to_dataframe(), x='X', y='LDOS')
plt.show()

grid_LDOS_r32_crop.mean(dim='Y').LDOS.T.plot()
plt.show()

grid_topo_r32

grid_topo_r32_crop = grid_topo_r32.where((grid_LDOS_r32.X >1E-7)&
                                         (grid_LDOS_r32.X < 2.5E-7)&
                                         (grid_LDOS_r32.Y >2E-7)&
                                         (grid_LDOS_r32.Y < 3E-7), drop = True)
grid_topo_r32_crop

# Usage:
plot_overlay_with_grid_topo_N_crop = plot_overlay_with_crop(grid_topo_r32_crop, grid_topo_r32)
plot_overlay_with_grid_topo_N_crop
plt.show()

# +
# Flattening and Drift Compensation 
# -


# ## bounding box selection of grid_LDOS data 

# ## Flattening and Drift Compensation 
#
#
# * back to [**Preparation**](#Preparation)

# After  grid2xr function 
# grid_xr.bias_mv ==> interpolated 
grid_3D.bias_mV.plot.scatter()


# * **2024MayFeTe0.55Se0.45** dataset shows different bias_mV offset for every pixel
#
#     * find the bias_mV valuve where grid_3D.I_fb = 0
# * gird_3D.bias_V.isel(grid_3D.I_fb=0)

# ## <font color= blue >  1-3. Check bias_mV ofsset  </font> 
# ### <font color= blue >  1-3.1. check the current is zero when bias_mV is zero  </font> 
# * if bias offset is close to zero : leave it as it is 
# * "bias_offset_adjust ==  **Falas** " is default
#
# * if "bias_offset_adjust ==  **True** "  $\to$ offset adjustment for bias_mV. 
# * __select defect free area for bais_mV = 0  reference__

def hv_XY_slicing(xr_data,
                  ch = 'LIX',
                  slicing= 'X', 
                  frame_width = 200,
                  cmap = 'bwr'): 
    '''
    input : xarray dataset 
    output : holoview image 
    
    
    * slicing 3D data set in X-bias_mV or Y-bias_mV plane 
    * X or Y position is knob
    
    
    default channel  =  'LIX_fb',  or assgin 'I_fb'
    default setting for frame width and cmap  can be changed. 
    if you need to add color limit 
     
    add ".opts(clim=(0, 1E-10))"
    
    '''
    import holoviews as hv
    from holoviews import opts

    xr_data_hv = hv.Dataset(xr_data[ch])

    hv.extension('bokeh')
    ###############
    # bias_mV slicing
    if slicing == 'Y':
        dmap_plane  = [ "X","bias_mV"]

        dmap = xr_data_hv.to(hv.Image,
                             kdims = dmap_plane,
                             dynamic = True )
        dmap.opts(colorbar = True,
                  cmap = cmap,
                  frame_width = frame_width).relabel('X - bias_mV plane slicing: ')
    else : #slicing= 'X'
        dmap_plane  = [ "Y","bias_mV"]

        dmap = xr_data_hv.to(hv.Image,
                             kdims = dmap_plane,
                             dynamic = True )
        dmap.opts(colorbar = True,
                  cmap = cmap,
                  frame_width = frame_width).relabel('Y - bias_mV plane slicing: ')
    fig = hv.render(dmap)
    return dmap   


# +
#grid_3D_bias_zm 
# close up the bias range near zero 

grid_LDOS_bias_zm = grid_LDOS.where(grid_LDOS.bias_mV>-0.5, drop=True).where(grid_LDOS.bias_mV<0.5, drop=True)
hv_XY_slicing(grid_LDOS_bias_zm, ch = 'LDOS',slicing= 'X')

# to check bias offset near low bais region only. 
# -

# ### <font color= blue >  1-3.2. Select bounding box with holoview  </font> 
#
# * if there is large near-zero state, bias offset adjustment is affecte 
#     $\to$ avoid defective area 
# * make it as a function later 
# * ATM, I could not find the way to add an interactive bbox options to functions. 
#     * fail to callback(or stream) of selected bbox points. 

# +
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')

xr_data = grid_LDOS; ch = 'LDOS'; frame_width = 400
xr_data_channel_hv = hv.Dataset(xr_data[ch])

# bias_mV slicing
dmap_plane  = ["X","Y"]
dmap = xr_data_channel_hv.to(hv.Image,
                          kdims = dmap_plane,
                          dynamic = True )
dmap.opts(colorbar = True,
          cmap = 'bwr',
          frame_width = frame_width,
          aspect = 'equal')#.relabel('XY plane slicing: ')
xr_data_channel_hv_image  = hv.Dataset(
    xr_data[ch].isel(bias_mV = 0)).relabel('for BBox selection : ')

bbox_points = hv.Points(xr_data_channel_hv_image).opts(frame_width = frame_width,
                                                    color = 'k',
                                                    aspect = 'equal',
                                                    alpha = 0.1,                                   
                                                    tools=['box_select'])

bound_box = hv.streams.BoundsXY(source = bbox_points,
                                bounds=(0,0,0,0))
#dmap.opts(clim = (0,1E-10))*bbox_points
dmap.opts()*bbox_points

# use the bound_box 
# -

bound_box.bounds


# ### <font color= blue >  1-3.3. Crop original xr data with selected bounding box </font> 
#

def crop_xr_data_with_bounding_box(xr_data, bound_box= bound_box):
    """
    Crop a region from a DataArray based on a bounding box.

    Parameters:
    - xr_data (xr.DataArray): The input DataArray to be cropped.
    - bound_box (tuple): A tuple containing the coordinates (x0, y0, x1, y1) 
    of the bounding box.
    - bound_box is from hv bbox points 
    

    Returns:
    - cropped_data (xr.DataArray): The cropped DataArray.
    """
    
    if (bound_box.bounds[0]==0)&(bound_box.bounds[1]==0)&(bound_box.bounds[2]==0)&(bound_box.bounds[3]==0):
    # No box selection
        x0, y0, x1, y1 = xr_data.X.min(), xr_data.Y.min(), xr_data.X.max(), xr_data.Y.max()
        print ("no bound_box")
    else: 
        x0, y0, x1, y1 = bound_box.bounds
        print ("bound_box selected")
    cropped_data = xr_data.sel(X=slice(x0, x1), Y=slice(y0, y1))
    return cropped_data


grid_3D_bias_zm_crop = crop_xr_data_with_bounding_box(grid_3D_bias_zm,bound_box)
grid_3D_bias_zm_crop


# ### <font color= blue >  1-3.4. Bias range offset using offest for zero from selected bounding box </font> 
#
#
# ####   In case of <font color= red > **superconducting** STS, </font> bias offset using I_fb_avg is not correct!

##
# find neares I =0 bias_mV 
def Bias_mV_offset_crop_area_avg(originalxr = grid_3D,
                                 cropped_xr = grid_3D_bias_zm_crop,
                                 bais_offset_adjust = False):
    originalxr_offset = originalxr.copy()
    I_fb_avg_df = cropped_xr.I_fb.mean (dim = ['X','Y']).to_dataframe().abs()
    if I_fb_avg_df.I_fb.idxmin() == 0:
        print ('Bias_mV is set to I = 0')
    else:
        print ('need to adjust Bias_mV Zero')
        print ('Bias_mV Zero shifted : '+ str( round(I_fb_avg_df.I_fb.idxmin(),3)  )+ ' mV')
    if bais_offset_adjust == True : 
        # when adjust bias offset only when it is True.         
        originalxr_offset = originalxr.assign_coords(bias_mV= (originalxr.bias_mV + I_fb_avg_df.I_fb.idxmin()  ))   
    else: pass
    return originalxr_offset


grid_3D = Bias_mV_offset_crop_area_avg(originalxr = grid_3D, cropped_xr = grid_3D_bias_zm_crop,bais_offset_adjust = False)
#grid_3D_0.where(grid_3D_0.bias_mV>-1, drop = True).where(grid_3D_0.bias_mV<1, drop = True)
#use bias_mV crop if needed
# check low bais region only. 

# +
#grid_3D
# -

grid_3D.attrs['bias_mV_step'] = np.abs(np.unique(np.diff(grid_3D.bias_mV.values))).mean().round(3)
grid_3D

#hv_XY_slicing(grid_3D_0, ch = 'LIX_fb',slicing= 'X')#.opts(clim=(0, 0.8E-10)) # Use climn if needed
hv_XY_slicing(grid_3D.where(grid_3D.bias_mV>-0.6, drop = True).where(grid_3D.bias_mV<0.6, drop = True), ch = 'LIX_fb',slicing= 'X').opts(clim=(0, 0.3E-12)) # Use climn if needed

# ### 1.2.3. Unit calculation (LDOS_fb)
#     * for semiconductor: CBM,VBM check. gap_map check
#     * add gap_maps to grid_2D

# +
grid_3D_gap =  grid_3D_SCgap(grid_3D)
grid_3D_gap

grid_LDOS = grid_3D_gap[['LDOS_fb' ]]
grid_LDOS
# -


# ### 1.4 Topography view 

# +
grid_topo =  plane_fit_y_xr(plane_fit_surface_xr(grid_topo))
grid_topo

#isns.imshow(plane_fit_y_xr(grid_topo).where(grid_topo.Y < 0.7E-9, drop=True).topography)

#grid_topo = grid_topo.drop('gap_map_I').drop('gap_map_LIX')

isns.imshow(grid_topo.topography, cmap ='copper')
#isns.imshow(grid_topo.topography.T, cmap ='copper')
plt.show()
# -


# # Analysis 
#
# back to * [**Preparation**](#Preparation)

# ### add z drift compensation with 1D plot 

# ####  check topography drift & compensate it 

# +
# make a 1D topo curve (z vs data acquisition time)
topo1D = grid_xr.topography.values.ravel()

# Convert to a different endianness and create a Series
topo1D = topo1D.byteswap().newbyteorder()
topo1Ddf = pd.Series(topo1D)

# Calculate the differential
topo1D_diff_df = topo1Ddf.diff(periods=1)
sns.scatterplot(topo1D_diff_df)


# +
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def perform_polyfit_and_save_difference(topo1Ddf_0, degree=2):
    """
    Perform polynomial fitting on the given pandas Series and save the difference
    between the original data and the fitted polynomial.

    Args:
        topo1Ddf_0 (pandas.Series): The input pandas Series for polynomial fitting.
        degree (int): The degree of the polynomial fit (default is 2 for a quadratic fit).

    Returns:
        pandas.Series: A new pandas Series containing the difference between the original data
                      and the fitted polynomial.
    """
    # Perform polyfit
    coefficients = np.polyfit(topo1Ddf_0.index, topo1Ddf_0, degree)
    p = np.poly1d(coefficients)

    # Calculate the difference between the original data and polyfit result
    topo1Ddf_0_sub_polyfit = topo1Ddf_0 - p(topo1Ddf_0.index)

    # Create a new figure
    fig, ax = plt.subplots()

    # Plot the original data
    topo1Ddf_0.plot(ax=ax, label='Original Data')

    # Plot the polyfit result
    ax.plot(topo1Ddf_0.index, p(topo1Ddf_0.index), color='green', label=f'{degree}th-degree Regression')

    # Set labels and legend
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.legend()

    # Show the plot
    plt.show()

    # Return the difference between original data and polyfit result
    return topo1Ddf_0_sub_polyfit

# Example usage:
# Replace topo1Ddf_0 with your pandas Series
# By calling the function, you will get the topo1Ddf_0_sub_polyfit Series
# -


# #### If there is outliers, split Z topo change for separate fitting

# +
## check outliers in Topography 
## for better background substraction 


def find_outlier_indices(series):
    """
    Finds the indices of outliers in the given Series using the IQR method.

    Args:
        series (pandas.Series): The input 1D Series containing potential outliers.

    Returns:
        pandas.Index: A pandas Index object containing the indices of the outliers.
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 -8* IQR
    upper_bound = Q3 +8 * IQR
    return series[(series < lower_bound) | (series > upper_bound)].index



# -

find_outlier_indices(topo1D_diff_df.diff())
print(find_outlier_indices(topo1D_diff_df.diff()))
# check number outliers 

# #### if there is no outlier (= no abrupt Z change), skip this section 

"""
#
topo1D_diff_df.iloc[find_outlier_indices(topo1D_diff_df.diff())[0]-3:
                    find_outlier_indices(topo1D_diff_df.diff())[0]+3].plot()

# domain selection --> after outlier detection 
#############
#  split topo 1D 
##############
topo1Ddf_0 = topo1Ddf.iloc[:find_outlier_indices(topo1D_diff_df.diff())[0]]
topo1Ddf_1 = topo1Ddf.iloc[find_outlier_indices(topo1D_diff_df.diff())[0]:]
# area selection 
"""


"""
# substract polyfit results from original topography 
# separeate polyfit applied for each area (divided by outliers) 

topo1Ddf_0_sub = perform_polyfit_and_save_difference(topo1Ddf_0)
topo1Ddf_1_sub = perform_polyfit_and_save_difference(topo1Ddf_1)

# connect substracted topography & reshape the topography 

topo1Ddf_sub= pd.concat([topo1Ddf_0_sub,topo1Ddf_1_sub])

"""


sns.lineplot(topo1D_diff_df)

# +
# connect substracted topography & reshape the topography 

topo1Ddf_sub= perform_polyfit_and_save_difference(topo1Ddf)

topo1Ddf_sub_xr = grid_topo.copy()
topo1Ddf_sub_xr.topography.values = topo1Ddf_sub.to_numpy().reshape(grid_topo.topography.shape)


# +
plane_fit_y_xr(topo1Ddf_sub_xr).topography.plot()
plt.show()

grid_topo = plane_fit_y_xr(topo1Ddf_sub_xr)
isns.imshow(grid_topo.topography)
# -

# ##  Grid area extract 
#
# ### grid 3D_LDOS
#
#
#

# # Crop X 

grid_LDOS = grid_LDOS.isel(X=slice(0, 40))

# ## bias range selection to reduce calculation time 

# +
data_array_slct = grid_LDOS.where(grid_LDOS.bias_mV<1, drop=True).where(grid_LDOS.bias_mV>-1, drop=True)
data_array_slct
# check PCA_DBSCAN_clstr channel is broadcasting 

# check XY slicing 
hv.extension('bokeh')
hv_XY_slicing(data_array_slct, ch = 'LDOS_fb',slicing= 'X').opts(clim=(0, 1E-10)) # check low intensity area


# +
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')

xr_data = data_array_slct; ch = 'LDOS_fb'; frame_width = 400
xr_data_channel_hv = hv.Dataset(xr_data[ch])

# bias_mV slicing
dmap_plane  = ["X","Y"]
dmap = xr_data_channel_hv.to(hv.Image,
                          kdims = dmap_plane,
                          dynamic = True )
dmap.opts(colorbar = True,
          cmap = 'bwr',
          frame_width = frame_width,
          aspect = 'equal')#.relabel('XY plane slicing: ')
xr_data_channel_hv_image  = hv.Dataset(
    xr_data[ch].isel(bias_mV = 0)).relabel('for BBox selection : ')

bbox_points = hv.Points(xr_data_channel_hv_image).opts(frame_width = frame_width,
                                                    color = 'k',
                                                    aspect = 'equal',
                                                    alpha = 0.1,                                   
                                                    tools=['box_select'])

bound_box = hv.streams.BoundsXY(source = bbox_points,
                                bounds=(0,0,0,0))
#dmap.opts(clim = (0,1E-10))*bbox_points
dmap.opts()*bbox_points

# use the bound_box 
# -

# ### bound_box region based on hv selection tool

bound_box
print (bound_box.bounds)

# +
###########
# select from boundbox

#grid_3D_crop = crop_xr_data_with_bounding_box(data_array_slct, bound_box)

# or use original setup

grid_3D_crop = data_array_slct

# +
#grid_3D_crop= grid_3D_crop.drop_vars('PCA_DBSCAN_clstr')
# "drop_vars" can be Still DataSet,
# Choose ['LDOS_fb'] is DataArray
grid_3D_crop

grid_3D_crop.to_dataframe().describe()
# -

# grid_LDOS_crop= grid_LDOS.where(grid_LDOS.bias_mV<1.5, drop = True).where(grid_LDOS.bias_mV>-1.5, drop = True)


# ### 1.6.Data Selection with HoloView
# * using Bounding Box or Lasso
#
# * currently only Bounding Box plot is working. 
# * check the Lass selection later. 
# * use stream pipe line (not a functino yet..)
#

# #### 1.6.2 bokeh plot & Bound box selection 
# ####       $\to$ selected points = Bound Box 

grid_LDOS

# +
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')

xr_data = grid_LDOS
ch_name = 'LDOS'
frame_width = 400

xr_data_channel_hv = hv.Dataset(xr_data[ch_name])

# bias_mV slicing
dmap_plane  = ["X","Y"]
dmap = xr_data_channel_hv.to(hv.Image,
                          kdims = dmap_plane,
                          dynamic = True )
dmap.opts(colorbar = True,
          cmap = 'bwr',
          frame_width = frame_width,
          aspect = 'equal')#.relabel('XY plane slicing: ')

xr_data_channel_hv_image  = hv.Dataset(xr_data[ch_name].isel(bias_mV = 0)).relabel('for BBox selection : ')

bbox_points = hv.Points(xr_data_channel_hv_image).opts(frame_width = frame_width,
                                                    color = 'k',
                                                    aspect = 'equal',
                                                    alpha = 0.1,                                   
                                                    tools=['box_select'])

bound_box = hv.streams.BoundsXY(source = bbox_points,
                                bounds=(0,0,0,0))
#dmap.opts(clim = (0,1E-10))*bbox_points
dmap.opts()*bbox_points


## hv.DynamicMap( 뒤에는 function 이 와야함), streams  로 해당 영역을 지정.( or 함수의 입력정보 지정) 
# averaged curve 를 그리기 위해서 해당영역을  xr  에서 average  해야함.. 
# curve 의 area 로 error bar도 같이 그릴것.. 
# -

bbox_1_BoundsXY = bound_box


bbox_2_BoundsXY = bound_box

bbox_3_BoundsXY = bound_box

bbox_2, _ = hv_bbox_avg(grid_LDOS, bound_box= bound_box, ch = ch_name,slicing_bias_mV = 0.2)
#bbox_3, _ = hv_bbox_avg(grid_LDOS, bound_box= bound_box, ch = ch_name,slicing_bias_mV = 0.4)
#bbox_4, _ = hv_bbox_avg(grid_LDOS, bound_box= bound_box, ch ='ch_name,slicing_bias_mV = 0.4)

#bbox_list = [bbox_1_BoundsXY, bbox_2_BoundsXY]
bbox_list = [bbox_1_BoundsXY, bbox_2_BoundsXY,bbox_3_BoundsXY]
bbox_list

grid_LDOS

# +
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

def plot_ldos_with_bboxes(grid_LDOS, bbox_list, slicing_bias_mV):
    # 그래프 설정
    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    # LDOS 이미지 그리기
    ldos_slice = grid_LDOS.LDOS.sel(bias_mV=slicing_bias_mV, method = 'nearest')
    im = ax1.imshow(ldos_slice, extent=[grid_LDOS.X.min(), grid_LDOS.X.max(), 
                                        grid_LDOS.Y.min(), grid_LDOS.Y.max()],
                    origin='lower', aspect='equal')
    plt.colorbar(im, ax=ax1, label='LDOS')
    
    # 색상 설정
    colors = plt.cm.rainbow(np.linspace(0, 1, len(bbox_list)))
    
    # bbox 그리기 및 평균 LDOS 계산
    for i, (bbox, color) in enumerate(zip(bbox_list, colors), 1):
        x, y, w, h = bbox.bounds
        rect = Rectangle((x, y), w-x, h-y, fill=False, edgecolor=color, linewidth=2)
        ax1.add_patch(rect)
        ax1.text(x, y, str(i), color=color, fontweight='bold')
        
        # bbox 내 평균 LDOS 계산
        mask = ((grid_LDOS.X >= x) & (grid_LDOS.X <= w) & 
                (grid_LDOS.Y >= y) & (grid_LDOS.Y <= h))
        avg_ldos = grid_LDOS.LDOS.where(mask)
        
        # 평균 LDOS 선 그래프
        sns.lineplot(data=avg_ldos.to_dataframe(), x='bias_mV', y='LDOS', ax=ax2, 
                     label=f'bbox {i}', color=color)
    
    # 그래프 설정
    ax1.set_title(f'LDOS at bias_mV = {slicing_bias_mV}')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    
    ax2.set_title('Average LDOS in bboxes')
    ax2.set_xlabel('bias_mV')
    ax2.set_ylabel('Average LDOS')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# 함수 사용 예시
# plot_ldos_with_bboxes(grid_LDOS, bbox_list, slicing_bias_mV=0)


# -

plot_ldos_with_bboxes(grid_LDOS, bbox_list, slicing_bias_mV= 0 )

# %matplotlib inline
hv.extension('bokeh')

# #### 1.6.5. Lasso area selection 
# * it works if I using grid_3D 
#     * with multiple data channels 
#     * but not with grid_LDOS (only 1 data channel) 
# * later.... lasso pts combine_by_coords need to repaired.. 
#     * but I can extract the target area anyway.. 
#
#

# +
hv.extension('bokeh')


grid_channel_hv = hv.Dataset(grid_LDOS.LDOS)

# bias_mV slicing
dmap_plane  = ["X","Y"]
dmap = grid_channel_hv.to(hv.Image,
                          kdims = dmap_plane,
                          dynamic = True )
dmap.opts(colorbar = True,
          cmap = 'bwr',
          frame_width = 400,
          aspect = 'equal').relabel('XY plane slicing: ')



grid_channel_hv_image = hv.Dataset(grid_LDOS.LDOS.isel(bias_mV = 0))

grid_channel_hv_points = hv.Points(grid_channel_hv_image).opts(frame_width = 400,  
                                   aspect = 'equal', alpha = 0.1,                                   
                                   tools=['box_select', 'lasso_select']
                                  )

slct_pts = hv.streams.Selection1D(source=grid_channel_hv_points)

dmap*grid_channel_hv_image*grid_channel_hv_points

## 
# currently I_fb  crop is working not LDOS_fb 
# need to change later 


# +
#slct_pts
pts = grid_channel_hv_points.iloc[slct_pts.index].dframe().set_index(['X', 'Y'])

pts_xr = xr.Dataset.from_dataframe(pts)
pts_xr

# +
# lasso selection avg  실패 
# -

#grid_LDOS
lss_v1_XY = lss_v1.to_dataframe().reset_index()
lss_v1_XY
#lss_v1_XY[lss_v1_XY.LDOS ==True]

#
grid_filter = np.ones_like(grid_LDOS.LDOS.values)
for 
grid_LDOS['filter'] = grid_LDOS.LDOS.copy()
grid_LDOS['filter'].values


lss_v1 = pts_xr

lss_v2 = pts_xr

lss_v3 = pts_xr

lss_v4 = pts_xr

lss_v5 = pts_xr

~lss_v1.to_dataframe().isna()

grid_LDOS['lss_V1']=

true_idx = lss_v1[lss_v1].stack().reset_index().loc[:, ['level_0', 'level_1']].values
true_idx



# +
lss_v1_XY = ~lss_v1.to_dataframe().isna()
lss_v1_XY= lss_v1_XY.reset_index()

B = grid_LDOS.to_dataframe().unstack('bias_mV')#[~lss_v1.to_dataframe().isna():]

#A.reset_index()
B.loc[list(zip(lss_v1_XY['X'], lss_v1_XY['Y']))]

# +
A.columns = pd.MultiIndex.from_product([A.columns, ['']])

print (A)
# -

A.join(B, how ='inner')
#pd.merge(A,B, how = 'inner')

grid_LDOS.to_dataframe()#groupby(['Y','X']).mean()


lasso_list = [lss_v1,lss_v2,lss_v3,lss_v4,lss_v5]

lasso = lss_v1

# +
lasso_coords

np.isnan(lss_v1.LDOS)
# -

sns.lineplot(data=lasso.to_dataframe(), x='bias_mV', y='LDOS',label='1')

# +
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Polygon
import numpy as np

def plot_ldos_with_lassos(grid_LDOS, lasso_list, slicing_bias_mV):
    # 그래프 설정
    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    # LDOS 이미지 그리기
    ldos_slice = grid_LDOS.LDOS.sel(bias_mV=slicing_bias_mV, method = 'nearest')
    im = ax1.imshow(ldos_slice, extent=[grid_LDOS.X.min(), grid_LDOS.X.max(), 
                                        grid_LDOS.Y.min(), grid_LDOS.Y.max()],
                    origin='lower', aspect='equal')
    plt.colorbar(im, ax=ax1, label='LDOS')
    
    # 색상 설정
    colors = plt.cm.rainbow(np.linspace(0, 1, len(lasso_list)))
    
    # lasso 그리기 및 평균 LDOS 계산
    for i, (lasso, color) in enumerate(zip(lasso_list, colors), 1):
        lasso_x = lasso.X.values
        lasso_y = lasso.Y.values
        
        # 데이터가 있는 좌표 인덱스 구하기
        valid_idx = ~np.isnan(lasso.LDOS.values)
        valid_x, valid_y = np.where(valid_idx)
        
        # 데이터가 있는 좌표 값 추출
        lasso_coords = np.column_stack((lasso.X.values[valid_x], lasso.Y.values[valid_y]))
        
        polygon = Polygon(lasso_coords, edgecolor=color, facecolor='none', linewidth=2)
        ax1.add_patch(polygon)
        ax1.text(lasso_x[0], lasso_y[0], str(i), color=color, fontweight='bold')
        """
        # lasso 내 평균 LDOS 계산
        x_coords = grid_LDOS.LDOS.coords['X'].values
        y_coords = grid_LDOS.LDOS.coords['Y'].values
        mask = np.isin(x_coords, lasso_x) & np.isin(y_coords, lasso_y)
        avg_ldos = grid_LDOS.LDOS.where(mask)
        """
        # 평균 LDOS 선 그래프
        sns.lineplot(data=lasso.to_dataframe(), x='bias_mV', y='LDOS', ax=ax2, 
                     label=f'lasso {i}', color=color)
    
    # 그래프 설정
    ax1.set_title(f'LDOS at bias_mV = {slicing_bias_mV}')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    
    ax2.set_title('Average LDOS in lassos')
    ax2.set_xlabel('bias_mV')
    ax2.set_ylabel('Average LDOS')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# 함수 사용 예시
# lasso_list = [lasso_v1, lasso_v2, lasso_v3, lasso_v4, lasso_v5]
# plot_ldos_with_lassos(grid_LDOS, lasso_list, slicing_bias_mV=0)


# -

plot_ldos_with_lassos(grid_LDOS, lss_list, slicing_bias_mV=0)

pts_xr.LDOS.plot()
plt.show()

# +
grid_3D_slct_pts = xr.combine_by_coords ([grid_LDOS, pts_xr], compat = 'override', join = 'inner')
#y_pts = points.iloc[slct_pts.index].dframe().Y
#grid_3D.sel(X = x_pts,Y = y_pts)
#grid_3D.I_fb.isel(bias_mV = 0).plot()

fig, axs = plt.subplots(ncols = 2, nrows = 1, figsize = (10,3))

grid_3D_slct_pts.LDOS.T.plot(ax = axs[0], robust = True) 
axs[0].set_aspect= 0.5

sns.lineplot(x = "bias_mV",            
             y = "LDOS", 
             data = grid_3D_slct_pts.to_dataframe(),
             ax = axs[1])
plt.show()
#grid_LDOS_slct_pts
#
#sns.relplot(x="bias_mV",
#            y="LIX_fb", 
#            kind="line",
#            data=grid_3D_slct_pts.to_dataframe())
# check. sn.relplot is  figure-level function
# -

grid_3D_slct_pts

Impurity_surrounding = grid_3D_slct_pts

#Impurity_center
Impurity_surrounding

grid_3D_slct_pts.I_fb.plot()
## I_fb area is selected region, no bias_mV info. 
plt.show()

# +

LDOS_fb_area1_df =  Impurity_center.LIX_fb.to_dataframe()
LDOS_fb_area2_df =  Impurity_surrounding.LIX_fb.to_dataframe() 


'''
LDOS_fb_area1_df =  bbox_1.LDOS_fb.to_dataframe()
LDOS_fb_area2_df =  bbox_2.LDOS_fb.to_dataframe() 
'''


# xr to dataframe
LDOS_fb_area1_df.columns = ['Area1']
LDOS_fb_area2_df.columns = ['Area2']# change df names 

LDOS_fb_area_df = pd.concat( [LDOS_fb_area1_df,LDOS_fb_area2_df], axis= 1)
LDOS_fb_area_df# = LDOS_fb_area_df.swaplevel(0,2)
#LDOS_fb_area_df.swaplevel(0,2) # index level swap. w.r.t. 'bias_mV'
#LDOS_fb_area_df = LDOS_fb_area_df.swaplevel(0,2).unstack().unstack() # unstack X& Y


##sns.lineplot(x= 'bias_mV', y ='LDOS1', data= LDOS_fb_area_df, label = 'area1')
#sns.lineplot(x= 'bias_mV', y ='LDOS2', data= LDOS_fb_area_df, label = 'area2')
#plt.show()
# use the below sns plot instead 
# -

LDOS_fb_area_df = LDOS_fb_area_df.reset_index()
LDOS_fb_area_df_melt = LDOS_fb_area_df.melt(id_vars = ['Y','X','bias_mV'], value_vars = ['Area1','Area2'])
LDOS_fb_area_df_melt.columns = ['Y','X','bias_mV', 'Area','LDOS']
LDOS_fb_area_df_melt

sns.lineplot(x= 'bias_mV', y = 'LDOS', data = LDOS_fb_area_df_melt, hue ='Area')
plt.show()

# #### 1.7. area selection based on special selection 
#     * tresholds_xxxx_xr = LDOS_fb channel th + use threshold_fiip   
#         * th_otsu_roi_label_2D_xr
#         * th_multiotsu_roi_label_2D_xr
#         * th_mean_roi_label_2D_xr
#         
#         

# +

#grid_LDOS.rolling(X=3, Y=3,min_periods=2,center= True).mean().isel(bias_mV=0).LDOS_fb.plot()
#plt.show()
grid_LDOS_th= th_mean_roi_label_2D_xr(grid_LDOS.rolling(X=4, Y=2,min_periods=2,center= True).mean(),
                                      bias_mV_th = 0.0,threshold_flip=False)

# +
#grid_LDOS_th= th_otsu_roi_label_2D_xr(equalize_hist_xr(grid_LDOS), bias_mV_th = 0,  threshold_flip=False)
# use Otsu 

#grid_LDOS_th= th_multiotsu_roi_label_2D_xr(grid_LDOS, window_length=51, polyorder=3), bias_mV_th = 0.5, multiclasses = 5)
# in case of multiotsu

grid_LDOS_th= th_mean_roi_label_2D_xr(grid_LDOS.rolling(X=4, Y=2,min_periods=2,center= True).mean(),
                                      bias_mV_th = 0,threshold_flip=False)
# in case of mean_roi

# results. 
    #grid_LDOS_th

isns.imshow (grid_LDOS_th.LDOS_fb_th_label, aspect =1)
isns.imshow(grid_LDOS_th.LDOS_fb_th)
plt.show()




# +
#plot with labes 
import matplotlib.patches as mpatches
from skimage.segmentation import clear_border
from skimage.morphology import closing, square
from skimage.measure import label, regionprops

fig,ax =  plt.subplots(figsize =  (5,5))

isns.imshow (grid_LDOS_th.LDOS_fb_th_label, ax =ax, aspect = 1)
label_map = skimage.morphology.closing (grid_LDOS_th.LDOS_fb_th_label, skimage.morphology.square(2))
# use closing for ROI selection 
clear_border = False 

if clear_border == True :
    label_map_clear_border = skimage.segmentation.clear_border( label_map)
else :     label_map_clear_border =label_map
# clear border in the label map 

for region in skimage.measure.regionprops(label_map_clear_border):
    # take regions with large enough areas
    if region.area >= 5:
        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
        
        (center_y,center_x) = region.centroid
        ax.annotate (region.label, xy= (center_x,center_y), c = 'r')
        # Anonotate region info by using region properties  
        
# use region properties to extract ROI info
plt.show()

# +
fig, ax = plt.subplots(figsize = (4,3))
#slctd_lables = [3,16,15,9,22]
slctd_lables = [1,2]

#for labels in range (int(grid_LDOS_th.LDOS_fb_th_label.max())):
for labels in slctd_lables:    
    sns.lineplot (x = 'bias_mV', y= 'LDOS_fb', 
              data = grid_LDOS_th.LDOS_fb.where(grid_LDOS_th.LDOS_fb_th_label ==labels ).mean(['X','Y']).to_dataframe(),
              ax =ax, label = str(labels))
plt.show()
# -

LDOS_fb_0_df = grid_LDOS_th.LDOS_fb.where( grid_LDOS_th.LDOS_fb_th_label ==0 ).mean(["X","Y"]).to_dataframe()
LDOS_fb_1_df = grid_LDOS_th.LDOS_fb.where( grid_LDOS_th.LDOS_fb_th_label !=0 ).mean(["X","Y"]).to_dataframe()
LDOS_fb_0_1_df = pd.concat( [LDOS_fb_0_df,LDOS_fb_1_df], axis= 1)
LDOS_fb_0_1_df.columns = ['(Area0)','(Area1)']
#LDOS_fb_0_1_df

# +
LDOS_fb_0_df = grid_LDOS_th.LDOS_fb.where( grid_LDOS_th.LDOS_fb_th_label ==0 ).to_dataframe()
LDOS_fb_0_df= LDOS_fb_0_df.rename( columns ={'LDOS_fb':'LDOS_Area0'})
LDOS_fb_1_df = grid_LDOS_th.LDOS_fb.where( grid_LDOS_th.LDOS_fb_th_label !=0 ).to_dataframe()
LDOS_fb_1_df= LDOS_fb_1_df.rename( columns ={'LDOS_fb':'LDOS_Area1'})
# rename columns 

LDOS_fb_0_1_df = pd.concat( [LDOS_fb_0_df,LDOS_fb_1_df], axis= 1)
#LDOS_fb_0_1_df = pd.concat( [LDOS_fb_0_df,LDOS_fb_1_df], axis= 1, join='outer')

LDOS_fb_0_1_df = LDOS_fb_0_1_df.reset_index()
#LDOS_fb_0_1_df
# -

####################33
# melt dataframe for avg plot
#####################
LDOS_fb_0_1_df_area_df_melt = LDOS_fb_0_1_df.melt(id_vars = ['Y','X', 'bias_mV'], value_vars = ['LDOS_Area0','LDOS_Area1'] )
LDOS_fb_0_1_df_area_df_melt.columns = ['Y','X','bias_mV', 'Area','LDOS']
LDOS_fb_0_1_df_area_df_melt

# +
fig,ax = plt.subplots(ncols = 3, figsize=(9,3))
isns.imshow (grid_LDOS_th.LDOS_fb_th, ax = ax[0]) 
ax[0].set_title('Thresholds')
isns.imshow (grid_LDOS_th.LDOS_fb_th.isnull(), ax = ax[1]) 
ax[1].set_title('Area Selection 0 or 1')

sns.lineplot(LDOS_fb_0_1_df_area_df_melt,x = 'bias_mV', y = 'LDOS', ax = ax[2], hue = 'Area')
#sns.lineplot( x  =LDOS_fb__1_df, data = LDOS_fb__1_df, ax = ax[2])
#sns.lineplot(grid_LDOS_th.LDOS_fb.where( grid_LDOS_th.LDOS_fb_th_label !=0 ).mean(["X","Y"]).to_dataframe(), ax = ax[2], label ='1')
ax[2].set_title('LDOS at Area 0 or 1')
plt.tight_layout()
plt.show()
# -





# # grid_LDOS_pks analysis 
#

# ### DBSCAN 

grid_LDOS_pks_df.to_csv('Grid Spectroscopy(X0.01)0T_40mK_001_LDOS_pks.cvs')

df =  grid_LDOS_pks_df

# +
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 데이터 준비
data = df[['X', 'Y', 'bias_mV']].values

# DBSCAN 파라미터 설정
eps = 0.01 # 데이터 스케일에 맞게 조정
min_samples = 5  # 노이즈와 클러스터를 구분하는 역할

# DBSCAN 실행
dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
labels = dbscan.labels_

# +
# 데이터 준비
df['label'] = labels

# 인터렉티브 3D 산점도 플롯
fig = px.scatter_3d(df, x='X', y='Y', z='bias_mV', color='label',
                    color_continuous_scale='Viridis', opacity=0.05,
                    hover_name='label', hover_data=['X', 'Y', 'bias_mV'])

# 축 레이블 설정
fig.update_layout(scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='bias_mV'),
                  width=500, height=500)

# 노이즈 데이터 포인트 별도 표시
noise_points = df[df['label'] == -1]
fig.add_trace(go.Scatter3d(x=noise_points['X'], y=noise_points['Y'], z=noise_points['bias_mV'],
                           mode='markers', marker=dict(color='black', size=1, symbol='x'),
                           name='Noise'))

fig.show()
# -

grid_LDOS_pks_df


grid_LDOS_pks_df

# +

# Creating a 3D scatter plot using Plotly Express
# Define the range for the color scale
color_range = [df['peak_heights'].quantile(0.05), df['peak_heights'].quantile(0.95)]


fig = px.scatter_3d(grid_LDOS_pks_df, x='Y', y='X', z='peak_bias', color='peak_heights', 
                    title='3D Scatter Plot of Y, X, peak_bias with peak_heights as color',
                    labels={'Y': 'Y', 'X': 'X', 'peak_bias': 'Peak Bias', 'peak_heights': 'Peak peak_heights'},
                    color_continuous_scale=px.colors.sequential.Viridis, opacity= 0.05,range_color=color_range)

# Show the plot
fig.show()

# +
fig, axs = plt.subplots(2, 2, figsize=(15, 12))
scatter = axs[0, 0].scatter(df['X'], df['Y'], c=df['peak_heights'], cmap='viridis',alpha = 0.3)
fig.colorbar(scatter, ax=axs[0, 0])
axs[0, 0].set_title('X-Y plane with bias_mV')

scatter = axs[0, 1].scatter(df['X'], df['Y'], c=df['peak_prominences'], cmap='coolwarm',alpha = 0.3)
fig.colorbar(scatter, ax=axs[0, 1])
axs[0, 1].set_title('X-Y plane with peak_prominences')

scatter = axs[1, 0].scatter(df['X'], df['Y'], c=df['peak_widths'], cmap='coolwarm',alpha = 0.3)
fig.colorbar(scatter, ax=axs[1, 0])
axs[1, 0].set_title('X-Y plane with peak_widths')

scatter = axs[1, 1].scatter(df['X'], df['Y'], c=df['peak_width_heights'], cmap='coolwarm',alpha = 0.3)
fig.colorbar(scatter, ax=axs[1, 1])
axs[1, 1].set_title('X-Y plane with peak_width_heights')

for ax in axs.flat:
    ax.set(xlabel='X', ylabel='Y')
    
plt.tight_layout()
plt.show()


# +

# Create 3D scatter plot
fig = go.Figure()

# Add traces for each cluster
for cluster in range(num_clusters):
    cluster_data = df[df['cluster'] == cluster]
    fig.add_trace(go.Scatter3d(
        x=cluster_data['X'],
        y=cluster_data['Y'],
        z=cluster_data['peak_bias'],
        mode='markers',
        marker=dict(size=5, color=cluster, opacity=0.05),
        name=f'Cluster {cluster}'
    ))

# Layout settings
fig.update_layout(
    scene=dict(
        xaxis=dict(title='X-axis'),
        yaxis=dict(title='Y-axis'),
        zaxis=dict(title='peak_bias'),
    ),
    title='3D Scatter Plot with Clustering',
    margin=dict(l=0, r=0, b=0, t=10)
)

# Show interactive plot
fig.show()

# +
## PCA analysis and KNN clusterin g
# -


# #### GapMap for SC
#

grid_LDOS_pks_df

metallic_mask

grid_LDOS_pks_df

# +
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

def create_gapmap_by_max_LDOS(grid_LDOS_pks):
    # bias_mV의 양수와 음수 부분 나누기
    bias_negative = grid_LDOS_pks.bias_mV < 0
    bias_positive = grid_LDOS_pks.bias_mV > 0

    # LDOS_smoothed_pks 값 중 양수와 음수 부분에서 가장 큰 값의 인덱스 찾기
    ldos_neg = grid_LDOS_pks.LDOS_smoothed_pks.isel(bias_mV=bias_negative)
    ldos_pos = grid_LDOS_pks.LDOS_smoothed_pks.isel(bias_mV=bias_positive)

    # 각 위치별로 최대값의 인덱스를 찾고, 해당 bias_mV 값 추출
    cohpk_0Neg = ldos_neg.idxmax(dim='bias_mV')
    cohpk_0Pos = ldos_pos.idxmax(dim='bias_mV')

    # Gapmap 채널 생성 (두 값의 차이)
    gapmap = cohpk_0Pos - cohpk_0Neg

    # 새로운 데이터셋 생성
    new_ds = xr.Dataset({
        'cohpk_0Neg': cohpk_0Neg,
        'cohpk_0Pos': cohpk_0Pos,
        'gapmap': gapmap
    })

    # 새로운 채널들을 원래 데이터셋에 추가
    combined_ds = grid_LDOS_pks.merge(new_ds)
    #mask 추가 
    
    # Plotting with seaborn
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    isns.imshow(combined_ds.cohpk_0Neg, ax=axes[0])
    axes[0].set_title('cohpk_0Neg')

    isns.imshow(combined_ds.cohpk_0Pos, ax=axes[1])
    axes[1].set_title('cohpk_0Pos')

    isns.imshow(combined_ds.gapmap, ax=axes[2])
    axes[2].set_title('gapmap')

    plt.tight_layout()
    plt.show()

    return combined_ds

# 사용 예제
# grid_LDOS_pks = xr.open_dataset('path_to_your_dataset.nc')
gapmap_xr = create_gapmap(grid_LDOS_pks)
print(gapmap_xr)
# -


gapmap_xr


gapmap_xr.cohpk_0Pos.plot()

grid_LDOS_zm= grid_LDOS.where(grid_LDOS.bias_mV<2.4,drop=True).where(grid_LDOS.bias_mV>-2.4,drop=True)


def find_2deriv_sg_peaks(grid_LDOS, 
                         ch_name='LDOS',
                         window_length=15,
                         window_length1=15,
                         window_length2=15,
                         polyorder=5,
                         polyorder1=4,
                         polyorder2 = 3,
                         X_point=10,
                         Y_point=10,
                         fnd_pks_distance=None, 
                         fnd_pks_width=None, 
                         fnd_pks_threshold=None,
                         fnd_pks_prominence=None):
    """
    Find peaks in the second derivative of the given grid_LDOS.

    Parameters:
    -----------
    grid_LDOS : xarray Dataset
        Input grid_LDOS data.
    window_length : int, optional
        Length of the filter window, default is 21.
    polyorder : int, optional
        Polynomial order, default is 3.
    X_point : int, optional
        X-coordinate point for plotting, default is 10.
    Y_point : int, optional
        Y-coordinate point for plotting, default is 10.
    fnd_pks_distance : int, optional
        Minimum distance between peaks, default is 10.
    fnd_pks_width : int, optional
        Minimum width of peaks, default is 10.
    fnd_pks_threshold : float, optional
        Minimum peak height threshold, default is 1E-11.
    fnd_pks_prominence : float, optional
        Minimum peak prominence, default is 2E-11.

    Returns:
    --------
    grid_LDOS_pks : xarray Dataset
        Dataset containing information about peaks in the second derivative
    ################################
    # as for a final return,  
    # LDOS_fb :(Y, X, bias_mV) original 
    # LDOS_fb_sg :(Y, X, bias_mV) original + sg 
    # LDOS_2deriv_sg : (Y, X, bias_mV) 2nd derivative of original + sg 
    # LDOS_fb_peaks_pad: (X, Y, peaks)  list of peak points 
    # LDOS_fb_peaks_mV: (Y, X, bias_mV) boolean of peak positions (True/False) --> multiply bias_mV to obtain peak positions
    # LDOS_fb_pk_mV: (Y, X, bias_mV) (origianl) LDOS value at the peak position 
    ################################

    """
    '''window_length
    window_length1 = window_length +4
    window_length2 = window_length +8
    polyorder
    polyorder1 = polyorder +0 
    polyorder2 = polyorder +0'''
    # Smoothing and differentiation to find peaks
    grid_LDOS_sg = savgolFilter_xr(grid_LDOS, window_length=window_length, polyorder=polyorder)
    grid_LDOS_1deriv = grid_LDOS_sg.differentiate('bias_mV')
    grid_LDOS_1deriv_sg = savgolFilter_xr(grid_LDOS_1deriv, window_length=window_length1, polyorder=polyorder1)
    grid_LDOS_2deriv = grid_LDOS_1deriv_sg.differentiate('bias_mV')
    grid_LDOS_2deriv_sg = savgolFilter_xr(grid_LDOS_2deriv, window_length=window_length2, polyorder=polyorder2)
    grid_LDOS_2deriv_sg_dps = find_peaks_xr(-1 * grid_LDOS_2deriv_sg, distance=fnd_pks_distance,
                                             width=fnd_pks_width, threshold=fnd_pks_threshold,
                                             prominence=fnd_pks_prominence)
    grid_LDOS_2deriv_sg_dps_pad = peak_pad(grid_LDOS_2deriv_sg_dps)
    print (grid_LDOS_2deriv_sg_dps_pad)
    grid_LDOS_2deriv_sg_dps_pad_mV = peak_mV_3Dxr(grid_LDOS_2deriv_sg_dps_pad, ch=ch_name)

    # Calculate peak values
    grid_LDOS_pks = grid_LDOS_2deriv_sg_dps_pad_mV.copy()
    grid_LDOS_pks['LDOS_pk_mV'] = (grid_LDOS_2deriv_sg_dps_pad_mV.LDOS_peaks_mV * grid_LDOS[ch_name]).astype(float)
    grid_LDOS_pks[ch_name].values = grid_LDOS[ch_name].values
    grid_LDOS_pks['LDOS_sg'] = grid_LDOS_sg[ch_name].copy()
    grid_LDOS_pks['LDOS_2deriv'] = grid_LDOS_2deriv[ch_name].copy()
    grid_LDOS_pks['LDOS_2deriv_sg'] = grid_LDOS_2deriv_sg[ch_name].copy()

    # Plotting
    fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
    grid_LDOS_pks[ch_name].isel(X=X_point, Y=Y_point).plot(ax=axs[0], color='k')
    grid_LDOS_pks.LDOS_sg.isel(X=X_point, Y=Y_point).plot(ax=axs[0], color='orange')
    grid_LDOS_pks[ch_name].where(grid_LDOS_pks.LDOS_pk_mV).isel(X=X_point, Y=Y_point).plot.scatter(ax=axs[0],
                                                                                                       x='bias_mV',
                                                                                                       y='LDOS',
                                                                                                       color='b')
    axs[0].set_ylabel('LDOS')

    grid_LDOS_pks.LDOS_2deriv.isel(X=X_point, Y=Y_point).plot(ax=axs[1], color='k')
    grid_LDOS_pks.LDOS_2deriv_sg.isel(X=X_point, Y=Y_point).plot(ax=axs[1], color='b')
    grid_LDOS_pks.LDOS_2deriv_sg.where(grid_LDOS_pks.LDOS_pk_mV).isel(X=X_point, Y=Y_point).plot.scatter(
        ax=axs[1], x='bias_mV', y='LDOS_2deriv_sg')
    axs[1].set_ylabel('LDOS (2nd derivative)')

    fig.tight_layout()
    plt.rcParams.update({'font.size': 10})
    plt.show()

    return grid_LDOS_pks


# ### PCA analysis for SC gap  and extract gapmap for SC by using SC  

def PCA_for_SC_gap(grid_LDOS_pks_df,
                   n_clusters=7):
    """
    Perform Principal Component Analysis (PCA) followed by K-Means clustering
    to identify superconducting (SC) gaps in a dataset.

    Parameters:
    -----------
    grid_LDOS_pks_df : pandas DataFrame
        DataFrame containing the data points to be analyzed.

    n_clusters : int, optional
        Number of clusters for K-Means clustering. Default is 6.

    Returns:
    --------
    grid_LDOS_pks_df : pandas DataFrame
        DataFrame with an additional column 'PCA_labels' containing cluster labels.

    Example:
    --------
    >>> import pandas as pd
    >>> from PCA_for_SC_gap import PCA_for_SC_gap
    
    # Assuming grid_LDOS_pks_df is a pandas DataFrame containing the data
    # Call the function with the DataFrame
    >>> result_df = PCA_for_SC_gap(grid_LDOS_pks_df)
    
    # 'result_df' now contains an additional column 'PCA_labels' with cluster labels
    # grid_LDOS_pks_df_pca = PCA_for_SC_gap( grid_LDOS_pks_df)
    # Further operations can be performed based on the clustering results.
    """
    import numpy as np
    import plotly.graph_objects as go
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    # Extract data from DataFrame
    X = grid_LDOS_pks_df.to_numpy()

    # Perform PCA to reduce dimensionality to 2D
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    # Initialize K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    # Train the model
    kmeans.fit(X_pca)

    # Get cluster labels
    labels = kmeans.labels_

    # Plot 2D Scatter Plot (PCA)
    fig = go.Figure()

    # Add data points for each cluster with different colors
    for cluster_num in range(n_clusters):
        cluster_points = X_pca[labels == cluster_num]
        fig.add_trace(go.Scatter(
            x=cluster_points[:, 0],
            y=cluster_points[:, 1],
            mode='markers',
            marker=dict(size=8, opacity=0.8),
            name=f'Cluster {cluster_num}'
        ))

    # Add cluster centers
    centers_pca = kmeans.cluster_centers_
    fig.add_trace(go.Scatter(
        x=centers_pca[:, 0],
        y=centers_pca[:, 1],
        mode='markers',
        marker=dict(size=5, color='red', opacity=0.4),
        name='Cluster Centers'
    ))

    # Set layout
    fig.update_layout(
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2',
        title='K-Means Clustering in 2D (PCA)',
        showlegend=True
    )

    # Show interactive plot
    fig.show()

    # Perform PCA inverse transform
    X_pca_inverse = pca.inverse_transform(X_pca)

    # Assign cluster labels to the DataFrame
    grid_LDOS_pks_df['PCA_labels'] = labels

    # Return the DataFrame with added 'PCA_labels' column
    return grid_LDOS_pks_df


grid_LDOS_pks_df_pca = PCA_for_SC_gap( grid_LDOS_pks_df,n_clusters = 7)

extract_gap_map(grid_LDOS_pks_df_pca,pk_labl_Neg =6,pk_labl_Pos =2)

# +
import pandas as pd

def extract_gap_map(grid_LDOS_pks_df_pca,pk_labl_Neg =1,pk_labl_Pos =2):
    """
    Extracts the superconducting (SC) gap map from preprocessed data.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Calculate bias step
    bias_mV_step = np.unique(grid_LDOS.bias_mV[1:].values - grid_LDOS.bias_mV[:-1].values)[0]
    
    # Filter data for each coherent peak and remove duplicates based on X, Y, LDOS_fb_peaks_pad
    grid_LDOS_pks_df_coherentPK_0 = grid_LDOS_pks_df_pca[grid_LDOS_pks_df_pca.PCA_labels == pk_labl_Neg].drop_duplicates(subset=['X', 'Y'], keep='first').drop(columns=['PCA_labels'])
    grid_LDOS_pks_df_coherentPK_1 = grid_LDOS_pks_df_pca[grid_LDOS_pks_df_pca.PCA_labels == pk_labl_Pos].drop_duplicates(subset=['X', 'Y'], keep='first').drop(columns=['PCA_labels'])
    
    # Convert data to 2D grid
    grid_LDOS_pks_df_coherentPK_0_2D = grid_LDOS_pks_df_coherentPK_0.pivot(index='X', columns='Y', values='LDOS_peaks_pad') * bias_mV_step + grid_LDOS.bias_mV.values[0]
    grid_LDOS_pks_df_coherentPK_1_2D = grid_LDOS_pks_df_coherentPK_1.pivot(index='X', columns='Y', values='LDOS_peaks_pad') * bias_mV_step + grid_LDOS.bias_mV.values[0]
    
    # Calculate SC gap map
    grid_LDOS_pks_df_coherentPK_Gapmap_2D = (grid_LDOS_pks_df_coherentPK_1_2D - grid_LDOS_pks_df_coherentPK_0_2D).abs()
    
    # Concatenate data into a single DataFrame
    grid_LDOS_pks_df_Gapmap = pd.concat([grid_LDOS_pks_df_coherentPK_0_2D.unstack().rename('SC_coherence_del0'),
                                          grid_LDOS_pks_df_coherentPK_1_2D.unstack().rename('SC_coherence_del1'),
                                          grid_LDOS_pks_df_coherentPK_Gapmap_2D.unstack().rename('SC_coherence_2delta')],
                                         axis=1)
    
    # Create subplots for each map
    fig, axs = plt.subplots(ncols=3, figsize=(8, 3))
    isns.imshow(grid_LDOS_pks_df_coherentPK_0_2D, ax=axs[0])
    axs[0].set_title('+$\Delta$')
    isns.imshow(grid_LDOS_pks_df_coherentPK_1_2D, ax=axs[1])
    axs[1].set_title('-$\Delta$')
    isns.imshow(grid_LDOS_pks_df_coherentPK_Gapmap_2D, ax=axs[2])
    axs[2].set_title('2$\Delta$')
    fig.tight_layout()
    
    return grid_LDOS_pks_df_Gapmap, fig


# +
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
data = grid_LDOS_pks_df
# PCA 수행 (3개의 주성분을 추출)
pca = PCA(n_components=3)
principal_components = pca.fit_transform(data)

# PCA 결과를 데이터 프레임으로 변환
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])
import plotly.graph_objs as go

# 3D 산점도 생성
scatter = go.Scatter3d(
    x=pca_df['PC1'],
    y=pca_df['PC2'],
    z=pca_df['PC3'],
    mode='markers',
    marker=dict(
        color='red',
        size=5,
        symbol='circle'
    )
)

# 레이아웃 설정
layout = go.Layout(
    scene=dict(
        xaxis=dict(title='Principal Component 1'),
        yaxis=dict(title='Principal Component 2'),
        zaxis=dict(title='Principal Component 3')
    ),
    title='3D PCA Visualization'
)

# 그래프 객체 생성
fig = go.Figure(data=[scatter], layout=layout)

# 그래프 출력
fig.show()
# -


pca_df

extract_gap_map(grid_LDOS_pks_df_pca,pk_labl_Neg =1,pk_labl_Pos =2)


def extract_gap_map(grid_LDOS_pks_df_pca,pk_labl_Neg =1,pk_labl_Pos =2):
    """
    Extracts the superconducting (SC) gap map from preprocessed data.

    Parameters:
    -----------
    grid_LDOS_pks_df_pca : pandas DataFrame
        DataFrame containing preprocessed data with PCA cluster labels.

    Returns:
    --------
    grid_LDOS_pks_df_Gapmap : pandas DataFrame
        DataFrame containing SC gap map.

    fig : matplotlib Figure
        Matplotlib Figure object containing three subplots: 
        +$\Delta$ map, -$\Delta$ map, and 2$\Delta$ map.

    Example:
    --------
    >>> import matplotlib.pyplot as plt
    >>> import seaborn as sns
    >>> from extract_gap_map import extract_gap_map
    
    # Assuming grid_LDOS_pks_df_pca is a pandas DataFrame containing preprocessed data
    # Call the function to extract the SC gap map
    >>> grid_LDOS_pks_df_Gapmap, fig = extract_gap_map(grid_LDOS_pks_df_pca)
    # or 
    >>> grid_LDOS_pks_df_Gapmap, _ = extract_gap_map(grid_LDOS_pks_df_pca)
    
    # Show the extracted gap map and plots
    >>> plt.show(fig)
    
    # 'grid_LDOS_pks_df_Gapmap' contains the SC gap map for further analysis.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Calculate bias step
    bias_mV_step = np.unique(grid_LDOS.bias_mV[1:].values - grid_LDOS.bias_mV[:-1].values)[0]

    # Filter data for each coherent peak
    grid_LDOS_pks_df_coherentPK_0 = grid_LDOS_pks_df_pca[grid_LDOS_pks_df_pca.PCA_labels == pk_labl_Neg].drop(columns=['PCA_labels'])
    grid_LDOS_pks_df_coherentPK_1 = grid_LDOS_pks_df_pca[grid_LDOS_pks_df_pca.PCA_labels == pk_labl_Pos].drop(columns=['PCA_labels'])

    # Convert data to 2D grid
    grid_LDOS_pks_df_coherentPK_0_2D = grid_LDOS_pks_df_coherentPK_0.pivot(index='X',
                                                                             columns='Y',
                                                                             values='LDOS_peaks_pad') * bias_mV_step + grid_LDOS.bias_mV.values[0]
    grid_LDOS_pks_df_coherentPK_1_2D = grid_LDOS_pks_df_coherentPK_1.pivot(index='X',
                                                                             columns='Y',
                                                                             values='LDOS_peaks_pad') * bias_mV_step + grid_LDOS.bias_mV.values[0]

    # Calculate SC gap map
    grid_LDOS_pks_df_coherentPK_Gapmap_2D = (grid_LDOS_pks_df_coherentPK_1_2D - grid_LDOS_pks_df_coherentPK_0_2D).abs()

    # Concatenate data into a single DataFrame
    grid_LDOS_pks_df_Gapmap = pd.concat([grid_LDOS_pks_df_coherentPK_0_2D.unstack().rename('SC_coherence_del0'),
                                          grid_LDOS_pks_df_coherentPK_1_2D.unstack().rename('SC_coherence_del1'),
                                          grid_LDOS_pks_df_coherentPK_Gapmap_2D.unstack().rename('SC_coherence_2delta')],
                                         axis=1)

    # Create subplots for each map
    fig, axs = plt.subplots(ncols=3, figsize=(8, 3))
    isns.imshow(grid_LDOS_pks_df_coherentPK_Neg_2D, ax=axs[0])
    axs[0].set_title('+$\Delta$')
    isns.imshow(grid_LDOS_pks_df_coherentPK_Pos_2D, ax=axs[1])
    axs[1].set_title('-$\Delta$')
    isns.imshow(grid_LDOS_pks_df_coherentPK_Gapmap_2D, ax=axs[2])
    axs[2].set_title('2$\Delta$')

    fig.tight_layout()

    return grid_LDOS_pks_df_Gapmap, fig


# +
pk_labl_Neg =1
pk_labl_Pos =2

# Calculate bias step
bias_mV_step = np.unique(grid_LDOS.bias_mV[1:].values - grid_LDOS.bias_mV[:-1].values)[0]

# Filter data for each coherent peak
grid_LDOS_pks_df_coherentPK_0 = grid_LDOS_pks_df_pca[grid_LDOS_pks_df_pca.PCA_labels == pk_labl_Neg].drop(columns=['PCA_labels'])
grid_LDOS_pks_df_coherentPK_1 = grid_LDOS_pks_df_pca[grid_LDOS_pks_df_pca.PCA_labels == pk_labl_Pos].drop(columns=['PCA_labels'])
grid_LDOS_pks_df_coherentPK_0
# -

# Filter data for each coherent peak
grid_LDOS_pks_df_coherentPK_0 = grid_LDOS_pks_df_pca[grid_LDOS_pks_df_pca.PCA_labels == pk_labl_Neg]
grid_LDOS_pks_df_coherentPK_1 = grid_LDOS_pks_df_pca[grid_LDOS_pks_df_pca.PCA_labels == pk_labl_Pos]
grid_LDOS_pks_df_coherentPK_0

grid_LDOS_pks_df_coherentPK_1.iloc[:400,:].pivot(index='X',columns='Y',
                                                 values='LDOS_peaks_pad')

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from lmfit.models import LorentzianModel, GaussianModel, VoigtModel, ConstantModel
from functools import reduce

def plot_region_fitting_result_from_dsfit(
        ds_fit,
        model_type=None,
        allowed_models=['Lorentzian', 'Gaussian', 'Voigt'],
        y_idx=None, x_idx=None,
        weight_function_show=False,
        use_zb_mask=False,
        zb_mask_key='ZB_mask',
        show_shade=True,
        return_fig=False):
    """
    Reconstruct and plot the best-fit curve for a selected pixel from a combined dataset,
    including optional CdGM level proximity and guide lines, and collect all curves into a pandas DataFrame.

    Parameters
    ----------
    ds_fit : xarray.Dataset
        Combined dataset with dims (Y, X, bias_mV, peak) containing:
          - Variables: 
              • 'bias_mV' (axis values)  
              • 'LDOS' (data to fit)  
              • 'peak_center', 'peak_amplitude', 'peak_sigma', 'redchi'  
          - Optional Variables: 
              • 'background_value'  
              • 'model_type'  
              • zb_mask_key (e.g. 'ZB_mask')  
          - Attributes:
              • 'weight_sigma'  
              • 'Ef' (Fermi energy)  
              • 'SCgap' (superconducting gap)  
    model_type : {None, str}, optional
        If None, uses per-pixel model from ds_fit['model_type']; 
        else forces the given model (must be in allowed_models).
    allowed_models : list of str, optional
        Supported models when model_type is None.
    y_idx, x_idx : int, optional
        Pixel indices. If None and use_zb_mask=True, picks a random valid pixel.
    weight_function_show : bool, default False
        If True, plot also the convoluted fit and weight function.
    use_zb_mask : bool, default False
        If True, applies zb_mask_key to bias-axis masking.
    zb_mask_key : str, default 'ZB_mask'
        Name of zero-bias mask variable in ds_fit.
    show_shade : bool, default True
        If True, plot CdGM level proximity lines and shading under peaks.
        If False, restrict x-axis to valid-bias region only.
    return_fig : bool, default False
        If True, return (fig, df); else show plot and return df.

    Returns
    -------
    df : pandas.DataFrame
        Indexed by bias_mV, columns include:
          - 'LDOS', 'best_fit', 'peak0'… , 'bkg' (if present),
            'convoluted_fit', 'weight_function' (if requested)
    fig : matplotlib.figure.Figure, optional
        Returned if return_fig=True.
    """
    # 1) Determine pixel indices
    ny, nx = ds_fit.dims['Y'], ds_fit.dims['X']
    if use_zb_mask and y_idx is None and x_idx is None and zb_mask_key in ds_fit:
        raw_mask = ds_fit[zb_mask_key].values
        valid_mask = np.any(~np.isnan(raw_mask), axis=2) if raw_mask.ndim == 3 else raw_mask.astype(bool)
        rc = ds_fit['redchi'].values
        valid_fit = ~np.isnan(rc)
        ys, xs = np.where(valid_mask & valid_fit)
        if len(ys) == 0:
            raise RuntimeError("No valid pixel found with ZB mask and redchi")
        sel = np.random.randint(len(ys))
        y_idx, x_idx = int(ys[sel]), int(xs[sel])
    if y_idx is None:
        y_idx = np.random.randint(ny)
    if x_idx is None:
        x_idx = np.random.randint(nx)

    print(f"Using pixel Y={y_idx}, X={x_idx} for plotting")

    # 2) Extract data
    bias = ds_fit['bias_mV'].values
    ldos = ds_fit['LDOS'].isel(Y=y_idx, X=x_idx).values

    # 3) Model selection
    if model_type is None:
        chosen = ds_fit['model_type'].isel(Y=y_idx, X=x_idx).item().capitalize()
    else:
        chosen = model_type.capitalize()
    if chosen not in allowed_models:
        raise ValueError(f"Model '{chosen}' not supported. Choose from {allowed_models}.")
    if chosen == 'Lorentzian':
        mc, fit_color = LorentzianModel, 'r'
    elif chosen == 'Gaussian':
        mc, fit_color = GaussianModel, 'b'
    else:
        mc, fit_color = VoigtModel, 'g'

    # 4) Bias mask
    mask = np.ones_like(bias, bool)
    if use_zb_mask and zb_mask_key in ds_fit:
        raw = ds_fit[zb_mask_key].isel(Y=y_idx, X=x_idx).values
        if isinstance(raw, np.ndarray) and raw.shape == bias.shape:
            mask = ~np.isnan(raw) if np.issubdtype(raw.dtype, np.floating) else raw.astype(bool)
        elif np.ndim(raw) == 0:
            mask = np.full_like(bias, bool(raw), bool)
    mask = mask.astype(bool)

    # 5) Load fit parameters
    redchi = ds_fit['redchi'].isel(Y=y_idx, X=x_idx).item()
    n_peaks = ds_fit.dims['peak']
    centers = ds_fit['peak_center'].isel(Y=y_idx, X=x_idx).values
    amps    = ds_fit['peak_amplitude'].isel(Y=y_idx, X=x_idx).values
    sigmas  = ds_fit['peak_sigma'].isel(Y=y_idx, X=x_idx).values
    idxs = [i for i in range(n_peaks)
            if not (np.isnan(centers[i]) or np.isnan(amps[i]) or np.isnan(sigmas[i]))]
    print("Valid peak indices:", idxs)

    # 6) Construct composite model
    models = []
    if 'background_value' in ds_fit:
        models.append(ConstantModel(prefix='bkg_'))
        bgv = ds_fit['background_value'].isel(Y=y_idx, X=x_idx).item()
    for i in idxs:
        models.append(mc(prefix=f'peak{i}_'))
    comp = reduce(lambda a, b: a + b, models)
    params = comp.make_params()
    if 'background_value' in ds_fit:
        params['bkg_c'].set(value=bgv)
    for i in idxs:
        params[f'peak{i}_center'].set(value=centers[i])
        params[f'peak{i}_amplitude'].set(value=amps[i])
        params[f'peak{i}_sigma'].set(value=sigmas[i])

    # 7) Evaluate fit & components
    best_fit  = comp.eval(params=params, x=bias)
    comps_vals = comp.eval_components(params=params, x=bias)

    # 8) Optional weight convolution
    if weight_function_show:
        wsig = ds_fit.attrs.get('weight_sigma', 1.0)
        wfunc = np.exp(-bias**2 / (2 * wsig**2))
        conv = np.convolve(best_fit, wfunc, mode='same') / np.sum(wfunc)

    # 9) Plot
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(bias, ldos, 'k-', lw=1.5, alpha=0.8, label='LDOS', zorder=1)
    ax.plot(bias, best_fit, fit_color+'-', lw=4, alpha=1.0,
            label=f"{chosen} Fit (redchi={redchi:.2e})", zorder=10)
    if weight_function_show:
        ax.plot(bias, conv, fit_color+'-', lw=3, alpha=0.8, label='Convoluted Fit', zorder=9)
        ax2 = ax.twinx()
        ax2.plot(bias, wfunc, '--', lw=1.5, alpha=0.5, color='gray', label='Weight Function')
        ax2.set_ylabel('Weight Function', color='gray')
    for i in idxs:
        ax.plot(bias, comps_vals[f'peak{i}_'], '--', lw=1.5, alpha=0.4,
                label=f'Peak {i}', zorder=2)
    if 'bkg_' in comps_vals:
        ax.plot(bias, comps_vals['bkg_'], '--', lw=1.5, alpha=0.4,
                color='gray', label='Background', zorder=2)

    # CdGM levels & shading
    if show_shade:
        lvl_prox = ds_fit['level_proximity'].isel(Y=y_idx, X=x_idx).values
        Ef = ds_fit.attrs.get('Ef', 1.0)
        SCgap = ds_fit.attrs.get('SCgap', 1.0)
        E_mu = SCgap**2 / Ef
        for i in idxs:
            lp = lvl_prox[i]
            if not np.isnan(lp) and abs(lp) < SCgap:
                if lp == 0:
                    lc, ls, fill = 'gray', '-', True
                else:
                    frac = abs((lp/E_mu) % 1)
                    frac = 1 - frac if frac > 0.5 else frac
                    if np.isclose(frac, 0, atol=1e-2):
                        lc, ls, fill = 'cyan', '--', True
                    elif np.isclose(frac, 0.5, atol=1e-2):
                        lc, ls, fill = 'magenta', '--', True
                    else:
                        lc, ls, fill = 'gray', '--', False
                ax.axvline(lp, color=lc, linestyle=ls, linewidth=1)
                if fill:
                    ax.fill_between(bias, comps_vals[f'peak{i}_'], color=lc, alpha=0.3)
        cand = []
        nmin = int(np.floor(bias.min()/E_mu))
        nmax = int(np.ceil(bias.max()/E_mu))
        for n in range(nmin, nmax+1):
            for lvl in (n*E_mu, (n+0.5)*E_mu):
                if abs(lvl) < SCgap:
                    cand.append(lvl)
        for lvl in sorted(cand):
            frac = abs((lvl/E_mu) % 1)
            frac = 1 - frac if frac > 0.5 else frac
            if np.isclose(frac, 0, atol=1e-2):
                col, lab = 'cyan', 'Integer CdGM Level'
            elif np.isclose(frac, 0.5, atol=1e-2):
                col, lab = 'magenta', 'Half-Integer CdGM Level'
            else:
                col, lab = 'gray', None
            ax.axvline(lvl, color=col, ls='--', lw=1, label=lab)
    else:
        valid_bias = bias[mask]
        if valid_bias.size > 0:
            ax.set_xlim(valid_bias.min(), valid_bias.max())

    ax.axvline(0, color='gray', ls='-', lw=1)
    ax.set_xlabel('Bias (mV)')
    ax.set_ylabel('LDOS')

    # Title with pixel indices and physical coordinates
    y_phys = ds_fit['Y'].isel(Y=y_idx).item() * 1e9
    x_phys = ds_fit['X'].isel(X=x_idx).item() * 1e9
    ax.set_title(
        f'Y_idx={y_idx}, X_idx={x_idx}, model={chosen}\n'
        f'Y={y_phys:.2f} nm, X={x_phys:.2f} nm',
        fontsize=10
    )

    # Legend
    if weight_function_show:
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='upper left')
    else:
        h, l = ax.get_legend_handles_labels()
        unique = dict(zip(l, h))
        ax.legend(unique.values(), unique.keys(), loc='best')

    plt.tight_layout()

    # Collect into DataFrame
    data = {'LDOS': ldos, 'best_fit': best_fit}
    for name, arr in comps_vals.items():
        data[name.rstrip('_')] = arr
    if weight_function_show:
        data['convoluted_fit'] = conv
        data['weight_function'] = wfunc
    df = pd.DataFrame(data, index=bias)
    df.index.name = 'bias_mV'

    if return_fig:
        return fig, df
    else:
        plt.show()
        return df



