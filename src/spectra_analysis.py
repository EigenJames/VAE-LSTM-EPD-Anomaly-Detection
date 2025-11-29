import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from scipy.signal import find_peaks, peak_widths
from scipy.integrate import simps
from mpl_toolkits.mplot3d import Axes3D
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_file(filepath):
    """
    Processes a single file to extract intensity data.
    
    Parameters:
    filepath (str): Path to the file to be processed.
    
    Returns:
    tuple: (wavelength, intensity) if successful, None otherwise.
    """
    try:
        raw_data = pd.read_csv(filepath, skiprows=11, header=None, sep=r'\s+', comment='#', engine='python')
        if raw_data.empty or raw_data.shape[1] < 1:
            print(f"Warning: {filepath} is empty or not properly formatted.")
            return None
        
        # Split and convert data
        data = raw_data[0].str.split(';', expand=True)
        data = data.apply(pd.to_numeric, downcast="float", errors='coerce', axis=1)
        
        # Check data integrity
        if data.isnull().values.any():
            print(f"Warning: {filepath} contains non-numeric data.")
            return None

        wavelength = data.iloc[:, 0].values
        intensity = data.iloc[:, 1].values
        print('Processed file:', filepath)
        
        return wavelength, intensity
    
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def read_spectra(path, export_filename, max_workers=4):
    """
    Reads transmittance data from text files in the specified directory and 
    compiles them into a DataFrame, then saves to a pickle file.
    
    Parameters:
    path (str): Directory path containing .txt files.
    export_filename (str): The filename to export the DataFrame as a pickle file.
    max_workers (int): The maximum number of workers for parallel processing.
    
    Returns:
    DataFrame: The compiled DataFrame with intensity values.
    """
    txt_files = glob.glob(os.path.join(path, "*.txt"))
    if not txt_files:
        raise FileNotFoundError("No .txt files found in the specified directory.")
    
    print(f"Found {len(txt_files)} files. Processing...")

    intensity_list = []
    wavelength = None

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create a list of futures
        futures = {executor.submit(process_file, f): f for f in txt_files}

        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                w, i = result
                if wavelength is None:
                    wavelength = w
                intensity_list.append(i)
    
    if not intensity_list:
        raise ValueError("No valid data found in the .txt files.")

    # Combine all intensities into a DataFrame
    intensity_array = np.array(intensity_list).T
    column_names = ['Spectrum_' + str(i+1) for i in range(intensity_array.shape[1])]
    df = pd.DataFrame(intensity_array, columns=column_names)
    df.insert(0, 'Wavelength', wavelength)

    df.set_index('Wavelength', inplace=True)
    df.to_pickle(export_filename)
    print("Data processing complete. Data saved to:", export_filename)
    return df

def load_spectra_series(import_filename):
    """
    Loads transmittance data from a pickle file.
    
    Parameters:
    import_filename (str): The filename to load the DataFrame from a pickle file.
    
    Returns:
    DataFrame: The loaded DataFrame with intensity values.
    """
    if not os.path.exists(import_filename):
        raise FileNotFoundError(f"File {import_filename} not found.")
    
    df = pd.read_pickle(import_filename)
    print("Data loaded from:", import_filename)
    return df

def wavelength_slicing(df, wlr_low, wlr_high):
    """
    Slices the DataFrame for a specific wavelength range.
    
    Parameters:
    df (DataFrame): The DataFrame to slice.
    wlr_low (float): The lower limit of the wavelength range.
    wlr_high (float): The upper limit of the wavelength range.
    
    Returns:
    DataFrame: The sliced DataFrame within the specified wavelength range.
    """
    df.index = pd.to_numeric(df.index, errors='coerce')
    sliced_data = df.loc[(df.index > wlr_low) & (df.index < wlr_high)]
    return sliced_data


def multi_overview(df, sample_step=1, df_name='DataFrame'):
    """
    Creates a 3D overview plot of the spectra time-series data using Matplotlib.
    
    Parameters:
    df (DataFrame): The DataFrame to plot.
    sample_step (int): The step size for sampling the spectra columns for plotting.
    df_name (str): The name of the DataFrame.
    
    Returns:
    None
    """
    idx = np.arange(0, len(df.columns), sample_step)
    x = df.index.values
    y = np.arange(len(idx))
    z = df.iloc[:, idx].values.T

    X, Y = np.meshgrid(x, y)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, z, cmap='viridis')

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    ax.set_title(f'Spectra Time-Series Overview on {df_name}', fontsize=15)
    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Spectra Series', fontsize=12)
    ax.set_zlabel('Intensity (A.U.)', fontsize=12)

    plt.show()

def multi_overview_3D(data, sample_step=1, df_name='DataFrame'):
    """
    Creates a 3D overview plot of the spectra time-series data using Plotly.
    
    Parameters:
    data (DataFrame): The DataFrame to plot.
    sample_step (int): The step size for sampling the spectra columns for plotting.
    df_name (str): The name of the DataFrame.
    
    Returns:
    None
    """
    idx = np.arange(0, len(data.columns.values), sample_step)
    x = data.index.values
    y = idx
    z = data.iloc[:, idx].values.T

    fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, colorscale='Viridis')])

    fig.update_layout(
        title=f'Spectra Time-Series Overview on {df_name}',
        autosize=True,
        width=1000,
        height=1000,
        margin=dict(l=65, r=50, b=65, t=90),
        scene=dict(
            xaxis=dict(title='Wavelength (nm)'),
            yaxis=dict(title='Spectra Series'),
            zaxis=dict(title='Intensity (A.U.)'),
            camera_eye=dict(x=0, y=-1, z=0.5),
            aspectratio=dict(x=1, y=1, z=0.3)
        )
    )

    return fig.show()

def sample_plt(data, start_spectra, end_spectra, df_name='DataFrame'):
    """
    Plots the average and standard deviation of selected spectra.
    
    Parameters:
    data (DataFrame): The DataFrame to plot.
    start_spectra (int): The starting index of the spectra to plot.
    end_spectra (int): The ending index of the spectra to plot.
    df_name (str): The name of the DataFrame.
    
    Returns:
    avg_spectra (Series): The average spectra.
    sd_spectra (Series): The standard deviation of the spectra.
    """
    indx_list = list(range(start_spectra, end_spectra))
    spectra = data.iloc[:, indx_list]
    avg_spectra = spectra.mean(axis=1)
    sd_spectra = spectra.std(axis=1)
    variation_level = sd_spectra.mean()

    fig, ax = plt.subplots(figsize=(15, 9))
    x = avg_spectra.index.values
    y = avg_spectra.values

    ax.plot(x, y, 'k--', label='Averaged')
    ax.errorbar(x, y, yerr=sd_spectra.values, alpha=0.3, label='Standard Deviation')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Intensity (A.U.)')
    ax.set_title(f'Spectra Variation on {df_name}\n({len(indx_list)} Spectra Averaged)', fontsize=20)
    ax.text(
        x.mean(), y.max() * 0.8,
        f'Spectrum range:\n {round(x.min(), 0)} nm to {round(x.max(), 0)} nm\n\nVariation level: {round(variation_level, 2)}',
        fontsize=15
    )

    ax.legend(loc='upper right', shadow=True, fontsize='x-large')
    plt.show()

    return avg_spectra, sd_spectra




def detect_peaks(data, min_height=1000, min_prominence=100, plot_peaks=True, plot_columns=None):
    """
    Detects strong peaks in the provided intensity columns, calculates peak center, intensity, and width.
    Returns the detected strong peaks in a DataFrame and optionally plots them.
    
    Parameters:
    data (DataFrame): The DataFrame containing the spectra, with wavelength as the index.
    min_height (float): Minimum peak height to be considered as a peak.
    min_prominence (float): Minimum prominence for a peak to be considered strong.
    plot_peaks (bool): Whether to plot the detected peaks (default is True).
    plot_columns (list or None): List of column indices or names to plot. If None, plot first and last spectra.
    
    Returns:
    peaks_df (DataFrame): DataFrame containing the time, peak center (wavelength), intensity, and FWHM for each column.
    """
    # Initialize a list to store peak data
    peak_data = []
    
    # Get the wavelength from the DataFrame index
    wavelengths = data.index.values
    
    # If no specific columns are provided for plotting, plot the first and last columns
    if plot_columns is None:
        plot_columns = [data.columns[0], data.columns[-1]]  # First and last columns
    
    # Loop through each intensity column
    for col in data.columns:
        # Extract intensity data
        intensities = data[col].values
        
        # Find strong peaks in the intensity data using height and prominence
        peak_indices, properties = find_peaks(intensities, height=min_height, prominence=min_prominence)
        
        # Calculate peak properties (widths) using FWHM
        results_full = peak_widths(intensities, peak_indices, rel_height=0.5)
        
        # Store peak information
        for i, peak_index in enumerate(peak_indices):
            peak_wavelength = wavelengths[peak_index]
            peak_intensity = intensities[peak_index]
            peak_width = results_full[0][i]
            
            # Append peak information to the list with time
            peak_data.append({
                'Time': col,  # Assuming the column names represent time
                'Peak Center (nm)': peak_wavelength,
                'Peak Intensity (A.U.)': peak_intensity,
                'Peak Width (nm)': peak_width
            })
        
        # Plot only if the column is in the plot_columns list
        if plot_peaks and col in plot_columns:
            plt.figure(figsize=(12, 6))
            plt.plot(wavelengths, intensities, label=f'Spectra: {col}', color='blue')
            
            # Annotate the strong peaks with their intensity and width (FWHM)
            for i, peak_index in enumerate(peak_indices):
                peak_wavelength = wavelengths[peak_index]
                peak_intensity = intensities[peak_index]
                peak_width = results_full[0][i]
                
                # Label the peak with its intensity and width
                plt.annotate(f'Intensity: {peak_intensity:.1f}\nWidth: {peak_width:.1f} nm',
                             (peak_wavelength, peak_intensity),
                             textcoords="offset points",
                             xytext=(0,10), ha='center', fontsize=8, color='red')
            
            # Label the plot
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Intensity (A.U.)')
            plt.title(f'Spectra with Strong Peaks (Height > {min_height}, Prominence > {min_prominence}) at Time {col}')
            plt.legend()
            plt.show()
    
    # Convert the list of peaks into a DataFrame
    peaks_df = pd.DataFrame(peak_data)
    
    return peaks_df


def element_trace(peaks_df, spectra_data):
    """
    Tracks the integrated intensity for each detected peak over time by integrating the intensity
    values inside the FWHM (Full Width at Half Maximum) range of each peak across different time points.
    
    Parameters:
    - peaks_df (DataFrame): A DataFrame containing the peak information with columns such as:
                            'Peak Center (nm)', 'Peak Width (nm)', and others.
                            Each row represents a detected peak at a specific time point.
    - spectra_data (DataFrame): A DataFrame containing the original spectra data with wavelengths as the index.
                                Each column represents the intensity values at a specific time point.
    
    Returns:
    - trace_df (DataFrame): A DataFrame where each column represents a peak (identified by its center wavelength in nm),
                            and each row represents the integrated intensity for that peak over time (one row per time point).
    
    How to Use:
    1. The `peaks_df` should contain information about the peaks detected in the spectra, including:
       - 'Peak Center (nm)': The wavelength where the peak occurs.
       - 'Peak Width (nm)': The width of the peak (FWHM).
    2. The `spectra_data` should have wavelengths as the **index** and time points as the **columns**.
       - Each column represents the intensities at different time points.
       - Each row represents the intensity values for a specific wavelength.
    3. The function will calculate the area under the curve (integrated intensity) within the FWHM range for each peak,
       for all time points.
    
    Example:
    If `peaks_df` contains peaks at wavelengths 400 nm, 500 nm, and 600 nm, and `spectra_data` contains
    intensity values at times t1, t2, and t3, the output will be a DataFrame with:
    - Columns: 400 nm, 500 nm, 600 nm (representing each peak).
    - Rows: t1, t2, t3 (representing different time points).
    The values will represent the integrated intensities for each peak over the specified range (FWHM).
    
    Example Usage:
    >>> trace_df = element_trace(peaks_df, spectra_data)
    >>> print(trace_df)
    """
    
    # Initialize a dictionary to store the integrated intensity values for each peak over time
    integrated_data = {}

    # Loop through each row in the peaks_df (each detected peak)
    for index, peak_info in peaks_df.iterrows():
        peak_center = np.round(peak_info['Peak Center (nm)'], 3)  # Round peak center to 3 decimal places
        peak_width = np.round(peak_info['Peak Width (nm)'], 3)    # Round peak width to 3 decimal places
        
        # Calculate the FWHM boundaries (start and end points) around the peak center
        lower_bound = peak_center - (peak_width / 2)
        upper_bound = peak_center + (peak_width / 2)
        
        # Find the corresponding wavelength indices in the spectra data for integration
        wavelength_values = spectra_data.index.values
        lower_idx = np.searchsorted(wavelength_values, lower_bound, side='left')
        upper_idx = np.searchsorted(wavelength_values, upper_bound, side='right')
        
        # Extract the relevant wavelengths for integration
        wavelengths_in_range = wavelength_values[lower_idx:upper_idx]

        # Create a list to store integrated values for each time point
        integrated_intensity_series = []

        # Loop through each time column in spectra_data (each time point)
        for time_point in spectra_data.columns:
            # Extract the intensities for the current time point within the FWHM range
            intensities_in_range = spectra_data.loc[wavelengths_in_range, time_point]

            # Integrate the intensity over the range defined by FWHM (using Simpson's rule)
            integrated_intensity = np.round(simps(intensities_in_range, wavelengths_in_range),3)

            # Append the integrated intensity for this time point
            integrated_intensity_series.append(integrated_intensity)
        
        # Store the integrated intensities for this peak (keyed by the peak center wavelength)
        integrated_data[peak_center] = integrated_intensity_series
    
    # Convert the integrated data into a DataFrame, where each column is a peak and rows are time points
    trace_df = pd.DataFrame(integrated_data, index=spectra_data.columns)
    
    # Return the DataFrame containing the integrated intensities for each peak over time
    return trace_df


