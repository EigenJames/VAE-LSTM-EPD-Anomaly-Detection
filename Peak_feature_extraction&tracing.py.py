from spectra_analysis import *
import pandas as pd


# Corrected file path with forward slashes
file_path = 'D:/James_archive/OneDrive/On_Going/VASSCAA_submission/Paper draft/data copy/Comp_1.pkl'

# Read the pickle file
df = pd.read_pickle(file_path)

# Display the DataFrame

sample_plt(df,1,3)
sample_data=df.iloc[:, :1]


# Detect strong peaks and get peak information as a DataFrame
peaks_df = detect_peaks(sample_data, min_height=500, min_prominence=100, plot_peaks=True)

# Display the first few rows of the peaks DataFrame

trace_df = element_trace(peaks_df, df)
