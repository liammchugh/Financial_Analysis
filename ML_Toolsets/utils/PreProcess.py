
import numpy as np
import pandas as pd
import random
import torch

def GrabData(filepath, db='training', weather_path=None, data_path=None, input_path=None):
    '''
    take filepaths for data, weather, and inputs 
    can be api format
    '''
    if db == 'api' and data_path is not None:

        data = pd.read_csv(data_path) 
        weather = pd.read_csv(weather_path) 
        inputs = pd.read_csv(input_path) 
        data.columns = data.columns.str.lower() # Convert column names to lowercase
        columns_to_drop = ['date', 'dateStr', 'zip_code'] # Drop columns that cant be used in math operations
        data = data.drop(columns=[col for col in columns_to_drop if col in data.columns], axis=1)
        data = ReShuffle(data)
        data.fillna(data.mean(), inplace=True)
        inputs = pd.concat([inputs] * len(data), ignore_index=True)
        # Merge the dataframes
        data = pd.concat([weather, data, inputs], axis=1)
        # print(data.head())
    else:
        # make sure there are no commas in the data file
        data = pd.read_csv(filepath) # Read CSV file into pandas dataframes
        data.columns = data.columns.str.lower() # Convert column names to lowercase

        # # Convert 'date' column to datetime format
        data['date'] = data['date'].apply(add_seconds)
        data['date'] = pd.to_datetime(data['date'], format='mixed')
        data = data.dropna(subset=['date'])

        # # Extract the day of the year
        data['dayofyear'] = data['date'].dt.dayofyear
        print('period length:', len(data['period']))
        data['time'] = data['date'].dt.hour + data['date'].dt.minute/60 # Convert time to hours
        print('time length:', len(data['time']))

        columns_to_drop = ['date', 'dateStr', 'zip_code'] # Drop columns that cant be used in math operations
        data = data.drop(columns=[col for col in columns_to_drop if col in data.columns], axis=1)
        data.fillna(data.mean(), inplace=True)
        # print(data.head())
    return data


def ReShuffle(data):
    ''' 
    Takes data in form of direct api download
    Returns data in form of columns of variables
    '''
    # Convert 'time' column to datetime format
    data['time'] = pd.to_datetime(data['time'])
    data.drop(columns=['tags'], inplace=True)
    pivot_data = data.pivot(index='time', columns='name', values='value')
    pivot_data.reset_index(inplace=True)
    # Print the reorganized DataFrame
    print(pivot_data.head())
    return pivot_data


# Convert DataFrames to lists of DataFrames
def process_lists(sample_df_list, target_df_list, zero_indices, max_context, filter_cols, filter_params):
    '''
    Process the input and target data to be used in model training or testing.
    Inputs:
        sample_df_list: List of DataFrames containing the input data
        target_df_list: List of DataFrames containing the target data
        zero_indices: Indices of the zero time values in the time column
        filter_params: Dictionary containing the series filter parameters
    Outputs:
        X_list: List of DataFrames containing the input data
            Energy-related values are clipped to be no less than 0
        y_list: List of DataFrames containing the target data
    '''
    X_list = []
    X_list_og = []
    y_list = [] 
    f = 0
    samples = len(zero_indices)
    for k in range(samples-1):
        # if k == samples:
        #     sample_x = sample_df_list.iloc[zero_indices[k]:, :]
        #     sample_y = target_df_list.iloc[zero_indices[k]:, :]
        # else:
        if zero_indices[k+1] - zero_indices[k] > max_context:
            print(f"Skipping Sample {k} with {zero_indices[k+1] - zero_indices[k]} length")
            continue
        sample_x = sample_df_list.iloc[zero_indices[k]:zero_indices[k+1], :]
        sample_y = target_df_list.iloc[zero_indices[k]:zero_indices[k+1], :]
       
        # Filter for infeasable data
        if sample_x.iloc[:, 5].min() < -20000 or sample_x.iloc[:, 5].max() > 500000:
            f+=1
        else:
            sample_x_np = sample_x.to_numpy()
            
            sample_x_filtered_np = SeriesFilter(sample_x_np, filter_cols, **filter_params)

            # Get the index of the column 'heatenergy_cumulat'
            col_idx = sample_x.columns.get_loc('heatenergy_cumulat')
            # Subtract the first element of the 'heatenergy_cumulat' column from all elements in that column
            sample_x_filtered_np[:, col_idx] = sample_x_filtered_np[:, col_idx] - sample_x_filtered_np[0, col_idx]
            # Clip all values to be no less than 0, except for 'heatenergy_cumulat' column
            for i, col in enumerate(sample_x.columns):
                if col != 'heatenergy_cumulat':
                    sample_x_filtered_np[:, i] = np.clip(sample_x_filtered_np[:, i], 0, None)

            sample_x_filtered_df = pd.DataFrame(sample_x_filtered_np, columns=sample_x.columns)
            X_list.append(sample_x_filtered_df)
            y_list.append(sample_y) 
            X_list_og.append(sample_x)
    print(f"{f} Samples Breach Feasibility Bounds")
    print(f"Useable Samples: {len(X_list)}")

    return X_list, X_list_og, y_list


## Padding and Masking - Training Data Prep ##
def pad_sequences(sequences, max_length):
    padded_sequences = torch.zeros(len(sequences), max_length, sequences[0].size(1))
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        padded_sequences[i, -length:] = seq
    return padded_sequences

def generate_masks(src, tgt):
    src_mask = (src.sum(dim=-1) != 0).unsqueeze(1).unsqueeze(2)
    tgt_mask = (tgt.sum(dim=-1) != 0).unsqueeze(1).unsqueeze(2)
    seq_len = tgt.size(1)
    nopeak_mask = (1 - torch.triu(torch.ones((1, seq_len, seq_len), device=tgt.device), diagonal=1)).bool()
    tgt_mask = tgt_mask & nopeak_mask
    return src_mask, tgt_mask

def plot_data(data, features, indices, data2=None, title='Random Sample'):
    import matplotlib.pyplot as plt
    if len(data) == 0:
        print("No samples to plot.")
    selector = 'radiatorsize'
    first_sample = data[1]
    if data2 is not None:
        second_sample = data2[1]
    # Extract x and y columns
    time = first_sample.iloc[:, 0]
    energy = first_sample.iloc[:, indices]

    fig, axs = plt.subplots(len(features), 1, figsize=(12, 15), sharex=True)
    for i, label in enumerate(features):
        axs[i].plot(time, energy.iloc[:, i], label=label)
        if data2 is not None:
            time2 = second_sample.iloc[:, 0]
            energy2 = second_sample.iloc[:, indices]
            axs[i].plot(time2, energy2.iloc[:, i], label='(Unfiltered)', linestyle='--', color='red', linewidth=1)
        axs[i].set_ylabel(label)
        axs[i].grid(True)
        axs[i].legend(loc='upper right')
    axs[-1].set_xlabel('Time (hours)')
    fig.suptitle(title, fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the title
    return


def scale_data(train_data, val_data, train_target, val_target):
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    # Combine the lists of DataFrames into single DataFrames
    train_data_combined = pd.concat(train_data, ignore_index=True)
    val_data_combined = pd.concat(val_data, ignore_index=True)
    train_target_combined = pd.concat(train_target, ignore_index=True)
    val_target_combined = pd.concat(val_target, ignore_index=True)

    # Fit and transform the combined data
    Xscaler = StandardScaler()
    Yscaler = StandardScaler()
    train_data_scaled = Xscaler.fit_transform(train_data_combined)
    val_data_scaled = Xscaler.transform(val_data_combined)
    train_target_scaled = Yscaler.fit_transform(train_target_combined)
    val_target_scaled = Yscaler.transform(val_target_combined)

    # Split the scaled data back into the original list of DataFrames
    train_data_scaled_list = []
    val_data_scaled_list = []
    train_target_scaled_list = []
    val_target_scaled_list = []

    train_start = 0
    val_start = 0
    train_target_start = 0
    val_target_start = 0

    for df in train_data:
        train_end = train_start + len(df)
        train_data_scaled_list.append(pd.DataFrame(train_data_scaled[train_start:train_end], columns=df.columns))
        train_start = train_end

    for df in val_data:
        val_end = val_start + len(df)
        val_data_scaled_list.append(pd.DataFrame(val_data_scaled[val_start:val_end], columns=df.columns))
        val_start = val_end

    for df in train_target:
        train_target_end = train_target_start + len(df)
        train_target_scaled_list.append(pd.DataFrame(train_target_scaled[train_target_start:train_target_end], columns=df.columns))
        train_target_start = train_target_end

    for df in val_target:
        val_target_end = val_target_start + len(df)
        val_target_scaled_list.append(pd.DataFrame(val_target_scaled[val_target_start:val_target_end], columns=df.columns))
        val_target_start = val_target_end

    return train_data_scaled_list, val_data_scaled_list, train_target_scaled_list, val_target_scaled_list, Xscaler, Yscaler

def inverse_transform(data, scaler):
    """
    Inverse transform the data using the provided scaler.

    Parameters:
    scaler: The scaler used for the original scaling (e.g., StandardScaler).
    data: A list of numpy arrays or DataFrames representing the batches to inverse transform,
          or a single numpy array or DataFrame.

    Returns:
    inverse_transformed_data: The inverse transformed data.
    """
    def handle_batch(batch):
        if isinstance(batch, pd.DataFrame):
            batch_values = batch.values
        else:
            batch_values = batch
        
        # Save mask of zero values
        zero_mask = (batch_values == 0)

        # Check if batch is 3D
        if batch_values.ndim == 3:
            batch_shape = batch_values.shape
            batch_values = batch_values.reshape(-1, batch_shape[-1])

            inverse_transformed_values = scaler.inverse_transform(batch_values)
            inverse_transformed_values = inverse_transformed_values.reshape(batch_shape)
        else:
            inverse_transformed_values = scaler.inverse_transform(batch_values)
        
        # Apply mask to set padded values back to zero
        inverse_transformed_values[zero_mask] = 0

        if isinstance(batch, pd.DataFrame):
            return pd.DataFrame(inverse_transformed_values, columns=batch.columns)
        else:
            return inverse_transformed_values

    # Check if data is a list (multiple batches)
    if isinstance(data, list):
        inverse_transformed_data = []
        for batch in data:
            inverse_transformed_data.append(handle_batch(batch))
        return inverse_transformed_data
    else:
        # Single sample case
        return handle_batch(data)


def add_seconds(date_str):
    # add seconds to datatime if missing to standardize format
    if len(date_str.split()) == 2:
        date_part, time_part = date_str.split()
        if len(time_part.split(':')) == 2:
            return f"{date_part} {time_part}:00"
    return date_str



def NormalizeData(data, avg=None, stdev=None):
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a NumPy array.")
    
    normalized_data = np.copy(data)
    # Replace missing/nan values with column mean
    # mask = np.isnan(normalized_data)
    # for i in range(normalized_data.shape[1]):
    #     col_mask = mask[:, i]
    #     for j in range(len(col_mask)):
    #         if col_mask[j]:
    #             normalized_data[col_mask[j], i] = np.nanmean(normalized_data[:, i])
    
    if avg is None:
        avg = data.mean(axis=0)
    if stdev is None:
        stdev = data.std(axis=0)
    
    # Replace zero stdev with 1
    stdev = np.where(stdev == 0, 1, stdev)
    
    normalized_data = (data - avg) / stdev
    
    return normalized_data, avg, stdev

def DenormalizeData(data, avg, stdev):
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a NumPy array.")
    
    denormalized_data = np.copy(data)
    
    denormalized_data = (data * stdev) + avg
    
    return denormalized_data

def getmedian(data):
    data_values = data.values if isinstance(data, pd.DataFrame) else data
    
    med_single = np.median(data_values, axis=0)
    med_single = np.where(med_single == 0, 1, med_single)
    return med_single

def medNormalizeData(data, med=None):
    # Helper function to normalize a single DataFrame or NumPy array
    def normalize_single(data_single, med_single=None):
        data_values = data_single.values if isinstance(data_single, pd.DataFrame) else data_single
        
        if med_single is None:
            med_single = np.median(data_values, axis=0)
        
        med_single = np.where(med_single == 0, 1, med_single)
        normalized_values = data_values / med_single
        
        if isinstance(data_single, pd.DataFrame):
            normalized_data_single = pd.DataFrame(normalized_values, index=data_single.index, columns=data_single.columns)
        else:
            normalized_data_single = normalized_values
        
        return normalized_data_single, med_single
    
    # Normalize depending on the type of input
    if isinstance(data, list):
        normalized_data_list = []
        med_list = []
        col_meds = []
        if med is None:
            for df in data:
                med_df = getmedian(df)
                col_meds.append(med_df)
            for i in range(len(df.columns)):
                med_list.append(np.median([col_med[i] for col_med in col_meds]))
        else: 
            med_list = med
        for df in data:
            normalized_df, med_df = normalize_single(df, med_list)
            normalized_data_list.append(normalized_df)
        return normalized_data_list, med_list
    else:
        return normalize_single(data, med)


def medDenormalizeData(data, med):
    # Helper function to denormalize a single DataFrame or NumPy array
    def denormalize_single(data_single, med_single):
        # Convert med_single to a NumPy array if it isn't already
        med_single = np.array(med_single)
        
        # Extract values from DataFrame or use the array as is
        if isinstance(data_single, pd.DataFrame):
            data_values = data_single.values
        else:
            data_values = data_single
        
        # Ensure med_single can be broadcast to the shape of data_values
        if med_single.ndim == 1:
            med_single = med_single.reshape(1, -1)
        
        denormalized_values = data_values * med_single
        
        if isinstance(data_single, pd.DataFrame):
            denormalized_data_single = pd.DataFrame(denormalized_values, index=data_single.index, columns=data_single.columns)
        else:
            denormalized_data_single = denormalized_values
        
        return denormalized_data_single
    
    # Denormalize depending on the type of input
    if isinstance(data, list):
        denormalized_data_list = []
        for df in data:
            denormalized_df = denormalize_single(df, med)
            denormalized_data_list.append(denormalized_df)
        return denormalized_data_list
    else:
        return denormalize_single(data, med)


def SeriesFilter(data, filter_columns, time=None, filter_type='biquadratic', cutoff_freq=1, order=2):
    """
    Apply a specified filter to each dimension of the input data.
    
    Parameters:
        data (np.array): 2D array where the first column is time in hours and the subsequent columns are the signals to be filtered.
        filter_cols (list): List of column indices to filter.
        filter_type (str): Type of filter to apply ('biquadratic' or 'butter').
        cutoff_freq (float): Cutoff frequency in 1/days (e.g., 0.1 for a 10-day cutoff period).
        order (int): Order of the filter.
    
    Returns:
        np.array: The filtered signals.
    """
    from scipy.signal import butter, filtfilt, sosfilt, sosfilt_zi, lfilter
    from scipy.interpolate import interp1d

    def butter_lowpass(cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def biquad_lowpass(cutoff, fs, order=2):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        sos = butter(order, normal_cutoff, btype='low', output='sos')
        return sos

    def apply_filter(data, filter_coefficients, filter_type='biquad'):
        if filter_type == 'biquad':
            sos = filter_coefficients
            zi = sosfilt_zi(sos)
            filtered_data, _ = sosfilt(sos, data, zi=zi*data[0])
            return filtered_data
        else:
            b, a = filter_coefficients
            filtered_data = lfilter(b, a, data)
            return filtered_data

    # Extract the time and signal columns
    if time is None:
        time = data[:, 0]
        signals = data[:, 1:]
    else:
        signals = data

    # Determine the sampling frequency (fs) in Hz (1/seconds)
    # Calculate the sampling rate based on the time increments in hours
    time_diff = np.diff(time)
    # if not np.allclose(time_diff, time_diff[0]):
    #     # Interpolate data to a uniform time grid
    #     print("Interpolating data to a uniform time grid.")
    #     time_uniform = np.linspace(time.min(), time.max(), len(data))
    #     interpolator = interp1d(time, signals, kind='linear', fill_value='extrapolate')
    #     signals = interpolator(time_uniform)
    #     time_diff = np.diff(time_uniform)
    #     fs = 1.0 / (time[1] - time[0])  # Uniform sampling frequency in 1/hours
    # else:
    fs = 1.0 / time_diff[0]

    # Convert cutoff frequency from 1/days to 1/hours
    cutoff_freq_per_hour = cutoff_freq / 24.0

    # Initialize the filtered signals array
    filtered_signals = signals.copy()

    if filter_type == 'biquadratic':
        sos = biquad_lowpass(cutoff_freq_per_hour, fs, order=order)
        for col in filter_columns:
            filtered_signals[:, col-1] = apply_filter(data[:, col], sos, filter_type='biquad')
    elif filter_type == 'butter':
        b, a = butter_lowpass(cutoff_freq_per_hour, fs, order=order)
        for col in filter_columns:
            filtered_signals[:, col-1] = apply_filter(data[:, col], (b, a), filter_type='butter')
    elif filter_type == 'filtfilt':
        b, a = butter_lowpass(cutoff_freq_per_hour, fs, order=order)
        for col in filter_columns:
            filtered_signals[:, col-1] = filtfilt(b, a, data[:, col])
        
    else:
        raise ValueError("Unsupported filter type: {}".format(filter_type))

    return np.column_stack((time, filtered_signals))
