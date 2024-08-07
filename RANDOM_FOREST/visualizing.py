import mne
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt

# Define the path to the EEG data file
path = r'C:\Users\prabh\Desktop\BCIs\Datasets\Schizophrenia\HC\sub-000\eeg\sub-000_task-responder_run-1_eeg.vhdr'

# Load the raw EEG data
raw = mne.io.read_raw_brainvision(path, preload=True)
raw.filter(1., 30., fir_design='firwin')
raw.pick_channels(['FCz', 'Cz', 'Pz', 'AFz', 'Fz', 'Fp1', 'Fp2'])
onsets = []

# Extract the annotations
annotations = raw.annotations
# Find the first stimulus event
stimulus_events = [annot for annot in annotations if 'Stimulus/' in annot['description']]
for event in stimulus_events:
    first_stimulus_onset = event['onset']
    if first_stimulus_onset in onsets:
        break   
    onsets.append(first_stimulus_onset)
    print(first_stimulus_onset)
    print('-'*60)

    # Convert annotations to events array
    events = mne.events_from_annotations(raw, event_id=None, regexp='Stimulus/')[0]

    # Define the baseline interval (set to a single sample)
    baseline = (0, 0)

    # Extract the data from the raw EEG recordings
    evoked = mne.Epochs(raw, events=events, tmin=first_stimulus_onset, tmax=first_stimulus_onset+0.5, baseline=None, preload=True).average()

    # Plot the evoked data
    evoked_df = evoked.to_data_frame()
    print(evoked_df.head())

    # Plotting
    times = evoked_df['time']
    FCz_values = evoked_df['AFz']
    Cz_values = evoked_df['Pz']

    plt.figure(figsize=(12, 6))
    plt.plot(times, FCz_values, label='FCz')
    plt.plot(times, Cz_values, label='Cz')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (uV)')
    plt.title('Evoked Response - FCz and Cz')

    # Define P2 and FRN windows
    p2_start = 0.15+first_stimulus_onset
    p2_end = first_stimulus_onset + 0.25
    frn_start = 0.25+first_stimulus_onset
    frn_end = first_stimulus_onset + 0.35

    # Shaded areas for P2 and FRN windows
    plt.axvspan(p2_start, p2_end, color='yellow', alpha=0.3, label='P2 window')
    plt.axvspan(frn_start, frn_end, color='red', alpha=0.3, label='FRN window')

    plt.legend()
    plt.show()
