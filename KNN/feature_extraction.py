import mne
import os
import pandas as pd

def get_features(evoked_df, channel, tmin, tmax, erp, onset):
    data = evoked_df[channel]
    times = evoked_df['time']
    idx = (times >= tmin) & (times <= tmax)

    component_data = data[idx]
    if erp == 'FRN':
        amplitude = component_data.min()
    else:
        amplitude = component_data.max()
    amp_idx = component_data.index[component_data==amplitude].tolist()
    latency = times.iloc[amp_idx].values[0]
    latency = latency - onset
    
    return amplitude, latency

# Define the path to the EEG data file
path = r'C:\Users\prabh\Desktop\BCIs\Datasets\Schizophrenia'

data = []

channels = ['FCz', 'Cz', 'Pz', 'AFz', 'Fz', 'Fp1', 'Fp2']

for group in ['HC', 'P']:
    path = r'C:\Users\prabh\Desktop\BCIs\Datasets\Schizophrenia'
    group_path = os.path.join(path, group)
    for subject in os.listdir(group_path):
        subject_path = os.path.join(group_path, subject, 'eeg')
        for file in os.listdir(subject_path):
            if 'responder' in file and file.endswith('_eeg.vhdr'):
                # Load the raw EEG data
                path = os.path.join(subject_path, file)
                raw = mne.io.read_raw_brainvision(path, preload=True)
                raw.pick_channels(channels)
                raw.filter(1., 30., fir_design='firwin')
                # Load the raw EEG data

                # Extract the annotations
                annotations = raw.annotations

                # Find the first stimulus event
                stimulus_events = [annot for annot in annotations if 'Stimulus/' in annot['description']]

                for i in stimulus_events:
                    temp = []
                    print(i)
                    onset = i['onset']

                    # Convert annotations to events array
                    events = mne.events_from_annotations(raw, event_id=None, regexp='Stimulus/')[0]

                    # Extract the data from the raw EEG recordings
                    epoched = mne.Epochs(raw, events=events, tmin=onset, tmax=onset+0.5, baseline=None, preload=True, event_repeated='drop')
                    if len(epoched) > 0:    
                        evoked = epoched.average()      

                        # Plot the evoked data
                        evoked_df = evoked.to_data_frame()

                        # Define P2 and FRN windows
                        p2_tmin = 0.15+onset
                        p2_tmax = 0.25+onset
                        frn_tmin = 0.25+onset
                        frn_tmax = 0.35+onset

                        for channel in channels:
                            FRN_amp, FRN_lat = get_features(evoked_df, channel, frn_tmin, frn_tmax, 'FRN', onset)
                            P2_amp, P2_lat = get_features(evoked_df, channel, frn_tmin, frn_tmax, 'P2', onset)
                            temp.append(FRN_amp)
                            temp.append(FRN_lat)
                            temp.append(P2_amp)
                            temp.append(P2_lat)

                        if group == 'HC':
                            temp.append(0)
                        else:
                            temp.append(1)
                        data.append(temp)
                    
# Create a DataFrame and save to CSV
columns = ['FCz_FRN_Amp', 'FCz_FRN_Lat', 'FCz_P2_Amp', 'FCz_P2_Lat',
           'Cz_FRN_Amp', 'Cz_FRN_Lat', 'Cz_P2_Amp', 'Cz_P2_Lat',
           'Pz_FRN_Amp', 'Pz_FRN_Lat', 'Pz_P2_Amp', 'Pz_P2_Lat',
           'AFz_FRN_Amp', 'AFz_FRN_Lat', 'AFz_P2_Amp', 'AFz_P2_Lat',
           'Fz_FRN_Amp', 'Fz_FRN_Lat', 'Fz_P2_Amp', 'Fz_P2_Lat',
           'Fp1_FRN_Amp', 'Fp1_FRN_Lat', 'Fp1_P2_Amp', 'Fp1_P2_Lat',
           'Fp2_FRN_Amp', 'Fp2_FRN_Lat', 'Fp2_P2_Amp', 'Fp2_P2_Lat',
           'Class']

df = pd.DataFrame(data, columns=columns)
output_file = r'C:\Users\prabh\Desktop\BCIs\Datasets\Schizophrenia\responder_erp_features.csv'
df.to_csv(output_file,    index=False)

print(f"Dataset saved to {output_file}")