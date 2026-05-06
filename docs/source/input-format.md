
# Input data formats

Different inputs are accepted 

| Domain | Space | Analysis | Input data |
|--------|-------|----------|------------|
| Sensor | Singlesubject | Time | Single `mne.Epochs` |
| Sensor | Singlesubject | Frequency | Single `mne.time_frequency.EpochsSpectrum` |
| Sensor | Singlesubject | Timefrequency | Single `mne.time_frequency.EpochsTFR` |
| Sensor | Group | Time | List of `mne.Evoked` (one per participant x cond) |
| Sensor | Group | Frequency | List of `mne.time_frequency.Spectrum` (one per participant x cond) |
| Sensor | Group | Timefrequency | List of `mne.time_frequency.AverageTFR` (one per participant x cond) |
| Source | Singlesubject | Time | List of `mne.[Volume]SourceEstimate` (one per trial) |
| Source | Singlesubject | Frequency | List of `mne.[Volume]SourceEstimate` (one per trial) |
| Source | Singlesubject | Timefrequency | List of lists of `mne.[Volume]SourceEstimate` (one per trial in outer list, one per frequency in inner list) | 
| Source | Group | Time | List of `mne.[Volume]SourceEstimate` (one per participant x cond) |
| Source | Group | Frequency | List of `mne.[Volume]SourceEstimate` (one per participant x cond) |
| Source | Group | Timefrequency | List of lists of `mne.[Volume]SourceEstimate` (one per participant x cond in outer list, one per frequency in inner list) |
