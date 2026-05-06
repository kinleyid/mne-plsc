
# Input data formats

Different inputs are used for different types of analysis, depending on the . Below is a comprehensive list. Note that "participant x cond" means a conjunction of participant and within-participants condition. For example, this could mean a given participant's average for the trials in condition A.

| Space | Analysis | Domain | Input data |
|-------|--=-------|--------|------------|
| Sensor | Single-subject | Time | Single `mne.Epochs` |
| Sensor | Single-subject | Frequency | Single `mne.time_frequency.EpochsSpectrum` |
| Sensor | Single-subject | Time-frequency | Single `mne.time_frequency.EpochsTFR` |
| Sensor | Group | Time | List of `mne.Evoked` (one per participant x cond) |
| Sensor | Group | Frequency | List of `mne.time_frequency.Spectrum` (one per participant x cond) |
| Sensor | Group | Time-frequency | List of `mne.time_frequency.AverageTFR` (one per participant x cond) |
| Source | Single-subject | Time | List of `mne.[Volume]SourceEstimate` (one per trial) |
| Source | Single-subject | Frequency | List of `mne.[Volume]SourceEstimate` (one per trial) |
| Source | Single-subject | Time-frequency | List of lists of `mne.[Volume]SourceEstimate` (one per trial in outer list, one per frequency in inner list) | 
| Source | Group | Time | List of `mne.[Volume]SourceEstimate` (one per participant x cond) |
| Source | Group | Frequency | List of `mne.[Volume]SourceEstimate` (one per participant x cond) |
| Source | Group | Time-frequency | List of lists of `mne.[Volume]SourceEstimate` (one per participant x cond in outer list, one per frequency in inner list) |
