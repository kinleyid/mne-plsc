
# Input data formats

Different inputs are accepted 

| Domain | Space | Analysis | Input data |
------------------------------------------
| Sensor | Singlesubject | Time | Single `mne.Epochs` |
| Sensor | Singlesubject | Frequency | Single `mne.EpochsSpectrum` |
| Sensor | Singlesubject | Timefrequency | Single `mne.EpochsTFR` |
| Sensor | Group | Time | List of `mne.Evoked` |
| Sensor | Group | Frequency | List of `mne.Spectrum` |
| Sensor | Group | Timefrequency | List of `mne.AverageTFR` |
| Source | Singlesubject | Time | Single `mne.Epochs` |
| Source | Singlesubject | Frequency | Single `mne.EpochsSpectrum` |
| Source | Singlesubject | Timefrequency | Single `mne.EpochsTFR` |
| Source | Group | Time | List of `mne.Evoked` |
| Source | Group | Frequency | List of `mne.Spectrum` |
| Source | Group | Timefrequency | List of `mne.AverageTFR` |