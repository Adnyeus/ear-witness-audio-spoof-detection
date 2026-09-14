# Dataset Setup Instructions

## ASVspoof 2019 Logical Access (LA) Dataset

This project uses the ASVspoof 2019 LA dataset for audio spoof detection.

### Download

1. Visit the official ASVspoof website: https://www.asvspoof.org/
2. Navigate to the "ASVspoof 2019" section
3. Download the Logical Access (LA) partition

### Expected Folder Structure

After downloading, organize your files as follows:

data/
├── ASVspoof2019_LA_train/
│ └── flac/
│ ├── LA_T_1000137.flac
│ ├── LA_T_1000406.flac
│ └── ... (25,380 files)
├── ASVspoof2019_LA_dev/
│ └── flac/
│ └── ... (24,986 files)
├── ASVspoof2019_LA_eval/
│ └── flac/
│ └── ... (71,933 files)
└── ASVspoof2019_LA_cm_protocols/
├── ASVspoof2019.LA.cm.train.trn.txt
├── ASVspoof2019.LA.cm.dev.trl.txt
└── ASVspoof2019.LA.cm.eval.trl.txt


### Protocol File Format

Each line in the protocol files contains:

SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID KEY

Where:
- `SPEAKER_ID`: Speaker identifier (LA_****)
- `AUDIO_FILE_NAME`: File name (without .flac extension)
- `SYSTEM_ID`: Spoofing system (A01-A19) or '-' for bonafide
- `KEY`: 'bonafide' or 'spoof'

### Note on Dataset Size

The full dataset is ~10 GB. **I did not commit the audio files to GitHub.** They are excluded by `.gitignore`. I only committed the derived feature matrices (`.mat` files) that were under GitHub's 100 MB file size limit.

### Quick Verification

After placing the dataset, verify with:
```matlab
audioPath = 'data/ASVspoof2019_LA_train/flac';
files = dir(fullfile(audioPath, '*.flac'));
fprintf('Found %d training files\n', length(files));
```

Expected: 25,380 files.