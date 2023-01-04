### A Machine Learning Pipeline for Laughter Detection on the ICSI Corpus

This repo is based on the laughter detection model by [Gillick et al.](https://github.com/jrgillick/laughter-detection) and retrains it on the
[ICSI Meeting corpus](https://ieeexplore.ieee.org/abstract/document/1198793)

The data pipeline uses [Lhotse](https://github.com/lhotse-speech/lhotse), a new Python library for speech and audio data preparation.

This repository consists of three main parts:
1. Evaluation Pipeline
2. Data Pipeline
3. Training Code

The following list outlines which parts of the repository belong to each of them and classifies the parts/files as one of three types:
1. `from scratch`: entirely written by myself
2. `adapted`: code taken from [Gillick et al.](https://github.com/jrgillick/laughter-detection) and adapted
3. `unmodified`: code taken from [Gillick et al.](https://github.com/jrgillick/laughter-detection) and not adapted or modified

- **Evalation Pipeline** (from scratch): 
    - `analysis`
        - `transcript_parsing/parse.py` +`preprocess.py`: parsing and preprocessing the ICSI transcripts
        - `analyse.py`: main function, that parses and evaluates predictions from .TextGrid files output by the model
        - `output_processing`: scripts for creating .wav files for the laughter occurrences to manually evaluate them
    - `visualise.py`: functions for visualising model performance (incl. prec-recall curve and confusion matrix)

- **Data Pipeline** (from scratch) - also see [diagram](#diagram-of-the-data-pipeline):
    - `compute_features`:  computes feature representing the whole corpus and specific subsets of the ICSI corpus
    - `create_data_df.py`: creates a dataframe representing training, development and test-set 

- **Training Code**:
    - `models.py` (unmodified): defines the model architecture
    - `train.py` (adapted): main training code
    - `segment_laughter.py` + `laugh_segmenter.py` (adpated): inference code to run laughter detection on audio files
    - `datasets.py` + `load_data.py` (from scratch): the new LAD (Laugh Activity Detection) Dataset + new inference Dataset and code for their creation

- **Misc**:
    - `Demo.ipynb` (from scratch): demonstration of using Lhotse to compute features from a dataframe defining laughter and non-laughter segments 
    - `config.py` (adapted): configurations for different parts of the pipeline
    - `results.zip` (N/A): contains the model predictions from experiments presented in my thesis
### Diagram of the Data Pipeline
![Data Pipeline](./docs/data-pipeline.png)

# Getting started
_Steps to get the environment setup from scratch such that training and evaluation can be run_

1. Clone this repo
2. `cd` into the repo    
3. create a python env and install all packages listed below. Put them in a `requirments.txt` file and run `pip install -r requirments.txt`
4. we use Lhotse's available recipe for the ICSI-corpus to download the corpus' audio + transcripts  
    - run the python script `get_icsi_data.py`
      - this will take a while to complete - it downloads all audio and transcriptions for the icsi corpus
      - after completion 
        - you should have a `data/icsi/speech` folder with all your audio files grouped by meeting 
        - you should have a `data/icsi/transcripts` folder with all the `.mrt` transcripts

5. Now create a `.env` file by copying the `.sample.env`-file to an `.env` file.
  - you can configure the folders to match your desired folder structure
6. Now you run `compute_features.py` once to compute the features for the whole corpus 
  - the first time this will parse the transcripts and create indices with laughter and non-laughter segments (see `Other documentation` section below). 
    - This will take a while (e.g. it took one hour for me) 
      - after initial creation the indices are cached and they are loaded from disk
  - that's done by the `compute_features_per_split()` method in the main() function
  - you can comment out the call to `compute_features_for_cuts()` in the main() function if you just want to create the features for the whole corpus for now
7. Then you run `create_data_df` to create a set of training samples 
8. Then you need to run `compute_features.py` to create the cutset
  - this is done by the `compute_features_for_cuts()` function in the main() function

# Other documentation
### analysis-folder:
- `parse.py`: 
  - functions for creating dataframes each containing all audio segments of a certain type (e.g. laughter, speech, etc.) - one per row. The columns for all these "segment dataframes" are the same 
    - Columns: ['meeting_id', 'part_id', 'chan_id', 'start', 'end', 'length', 'type', 'laugh_type']  

  - Additionally `parse.py` creates one other dataframe, called `info_df`. This dataframe contains general information about each meeting.
    - Columns of `info_df`: ['meeting_id', 'part_id', 'chan_id', 'length', 'path'] 

- `preproces.py`: functions for creating all the indices. An _index_ in this context is a nested mapping. Each index maps a participant in a certain meeting to all transcribed "audio segments" of a certain type (e.g. laughter, speech, etc.) recorded by this participant's microphone.
The "audio segments" are taken from the dataframes created in `parse.py`. Each segment is turned into an "openclosed"-interval which is a datatype provided by the Portion library (TODO link portion library).
These intervals are joined into a single disjunction. Portion allows normal interval operations on such disjunctions which simplifies future logic, e.g. detecting whether predictions overlap with transcribed events. 

All indices follow the same structure. They are defined as python dictionary of the following shape: 
```
  {
      meeting_id: {
          tot_len: INT,
          tot_events: INT,
          part_id: P.openclosed(start,end) | P.openclosed(start,end),
          part_id: P.openclosed(start,end) | P.openclosed(start,end)
          ...
      }
      meeting_id: {
        ...
      }
      ...
  }
```

