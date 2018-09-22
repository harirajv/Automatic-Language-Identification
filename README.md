# Automatic Language Identification
This project makes use of Mel Frequency Cepstral coefficients extracted from WAV files of speech recordings to identify the language used by the speaker.

## Mel Frequency Cepstral Coefficients
The Mel Frequency cepstral coefficients are one of the best feature representations for speech signals and most suited for speech recognition tasks. It is based on a cosine transform on the short term power spectrum of the signal. 

The mel.py file extracts MFCC features from a WAV file. These are exported to a CSV file which can then be used for classification tasks. Two classifiers that use these features - SVM and Naive-Bayes are included in the repository.
