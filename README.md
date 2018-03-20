# Speech-to-text
Acoustic model development.
Trained several different neural networks aiming to improve speech-to-text recognition.
Input is a sound file that was converted to spectogram.
Output is a printed text of what was said in the audio file.

Models of neural networks based on RNN, CNN and combination of RNN and CNN were trained.
Addition of batch normalization and TimeDistributed wrappers significantly improved the performance.

Found out that including CNN layer for spatial  feature extraction from the input layer significantly improves the model performance. Various kernel sizes and strides were tested. Optimal values were 11 or 13 for kernel size and 2 for the stride.

The best model performance (the lowest Loss on the Validation data set) was obtained using a deep learning model that combined a single CNN layer and 3 bidirectional RNN layers using GRU cells. Using dropouts was critical to avoid overfitting (model_5 demonstrates overfitting).
