IEEE AASP Challenge: Detection and Classification of Acoustic Scenes and Events 2016
http://www.cs.tut.fi/sgn/arg/dcase2016/

Task 2 - Synthetic audio sound event detection - Test Dataset (+annotations)
http://www.cs.tut.fi/sgn/arg/dcase2016/task-synthetic-sound-event-detection

Credits: Grégoire Lafay (IRCCYN, Ecole Cntrale de Nantes, France)

Contact: dcase-discussions@googlegroups.com

--

This README file documents the test dataset of the Synthetic Audio Sound Event Detection (SASED) task, including ground truth annotations.

--

Folder sound/ contains 54 audio files of 2min duration each. The format of the audio files is: raw PCM sampled at 44100 Hz, 16 bits (CD quality).

Folder annotation/ contains 54 .txt files, corresponding to the ground truth annotation for each respective audio recording. 
Each .txt file contains a list of detected sound events, specified by the onset, offset and the event class separated by a tab.

--

More information on Task 2:
http://www.cs.tut.fi/sgn/arg/dcase2016/task-sound-event-detection-in-synthetic-audio
