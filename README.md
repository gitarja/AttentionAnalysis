# AttentionAnalysis

We use this program to analyze the response and gaze behavior of subjects when they play Go/NoGo task.
We represent each subject using two-type of features: spatial and gaze-adjustment features.

## Spatial features

Spatial features comprised of 24 index.

| No | ID               | Detail                                                                           |
|----|------------------|----------------------------------------------------------------------------------|
| 1  | Go-positive      | The percentage of Go response                                                    |
| 2  | Go-negative      | The percentage of Go-negative response                                           |
| 3  | NoGo-positive    | The percentage of NoGo response                                                  |
| 4  | NoGo-negative    | The percentage of NoGo-negative response                                         |
| 5  | RT               | The average of a subject response time                                           |
| 6  | RT-var           | The standard deviation of a subject response time                                |
| 7  | Trajectory-area  | The gaze trajectory area                                                         |
| 8  | Velocity-avg     | The average instantaneous velocity of subjects' gaze                             |
| 9  | Velocity-var     | The standard deviation of the instantaneous velocity  of subjects' gaze          |
| 10 | Acceleration-avg | The average acceleration of subjects' gaze                                       |
| 11 | Acceleration-var | The standard deviation of the velocity    of subjects' gaze along the y-axis     |
| 12 | Fixation-avg     | The average of subjects' fixation time                                           |
| 13 | Fixation-var     | The standard deviation  of   subjects' fixation time                             |
| 14 | Distance-avg     | The average of gaze distance                                                     |
| 15 | Distance-var     | The standard deviation of gaze distance                                          |
| 16 | Angle-avg        | The average of gaze angle                                                        |
| 17 | Angle-var        | The standard deviation of gaze angle                                             |
| 18 | Distance-sen     | Sample entropy of subjects' gaze distance                                        |
| 19 | Angle-sen        | Sample entropy of subjects' gaze angle                                           |
| 20 | Velocity-sen     | Sampe entropy of gaze velocity                                                   |
| 21 | Spatial-en       | The entropy of subjects' gaze                                                    |
| 22 | Gaze-obj-en      | The entropy of the distance between subjects' gaze and stimulus position         |
| 23 | Gaze-obj-sen     | Sample entropy  of the distance   between subjects' gaze and stimulus position   |
| 24 | Gaze-obj-spe     | Spectral entropy of  the distance   between subjects' gaze and stimulus position |
## Gaze-adjustment Features
We analyzed the gaze-adjustment of participants for every type of response. Gaze-adjustment was the Euclidean distance between stimulus position and the subject's gaze when the stimulus appeared until it disappeared; its value ranged from 0 to sqrt(2). Since each gaze-adjustment's span differed, we represented each gaze-adjustment as Auto-regressive parameters. The model's lag **L** was set to five (average of AIC: 126.77), thereby each gaze-adjustment was represented by six variables.

## Data analysis
We performed classification using AdaBoost with DecisionTree as the base classifier. The hyperparameter values of the AdaBoost algorithm and the classifier were optimized using grid-search. Three-fold cross validation was performed to evaluate the algorithm's performance.

## Folder structure
1. Analysis: codes to perform statistical analysis and classification
2. Conf: constant variables (please modify the variables to match with your environment)
3. DNNModels: Siamese neural network to discriminate ASD from typical ones (under progress)
4. GazeAdjustment: codes to extract gaze adjustment features
5. Spatial: codes to extract spatial features
6. Testing: testing files
7. Utils: library 