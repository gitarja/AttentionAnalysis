# AttentionAnalysis

We use this program to analyze the response and gaze behavior of subjects when they play Go/NoGo task.
We represent each subject using two-type of features: spatial and gaze-adjustment features.

## Spatial features

Spatial features comprised of 19 index.

| No | ID               | Detail                                                                             |
|----|------------------|------------------------------------------------------------------------------------|
| 1  | Go-positive      | The percentage of Go response                                                      |
| 2  | Go-negative      | The percentage of Go-negative response                                             |
| 3  | NoGo-positive    | The percentage of NoGo response                                                    |
| 4  | NoGo-negative    | The percentage of NoGo-negative response                                           |
| 5  | RT               | The average of a subject response time                                             |
| 6  | RTVar            | The standard deviation of a subject response time                                  |
| 7  | TrajectoryArea   | The gaze trajectory area                                                           |
| 8  | VelocityX_avg    | The average velocity of a subject's gaze along the x-axis                          |
| 9  | VelocityY_avg    | The average velocity  of a   subject's gaze along the y-axis                       |
| 10 | VelocityX_std    | The standard deviation of the velocity    of a subject's gaze along the x-axis     |
| 11 | VelocityY_std    | The standard deviation of the velocity    of a subject's gaze along the y-axis     |
| 12 | Fixation_avg     | The average of a subject's fixation time                                           |
| 13 | Fixation_std     | The standard deviation  of a   subject's fixation time                             |
| 14 | Sampen_dist      | Sample entropy of a subject's gaze distance                                        |
| 15 | Sampen_angle     | Sample entropy of a subject's gaze angle                                           |
| 16 | Spatial_entropy  | The entropy of a subject's gaze                                                    |
| 17 | GazeObj_entropy  | The entropy of the distance between a subject's gaze and stimulus   position       |
| 18 | Sampen_gaze_obj  | Sample entropy  of the distance   between a subject's gaze and stimulus position   |
| 19 | Spectral_entropy | Spectral entropy of  the distance   between a subject's gaze and stimulus position |


## Gaze-adjustment Features
We analyzed the gaze-adjustment of participants for every type of response. Gaze-adjustment was the Euclidean distance between stimulus position and the subject's gaze when the stimulus appeared until it disappeared; its value ranged from 0 to sqrt(2). Since each gaze-adjustment's span differed, we represented each gaze-adjustment as Auto-regressive parameters. The model's lag **L** was set to five (average of AIC: 126.77), thereby each gaze-adjustment was represented by six variables.
