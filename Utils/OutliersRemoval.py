class OutliersRemoval:

    def __init__(self, cutoff=100):
        self.cutoff = cutoff

    def transformGameResult(self, data):
        '''
        :param data: game results data
        :return: game results data without outlier
        1. Varying Target Prevalence Reveals Two Dissociable Decision Criteria in Visual Search
        # 2. Motor processes in simple, go/no-go, and choice reaction time tasks: A psychophysiological analysis.
        '''
        transform_data = []
        for d in data:
            transform_data.append(d[(d["RT"] == -1) | (d["RT"] >= self.cutoff)])

        return transform_data

    def transformGazeHead(self, data, only_gaze=False):
        '''
        :param data: gaze data
        :param only_gaze: if only gaze, data are filtered only using GazeX and GazeY parameters
        :return: gaze data without outlier (Gaze >= 0, Object != 1)
        '''
        transform_data = []
        for d in data:
            if only_gaze:
                transform_data.append(d[(d["GazeX"] >= 0) & (
                        d["GazeY"] >= 0)])
            else:
                transform_data.append(d[(d["ObjectX"] != -1) & (d["ObjectY"] != -1) & (d["GazeX"] >= 0) & (
                        d["GazeY"] >= 0)])

        return transform_data
