import unittest
from Utils.DataReader import DataReader
from Utils.OutliersRemoval import OutliersRemoval
from Conf.Settings import TYPICAL_PATH, ASD_PATH
class MyTestCase(unittest.TestCase):
    def test_dataReader(self):
        reader = DataReader()
        removal = OutliersRemoval(cutoff=150)
        data_game, _ = reader.readGameResults(ASD_PATH)
        data_gaze, _, _ = reader.readGazeData(ASD_PATH)
        self.assertEqual(len(data_game), len(data_gaze))



if __name__ == '__main__':
    unittest.main()
