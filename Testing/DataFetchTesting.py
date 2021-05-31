import unittest
from DNNModels.DataFetch import DataFetch

class MyTestCase(unittest.TestCase):
    def test_something(self):
        data_fetch = DataFetch("D:\\usr\\pras\\data\\AttentionTestData\\Collaboration\\High-risk\\", "_gazeHeadPose.csv", "_gameResults.csv", "D:\\usr\\pras\\data\\AttentionTestData\\Collaboration\\Turns_children.csv")
        self.assertNotEqual(data_fetch.loadData(), None)


if __name__ == '__main__':
    unittest.main()
