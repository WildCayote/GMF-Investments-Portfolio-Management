import unittest

class TestEquality(unittest.TestCase):
    '''
    This test is created for the sake of having a test to run during ci
    '''
    def test_numbers_equal(self):
        self.assertEqual(9, 9)

if __name__ == '__main__':
    unittest.main()