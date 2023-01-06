# import the necessary modules (e.g. unittest, mock, etc.)
import unittest
import search_frontend
from search_frontend import search_body, search_title, search_anchor, get_pagerank
from search_backend import SearchMaster


# define a test case class that subclasses unittest.TestCase
class MyTestCase(unittest.TestCase):

    # define a setUp method to create any necessary test resources
    def setUp(self):
        # create test resources here
        self.search_master = SearchMaster()

    # define one or more test methods that begin with "test_"
    def test_title(self):
        # call the function being tested and assert that the output is correct
        query = 'Anarchism'
        result = self.search_master.get_relevant_titles(query)
        self.assertEqual(result, [(12, 'Anarchism')])

    # define one or more test methods that begin with "test_"
    def test_body(self):
        # call the function being tested and assert that the output is correct
        query = 'academy of the year'
        result = self.search_master.body_search(query)
        len_result = len(result)
        self.assertTrue(2 < len_result <= 100)

    # define one or more test methods that begin with "test_"
    def test_anchor(self):
        # call the function being tested and assert that the output is correct
        query = 'Anarchism'
        result = self.search_master.get_relevant_anchors(query)
        print(result)
        self.assertGreater(len(result), 0)

    # define one or more test methods that begin with "test_"
    def test_pagerank(self):
        # call the function being tested and assert that the output is correct
        wiki_ids = [12, 25]
        result = self.search_master.get_pagerank(wiki_ids)
        print(result)
        self.assertEqual(len(result), 2)

    # define a tearDown method to clean up any test resources
    def tearDown(self):
        # clean up test resources here
        pass


# create a main function to run the tests
def main():
    unittest.main()


# run the main function if this module is run directly
if __name__ == '__main__':
    main()
