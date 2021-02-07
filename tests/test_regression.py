#!/usr/bin/env python3
import unittest

from src.regression import SimpleLinearRegression


class TestRegression(unittest.TestCase):

    @unittest.skip("Skip this test")
    def test_simple_objective_function(self):
        default_b=[1,1]
        slr = SimpleLinearRegression(default_b)
        # X=[1,2,3,4,5,6,7,8,9,10]
        # y=[2,5,5,8,11,12,14,16,17,21]
        b,X,y=[1,1],[1,2],[2,5]
        self.assertEqual(slr.objective(b,X,y),4)

        b,X,y=[0,2],[1,2],[2,5] # 1
        self.assertEqual(slr.objective(b,X,y),1)
    
    def test_simple_objective_train(self):
        slr = SimpleLinearRegression([1,1])
        X=[1,2,3,4,5,6,7,8,9,10]
        y=[2,5,5,8,11,12,14,16,17,21]

        self.assertTrue(slr.train(X,y))
    
    # @unittest.skip("Skip this test")
    def test_simple_objective_test(self):
        slr = SimpleLinearRegression([1,1])
        X=[1,2,3,4,5,6,7,8,9,10]
        y=[2,5,5,8,11,12,14,16,17,21]

        slr.train(X,y)
        
        self.assertTrue(21<=slr.test(11)<=23)
