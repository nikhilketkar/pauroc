import unittest
import pauroc

import sys
import numpy
from sklearn.metrics import roc_auc_score, roc_curve

class PAUROCTest(unittest.TestCase):
    def gen_valid_input(self):
        fpr = numpy.array([0.1, 0.2, 0.3])
        tpr = numpy.array([0.1, 0.2, 0.3])
        fpr_start = 0.1
        fpr_end = 0.4
        return fpr, tpr, fpr_start, fpr_end

    def test_nonfloat_fpr(self):
        fpr, tpr, fpr_start, fpr_end = self.gen_valid_input()
        fpr = numpy.array([1,2,3])
        with self.assertRaises(pauroc.FPRArrayNotFloat):
            pauroc.pauroc(tpr,fpr,fpr_start,fpr_end)

    def test_nonfloat_tpr(self):
        fpr, tpr, fpr_start, fpr_end = self.gen_valid_input()
        tpr = numpy.array([1,2,3])
        with self.assertRaises(pauroc.TPRArrayNotFloat):
            pauroc.pauroc(tpr,fpr,fpr_start,fpr_end)

    def test_nonfloat_fpr_range_start(self):
        fpr, tpr, fpr_start, fpr_end = self.gen_valid_input()
        fpr_start = 1
        with self.assertRaises(pauroc.FPRRangeStartNotFloat):
            pauroc.pauroc(tpr,fpr,fpr_start,fpr_end)

    def test_nonfloat_fpr_range_end(self):
        fpr, tpr, fpr_start, fpr_end = self.gen_valid_input()
        fpr_end = 1
        with self.assertRaises(pauroc.FPRRangeEndNotFloat):
            pauroc.pauroc(tpr,fpr,fpr_start,fpr_end)

    def test_nonempty_fpr(self):
        fpr, tpr, fpr_start, fpr_end = self.gen_valid_input()
        fpr = numpy.array([])
        with self.assertRaises(pauroc.FPRArrayLessThanTwo):
            pauroc.pauroc(tpr,fpr,fpr_start,fpr_end)

    def test_nonempty_tpr(self):
        fpr, tpr, fpr_start, fpr_end = self.gen_valid_input()
        tpr = numpy.array([])
        with self.assertRaises(pauroc.TPRArrayLessThanTwo):
            pauroc.pauroc(tpr,fpr,fpr_start,fpr_end)

    def test_tpr_fpr_equality(self):
        fpr, tpr, fpr_start, fpr_end = self.gen_valid_input()
        tpr = tpr[0:-1]
        with self.assertRaises(pauroc.TPRFPRArraySizeUnequal):
            pauroc.pauroc(tpr,fpr,fpr_start,fpr_end)

    def test_monotonic_fpr(self):
        fpr, tpr, fpr_start, fpr_end = self.gen_valid_input()
        fpr = numpy.array([0.3, 0.2, 0.1])
        with self.assertRaises(pauroc.FPRArrayNonMonotonic):
            pauroc.pauroc(tpr,fpr,fpr_start,fpr_end)

    def test_monotonic_tpr(self):
        fpr, tpr, fpr_start, fpr_end = self.gen_valid_input()
        tpr = numpy.array([0.3, 0.2, 0.1])
        with self.assertRaises(pauroc.TPRArrayNonMonotonic):
            pauroc.pauroc(tpr,fpr,fpr_start,fpr_end)

    def test_fpr_range_start_lower(self):
        fpr, tpr, fpr_start, fpr_end = self.gen_valid_input()
        fpr_start = -0.001
        with self.assertRaises(pauroc.FPRRangeStartPointNegative):
            pauroc.pauroc(tpr,fpr,fpr_start,fpr_end)

    def test_fpr_range_start_upper(self):
        fpr, tpr, fpr_start, fpr_end = self.gen_valid_input()
        fpr_start = 1.001
        with self.assertRaises(pauroc.FPRRangeStartPointGreaterThanOne):
            pauroc.pauroc(tpr,fpr,fpr_start,fpr_end)

    def test_fpr_range_end_lower(self):
        fpr, tpr, fpr_start, fpr_end = self.gen_valid_input()
        fpr_end = -0.001
        with self.assertRaises(pauroc.FPRRangeEndPointNegative):
            pauroc.pauroc(tpr,fpr,fpr_start,fpr_end)

    def test_fpr_range_end_upper(self):
        fpr, tpr, fpr_start, fpr_end = self.gen_valid_input()
        fpr_end = 1.0001
        with self.assertRaises(pauroc.FPRRangeEndPointGreaterThanOne):
            pauroc.pauroc(tpr,fpr,fpr_start,fpr_end)

    def test_tpr_fpr_range_invalid(self):
        fpr, tpr, fpr_end, fpr_start = self.gen_valid_input()
        with self.assertRaises(pauroc.FPRRangeInvalid):
            pauroc.pauroc(tpr,fpr,fpr_start,fpr_end)

    def test_rectange_exact(self):
        fpr = numpy.array([0.2,0.4,0.6,0.8])
        tpr = numpy.array([0.8,0.8,0.8,0.8])
        fpr_start = 0.4
        fpr_end = 0.6
        result = pauroc.pauroc(tpr,fpr,fpr_start,fpr_end)
        self.assertAlmostEqual(result, 0.16, 10)

    def test_rectange_left_interpolation(self):
        fpr = numpy.array([0.2,0.4,0.6,0.8])
        tpr = numpy.array([0.8,0.8,0.8,0.8])
        fpr_start = 0.3
        fpr_end = 0.6
        result = pauroc.pauroc(tpr,fpr,fpr_start,fpr_end)
        self.assertAlmostEqual(result, 0.24, 10)

    def test_rectange_right_interpolation(self):
        fpr = numpy.array([0.2,0.4,0.6,0.8])
        tpr = numpy.array([0.8,0.8,0.8,0.8])
        fpr_start = 0.4
        fpr_end = 0.7
        result = pauroc.pauroc(tpr,fpr,fpr_start,fpr_end)
        self.assertAlmostEqual(result, 0.24, 10)

    def test_rectange_both_side_interpolation(self):
        fpr = numpy.array([0.2,0.4,0.6,0.8])
        tpr = numpy.array([0.8,0.8,0.8,0.8])
        fpr_start = 0.3
        fpr_end = 0.7
        result = pauroc.pauroc(tpr,fpr,fpr_start,fpr_end)
        self.assertAlmostEqual(result, 0.32, 10)

    def test_left_triangle_interpolation(self):
        fpr = numpy.array([0.2,0.4,0.6,0.8])
        tpr = numpy.array([0.8,0.8,0.8,0.8])
        fpr_start = 0.1
        fpr_end = 0.4
        result = pauroc.pauroc(tpr,fpr,fpr_start,fpr_end)
        self.assertAlmostEqual(result, 0.22, 10)

    def test_two_points(self):
        fpr = numpy.array([0.2,0.4])
        tpr = numpy.array([0.8,0.8])
        fpr_start = 0.1
        fpr_end = 0.4
        result = pauroc.pauroc(tpr,fpr,fpr_start,fpr_end)
        self.assertAlmostEqual(result, 0.22, 10)

    def test_scikit_learn_auc(self):
        y_true = numpy.round(numpy.random.rand(1000,1))
        y_score = numpy.random.rand(1000,1)
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        fpr_start = 0.0
        fpr_end = 1.0
        pauroc_result = pauroc.pauroc(tpr,fpr,fpr_start,fpr_end)
        auc_result = roc_auc_score(y_true, y_score)
        self.assertAlmostEqual(pauroc_result, auc_result, 2)

    def test_nan_in_tpr(self):
        fpr = numpy.array([0.2,0.3,0.4])
        tpr = numpy.array([0.8,numpy.nan,0.8])
        fpr_start = 0.1
        fpr_end = 0.4
        with self.assertRaises(pauroc.TPRArrayHasNan):
            pauroc.pauroc(tpr,fpr,fpr_start,fpr_end)

    def test_nan_in_fpr(self):
        fpr = numpy.array([0.2,numpy.nan,0.4])
        tpr = numpy.array([0.8,0.8,0.8])
        fpr_start = 0.1
        fpr_end = 0.4
        with self.assertRaises(pauroc.FPRArrayHasNan):
            pauroc.pauroc(tpr,fpr,fpr_start,fpr_end)

    def test_fpr_lesser_than_zero(self):
        fpr = numpy.array([-0.001, 0.2,0.4])
        tpr = numpy.array([0.8,0.8,0.8])
        fpr_start = 0.1
        fpr_end = 0.4
        with self.assertRaises(pauroc.FPRArrayHasValuesLessThanZero):
            pauroc.pauroc(tpr,fpr,fpr_start,fpr_end)

    def test_fpr_greater_than_one(self):
        fpr = numpy.array([0.2,0.4, 1.001])
        tpr = numpy.array([0.8,0.8,0.8])
        fpr_start = 0.1
        fpr_end = 0.4
        with self.assertRaises(pauroc.FPRArrayHasValuesGreaterThanOne):
            pauroc.pauroc(tpr,fpr,fpr_start,fpr_end)

    def test_tpr_lesser_than_zero(self):
        fpr = numpy.array([0.1, 0.2,0.4])
        tpr = numpy.array([-0.001,0.8,0.8])
        fpr_start = 0.1
        fpr_end = 0.4
        with self.assertRaises(pauroc.TPRArrayHasValuesLessThanZero):
            pauroc.pauroc(tpr,fpr,fpr_start,fpr_end)

    def test_tpr_greater_than_one(self):
        fpr = numpy.array([0.2,0.4,0.6])
        tpr = numpy.array([0.8,0.8,1.001])
        fpr_start = 0.1
        fpr_end = 0.4
        with self.assertRaises(pauroc.TPRArrayHasValuesGreterThanOne):
            pauroc.pauroc(tpr,fpr,fpr_start,fpr_end)

if __name__ == '__main__':
    unittest.main()


