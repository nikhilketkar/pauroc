import unittest
import pauroc

import sys
import numpy

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
        with self.assertRaises(pauroc.FPRArrayEmpty):
            pauroc.pauroc(tpr,fpr,fpr_start,fpr_end)

    def test_nonempty_tpr(self):
        fpr, tpr, fpr_start, fpr_end = self.gen_valid_input()
        tpr = numpy.array([])
        with self.assertRaises(pauroc.TPRArrayEmpty):
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

if __name__ == '__main__':
    unittest.main()


