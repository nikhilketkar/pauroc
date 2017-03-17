import unittest
import pauroc

import numpy

class PAUROCTest(unittest.TestCase):
    def test_nonfloat_fpr(self):
        fpr = numpy.array([1,2,3])
        tpr = numpy.array([])
        fpr_start = 0.0
        fpr_end = 0.0
        with self.assertRaises(pauroc.FPRArrayNotFloat):
            pauroc.pauroc(tpr,fpr,fpr_start,fpr_end)

    def test_nonfloat_tpr(self):
        fpr = numpy.array([])
        tpr = numpy.array([1,2,3])
        fpr_start = 0.0
        fpr_end = 0.0
        with self.assertRaises(pauroc.TPRArrayNotFloat):
            pauroc.pauroc(tpr,fpr,fpr_start,fpr_end)

    def test_nonfloat_fpr_range_start(self):
        fpr = numpy.array([])
        tpr = numpy.array([])
        fpr_start = 1
        fpr_end = 0.0
        with self.assertRaises(pauroc.FPRRangeStartNotFloat):
            pauroc.pauroc(tpr,fpr,fpr_start,fpr_end)

    def test_nonfloat_fpr_range_end(self):
        fpr = numpy.array([])
        tpr = numpy.array([])
        fpr_start = 0.0
        fpr_end = 1
        with self.assertRaises(pauroc.FPRRangeEndNotFloat):
            pauroc.pauroc(tpr,fpr,fpr_start,fpr_end)

    def test_nonempty_fpr(self):
        fpr = numpy.array([])
        tpr = numpy.array([])
        fpr_start = 0.0
        fpr_end = 0.0
        with self.assertRaises(pauroc.FPRArrayEmpty):
            pauroc.pauroc(tpr,fpr,fpr_start,fpr_end)

    def test_nonempty_tpr(self):
        fpr = numpy.array([0.1,0.2])
        tpr = numpy.array([])
        fpr_start = 0.0
        fpr_end = 0.0
        with self.assertRaises(pauroc.TPRArrayEmpty):
            pauroc.pauroc(tpr,fpr,fpr_start,fpr_end)

    def test_tpr_fpr_equality(self):
        fpr = numpy.array([0.1,0.2])
        tpr = numpy.array([0.1])
        fpr_start = 0.0
        fpr_end = 0.0
        with self.assertRaises(pauroc.TPRFPRArraySizeUnequal):
            pauroc.pauroc(tpr,fpr,fpr_start,fpr_end)

    def test_fpr_range_start_lower(self):
        fpr = numpy.array([0.1,0.2])
        tpr = numpy.array([0.1,0.2])
        fpr_start = -0.001
        fpr_end = 0.0
        with self.assertRaises(pauroc.FPRRangeStartPointNegative):
            pauroc.pauroc(tpr,fpr,fpr_start,fpr_end)

    def test_fpr_range_start_upper(self):
        fpr = numpy.array([0.1,0.2])
        tpr = numpy.array([0.1,0.2])
        fpr_start = 1.001
        fpr_end = 0.0
        with self.assertRaises(pauroc.FPRRangeStartPointGreaterThanOne):
            pauroc.pauroc(tpr,fpr,fpr_start,fpr_end)

    def test_fpr_range_end_lower(self):
        fpr = numpy.array([0.1,0.2])
        tpr = numpy.array([0.1,0.2])
        fpr_start = 0.001
        fpr_end = -0.001
        with self.assertRaises(pauroc.FPRRangeEndPointNegative):
            pauroc.pauroc(tpr,fpr,fpr_start,fpr_end)

    def test_fpr_range_end_upper(self):
        fpr = numpy.array([0.1,0.2])
        tpr = numpy.array([0.1,0.2])
        fpr_start = 0.001
        fpr_end = 1.0001
        with self.assertRaises(pauroc.FPRRangeEndPointGreaterThanOne):
            pauroc.pauroc(tpr,fpr,fpr_start,fpr_end)

    def test_tpr_fpr_range_invalid(self):
        fpr = numpy.array([0.1,0.2])
        tpr = numpy.array([0.1,0.2])
        fpr_start = 0.002
        fpr_end = 0.001
        with self.assertRaises(pauroc.FPRRangeInvalid):
            pauroc.pauroc(tpr,fpr,fpr_start,fpr_end)

if __name__ == '__main__':
    unittest.main()


