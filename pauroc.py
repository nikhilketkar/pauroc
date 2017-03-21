import numpy
from scipy.interpolate import interp1d
from scipy.integrate import quad

class PAUROCException(Exception): pass
class FPRArrayNotFloat(PAUROCException): pass
class TPRArrayNotFloat(PAUROCException): pass
class FPRRangeStartNotFloat(PAUROCException): pass
class FPRRangeEndNotFloat(PAUROCException): pass
class FPRArrayNonMonotonic(PAUROCException): pass
class TPRArrayNonMonotonic(PAUROCException): pass
class TPRFPRArraySizeUnequal(PAUROCException): pass
class FPRArrayLessThanTwo(PAUROCException): pass
class TPRArrayLessThanTwo(PAUROCException): pass
class FPRRangeStartPointNegative(PAUROCException): pass
class FPRRangeStartPointGreaterThanOne(PAUROCException):pass
class FPRRangeEndPointNegative(PAUROCException):pass
class FPRRangeEndPointGreaterThanOne(PAUROCException):pass
class FPRRangeInvalid(PAUROCException):pass
class FPRArrayHasNan(PAUROCException):pass
class TPRArrayHasNan(PAUROCException):pass

def pauroc(tpr, fpr, fpr_range_start, fpr_range_end):
    if not numpy.issubdtype(fpr.dtype,numpy.float): raise FPRArrayNotFloat
    if not numpy.issubdtype(tpr.dtype,numpy.float): raise TPRArrayNotFloat
    if not isinstance(fpr_range_start, float): raise FPRRangeStartNotFloat
    if not isinstance(fpr_range_end, float): raise FPRRangeEndNotFloat
    if len(fpr) < 2: raise FPRArrayLessThanTwo
    if len(tpr) < 2: raise TPRArrayLessThanTwo
    if len(tpr) != len(fpr): raise TPRFPRArraySizeUnequal
    if numpy.all(numpy.diff(fpr) < 0.0): raise FPRArrayNonMonotonic
    if numpy.all(numpy.diff(tpr) < 0.0): raise TPRArrayNonMonotonic
    if fpr_range_start < 0: raise FPRRangeStartPointNegative
    if fpr_range_start > 1: raise FPRRangeStartPointGreaterThanOne
    if fpr_range_end < 0: raise FPRRangeEndPointNegative
    if fpr_range_end > 1: raise FPRRangeEndPointGreaterThanOne
    if fpr_range_start > fpr_range_end: raise FPRRangeInvalid
    if numpy.any(numpy.isnan(fpr)): raise FPRArrayHasNan
    if numpy.any(numpy.isnan(tpr)): raise TPRArrayHasNan

    fpr_padded = numpy.concatenate([[0.0], fpr, [1.0]])
    tpr_padded = numpy.concatenate([[0.0], tpr, [1.0]])
    f = interp1d(fpr_padded, tpr_padded)
    result, error = quad(f, fpr_range_start, fpr_range_end, limit=1000+2)
    return result

