import numpy

class PAUROCException(Exception): pass
class FPRArrayNotFloat(PAUROCException): pass
class TPRArrayNotFloat(PAUROCException): pass
class FPRRangeStartNotFloat(PAUROCException): pass
class FPRRangeEndNotFloat(PAUROCException): pass
class FPRArrayNonMonotonic(PAUROCException): pass
class TPRArrayNonMonotonic(PAUROCException): pass
class TPRFPRArraySizeUnequal(PAUROCException): pass
class FPRArrayEmpty(PAUROCException): pass
class TPRArrayEmpty(PAUROCException): pass
class FPRRangeStartPointNegative(PAUROCException): pass
class FPRRangeStartPointGreaterThanOne(PAUROCException):pass
class FPRRangeEndPointNegative(PAUROCException):pass
class FPRRangeEndPointGreaterThanOne(PAUROCException):pass
class FPRRangeInvalid(PAUROCException):pass

def pauroc(tpr, fpr, fpr_range_start, fpr_range_end):
    if not numpy.issubdtype(fpr.dtype,numpy.float): raise FPRArrayNotFloat
    if not numpy.issubdtype(tpr.dtype,numpy.float): raise TPRArrayNotFloat
    if not isinstance(fpr_range_start, float): raise FPRRangeStartNotFloat
    if not isinstance(fpr_range_end, float): raise FPRRangeEndNotFloat
    if len(fpr) < 1: raise FPRArrayEmpty
    if len(tpr) < 1: raise TPRArrayEmpty
    if len(tpr) != len(fpr): raise TPRFPRArraySizeUnequal
    if numpy.all(numpy.diff(fpr) < 0.0): raise FPRArrayNonMonotonic
    if numpy.all(numpy.diff(tpr) < 0.0): raise TPRArrayNonMonotonic
    if fpr_range_start < 0: raise FPRRangeStartPointNegative
    if fpr_range_start > 1: raise FPRRangeStartPointGreaterThanOne
    if fpr_range_end < 0: raise FPRRangeEndPointNegative
    if fpr_range_end > 1: raise FPRRangeEndPointGreaterThanOne
    if fpr_range_start > fpr_range_end: raise FPRRangeInvalid

    def interpolate(given_x_array, given_y_array, given_x_value):
        insertion_point = numpy.searchsorted(given_x_array, given_x_value)

        if insertion_point == 0:
            x_before, x_after = 0.0, given_x_array[insertion_point]
            y_before, y_after = 0.0, given_y_array[insertion_point]
        elif insertion_point == len(given_x_array) - 1:
            x_before, x_after = given_x_array[insertion_point-1], given_x_array[insertion_point]
            y_before, y_after = given_y_array[insertion_point-1], given_x_array[insertion_point]
        elif insertion_point == len(given_x_array):
            x_before, x_after = given_x_array[insertion_point-1], 1.0
            y_before, y_after = given_y_array[insertion_point-1], 1.0
        else:
           x_before, x_after = given_x_array[insertion_point - 1], given_x_array[insertion_point + 1]
           y_before, y_after = given_y_array[insertion_point - 1], given_y_array[insertion_point + 1]

        m = (y_before - y_after)/(x_before - x_after)
        result = y_before + m * given_x_value
        return result, insertion_point

    def insert_2_sorted(given_array, element1, insertion_point1, element2, insertion_point2):
        temp = numpy.insert(given_array, insertion_point1, element1)
        return numpy.insert(temp, insertion_point2, element2)

    tpr_range_start, insertion_point1 = interpolate(fpr,tpr,fpr_range_start)
    tpr_range_end, insertion_point2 = interpolate(fpr,tpr,fpr_range_end)

    fpr_interpolated = insert_2_sorted(fpr, fpr_range_start, insertion_point1, fpr_range_end, insertion_point2)
    tpr_interpolated = insert_2_sorted(tpr, tpr_range_start, insertion_point1, tpr_range_end, insertion_point2)

    return numpy.trapz(tpr_interpolated[insertion_point1:insertion_point2+1],
                       fpr_interpolated[insertion_point1:insertion_point2+1])




