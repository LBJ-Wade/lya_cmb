#-----------------------------------------------------------------------------
# Cross correlation of CMB pixels with quasar lyman alpha flux
#
# Author: Hongyu Zhu
# Date: 20 Feb 2015
#-----------------------------------------------------------------------------

# --- Python std lib imports -------------------------------------------------
from time import time
import numpy as np

# --- Cython cimports --------------------------------------------------------
cimport cython
from libc.stdint cimport uint32_t, uint64_t, int32_t, int64_t
from libc.math cimport sin, asin, log, sqrt

# --- Ctypedefs --------------------------------------------------------------
ctypedef double double_t
ctypedef uint64_t ulong_t
ctypedef uint32_t uint_t
ctypedef int32_t int_t
ctypedef int64_t long_t

# --- Compile-time definitions -----------------------------------------------
DEF xoff = -1
DEF yoff = -1
DEF zoff = -1
DEF atr = (3.1415926536 / 180.) / 60.                  # arcmin to radians

# ----------------------------------------------------------------------------
# Cython functions
# ----------------------------------------------------------------------------
@cython.cdivision(True)
cdef double_t distance(double_t xp, double_t yp, double_t zp, double_t xq, \
        double_t yq, double_t zq) nogil:
    """
    Calculate angular distance.
    """
    cdef:
        double_t dx = xp - xq
        double_t dy = yp - yq
        double_t dz = zp - zq
    return 2 * asin(sqrt(dx * dx + dy * dy + dz * dz) / 2)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double_t mean(double_t[::1] x):
    """
    Calculate mean in C speed.
    """
    cdef:
        int_t i
        double_t data_sum = 0
        ulong_t ndata = len(x)
    for i in xrange(ndata):
        data_sum += x[i]
    return data_sum / ndata

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int_t intmin(int_t x, int_t y):
    """
    Bound check function: make sure lower bound >= 0.
    """
    cdef int_t minint = x
    if x > y:
        minint = y
    return minint

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int_t intmax(int_t x, int_t y):
    """
    Bound check function: make sure upper bound <= nhocells.
    """
    cdef int_t maxint = x
    if x < y:
        maxint = y
    return maxint

@cython.boundscheck(False)
@cython.wraparound(False)
def init_cmb(double_t[:,::1] xp, uint_t nhocells, double_t blen):
    """
    Apply a chaining mesh on the whole sky and initialize cmb pixels to a
    linked list.
    Return the linked list (ll) memoryview and head of chain (hoc) memoryview.
    Note: the data xp should be prepared in the shape (4, npixs).
    """
    cdef:
        ulong_t plength = xp.shape[1]

        long_t [::1] ll = np.zeros(plength, dtype = np.int64)
        long_t [:,:,::1] hoc = np.zeros((nhocells, nhocells, nhocells),\
                dtype = np.int64)
        ulong_t i
        uint_t ix, iy, iz
        double_t xpi, ypi, zpi

    for i in xrange(plength):
        xpi, ypi, zpi = xp[0,i], xp[1,i], xp[2,i]
        ix = int(((xpi - xoff) / blen) * nhocells)
        iy = int(((ypi - yoff) / blen) * nhocells)
        iz = int(((zpi - zoff) / blen) * nhocells)

        ll[i] = hoc[ix, iy, iz]
        hoc[ix, iy, iz] = i

    return ll, hoc


@cython.boundscheck(False)
@cython.wraparound(False)
def cross_correlation(double_t[:,::1] xp, double_t[:,::1] xq, double_t angmax,\
        uint_t nhocells, double_t blen, int_t nbins, uint_t islog):
    """
    Cross_correlation in lin/log bin.
    Return the number of quasar - pixel pairs in each bin (xc_bin), cross
    correlation values in each bin (xc_tot) and the running time.
    Note: the data xp/xq should be prepared in the shape (4, npix/nqso).
    """
    cdef:
        double_t dis_i
        double_t dis_f = 3.1415926536
        double_t rlim = 2. * sin(angmax * atr * 0.5)
        int_t ncb = int((rlim / blen) * float(nhocells)) + 1

        double_t y_mean = mean(xp[3,:])
        double_t lya_mean = np.dot(xq[3,:],xq[4,:])/np.sum(xq[4,:])

        ulong_t plength = xp.shape[1]
        ulong_t qlength = xq.shape[1]

        ulong_t i
        uint_t ix, iy, iz

        int_t xb, yb, zb, xbm, ybm, zbm
        double_t xqi, yqi, zqi, vqi, nqi, xpi, ypi, zpi, vpi, xc_dis, t0 = time()

        long_t[::1] ll_cmb = np.zeros(plength, dtype = np.int64)
        long_t[:,:,::1] hoc_cmb = np.zeros((nhocells, nhocells, nhocells), \
                dtype = np.int64)

        long_t[::1] xc_bin = np.zeros(nbins, dtype = np.int64)
        double_t[::1] xc_tot = np.zeros(nbins, dtype=np.float64)

    if islog:
        dis_i = 0.0001
    else:
        dis_i = 0

    print 'The mean of normalized lya:', lya_mean

    ll_cmb, hoc_cmb = init_cmb(xp, nhocells, blen)

    for iq in xrange(qlength):

        xqi, yqi, zqi, vqi, nqi = xq[:,iq]

        ix = int(((xqi - xoff) / blen) * nhocells)
        iy = int(((yqi - yoff) / blen) * nhocells)
        iz = int(((zqi - zoff) / blen) * nhocells)

        xb = intmax(ix - ncb, 0)
        yb = intmax(iy - ncb, 0)
        zb = intmax(iz - ncb, 0)
        xbm = intmin(ix + ncb, nhocells)
        ybm = intmin(iy + ncb, nhocells)
        zbm = intmin(iz + ncb, nhocells)

        for p in xrange(xb, xbm):
            for q in xrange(yb, ybm):
                for r in xrange(zb, zbm):
                    if hoc_cmb[p, q, r] != 0:
                        i = hoc_cmb[p, q, r]
                        while True:
                            xpi, ypi, zpi, vpi = xp[0,i], xp[1,i], xp[2,i], xp[3,i]
                            xc_dis = distance(xpi, ypi, zpi, xqi, yqi, zqi)

                            if islog:
                                k = int(log(xc_dis / dis_i) * nbins / log(dis_f / dis_i))
                                if k < 0:
                                    k = 0
                            else:
                                k = int((xc_dis - dis_i) * nbins / (dis_f - dis_i))

                            xc_bin[k] += 1;
                            xc_tot[k] += nqi * (vqi / lya_mean - 1) * (vpi - y_mean);

                            if ll_cmb[i] != 0:
                                i = ll_cmb[i]
                            else:
                                break

    return xc_bin, xc_tot, time() - t0
