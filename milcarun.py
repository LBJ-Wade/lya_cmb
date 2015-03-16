import numpy as np
import healpy as hp
import matplotlib.pylab as plt
import pyfits


def columnplot(ninbin, xcinbin, maxang, n, islog):
    """
    Create a column plot:
    1. Number of pixel - qso pairs in each bin vs. angular distance
    2. Cross correlation
    Input: xcinbin should in units of y; n is the number of bins
    """

    atr = np.pi / 180. / 60

    if islog:
        logx = np.linspace(np.log(0.0001), np.log(np.pi), n)
        x = np.exp(logx) / atr
    else:
        x = np.linspace(0, np.pi, n) / atr

    xinmK = xcinbin * 2. * 2.73e6 / ninbin

    plt.figure(figsize = (10,6))
    plt.subplot(1,2,1)
    plt.plot(x, ninbin)
    plt.xlabel("Angular distance [arcmin]")
    plt.ylabel("Number of pixel - qso pairs")
    plt.xlim((0, maxang))

    plt.subplot(1,2,2)
    plt.plot(x, xinmK)
    plt.axhline(0, color='black')
    plt.xlabel("Angular distance [arcmin]")
    plt.ylabel(r"$\langle\Delta T\;\Delta F\rangle\;[\mathrm{\mu K}]$")
    plt.xlim((0, maxang))
    plt.subplots_adjust(wspace = 0.3)

    plt.show()

# --- Specify constants ------------------------------------------------------

NSIDE = 2048

# --- Specify variables ------------------------------------------------------

argsmask = 3            # Foreground mask: 0 - 40%, 1 - 50%, 2 - 60%, 3 - 70%

argsislog = 0           # Log or lin bins

argsangmax = 30.        # Maximum angle to be measured

argsnhocells = 1000     # Number of cells in 1 dimension

argsblen = 2.           # Box length

if argsislog:
    argsnbins = 30
else:
    argsnbins = 4000    # Number of bins (dept. of argsislog)

# --- Load the data ----------------------------------------------------------

comb = np.load('/home/hongyuz/lya_cmb/data/qsosum.npy')
hdulist_mask = pyfits.open('/home/hongyuz/lya_cmb/COM_CompMap_YSZ_R2.00.fits/masks.fits')
hdulist_nilc = pyfits.open('/home/hongyuz/lya_cmb/COM_CompMap_YSZ_R2.00.fits/milca_ymaps.fits')

maps = hdulist_nilc[1].data.field(0)

mask_sky = hdulist_mask[1].data.field(argsmask)
# foreground mask: 0 - 40%, 1 - 50%, 2 - 60%, 3 - 70%
ptsrc = hdulist_mask[1].data.field(4)

# --- Mask cmb ---------------------------------------------------------------

mask = mask_sky * ptsrc
maps[mask == 0] = np.nan

# --- Mask qso ---------------------------------------------------------------

r = hp.rotator.Rotator(coord=['C','G'], deg=False)

thetac = (90 - comb[:,2]) / 180 * np.pi
phic = comb[:,1] / 180 * np.pi

thetag, phig = r(thetac, phic)
bosspix = hp.ang2pix(NSIDE, thetag, phig, nest=False)

comb_mask = comb[np.logical_not(np.isnan(maps[bosspix]))]
thetac_mask = (90 - comb_mask[:,2]) / 180 * np.pi
phic_mask = comb_mask[:,1] / 180 * np.pi

thetag_mask, phig_mask = r(thetac_mask, phic_mask)

# --- Plot cmb and qsos ------------------------------------------------------

ax = hp.mollview(maps, nest=False, min=-2e-6, max=2e-6)
ax = hp.projscatter(thetag_mask, phig_mask)

# --- Normalize along redshifts ----------------------------------------------

a = -2.910
b = 4.164
taueff_fg = 10. ** (a + b * np.log10(1 + comb_mask[:,0]))
ffg = np.exp(-1. * taueff_fg)

# --- Transform from (theta, phi) to (x, y, z) -------------------------------

thetap, phip = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)), nest=False)
maps_mask = maps[np.logical_not(np.isnan(maps))]
thetap_mask = thetap[np.logical_not(np.isnan(maps))]
phip_mask = phip[np.logical_not(np.isnan(maps))]

xp = np.sin(thetap_mask)*np.cos(phip_mask)
yp = np.sin(thetap_mask)*np.sin(phip_mask)
zp = np.cos(thetap_mask)

xq0 = np.sin(thetag_mask)*np.cos(phig_mask)
yq0 = np.sin(thetag_mask)*np.sin(phig_mask)
zq0 = np.cos(thetag_mask)
normflx0 = comb_mask[:,3]/ffg
pixlen0 = comb_mask[:,4]

# --- Possible filtering of data ---------------------------------------------

xq = xq0[np.logical_and(normflx0<5, normflx0>0)]
yq = yq0[np.logical_and(normflx0<5, normflx0>0)]
zq = zq0[np.logical_and(normflx0<5, normflx0>0)]
normflx = normflx0[np.logical_and(normflx0<5, normflx0>0)]
pixlen = pixlen0[np.logical_and(normflx0<5, normflx0>0)]

print 'Number of quasars:', xq.shape[0]

# --- IMPORTANT: Prepare data in the format (5/4, npix/nqso) -----------------

qso_pos_flux = np.row_stack((xq, yq, zq, normflx, pixlen))
cmb_pos_y = np.row_stack((xp, yp, zp, maps_mask))

# --- Run the .pyx file ------------------------------------------------------

from log_xc_mesh import cross_correlation

xcbin, xctot, time = cross_correlation(cmb_pos_y, qso_pos_flux, argsangmax, \
        argsnhocells, argsblen, argsnbins, argsislog)
print "Compute time: %fs" % time

# --- Save and Plot ----------------------------------------------------------

np.savetxt("xc.txt", np.column_stack((xcbin, xctot)))
columnplot(np.array(xcbin), np.array(xctot), argsangmax, argsnbins, argsislog)
