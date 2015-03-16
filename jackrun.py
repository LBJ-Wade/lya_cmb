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

    dis_f = 3.1415926536

    if islog:
        logx = np.linspace(np.log(0.0001), np.log(dis_f), n)
        x = np.exp(logx) / atr
    else:
        x = np.linspace(0, dis_f, n)

    xinmK = xcinbin * 2. * 2.73e6 / ninbin

    plt.figure(figsize = (10,6))
    plt.subplot(1,2,1)
    plt.plot(x, ninbin)
    plt.xlabel("Angular distance [arcmin]")
    plt.ylabel("Number of pixel - qso pairs")
    plt.xlim((0, maxang))
    plt.ylim((0,1.5e7))

    plt.subplot(1,2,2)
    plt.plot(x, xinmK)
    plt.xlabel("Angular distance [arcmin]")
    plt.ylabel(r"$\langle\Delta T\;\Delta F\rangle\;[\mathrm{\mu K}]$")
    plt.xlim((0, maxang))
    plt.subplots_adjust(wspace = 0.2)

    plt.show()


NSIDE = 2048

# --- Specifiy variables -----------------------------------------------------

argsmask = 3            # Foreground mask: 0 - 40%, 1 - 50%, 2 - 60%, 3 - 70%

argsangmax = 30.        # Maximum angle to be measured

argsnhocells = 1000     # Number of cells in 1 dimension

argsblen = 2.           # Box length

argsnbins = 4000        # Number of bins

argsislog = 0           # Log or lin bins

argsnchunk = 100        # Number of chunks

# --- Load the data ----------------------------------------------------------

comb = np.load('/home/hongyuz/lya_cmb/data/qsosumweight.npy')
hdulist_mask = pyfits.open('/home/hongyuz/lya_cmb/COM_CompMap_YSZ_R2.00.fits/masks.fits')
hdulist_nilc = pyfits.open('/home/hongyuz/lya_cmb/COM_CompMap_YSZ_R2.00.fits/nilc_ymaps.fits')

maps = hdulist_nilc[1].data.field(0)
nilc_mask = hdulist_nilc[1].data['MASK']

mask_sky = hdulist_mask[1].data.field(argsmask)
# foreground mask: 0 - 40%, 1 - 50%, 2 - 60%, 3 - 70%
ptsrc = hdulist_mask[1].data.field(4)

# --- Mask cmb ---------------------------------------------------------------

mask = nilc_mask * mask_sky * ptsrc
maps[mask == 0] = np.nan

# --- Mask qso ---------------------------------------------------------------

r = hp.rotator.Rotator(coord=['C','G'], deg=False)

thetac = (90-comb[:,2])/180*np.pi
phic = comb[:,1]/180*np.pi

thetag, phig = r(thetac, phic)
bosspix = hp.ang2pix(NSIDE, thetag, phig, nest=False)

comb_mask = comb[np.logical_not(np.isnan(maps[bosspix]))]
thetac_mask = (90 - comb_mask[:,2])/180*np.pi
phic_mask = comb_mask[:,1]/180*np.pi

thetag_mask, phig_mask = r(thetac_mask, phic_mask)

# --- Normalize along redshifts ----------------------------------------------

a = -2.910
b = 4.164
taueff_fg = 10.**(a+b*np.log10(1+comb_mask[:,0]))
ffg = np.exp(-1.*taueff_fg)

# --- Transform from (theta, phi) to (x, y, z) -------------------------------

thetap, phip = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)), nest=False)
maps_mask = maps[np.logical_not(np.isnan(maps))]
thetap_mask = thetap[np.logical_not(np.isnan(maps))]
phip_mask = phip[np.logical_not(np.isnan(maps))]

xp = np.sin(thetap_mask)*np.cos(phip_mask)
yp = np.sin(thetap_mask)*np.sin(phip_mask)
zp = np.cos(thetap_mask)

xq = np.sin(thetag_mask)*np.cos(phig_mask)
yq = np.sin(thetag_mask)*np.sin(phig_mask)
zq = np.cos(thetag_mask)
normflx = comb_mask[:,3]/ffg
pixlen = comb_mask[:,4]

# --- IMPORTANT: Prepare data in the format (4, npix/nqso) -------------------

qso_pos_flux0 = np.row_stack((xq, yq, zq, normflx, pixlen))
cmb_pos_y = np.row_stack((xp, yp, zp, maps_mask))

print "Data has been prepared"

# --- Run the .pyx file ------------------------------------------------------

from log_xc_mesh import cross_correlation

xcbin = np.zeros((argsnbins, argsnchunk))
xctot = np.zeros((argsnbins, argsnchunk))

# --- Jackknife --------------------------------------------------------------

qso_pos_flux = qso_pos_flux0[:,qso_pos_flux0[0,:].argsort()].copy()

ind = np.array_split(np.arange(qso_pos_flux.shape[1]), argsnchunk)

for i in range(argsnchunk):
    indind = range(argsnchunk)
    del indind[i]
    jackknifeind = np.hstack((ind[i] for i in indind))
    qso_pos_flux_chunk = qso_pos_flux[:,jackknifeind].copy()

    xcbin[:,i], xctot[:,i], time = cross_correlation(cmb_pos_y, qso_pos_flux_chunk,\
            argsangmax, argsnhocells, argsblen, argsnbins, argsislog)
    print "Compute time: %fs" % time

# --- Save and Plot ----------------------------------------------------------

np.savetxt("Jackknife.txt", xctot)
# columnplot(np.array(xcbin), np.array(xctot), argsangmax, argsnbins, argsislog)
