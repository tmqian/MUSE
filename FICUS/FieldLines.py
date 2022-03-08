import numpy as np
from netCDF4 import Dataset

import matplotlib.pyplot as plt
from matplotlib import cm

class FieldLine():
    
    def __init__(self, fin):
        
        self.field = SplineMagneticField.from_mgrid(fin)
        self.fname = fin
        
        
    def trace(self, N_line=12,  # 12 lines to be traced
                    N_circ=50,  # 50 revolutions of 2pi
                    N_save=4 ,  # 4 points per circle
                    r_min=0.34,
                    r_max=0.37  ):
    
        N_step = N_circ * N_save

        r0 = np.linspace(0.34,0.37,N_line)
        z0 = 0*r0
        phis = np.arange(N_step) *  (2*np.pi) / N_save
        
        r_lines, z_lines = field_line_integrate(r0,z0,phis, self.field)
        print( np.shape(r_lines) )
        
        self.r_lines = r_lines
        self.z_lines = z_lines
        self.phiaxis = phis
        
        self.N_line = N_line 
        self.N_circ = N_circ
        self.N_step = N_step
        self.r_min  = r_min
        self.r_max  = r_max
        
        

    def field_plot(self, phi=0, _legend=False, trim=0, skip_label=1, skip_plot=1, ms=2):

        nlines = self.N_line
        npoinc = int(self.N_step / self.N_circ)

        cmap = cm.rainbow(np.linspace(0, 1, nlines-trim+1))

        for surf in np.arange(0, nlines-trim, skip_plot):
            r = self.r_lines[phi::npoinc, surf]
            z = self.z_lines[phi::npoinc, surf]

            if (surf % skip_label == 0 and _legend):
                plt.plot(r, z, '.', color=cmap[surf], ms=ms, label='{} / {}'.format(surf, nlines))
            else:
                plt.plot(r, z, '.', color=cmap[surf], ms=ms)

        phiaxis = self.phiaxis
        angle = phi * phiaxis[-1]/np.pi/npoinc
        plt.title(r'$\varphi$ = %.2f $\pi$' % angle)

        if _legend:
            plt.legend(loc=2, fontsize=8)

        plt.axis('equal')
        plt.xlim(0.22,0.4)
        
    def show(self, size=(8,4), trim=0, skip=1):
        
        plt.figure(figsize=size)
        
        plt.subplot(1,2,1)
        self.field_plot(0, _legend=True, trim=trim, skip_label=skip)

        plt.subplot(1,2,2)
        self.field_plot(1 ,trim=trim,skip_label=skip)

        plt.suptitle(self.fname)



'''
    This code is borrowed from DESC

    13 December 2021
'''

from scipy.integrate import solve_ivp


class MagneticField():
    """Base class for all magnetic fields
    Subclasses must implement the "compute_magnetic_field" method
    """

    _io_attrs_ = []

    def __mul__(self, x):
        if np.isscalar(x):
            return ScaledMagneticField(x, self)
        else:
            return NotImplemented

    def __rmul__(self, x):
        return self.__mul__(x)

    def __add__(self, x):
        if isinstance(x, MagneticField):
            return SumMagneticField(self, x)
        else:
            return NotImplemented

    def __neg__(self):
        return ScaledMagneticField(-1, self)

    def __sub__(self, x):
        return self.__add__(-x)


    def compute_magnetic_field(self, coords, params=None, dR=0, dp=0, dZ=0):
        """Compute magnetic field at a set of points
        Parameters
        ----------
        coords : array-like shape(N,3) or Grid
            cylindrical coordinates to evaluate field at [R,phi,Z]
        params : tuple, optional
            parameters to pass to scalar potential function
        dR, dp, dZ : int, optional
            order of derivative to take in R,phi,Z directions
        Returns
        -------
        field : ndarray, shape(N,3)
            magnetic field at specified points, in cylindrical form [BR, Bphi,BZ]
        """

    def __call__(self, coords, params=None, dR=0, dp=0, dZ=0):
        return self.compute_magnetic_field(coords, params, dR, dp, dZ)


class SplineMagneticField(MagneticField):
    """Magnetic field from precomputed values on a grid
    Parameters
    ----------
    R : array-like, size(NR)
        R coordinates where field is specified
    phi : array-like, size(Nphi)
        phi coordinates where field is specified
    Z : array-like, size(NZ)
        Z coordinates where field is specified
    BR : array-like, shape(NR,Nphi,NZ)
        radial magnetic field on grid
    Bphi : array-like, shape(NR,Nphi,NZ)
        toroidal magnetic field on grid
    BZ : array-like, shape(NR,Nphi,NZ)
        vertical magnetic field on grid
    method : str
        interpolation method
    extrap : bool
        whether to extrapolate beyond the domain of known field values or return nan
    period : float
        period in the toroidal direction (usually 2pi/NFP)
    """

    _io_attrs_ = [
        "_R",
        "_phi",
        "_Z",
        "_BR",
        "_Bphi",
        "_BZ",
        "_method",
        "_extrap",
        "_period",
        "_derivs",
    ]

    def __init__(self, R, phi, Z, BR, Bphi, BZ, method="cubic", extrap=False, period=0):

        R, phi, Z = np.atleast_1d(R), np.atleast_1d(phi), np.atleast_1d(Z)
        assert R.ndim == 1
        assert phi.ndim == 1
        assert Z.ndim == 1
        BR, Bphi, BZ = np.atleast_3d(BR), np.atleast_3d(Bphi), np.atleast_3d(BZ)
        assert BR.shape == Bphi.shape == BZ.shape == (R.size, phi.size, Z.size)

        self._R = R
        self._phi = phi
        self._Z = Z
        self._BR = BR
        self._Bphi = Bphi
        self._BZ = BZ

        self._method = method
        self._extrap = extrap
        self._period = period

        self._derivs = {}
        self._derivs["BR"] = self._approx_derivs(self._BR)
        self._derivs["Bphi"] = self._approx_derivs(self._Bphi)
        self._derivs["BZ"] = self._approx_derivs(self._BZ)

    def _approx_derivs(self, Bi):
        tempdict = {}
        tempdict["fx"] = _approx_df(self._R, Bi, self._method, 0)
        tempdict["fy"] = _approx_df(self._phi, Bi, self._method, 1)
        tempdict["fz"] = _approx_df(self._Z, Bi, self._method, 2)
        tempdict["fxy"] = _approx_df(self._phi, tempdict["fx"], self._method, 1)
        tempdict["fxz"] = _approx_df(self._Z, tempdict["fx"], self._method, 2)
        tempdict["fyz"] = _approx_df(self._Z, tempdict["fy"], self._method, 2)
        tempdict["fxyz"] = _approx_df(self._Z, tempdict["fxy"], self._method, 2)
        return tempdict

    def compute_magnetic_field(self, coords, params=None, dR=0, dp=0, dZ=0):
        """Compute magnetic field at a set of points
        Parameters
        ----------
        coords : array-like shape(N,3) or Grid
            cylindrical coordinates to evaluate field at [R,phi,Z]
        params : tuple, optional
            parameters to pass to scalar potential function
        dR, dp, dZ : int, optional
            order of derivative to take in R,phi,Z directions
        Returns
        -------
        field : ndarray, shape(N,3)
            magnetic field at specified points, in cylindrical form [BR, Bphi,BZ]
        """

        if isinstance(coords, Grid):
            coords = coords.nodes
        coords = jnp.atleast_2d(coords)
        Rq, phiq, Zq = coords.T

        BRq = interp3d(
            Rq,
            phiq,
            Zq,
            self._R,
            self._phi,
            self._Z,
            self._BR,
            self._method,
            (dR, dp, dZ),
            self._extrap,
            (None, self._period, None),
            **self._derivs["BR"],
        )
        Bphiq = interp3d(
            Rq,
            phiq,
            Zq,
            self._R,
            self._phi,
            self._Z,
            self._Bphi,
            self._method,
            (dR, dp, dZ),
            self._extrap,
            (None, self._period, None),
            **self._derivs["Bphi"],
        )
        BZq = interp3d(
            Rq,
            phiq,
            Zq,
            self._R,
            self._phi,
            self._Z,
            self._BZ,
            self._method,
            (dR, dp, dZ),
            self._extrap,
            (None, self._period, None),
            **self._derivs["BZ"],
        )

        return jnp.array([BRq, Bphiq, BZq]).T

    @classmethod
    def from_mgrid(
        cls, mgrid_file, extcur=1, method="cubic", extrap=False, period=None
    ):
        """Create a SplineMagneticField from an "mgrid" file from MAKEGRID
        Parameters
        ----------
        mgrid_file : str or path-like
            path to mgrid file in netCDF format
        extcur : array-like
            currents for each subset of the field
        method : str
            interpolation method
        extrap : bool
            whether to extrapolate beyond the domain of known field values or return nan
        period : float
            period in the toroidal direction (usually 2pi/NFP)
        """
        mgrid = Dataset(mgrid_file, "r")
        ir = int(mgrid["ir"][()])
        jz = int(mgrid["jz"][()])
        kp = int(mgrid["kp"][()])
        nfp = mgrid["nfp"][()].data
        nextcur = int(mgrid["nextcur"][()])
        rMin = mgrid["rmin"][()]
        rMax = mgrid["rmax"][()]
        zMin = mgrid["zmin"][()]
        zMax = mgrid["zmax"][()]

        br = np.zeros([kp, jz, ir])
        bp = np.zeros([kp, jz, ir])
        bz = np.zeros([kp, jz, ir])
        extcur = np.broadcast_to(extcur, nextcur)
        for i in range(nextcur):

            # apply scaling by currents given in VMEC input file
            scale = extcur[i]

            # sum up contributions from different coils
            coil_id = "%03d" % (i + 1,)
            br[:, :, :] += scale * mgrid["br_" + coil_id][()]
            bp[:, :, :] += scale * mgrid["bp_" + coil_id][()]
            bz[:, :, :] += scale * mgrid["bz_" + coil_id][()]
        mgrid.close()

        # shift axes to correct order
        br = np.moveaxis(br, (0, 1, 2), (1, 2, 0))
        bp = np.moveaxis(bp, (0, 1, 2), (1, 2, 0))
        bz = np.moveaxis(bz, (0, 1, 2), (1, 2, 0))

        # re-compute grid knots in radial and vertical direction
        Rgrid = np.linspace(rMin, rMax, ir)
        Zgrid = np.linspace(zMin, zMax, jz)
        pgrid = 2.0 * np.pi / (nfp * kp) * np.arange(kp)
        if period is None:
            period = 2 * np.pi / (nfp)

        return cls(Rgrid, pgrid, Zgrid, br, bp, bz, method, extrap, period)

    @classmethod
    def from_field(
        cls, field, R, phi, Z, params=(), method="cubic", extrap=False, period=None
    ):
        """Create a splined magnetic field from another field for faster evaluation
        Parameters
        ----------
        field : MagneticField or callable
            field to interpolate. If a callable, should take a vector of
            cylindrical coordinates and return the field in cylindrical components
        R, phi, Z : ndarray
            1d arrays of interpolation nodes in cylindrical coordinates
        params : tuple, optional
            parameters passed to field
        method : str
            spline method for SplineMagneticField
        extrap : bool
            whether to extrapolate splines beyond specified R,phi,Z
        period : float
            period for phi coordinate. Usually 2pi/NFP
        """
        R, phi, Z = map(np.asarray, (R, phi, Z))
        rr, pp, zz = np.meshgrid(R, phi, Z, indexing="ij")
        shp = rr.shape
        coords = np.array([rr.flatten(), pp.flatten(), zz.flatten()]).T
        BR, BP, BZ = field(coords, *params).T
        return cls(
            R,
            phi,
            Z,
            BR.reshape(shp),
            BP.reshape(shp),
            BZ.reshape(shp),
            method,
            extrap,
            period,
        )


#####

def field_line_integrate(
    r0, z0, phis, field, params=(), rtol=1e-8, atol=1e-8, maxstep=1000
):
    """Trace field lines by integration
    Parameters
    ----------
    r0, z0 : array-like
        initial starting coordinates for r,z on phi=phis[0] plane
    phis : array-like
        strictly increasing array of toroidal angles to output r,z at
        Note that phis is the geometric toroidal angle for positive Bphi,
        and the negative toroidal angle for negative Bphi
    field : MagneticField
        source of magnetic field to integrate
    params: tuple
        parameters passed to field
    rtol, atol : float
        relative and absolute tolerances for ode integration
    maxstep : int
        maximum number of steps between different phis
    Returns
    -------
    r, z : ndarray
        arrays of r, z coordinates at specified phi angles
    """
    r0, z0, phis = map(jnp.asarray, (r0, z0, phis))
    assert r0.shape == z0.shape, "r0 and z0 must have the same shape"
    rshape = r0.shape
    r0 = r0.flatten()
    z0 = z0.flatten()
    x0 = jnp.array([r0, phis[0] * jnp.ones_like(r0), z0]).T

    @jit
    def odefun(rpz, s):
        rpz = rpz.reshape((3, -1)).T
        r = rpz[:, 0]
        br, bp, bz = field.compute_magnetic_field(rpz, params).T
        return jnp.array(
            [r * br / bp * jnp.sign(bp), jnp.sign(bp), r * bz / bp * jnp.sign(bp)]
        ).squeeze()

    intfun = lambda x: odeint(odefun, x, phis, rtol=rtol, atol=atol, mxstep=maxstep)
    x = jnp.vectorize(intfun, signature="(k)->(n,k)")(x0)
    r = x[:, :, 0].T.reshape((len(phis), *rshape))
    z = x[:, :, 2].T.reshape((len(phis), *rshape))
    return r, z

#####

from jax import numpy as jnp
from jax import jit
from jax.experimental.ode import odeint
from collections import OrderedDict
import numbers

def _approx_df(x, f, method, axis, **kwargs):
    """Approximates derivatives for cubic spline interpolation"""

    if method == "cubic":
        dx = jnp.diff(x)
        df = jnp.diff(f, axis=axis)
        dxi = jnp.where(dx == 0, 0, 1 / dx)
        if df.ndim > dxi.ndim:
            dxi = jnp.expand_dims(dxi, tuple(range(1, df.ndim)))
            dxi = jnp.moveaxis(dxi, 0, axis)
        df = dxi * df
        fx = jnp.concatenate(
            [
                jnp.take(df, [0], axis, mode="wrap"),
                1
                / 2
                * (
                    jnp.take(df, jnp.arange(0, df.shape[axis] - 1), axis, mode="wrap")
                    + jnp.take(df, jnp.arange(1, df.shape[axis]), axis, mode="wrap")
                ),
                jnp.take(df, [-1], axis, mode="wrap"),
            ],
            axis=axis,
        )
        return fx
    if method == "cubic2":
        dx = jnp.diff(x)
        df = jnp.diff(f, axis=axis)
        if df.ndim > dx.ndim:
            dx = jnp.expand_dims(dx, tuple(range(1, df.ndim)))
            dx = jnp.moveaxis(dx, 0, axis)
        dxi = jnp.where(dx == 0, 0, 1 / dx)
        df = dxi * df

        A = jnp.diag(
            jnp.concatenate(
                (
                    np.array([1.0]),
                    2 * (dx.flatten()[:-1] + dx.flatten()[1:]),
                    np.array([1.0]),
                )
            )
        )
        upper_diag1 = jnp.diag(
            jnp.concatenate((np.array([1.0]), dx.flatten()[:-1])), k=1
        )
        lower_diag1 = jnp.diag(
            jnp.concatenate((dx.flatten()[1:], np.array([1.0]))), k=-1
        )
        A += upper_diag1 + lower_diag1
        zero = jnp.zeros(tuple(df.shape[i] if i != axis else 1 for i in range(df.ndim)))
        b = jnp.concatenate(
            [
                2 * jnp.take(df, [0], axis, mode="wrap"),
                3
                * (
                    jnp.take(dx, jnp.arange(0, df.shape[axis] - 1), axis, mode="wrap")
                    * jnp.take(df, jnp.arange(1, df.shape[axis]), axis, mode="wrap")
                    + jnp.take(dx, jnp.arange(1, df.shape[axis]), axis, mode="wrap")
                    * jnp.take(df, jnp.arange(0, df.shape[axis] - 1), axis, mode="wrap")
                ),
                2 * jnp.take(df, [-1], axis, mode="wrap"),
            ],
            axis=axis,
        )
        b = jnp.moveaxis(b, axis, 0).reshape((b.shape[axis], -1))
        fx = jnp.linalg.solve(A, b)
        fx = jnp.moveaxis(fx.reshape(f.shape), 0, axis)
        return fx
    if method in ["cardinal", "catmull-rom"]:
        dx = x[2:] - x[:-2]
        df = jnp.take(f, jnp.arange(2, f.shape[axis]), axis, mode="wrap") - jnp.take(
            f, jnp.arange(0, f.shape[axis] - 2), axis, mode="wrap"
        )
        dxi = jnp.where(dx == 0, 0, 1 / dx)
        if df.ndim > dxi.ndim:
            dxi = jnp.expand_dims(dxi, tuple(range(1, df.ndim)))
            dxi = jnp.moveaxis(dxi, 0, axis)
        df = dxi * df
        fx0 = (
            (jnp.take(f, [1], axis, mode="wrap") - jnp.take(f, [0], axis, mode="wrap"))
            / (x[(1,)] - x[(0,)])
            if x[0] != x[1]
            else jnp.zeros_like(jnp.take(f, [0], axis, mode="wrap"))
        )
        fx1 = (
            (
                jnp.take(f, [-1], axis, mode="wrap")
                - jnp.take(f, [-2], axis, mode="wrap")
            )
            / (x[(-1,)] - x[(-2,)])
            if x[-1] != x[-2]
            else jnp.zeros_like(jnp.take(f, [0], axis, mode="wrap"))
        )
        if method == "cardinal":
            c = kwargs.get("c", 0)
        else:
            c = 0
        fx = (1 - c) * jnp.concatenate([fx0, df, fx1], axis=axis)
        return fx

def interp3d(
    xq,
    yq,
    zq,
    x,
    y,
    z,
    f,
    method="cubic",
    derivative=0,
    extrap=False,
    period=0,
    fx=None,
    fy=None,
    fz=None,
    fxy=None,
    fxz=None,
    fyz=None,
    fxyz=None,
    **kwargs,
):
    """Interpolate a 3d function
    Parameters
    ----------
    xq : ndarray, shape(Nq,)
        x query points where interpolation is desired
    yq : ndarray, shape(Nq,)
        y query points where interpolation is desired
    zq : ndarray, shape(Nq,)
        z query points where interpolation is desired
    x : ndarray, shape(Nx,)
        x coordinates of known function values ("knots")
    y : ndarray, shape(Ny,)
        y coordinates of known function values ("knots")
    z : ndarray, shape(Nz,)
        z coordinates of known function values ("knots")
    f : ndarray, shape(Nx,Ny,Nz,...)
        function values to interpolate
    method : str
        method of interpolation
        - `'nearest'`: nearest neighbor interpolation
        - `'linear'`: linear interpolation
        - `'cubic'`: C1 cubic splines (aka local splines)
        - `'cubic2'`: C2 cubic splines (aka natural splines)
        - `'catmull-rom'`: C1 cubic centripedal "tension" splines
        - `'cardinal'`: c1 cubic general tension splines. If used, can also pass keyword
            parameter `c` in float[0,1] to specify tension
    derivative : int, array-like
        derivative order to calculate, scalar values uses the same order for all
        coordinates, or pass a 3 element array or tuple to specify different derivatives
        in x,y,z directions
    extrap : bool, float, array-like
        whether to extrapolate values beyond knots (True) or return nan (False),
        or a specified value to return for query points outside the bounds. Can
        also be passed as an array or tuple to specify different conditions for
        [[xlow, xhigh],[ylow,yhigh],[zlow,zhigh]]
    period : float, None, array-like
        periodicity of the function. If given, function is assumed to be periodic
        on the interval [0,period]. Pass a 3 element array or tuple to specify different
        periods for x,y,z coordinates
    fx : ndarray, shape(Nx,Ny,Nz,...)
        specified x derivatives at knot locations. If not supplied, calculated internally
        using `method`. Only used for cubic interpolation
    fy : ndarray, shape(Nx,Ny,Nz,...)
        specified y derivatives at knot locations. If not supplied, calculated internally
        using `method`. Only used for cubic interpolation
    fz : ndarray, shape(Nx,Ny,Nz,...)
        specified z derivatives at knot locations. If not supplied, calculated internally
        using `method`. Only used for cubic interpolation
    fxy : ndarray, shape(Nx,Ny,Nz,...)
        specified mixed derivatives at knot locations. If not supplied, calculated internally
        using `method`. Only used for cubic interpolation
    fxz : ndarray, shape(Nx,Ny,Nz,...)
        specified mixed derivatives at knot locations. If not supplied, calculated internally
        using `method`. Only used for cubic interpolation
    fyz : ndarray, shape(Nx,Ny,Nz,...)
        specified mixed derivatives at knot locations. If not supplied, calculated internally
        using `method`. Only used for cubic interpolation
    fxyz : ndarray, shape(Nx,Ny,Nz,...)
        specified mixed derivatives at knot locations. If not supplied, calculated internally
        using `method`. Only used for cubic interpolation
    Returns
    -------
    fq : ndarray, shape(Nq,...)
        function value at query points
    """
    xq, yq, zq, x, y, z, f = map(jnp.asarray, (xq, yq, zq, x, y, z, f))
    period, extrap = map(np.asarray, (period, extrap))
    if len(x) != f.shape[0] or x.ndim != 1:
        raise ValueError("x and f must be arrays of equal length")
    if len(y) != f.shape[1] or y.ndim != 1:
        raise ValueError("y and f must be arrays of equal length")
    if len(z) != f.shape[2] or z.ndim != 1:
        raise ValueError("z and f must be arrays of equal length")

    periodx, periody, periodz = np.broadcast_to(
        np.where(period == None, 0, period), (3,)
    )

    derivative_x, derivative_y, derivative_z = np.broadcast_to(
        np.where(derivative == None, 0, derivative), (3,)
    )
    lowx, highx, lowy, highy, lowz, highz = np.broadcast_to(extrap, (3, 2)).flatten()

    if periodx not in [0, None]:
        xq, x, f, fx, fy, fz, fxy, fxz, fyz, fxyz = _make_periodic(
            xq, x, periodx, 0, f, fx, fy, fz, fxy, fxz, fyz, fxyz
        )
        lowx = highx = True
    if periody not in [0, None]:
        yq, y, f, fx, fy, fz, fxy, fxz, fyz, fxyz = _make_periodic(
            yq, y, periody, 1, f, fx, fy, fz, fxy, fxz, fyz, fxyz
        )
        lowy = highy = True
    if periodz not in [0, None]:
        zq, z, f, fx, fy, fz, fxy, fxz, fyz, fxyz = _make_periodic(
            zq, z, periodz, 2, f, fx, fy, fz, fxy, fxz, fyz, fxyz
        )
        lowz = highz = True

    if method == "nearest":
        if (
            derivative_x in [0, None]
            and derivative_y in [0, None]
            and derivative_z in [0, None]
        ):
            i = jnp.argmin(jnp.abs(xq[:, np.newaxis] - x[np.newaxis]), axis=1)
            j = jnp.argmin(jnp.abs(yq[:, np.newaxis] - y[np.newaxis]), axis=1)
            k = jnp.argmin(jnp.abs(zq[:, np.newaxis] - z[np.newaxis]), axis=1)
            fq = f[i, j, k]
        else:
            fq = jnp.zeros((xq.size, yq.size, zq.size, *f.shape[3:]))

    elif method == "linear":
        i = jnp.clip(jnp.searchsorted(x, xq, side="right"), 1, len(x) - 1)
        j = jnp.clip(jnp.searchsorted(y, yq, side="right"), 1, len(y) - 1)
        k = jnp.clip(jnp.searchsorted(z, zq, side="right"), 1, len(z) - 1)

        f000 = f[i - 1, j - 1, k - 1]
        f001 = f[i - 1, j - 1, k]
        f010 = f[i - 1, j, k - 1]
        f100 = f[i, j - 1, k - 1]
        f110 = f[i, j, k - 1]
        f011 = f[i - 1, j, k]
        f101 = f[i, j - 1, k]
        f111 = f[i, j, k]
        x0 = x[i - 1]
        x1 = x[i]
        y0 = y[j - 1]
        y1 = y[j]
        z0 = z[k - 1]
        z1 = z[k]
        dx = x1 - x0
        dxi = jnp.where(dx == 0, 0, 1 / dx)
        dy = y1 - y0
        dyi = jnp.where(dy == 0, 0, 1 / dy)
        dz = z1 - z0
        dzi = jnp.where(dz == 0, 0, 1 / dz)
        if derivative_x in [0, None]:
            tx = jnp.array([x1 - xq, xq - x0])
        elif derivative_x == 1:
            tx = jnp.array([-jnp.ones_like(xq), jnp.ones_like(xq)])
        else:
            tx = jnp.zeros((2, xq.size))
        if derivative_y in [0, None]:
            ty = jnp.array([y1 - yq, yq - y0])
        elif derivative_y == 1:
            ty = jnp.array([-jnp.ones_like(yq), jnp.ones_like(yq)])
        else:
            ty = jnp.zeros((2, yq.size))
        if derivative_z in [0, None]:
            tz = jnp.array([z1 - zq, zq - z0])
        elif derivative_z == 1:
            tz = jnp.array([-jnp.ones_like(zq), jnp.ones_like(zq)])
        else:
            tz = jnp.zeros((2, zq.size))
        F = jnp.array([[[f000, f010], [f100, f110]], [[f001, f011], [f101, f111]]])
        fq = dxi * dyi * dzi * jnp.einsum("il,ijkl,jl,kl->l", tx, F, ty, tz)

    elif method in ["cubic", "cubic2", "cardinal", "catmull-rom"]:
        if fx is None:
            fx = _approx_df(x, f, method, 0, **kwargs)
        if fy is None:
            fy = _approx_df(y, f, method, 1, **kwargs)
        if fz is None:
            fz = _approx_df(z, f, method, 2, **kwargs)
        if fxy is None:
            fxy = _approx_df(y, fx, method, 1, **kwargs)
        if fxz is None:
            fxz = _approx_df(z, fx, method, 2, **kwargs)
        if fyz is None:
            fyz = _approx_df(z, fy, method, 2, **kwargs)
        if fxyz is None:
            fxyz = _approx_df(z, fxy, method, 2, **kwargs)

        i = jnp.clip(jnp.searchsorted(x, xq, side="right"), 1, len(x) - 1)
        j = jnp.clip(jnp.searchsorted(y, yq, side="right"), 1, len(y) - 1)
        k = jnp.clip(jnp.searchsorted(z, zq, side="right"), 1, len(z) - 1)

        dx = x[i] - x[i - 1]
        deltax = xq - x[i - 1]
        dxi = jnp.where(dx == 0, 0, 1 / dx)
        tx = deltax * dxi

        dy = y[j] - y[j - 1]
        deltay = yq - y[j - 1]
        dyi = jnp.where(dy == 0, 0, 1 / dy)
        ty = deltay * dyi

        dz = z[k] - z[k - 1]
        deltaz = zq - z[k - 1]
        dzi = jnp.where(dz == 0, 0, 1 / dz)
        tz = deltaz * dzi

        fs = OrderedDict()
        fs["f"] = f
        fs["fx"] = fx
        fs["fy"] = fy
        fs["fz"] = fz
        fs["fxy"] = fxy
        fs["fxz"] = fxz
        fs["fyz"] = fyz
        fs["fxyz"] = fxyz
        fsq = OrderedDict()
        for ff in fs.keys():
            for kk in [0, 1]:
                for jj in [0, 1]:
                    for ii in [0, 1]:
                        fsq[ff + str(ii) + str(jj) + str(kk)] = fs[ff][
                            i - 1 + ii, j - 1 + jj, k - 1 + kk
                        ]
                        if "x" in ff:
                            fsq[ff + str(ii) + str(jj) + str(kk)] *= dx
                        if "y" in ff:
                            fsq[ff + str(ii) + str(jj) + str(kk)] *= dy
                        if "z" in ff:
                            fsq[ff + str(ii) + str(jj) + str(kk)] *= dz

        F = jnp.vstack([foo for foo in fsq.values()])

        coef = jnp.matmul(A_tricubic, F)

        coef = jnp.moveaxis(coef.reshape((4, 4, 4, -1), order="F"), -1, 0)

        ttx = _get_t_der(tx, derivative_x, dxi)
        tty = _get_t_der(ty, derivative_y, dyi)
        ttz = _get_t_der(tz, derivative_z, dzi)
        fq = jnp.einsum("lijk...,li,lj,lk->l...", coef, ttx, tty, ttz)

    else:
        raise ValueError(f"unknown method {method}")

    fq = _extrap(xq, fq, x, f, lowx, highx, axis=0)
    fq = _extrap(yq, fq, y, f, lowy, highy, axis=1)
    fq = _extrap(zq, fq, z, f, lowz, highz, axis=2)

    return fq

def _make_periodic(xq, x, period, axis, *arrs):
    """Make arrays periodic along a specified axis"""

    if period == 0:
        raise ValueError(f"period must be a non-zero value; got {period}")
    period = abs(period)
    xq = xq % period
    x = x % period
    i = jnp.argsort(x)
    x = x[i]
    x = jnp.concatenate([x[-1:] - period, x, x[:1] + period])
    arrs = list(arrs)
    for k in range(len(arrs)):
        if arrs[k] is not None:
            arrs[k] = jnp.take(arrs[k], i, axis, mode="wrap")
            arrs[k] = jnp.concatenate(
                [jnp.take(arrs[k], [-1], axis), arrs[k], jnp.take(arrs[k], [0], axis)],
                axis=axis,
            )
    return (xq, x, *arrs)

def _get_t_der(t, derivative, dxi):
    """Gets arrays of [1,t,t^2,t^3] for cubic interpolation"""

    if derivative == 0 or derivative is None:
        tt = jnp.array([jnp.ones_like(t), t, t ** 2, t ** 3]).T
    elif derivative == 1:
        tt = (
            jnp.array([jnp.zeros_like(t), jnp.ones_like(t), 2 * t, 3 * t ** 2]).T
            * dxi[:, np.newaxis]
        )
    elif derivative == 2:
        tt = (
            jnp.array(
                [jnp.zeros_like(t), jnp.zeros_like(t), 2 * jnp.ones_like(t), 6 * t]
            ).T
            * dxi[:, np.newaxis] ** 2
        )
    elif derivative == 3:
        tt = (
            jnp.array(
                [
                    jnp.zeros_like(t),
                    jnp.zeros_like(t),
                    jnp.zeros_like(t),
                    6 * jnp.ones_like(t),
                ]
            ).T
            * dxi[:, np.newaxis] ** 3
        )
    else:
        tt = jnp.array(
            [jnp.zeros_like(t), jnp.zeros_like(t), jnp.zeros_like(t), jnp.zeros_like(t)]
        ).T
    return tt

def _extrap(xq, fq, x, f, low, high, axis=0):
    """Clamp or extrapolate values outside bounds"""

    if isinstance(low, numbers.Number) or (not low):
        low = low if isinstance(low, numbers.Number) else np.nan
        fq = jnp.where(xq < x[0], low, fq)
    if isinstance(high, numbers.Number) or (not high):
        high = high if isinstance(high, numbers.Number) else np.nan
        fq = jnp.where(xq > x[-1], high, fq)
    return fq

# fmt: off
A_tricubic = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-3, 3, 0, 0, 0, 0, 0, 0,-2,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 2,-2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     -2,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0,-3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     -2, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 9,-9,-9, 9, 0, 0, 0, 0, 6, 3,-6,-3, 0, 0, 0, 0, 6,-6, 3,-3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     4, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-6, 6, 6,-6, 0, 0, 0, 0,-3,-3, 3, 3, 0, 0, 0, 0,-4, 4,-2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     -2,-2,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 2, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 2, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-6, 6, 6,-6, 0, 0, 0, 0,-4,-2, 4, 2, 0, 0, 0, 0,-3, 3,-3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     -2,-1,-2,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 4,-4,-4, 4, 0, 0, 0, 0, 2, 2,-2,-2, 0, 0, 0, 0, 2,-2, 2,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3, 3, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0,-2,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,-2, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3, 3, 0, 0, 0, 0, 0, 0,-2,-1, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,-2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3, 0, 3, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0,-3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0,-1, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9,-9,-9, 9, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 6, 3,-6,-3, 0, 0, 0, 0, 6,-6, 3,-3, 0, 0, 0, 0, 4, 2, 2, 1, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-6, 6, 6,-6, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0,-3,-3, 3, 3, 0, 0, 0, 0,-4, 4,-2, 2, 0, 0, 0, 0,-2,-2,-1,-1, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0,-2, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 2, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-6, 6, 6,-6, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0,-4,-2, 4, 2, 0, 0, 0, 0,-3, 3,-3, 3, 0, 0, 0, 0,-2,-1,-2,-1, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4,-4,-4, 4, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 2, 2,-2,-2, 0, 0, 0, 0, 2,-2, 2,-2, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    [-3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0, 0, 0,-1, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0,-3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0,-2, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 9,-9, 0, 0,-9, 9, 0, 0, 6, 3, 0, 0,-6,-3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6,-6, 0, 0, 3,-3, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 4, 2, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-6, 6, 0, 0, 6,-6, 0, 0,-3,-3, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-4, 4, 0, 0,-2, 2, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0,-2,-2, 0, 0,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     -3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0, 0, 0,-1, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9,-9, 0, 0,-9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     6, 3, 0, 0,-6,-3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6,-6, 0, 0, 3,-3, 0, 0, 4, 2, 0, 0, 2, 1, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-6, 6, 0, 0, 6,-6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     -3,-3, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-4, 4, 0, 0,-2, 2, 0, 0,-2,-2, 0, 0,-1,-1, 0, 0],
    [ 9, 0,-9, 0,-9, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 3, 0,-6, 0,-3, 0, 6, 0,-6, 0, 3, 0,-3, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 2, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 9, 0,-9, 0,-9, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     6, 0, 3, 0,-6, 0,-3, 0, 6, 0,-6, 0, 3, 0,-3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 2, 0, 2, 0, 1, 0],
    [-27,27,27,-27,27,-27,-27,27,-18,-9,18, 9,18, 9,-18,-9,-18,18,-9, 9,18,-18, 9,-9,-18,18,18,-18,-9, 9, 9,
     -9,-12,-6,-6,-3,12, 6, 6, 3,-12,-6,12, 6,-6,-3, 6, 3,-12,12,-6, 6,-6, 6,-3, 3,-8,-4,-4,-2,-4,-2,-2,-1],
    [18,-18,-18,18,-18,18,18,-18, 9, 9,-9,-9,-9,-9, 9, 9,12,-12, 6,-6,-12,12,-6, 6,12,-12,-12,12, 6,-6,-6,
     6, 6, 6, 3, 3,-6,-6,-3,-3, 6, 6,-6,-6, 3, 3,-3,-3, 8,-8, 4,-4, 4,-4, 2,-2, 4, 4, 2, 2, 2, 2, 1, 1],
    [-6, 0, 6, 0, 6, 0,-6, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3, 0,-3, 0, 3, 0, 3, 0,-4, 0, 4, 0,-2, 0, 2, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0,-2, 0,-1, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0,-6, 0, 6, 0, 6, 0,-6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     -3, 0,-3, 0, 3, 0, 3, 0,-4, 0, 4, 0,-2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0,-2, 0,-1, 0,-1, 0],
    [18,-18,-18,18,-18,18,18,-18,12, 6,-12,-6,-12,-6,12, 6, 9,-9, 9,-9,-9, 9,-9, 9,12,-12,-12,12, 6,-6,-6,
     6, 6, 3, 6, 3,-6,-3,-6,-3, 8, 4,-8,-4, 4, 2,-4,-2, 6,-6, 6,-6, 3,-3, 3,-3, 4, 2, 4, 2, 2, 1, 2, 1],
    [-12,12,12,-12,12,-12,-12,12,-6,-6, 6, 6, 6, 6,-6,-6,-6, 6,-6, 6, 6,-6, 6,-6,-8, 8, 8,-8,-4, 4, 4,-4,
     -3,-3,-3,-3, 3, 3, 3, 3,-4,-4, 4, 4,-2,-2, 2, 2,-4, 4,-4, 4,-2, 2,-2, 2,-2,-2,-2,-2,-1,-1,-1,-1],
    [ 2, 0, 0, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-6, 6, 0, 0, 6,-6, 0, 0,-4,-2, 0, 0, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3, 3, 0, 0,-3, 3, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0,-2,-1, 0, 0,-2,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 4,-4, 0, 0,-4, 4, 0, 0, 2, 2, 0, 0,-2,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,-2, 0, 0, 2,-2, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     2, 0, 0, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-6, 6, 0, 0, 6,-6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     -4,-2, 0, 0, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3, 3, 0, 0,-3, 3, 0, 0,-2,-1, 0, 0,-2,-1, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4,-4, 0, 0,-4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     2, 2, 0, 0,-2,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,-2, 0, 0, 2,-2, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
    [-6, 0, 6, 0, 6, 0,-6, 0, 0, 0, 0, 0, 0, 0, 0, 0,-4, 0,-2, 0, 4, 0, 2, 0,-3, 0, 3, 0,-3, 0, 3, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0,-1, 0,-2, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0,-6, 0, 6, 0, 6, 0,-6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     -4, 0,-2, 0, 4, 0, 2, 0,-3, 0, 3, 0,-3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0,-1, 0,-2, 0,-1, 0],
    [18,-18,-18,18,-18,18,18,-18,12, 6,-12,-6,-12,-6,12, 6,12,-12, 6,-6,-12,12,-6, 6, 9,-9,-9, 9, 9,-9,-9,
     9, 8, 4, 4, 2,-8,-4,-4,-2, 6, 3,-6,-3, 6, 3,-6,-3, 6,-6, 3,-3, 6,-6, 3,-3, 4, 2, 2, 1, 4, 2, 2, 1],
    [-12,12,12,-12,12,-12,-12,12,-6,-6, 6, 6, 6, 6,-6,-6,-8, 8,-4, 4, 8,-8, 4,-4,-6, 6, 6,-6,-6, 6, 6,-6,
     -4,-4,-2,-2, 4, 4, 2, 2,-3,-3, 3, 3,-3,-3, 3, 3,-4, 4,-2, 2,-4, 4,-2, 2,-2,-2,-1,-1,-2,-2,-1,-1],
    [ 4, 0,-4, 0,-4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0,-2, 0,-2, 0, 2, 0,-2, 0, 2, 0,-2, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 4, 0,-4, 0,-4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     2, 0, 2, 0,-2, 0,-2, 0, 2, 0,-2, 0, 2, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    [-12,12,12,-12,12,-12,-12,12,-8,-4, 8, 4, 8, 4,-8,-4,-6, 6,-6, 6, 6,-6, 6,-6,-6, 6, 6,-6,-6, 6, 6,-6,
     -4,-2,-4,-2, 4, 2, 4, 2,-4,-2, 4, 2,-4,-2, 4, 2,-3, 3,-3, 3,-3, 3,-3, 3,-2,-1,-2,-1,-2,-1,-2,-1],
    [ 8,-8,-8, 8,-8, 8, 8,-8, 4, 4,-4,-4,-4,-4, 4, 4, 4,-4, 4,-4,-4, 4,-4, 4, 4,-4,-4, 4, 4,-4,-4, 4,
     2, 2, 2, 2,-2,-2,-2,-2, 2, 2,-2,-2, 2, 2,-2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 1, 1, 1, 1, 1, 1, 1, 1]
])

###

class Grid():
    """Base class for collocation grids
    Unlike subclasses LinearGrid and ConcentricGrid, the base Grid allows the user
    to pass in a custom set of collocation nodes.
    Parameters
    ----------
    nodes : ndarray of float, size(num_nodes,3)
        node coordinates, in (rho,theta,zeta)
    sort : bool
        whether to sort the nodes for use with FFT method.
    """

    # TODO: calculate weights automatically using voronoi / delaunay triangulation
    _io_attrs_ = [
        "_L",
        "_M",
        "_N",
        "_NFP",
        "_sym",
        "_nodes",
        "_weights",
        "_axis",
        "_node_pattern",
    ]

    def __init__(self, nodes, sort=True):

        self._L = np.unique(nodes[:, 0]).size
        self._M = np.unique(nodes[:, 1]).size
        self._N = np.unique(nodes[:, 2]).size
        self._NFP = 1
        self._sym = False
        self._node_pattern = "custom"

        self._nodes, self._weights = self._create_nodes(nodes)

        self._enforce_symmetry()
        if sort:
            self._sort_nodes()
        self._find_axis()
        self._scale_weights()

    def _enforce_symmetry(self):
        """Enforces stellarator symmetry"""
        if self.sym:  # remove nodes with theta > pi
            non_sym_idx = np.where(self.nodes[:, 1] > np.pi)
            self._nodes = np.delete(self.nodes, non_sym_idx, axis=0)
            self._weights = np.delete(self.weights, non_sym_idx, axis=0)

    def _sort_nodes(self):
        """Sorts nodes for use with FFT"""

        sort_idx = np.lexsort((self.nodes[:, 1], self.nodes[:, 0], self.nodes[:, 2]))
        self._nodes = self.nodes[sort_idx]
        self._weights = self.weights[sort_idx]

    def _find_axis(self):
        """Finds indices of axis nodes"""
        self._axis = np.where(self.nodes[:, 0] == 0)[0]

    def _scale_weights(self):
        """Scales weights to sum to full volume and reduces weights for duplicated nodes"""

        nodes = self.nodes.copy().astype(float)
        nodes[:, 1] %= 2 * np.pi
        nodes[:, 2] %= 2 * np.pi / self.NFP
        _, inverse, counts = np.unique(
            nodes, axis=0, return_inverse=True, return_counts=True
        )
        self._weights /= counts[inverse]
        self._weights *= 4 * np.pi ** 2 / self._weights.sum()

    def _create_nodes(self, nodes):
        """Allows for custom node creation
        Parameters
        ----------
        nodes : ndarray of float, size(num_nodes,3)
            node coordinates, in (rho,theta,zeta)
        Returns
        -------
        nodes : ndarray of float, size(num_nodes,3)
            node coordinates, in (rho,theta,zeta)
        """
        nodes = np.atleast_2d(nodes).reshape((-1, 3))
        # make weights sum to 4pi^2
        weights = np.ones(nodes.shape[0]) / nodes.shape[0] * 4 * np.pi ** 2
        self._L = len(np.unique(nodes[:, 0]))
        self._M = len(np.unique(nodes[:, 1]))
        self._N = len(np.unique(nodes[:, 2]))
        return nodes, weights

    @property
    def L(self):
        """int: radial grid resolution"""
        return self.__dict__.setdefault("_L", 0)

    @property
    def M(self):
        """int: poloidal grid resolution"""
        return self.__dict__.setdefault("_M", 0)

    @property
    def N(self):
        """int: toroidal grid resolution"""
        return self.__dict__.setdefault("_N", 0)

    @property
    def NFP(self):
        """int: number of field periods"""
        return self.__dict__.setdefault("_NFP", 1)

    @property
    def sym(self):
        """bool: True for stellarator symmetry, False otherwise"""
        return self.__dict__.setdefault("_sym", False)

    @property
    def nodes(self):
        """ndarray: node coordinates, in (rho,theta,zeta)"""
        return self.__dict__.setdefault("_nodes", np.array([]).reshape((0, 3)))

    @nodes.setter
    def nodes(self, nodes):
        self._nodes = nodes

    @property
    def weights(self):
        """ndarray: weight for each node, either exact quadrature or volume based"""
        return self.__dict__.setdefault("_weights", np.array([]).reshape((0, 3)))

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    @property
    def num_nodes(self):
        """int: total number of nodes"""
        return self.nodes.shape[0]

    @property
    def axis(self):
        """ndarray: indices of nodes at magnetic axis"""
        return self.__dict__.setdefault("_axis", np.array([]))

    @property
    def node_pattern(self):
        """str: pattern for placement of nodes in rho,theta,zeta"""
        return self.__dict__.setdefault("_node_pattern", "custom")

    def __repr__(self):
        """string form of the object"""
        return (
            type(self).__name__
            + " at "
            + str(hex(id(self)))
            + " (L={}, M={}, N={}, NFP={}, sym={}, node_pattern={})".format(
                self.L, self.M, self.N, self.NFP, self.sym, self.node_pattern
            )
        )



