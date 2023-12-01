import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord


def equ_to_cart(equ_coord):
    return SkyCoord(
        x=equ_coord.cartesian.x,
        y=equ_coord.cartesian.y,
        z=equ_coord.cartesian.z,
        representation_type="cartesian",
    )


def cart_to_equ(cart_coord):
    cart_coord.representation_type = "unitspherical"
    return cart_coord


def radec_to_cart(ra, dec):
    coord = SkyCoord(ra, dec, unit="deg")
    return equ_to_cart(coord)


def xyz_to_equ(x, y, z):
    cart_coord = SkyCoord(
        x=x,
        y=y,
        z=z,
        representation_type="cartesian",
    )
    return cart_to_equ(cart_coord)


def fit_traj(ra, dec, jd, poly_exp=2):
    cart = radec_to_cart(ra, dec)
    popt_x = np.polyfit(jd, cart.x.value, poly_exp)
    popt_y = np.polyfit(jd, cart.y.value, poly_exp)
    popt_z = np.polyfit(jd, cart.z.value, poly_exp)

    return np.array([popt_x, popt_y, popt_z])


def fit_roid(pdf_ast, ast_name):
    pdf_roid = pdf_ast[pdf_ast["ssnamenr"] == ast_name]
    return fit_traj(
        pdf_roid["ra"].values, pdf_roid["dec"].values, pdf_roid["jd"].values
    )


def fit_ast_dataset(pdf_ast):
    fit_list = [
        [roid_name, *fit_roid(pdf_ast, roid_name)]
        for roid_name in pdf_ast["ssnamenr"].unique()
    ]
    return pd.DataFrame(fit_list, columns=["ssnamenr", "x_fit", "y_fit", "z_fit"])


def predict_cart(params, time):
    return np.poly1d(params)(time)


def predict_equ(xp, yp, zp, time):
    x_coord = np.poly1d(xp)(time)
    y_coord = np.poly1d(yp)(time)
    z_coord = np.poly1d(zp)(time)
    return xyz_to_equ(x_coord, y_coord, z_coord)


def predict_roid(pdf, roid, time):
    pdf_ast = pdf[pdf["ssnamenr"] == roid]
    return predict_equ(
        pdf_ast["xp"].values[0],
        pdf_ast["yp"].values[0],
        pdf_ast["zp"].values[0],
        time,
    )


def fit_expectation(pdf, roid):
    pdf_ast = pdf[pdf["ssnamenr"] == roid]
    return predict_roid(pdf, roid, pdf_ast["jd"].values)


def chi_fit(pdf, roid):
    pdf_ast = pdf[pdf["ssnamenr"] == roid]
    coord_exp = fit_expectation(pdf, roid)
    coord_obs = SkyCoord(pdf_ast["ra"], pdf_ast["dec"], unit="deg")
    return [
        chi_square(coord_obs.ra, coord_exp.ra).value,
        chi_square(coord_obs.dec, coord_exp.dec).value,
    ]


def chi_pdf(xp, yp, zp, ra, dec, jd):
    coord_obs = SkyCoord(ra, dec, unit="deg")
    coord_exp = predict_equ(
        xp,
        yp,
        zp,
        jd,
    )
    return [
        chi_square(coord_obs.ra, coord_exp.ra).value,
        chi_square(coord_obs.dec, coord_exp.dec).value,
    ]


def chi_square(obs, exp):
    return (((obs - exp) ** 2) / exp).sum()


def plot_fit(pdf, roid):
    pdf_ast = pdf[pdf["ssnamenr"] == roid]
    coord_exp = fit_expectation(pdf, roid)
    coord_obs = SkyCoord(pdf_ast["ra"], pdf_ast["dec"], unit="deg")

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(coord_obs.ra, coord_obs.dec, label="obs")
    ax1.scatter(coord_exp.ra, coord_exp.dec, label="exp")

    ax2.scatter(pdf_ast["jd"], coord_obs.separation(coord_exp).arcsecond)
    plt.legend()
    plt.show()


def sep_next_point(ra, dec, jd):
    sep_next_point = []
    for i in range(2, len(ra)):
        popt = fit_traj(
            ra[:i],
            dec[:i],
            jd[:i],
        )
        next_pred = predict_equ(*popt, jd[i])
        next_obs = SkyCoord(ra[i], dec[i], unit="deg")
        sep_next_point.append(next_obs.separation(next_pred).arcminute)
    return sep_next_point
