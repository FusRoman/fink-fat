import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp


def color_dict(mpc_database):
    orbit_color = [
        "gold",
        "red",
        "dodgerblue",
        "limegreen",
        "grey",
        "magenta",
        "chocolate",
        "blue",
        "orange",
        "mediumspringgreen",
        "deeppink",
    ]

    return {
        orbit_type: orbit_color
        for orbit_type, orbit_color in zip(
            np.unique(mpc_database["Orbit_type"]), orbit_color
        )
    }


def compute_residue(df):
    df = df.reset_index(drop=True)
    computed_elem = df[
        ["a_x", "e_x", "i_x", "long. node", "arg. peric", "mean anomaly"]
    ]
    known_elem = df[["a_y", "e_y", "i_y", "Node", "Peri", "M"]]

    df[["da", "de", "di", "dNode", "dPeri", "dM"]] = (
        computed_elem.values - known_elem.values
    )

    return df

def plot_residue(df, orbit_color, n_trajectories, n_points):
    df = compute_residue(df)
    orbit_type = np.unique(df["Orbit_type"])

    fig, axes = plt.subplots(3, 2, sharex=True)
    fig.suptitle(
        "Orbital elements residuals, {} trajectories, {} points".format(
            n_trajectories, n_points
        )
    )

    subplot_title = [
        "semi-major axis",
        "eccentricity",
        "inclination",
        "Longitude of the ascending node",
        "Argument of perihelion",
        "Mean anomaly",
    ]

    for ax, orb_elem, title in zip(
        axes.flatten(), ["da", "de", "di", "dNode", "dPeri", "dM"], subplot_title
    ):
        ax.set_title(title)
        ax.axhline(0, ls="--", color="grey")
        for otype in orbit_type:
            v = df[df["Orbit_type"] == otype]
            omean = np.mean(v[orb_elem].values)

            failed_orb = np.where(v["a_x"].values == -1)
            success_orb = np.where(v["a_x"].values != -1)
            ax.scatter(
                np.array(v.index)[success_orb],
                v[orb_elem].values[success_orb],
                label="{}: {}, mean : {}, fail: {}".format(
                    otype, len(v), np.around(omean, decimals=4), len(failed_orb[0])
                ),
                color=orbit_color[otype],
            )
            ax.scatter(
                np.array(v.index)[failed_orb],
                v[orb_elem].values[failed_orb],
                marker="x",
                color=orbit_color[otype],
            )

            ax.axhline(omean, ls=":", color=orbit_color[otype])
            ax.set_ylabel("$\delta$ {}".format(orb_elem[1:]))  # noqa: W605
        ax.legend(prop={"size": 7})

    plt.show()


def plot_cpu_time(all_time, n_trajectories, n_points):
    plt.plot(np.arange(1, mp.cpu_count() + 1), all_time)
    plt.xlabel("number of cpu")
    plt.ylabel("computation time")
    plt.title(
        "CPU Time analysis\nwith file write on ram\n {} trajectories with {} points".format(
            n_trajectories, n_points
        )
    )
    plt.show()
