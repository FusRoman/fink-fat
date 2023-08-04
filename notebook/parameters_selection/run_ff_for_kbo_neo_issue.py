import subprocess
import shutil


if __name__ == "__main__":
    from datetime import date, timedelta

    start_date = date(2020, 9, 1)
    end_date = date(2020, 10, 1)  # perhaps date.now()

    delta = end_date - start_date  # returns timedelta

    for i in range(delta.days + 1):
        curr_date = start_date + timedelta(days=i)

        print(curr_date)
        ff_assoc_command = f"fink_fat associations mpc --night {curr_date} --config ../fink_fat_experiments/kbo_neo_issue.conf --verbose"
        ff_orbit_command = "fink_fat solve_orbit mpc local --config ../fink_fat_experiments/kbo_neo_issue.conf --verbose"

        results_assoc = subprocess.run(
            ff_assoc_command, shell=True, universal_newlines=True, check=True
        )

        print()
        print()

        results_orbit = subprocess.run(
            ff_orbit_command, shell=True, universal_newlines=True, check=True
        )

        print("move data")
        shutil.copytree(
            "kbo_neo_issue_ff_output/mpc", f"save_ff_output_kbo_neo_issue/{curr_date}"
        )

        print()
        print("--- stdout ---")
        print(results_assoc.stdout)
        print()
        print(results_orbit.stdout)
        print()
        print()
