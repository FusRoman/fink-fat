import os
import signal
import subprocess
import numpy as np

import pandas as pd

from fink_fat.orbit_fitting.orbfit_local import final_clean, obs_clean, prep_orbitfit, read_oel, write_inp, write_observation_file


def write_oop(ram_dir, provisional_designation):
    with open(ram_dir + provisional_designation + ".oop", "w") as file:
        # write output options
        file.write("output.\n")
        file.write("\t.elements = 'KEP'\n")
        
        # write operations options
        file.write("operations.\n")
        file.write("\t.init_orbdet = 2\n")
        file.write("\t.diffcor = 2\n")
        file.write("\t.ident = 0\n")
        file.write("\t.ephem = 0\n")
        
        # write error model options
        file.write("error_model.\n")
        file.write("\t.name='fcct14'\n")
        
        # write additional options
        file.write("IERS.\n")
        file.write("\t.extrapolation = .T.\n")
        
        # write reject options
        file.write("reject.\n")
        file.write("\t.rejopp = .FALSE.\n")
        
        # write propagation options
        file.write("propag.\n")
        file.write("\t.iast = 17\n")
        file.write("\t.npoint = 600\n")
        file.write("\t.dmea = 0.2d0\n")
        file.write("\t.dter = 0.05d0\n")
        
        # write location files options
        file.write(".filbe=" + ram_dir + "AST17\n")
        file.write("\noutput_files.\n")
        file.write("\t.elem = " + ram_dir + provisional_designation + ".oel\n")
        file.write("object1.\n")
        file.write("\t.obs_dir = " + ram_dir + "mpcobs\n")
        file.write("\t.obs_fname = " + provisional_designation)


def call_orbitfit(ram_dir, provisional_designation):
    orbitfit_path = os.path.join("/opt", "OrbitFit", "bin/")

    command = (
        orbitfit_path
        + "orbfit.x < "
        + ram_dir
        + provisional_designation
        + ".inp "
        + ">/dev/null 2>&1"
    )

    with subprocess.Popen(
        command, shell=True, stdout=subprocess.DEVNULL, preexec_fn=os.setsid
    ) as process:
        try:
            output = process.communicate(timeout=5)[0]
            return output
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGINT)  # send signal to the process group
            output = process.communicate()[0]
            return output


@pandas_udf(ArrayType(DoubleType()))
def get_orbit_element(ra, dec, dcmag, band, date, traj_id, ram_dir):
    _pid = os.getpid()
    current_ram_path = os.path.join(ram_dir, str(_pid), "")
    if not os.path.isdir(current_ram_path):
        os.mkdir(current_ram_path)
        
    prep_orbitfit(current_ram_path)
    
    res = []
    for c_ra, c_dec, c_dcmag, c_band, c_date, c_traj_id in zip(ra, dec, dcmag, band, date, traj_id):
        prov_desig = write_observation_file(
            current_ram_path,
            c_ra,
            c_dec,
            c_dcmag,
            c_band,
            c_date,
            c_traj_id
        )
        write_inp(current_ram_path, prov_desig)
        write_oop(current_ram_path, prov_desig)
        
        call_orbitfit(current_ram_path, prov_desig)
        orb_elem = read_oel(current_ram_path, prov_desig)
        
        res.append(orb_elem)
        
        obs_clean(current_ram_path, prov_desig)
        
    final_clean(current_ram_path)
    os.rmdir(current_ram_path)
    return pd.Series(res)

if __name__=="__main__":
    import fink_fat.others.utils as ut
    ram_dir = "/tmp/ramdisk/"

    df = ut.load_data("Solar System MPC")
    gb = df.groupby(['ssnamenr']).count().reset_index()
    all_mpc_name = np.unique(df["ssnamenr"])
    df_traj = df[df["ssnamenr"].isin(gb[gb['ra'] == 7]["ssnamenr"][:2])]
    mpc_name = np.unique(df_traj["ssnamenr"])
    to_traj_id = {name:i for i, name in zip(np.arange(len(mpc_name)), mpc_name)}
    df_traj["trajectory_id"] = df_traj.apply(lambda x: to_traj_id[x["ssnamenr"]], axis=1)

    sparkDF = spark.createDataFrame(df_traj[["ra", "dec", "dcmag", "fid", "jd", "trajectory_id"]])

    spark_gb = sparkDF.groupby("trajectory_id") \
        .agg(F.sort_array(F.collect_list(F.struct("jd", "ra", "dec", "fid", "dcmag"))) \
        .alias("collected_list"))\
        .withColumn("ra", F.col("collected_list.ra"))\
        .withColumn("dec", F.col("collected_list.dec"))\
        .withColumn("fid", F.col("collected_list.fid"))\
        .withColumn("dcmag", F.col("collected_list.dcmag"))\
        .withColumn("jd", F.col("collected_list.jd"))\
        .drop("collected_list")

    spark_gb = spark_gb.repartition(sparkDF.rdd.getNumPartitions())

    spark_column = spark_gb.withColumn('coord', get_orbit_element(
        spark_gb.ra,
        spark_gb.dec,
        spark_gb.dcmag,
        spark_gb.fid,
        spark_gb.jd,
        spark_gb.trajectory_id
    ))