# source the intel fortran compiler and libraries, needed for OrbFit
source /opt/intel/oneapi/setvars.sh

wget http://adams.dm.unipi.it/orbfit/OrbFit5.0.7.tar.gz

if [[ ! -d OrbitFit/ ]]
then
    echo "OrbitFit/ does not exists on your filesystem."
    mkdir OrbitFit/
fi

tar -xf OrbFit5.0.7.tar.gz -C OrbitFit/

rm OrbFit5.0.7.tar.gz

cd OrbitFit/

./config -O intel

make

cd lib/

wget https://ssd.jpl.nasa.gov/ftp/eph/planets/Linux/de440/linux_p1550p2650.440

mv linux_p1550p2650.440 jpleph