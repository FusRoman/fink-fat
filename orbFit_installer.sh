ORBLOCATE=~/OrbitFit

aria2c -x8 http://adams.dm.unipi.it/orbfit/OrbFit5.0.7.tar.gz

if [[ ! -d $ORBLOCATE ]]
then
    echo "OrbitFit/ directory does not exists on your filesystem."
    mkdir $ORBLOCATE
fi

tar -xf OrbFit5.0.7.tar.gz -C $ORBLOCATE

rm OrbFit5.0.7.tar.gz

cd $ORBLOCATE

./config -O gfortran

make

cd lib/

aria2c -x8 https://ssd.jpl.nasa.gov/ftp/eph/planets/Linux/de440/linux_p1550p2650.440

mv linux_p1550p2650.440 jpleph
