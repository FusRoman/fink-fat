wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

rm GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

sudo add-apt-repository "deb https://apt.repos.intel.com/oneapi all main"

sudo apt update
sudo apt install intel-oneapi-compiler-fortran

source /opt/intel/oneapi/setvars.sh

ifort --version