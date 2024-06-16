source /spack/share/spack/setup-env.sh
ls /
echo "---"
ls /key4hep-spack/environments/
spack env activate /key4hep-spack/environments/key4hep-release-user
spack concretize -f
spack find
#source /cvmfs/sw.hsf.org/key4hep/setup.sh
#. /HiggsAnalysis/CombinedLimit/env_lcg.sh
#root-config --version
#uname -a
#gcc --version
#cd HiggsAnalysis/CombinedLimit
#/usr/bin/make LCG=1 -j 8
