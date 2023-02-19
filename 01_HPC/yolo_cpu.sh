#PBS -lwalltime=07:58:00
#PBS -lselect=1:ncpus=128:mem=512gb:ompthreads=128

module purge 
module load anaconda3/personal
source activate cpu_01
python $HOME/foodsnap-backend-model/01_HPC/train_cpu.py

cd $TMPDIR
mkdir $WORK/$PBS_JOBID
cp * $WORK/$PBS_JOBID
cp -R runs $WORK/$PBS_JOBID
