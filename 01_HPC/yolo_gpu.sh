#PBS -l walltime=01:59:00
#PBS -l select=1:ncpus=4:mem=64gb:ngpus=1:gpu_type=RTX6000

module purge

module add tools/dev
module add cuda/10.2

module load anaconda3/personal
source activate cpu_01

python $HOME/foodsnap-backend-model/01_HPC/train_gpu.py

cd $TMPDIR
mkdir $WORK/$PBS_JOBID
cp * $WORK/$PBS_JOBID
cp -r runs $WORK/$PBS_JOBID
