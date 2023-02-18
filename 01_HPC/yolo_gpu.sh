#PBS -l walltime=00:59:00
#PBS -l select=1:ncpus=32:mem=100gb:ngpus=1:gpu_type=RTX6000

module load anaconda3/personal
source activate yolo_01
module load cuda 
python $HOME//01_HPC/yolo_train.py

cd $TMPDIR
mkdir $WORK/$PBS_JOBID
cp * $WORK/$PBS_JOBID
cp -r runs $WORK/$PBS_JOBID
