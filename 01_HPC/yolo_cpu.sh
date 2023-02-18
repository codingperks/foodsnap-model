#PBS -lwalltime=08:00:00
#PBS -lselect=1:ncpus=256:mem=920gb:ompthreads=256
 
module load anaconda3/personal
source activate yolo_01
python $HOME/01_HPC/yolo_train.py

cd $TMPDIR
mkdir $WORK/$PBS_JOBID
cp * $WORK/$PBS_JOBID
cp -R runs $WORK/$PBS_JOBID
