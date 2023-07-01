Usage:


python3 analyse_dataset.py <smarter-labelme-image-folder> [<more folders>]
   Print statistics about a folder or a set of folders used as training data

python3 make_combined_dataset.py <smarter-labelme-image-folder> [<more image folders] <destination folder>
   Creates a classifier training dataset from one or multiple annotated datasets combined with MSCOCO 2017

mkdir snapshots
./train.sh <dataset_folder> snapshots
   Trains SSD Multibox with parameters suited for smarter-labelme usage and stores snapshots in snapshot folder


Please see README.md in SSD symlink/subfolder for requirements and prerequisits of NVIDIA DeepLearningExamples training code. Needs NVIDIA dali among others.
