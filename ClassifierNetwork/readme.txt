Usage:


python3 analyse_dataset.py <smarter-labelme-image-folder> [<more folders>]
   Print statistics about a folder or a set of folders used as training data

python3 make_training_data.py <smarter-labelme-image-folder> [<more image folders] <destination folder>
   Creates a classifier training dataset from one or multiple annotated datasets
   stores different datasets in destination_folder/train destination_folder/val and destination_folder/test

mkdir snapshots
python3 train.py --traindata <training_data_folder> --testdata <validation_data_folder> --save-model-directoy snapshots
   Trains a new network with default seed (see --help for other options) and stores it in folder "snapshots"

python3 test.py --testdata <validation_data_folder> <snapshot>
   Tests a specific trained network on the designated test data - see --help for additional options

mkdir heatmapfolder
python3 heatmaptest.py --testdata <validation_data_folder> <snapshot> <heatmapfolder>
   Same as test but stores CAM heatmaps in heatmapfolder.

