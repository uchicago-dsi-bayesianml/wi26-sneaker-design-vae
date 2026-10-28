# Data folder

The processed sneaker images are not committed. Notebook `01_data_preprocessing.ipynb` fills this folder.

1. Download the UT Zappos50K images from the [dataset page](https://vision.cs.utexas.edu/projects/finegrained/utzap50k/) (`ut-zap50k-images.zip`, 291 MB) or from [Kaggle](https://www.kaggle.com/datasets/aryashah2k/large-shoe-dataset-ut-zappos50k) and unzip them anywhere.
2. In notebook 01, set `RAW_DATA_DIR` to the `Shoes/Sneakers and Athletic Shoes` folder of the unzipped images.
3. Run the notebook. It writes 12,856 images at 64 x 64 pixels into this folder, one subfolder per brand, which the training notebooks read with `torchvision.datasets.ImageFolder`.

The dataset is for academic, non-commercial use only. See the dataset page for its citation and terms.
