# SAROS Dataset

*Sparsely Annotated Regions and Organs Segmentation* (SAROS)

## Download

1. Install the package manager [poetry](https://python-poetry.org/docs/#installation)

2. Clone the repository
```shell
git clone https://github.com/UMEssen/saros-dataset
cd saros-dataset
```
4. Install the dependencies
```shell
poetry install --no-dev
```
3. Run the download script
```shell
poetry run python3 download.py
```

By default, the CTs and the segmentations will be downloaded as NIfTIs and resampled to 5mm thickness. If this is not desired, please read the section [Command Line Parameters](#command-line-parameters). In the DICOMs, the slices that have not been annotated have been set to zero (background), but for training a CNN they should be set to an ignore label to ensure that the CNN does not learn them as background. The ignore label can be set with the parameter `--set-ignore` (default: None, i.e., will remain zero) and will result in NIfTIs files with the chosen label for the not annotated slices.

Please note that not all collections are freely available for download. Most collections can be accessed by [creating an account on TCIA](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=23691309). For the following collections you will need to separately ask for access by filling [TCIA Restricted License Agreement](https://wiki.cancerimagingarchive.net/download/attachments/4556915/TCIA%20Restricted%20License%2020220519.pdf?version=1&modificationDate=1652964581655&api=v2) and by sending it to [help@cancerimagingarchive.net](mailto:help@cancerimagingarchive.net):
* [Head-Neck Cetuximab](https://wiki.cancerimagingarchive.net/display/Public/Head-Neck+Cetuximab)
* [ACRIN-HNSCC-FDG-PET/CT](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52763679)
* [QIN-HEADNECK](https://wiki.cancerimagingarchive.net/display/Public/QIN-HEADNECK)
* [TCGA-HNSC](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=11829589)
* [HNSCC](https://wiki.cancerimagingarchive.net/display/Public/HNSCC)
* [Anti-PD-1_MELANOMA](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=37225348)

### Command Line Parameters
* `--info-csv`: The path to the CSV file containing the information about the collections to download, this file can be downloaded from our TCIA collection. If not present, the current directory and the filename `info.csv` will be used.
* `--set-ignore`:
* `--target-dir`: The directory where the CTs and the segmentations should be stored.
* `--save-original-image`: If present, the original CT (not resampled) will be saved in the target directory with the name `image_original.nii.gz`.
* `--save-meta-dicoms`: If present, the first and the last DICOM files containing DICOM meta data will be stored.
* `--save-dicoms`: If present, all DICOMs will be stored.
* `--force-download`: If present, the download will be downloaded even if the target directory already exists.
* `--no-login`: If present, the user will not be asked to login to TCIA.
* `--parallel-downloads`: The number of parallel downloads (default: 2), please use this value carefully in order not do overload the TCIA servers.
