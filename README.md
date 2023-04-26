# Cell-type prediction model for HubMAP datasets

This repo provides an interface to the `deepcelltypes` cell-type
classification model.

The `deepcelltypes` model requires a multiplexed image and a
corresponding segmentation mask; the latter can be generated with
Mesmer via [`deepcell-applications`](https://github.com/vanvalenlab/deepcell-applications).
The output of the model is a `.csv` file with four columns:
Cell mask ID (int), Cell centroid x, y in pix coords (float), and the
predicted cell type (string).

The packaged version of the model was developed for use with
Hickey CODEX data on Hubmap; sepcifically the `HBM742.NHHQ.357` dataset.

## Setup

The recommended way to run the model is via docker. As this model is
not yet publicly available, the image must be built:

```bash
docker build -f Dockerfile -t vanvalenlab/deepcelltypes-hubmap .
```

## Run

The available options for the containerized model can be viewed:

```bash
$ docker run --rm vanvalenlab/deepcelltypes-hubmap --help

Options:
  --data-dir TEXT     Path to hubmap data
  --image-fname TEXT  Filename of the multiplexed .tif image in processed
                      CODEX format.
  --segmask TEXT      Path to .tif file containing the segmentation mask.
  --z-slice INTEGER   Which z-slice in the image to run the 2D celltype
                      prediction on
  --output-file TEXT  Name of output csv file containing the celltype
                      predictions
  --help              Show this message and exit.
```

Note there are some hard-coded components that depend on the specific structure
of the `HBM742.NHHQ.357` dataset on HubMAP.
In principle, the model would work as-packaged on any CODEX dataset with the same
file structure and metadata, but this hasn't been tested.
The primary goal of this package is to aid in the preliminary development of
the data pipelines for applying the model across all the multiplexed datasets
available on hubmap.

## Example

The following example assumes the `HBM742.NHHQ.357` dataset has been downloaded
from Globus to `/data/HBM742.NHHQ.357`.
The segmentation mask was computed separately via `deepcell-applications` extracting
the 0th and 86th channel from the multiplexed image as the nuclear and membrane
channels, respectively, as recommended in the metadata.
The example assumes that the mask is stored in the `$HOME` directory, which also
serves as the output location for the celltype prediction `csv`.

The following applies the model to the CODEX slice at X=4, Y=3, Z=9:

```bash
docker run --rm --user $(id -u):$(id -g) \
  -v $HOME:/home -v /data/HBM742.NHHQ.357:/data \
  vanvalenlab/deepcelltypes-hubmap \
  --data-dir /data \
  --image-fname reg001_X04_Y03.tif \
  --segmask /home/mask_reg001_X04_Y03.tif \
  --z-slice 9 \
  --output-file /home/celltypes_reg001_X04_Y03.csv
```
