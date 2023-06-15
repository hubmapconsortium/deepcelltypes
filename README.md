# Cell-type prediction model for HubMAP datasets

This repo provides an interface to the `deepcelltypes` cell-type
classification model.

The interface is implemented as a CWL `CommandLineTool` for integration
with existing hubmap workflows.
The `deepcelltypes` model requires a multiplexed image and a
corresponding segmentation mask.
The current interface assumes that the data format and directory
structure conforms to that of the Hubmap Cytokit+SPRM pipeline:

```
└── pipeline_output
    ├── expr
    │   └── reg001_expr.ome.tiff
    └── mask
        └── reg001_mask.ome.tiff
```

The output of the model is a `.csv` file with four columns:
Cell mask ID (int), Cell centroid x, y in pix coords (float), and the
predicted cell type (string):

```
12,499.87,6633.12,ENDOTHELIAL
13,501.91,6388.53,ENDOTHELIAL
14,510.40,6608.69,NEUTROPHIL
15,514.16,6551.42,ENDOTHELIAL
16,519.01,7971.78,MACROPHAGE
17,519.66,8024.51,NEUTROPHIL
18,519.89,7050.21,ENDOTHELIAL
```

## Setup

The model is run in docker via CWL. As this model is
not yet publicly available, the image must be built:

```bash
docker build -f Dockerfile -t vanvalenlab/deepcelltypes-hubmap:latest .
```

## Run

The workflow can be run by providing a path to the output from the
Cytokit+SPRM pipeline.
`example_job.yml` contains the input required to run the model on the
[HBM636.GTZK.259](https://portal.hubmapconsortium.org/browse/dataset/35d53aa621e25665b1d239f1d89befb2)
dataset from the Hubmap data portal:

```
cwltool run_deepcelltypes.cwl example.json
```
