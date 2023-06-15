class: CommandLineTool
cwlVersion: v1.2
baseCommand: ["python3", "../main.py"]

requirements:
  DockerRequirement:
    dockerImageId: vanvalenlab/deepcelltypes-hubmap:latest
    dockerOutputDirectory: /output

inputs:
  data_dir:
    label: Directory containing pipeline outputs
    type: Directory
    inputBinding:
      prefix: "--data-dir"
  image_fname:
    label: Tiff file containing multiplexed image
    type: string
    default: "reg001_expr.ome.tiff"
    inputBinding:
      prefix: "--image-fname"
  segmask:
    label: Tiff file containing segmentation mask
    type: string
    default: "reg001_mask.ome.tiff"
    inputBinding:
      prefix: "--segmask"

outputs:
  celltypes:
    label: CSV file containing cell-type predictions from deepcelltypes
    type: File
    outputBinding:
      glob: /output/deepcelltypes_predictions.csv
