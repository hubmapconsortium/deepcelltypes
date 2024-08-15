class: CommandLineTool
cwlVersion: v1.1
baseCommand: ["python3", "../main.py"]

requirements:
  DockerRequirement:
    dockerPull: hubmap/deepcelltypes:1.0.1
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
  marker_info:
    label: JSON file containing metadata about marker panels used for prediction
    type: File
    outputBinding:
      glob: /output/marker_info.json
