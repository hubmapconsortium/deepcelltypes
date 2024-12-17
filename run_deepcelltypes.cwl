class: CommandLineTool
cwlVersion: v1.1
baseCommand: ["python3", "../main.py"]

requirements:
  DockerRequirement:
    dockerPull: hubmap/deepcelltypes:1.0.5
    dockerOutputDirectory: /output
  DockerGpuRequirement: {}

inputs:
  data_dir:
    label: Directory containing pipeline outputs
    type: Directory
    inputBinding:
      position: 0

outputs:
  celltypes:
    label: CSV file containing cell-type predictions from deepcelltypes
    type: Directory
    outputBinding:
      glob: deepcelltypes
  marker_info:
    label: JSON file containing metadata about marker panels used for prediction
    type: File
    outputBinding:
      glob: /output/marker_info.json
