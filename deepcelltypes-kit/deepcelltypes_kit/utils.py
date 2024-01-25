import yaml
from pathlib import Path


def flatten_nested_dict(nested_dict):
    flattened = []
    for key, value in nested_dict.items():
        if value:
            flattened.append(key)
            flattened.extend(flatten_nested_dict(value))
        else:
            flattened.append(key)
    return list(sorted(set(flattened)))


def get_ct_ch_across_files(dataset_path, keyword=None):
    core_dict = {}
    core_channels = []
    for file in dataset_path.iterdir():
        if "npz.dvc" in file.name:
            meta_file = file

            if keyword is not None:
                if not file.name.startswith(keyword):
                    continue

            with open(meta_file) as f:
                meta_info = yaml.load(f, Loader=yaml.FullLoader)

            try:
                celltype_mapper = meta_info["meta"]["file_contents"]["cell_types"][
                    "mapper"
                ]
            except KeyError:
                print(f"cell type mapper not found in {file.name}")
                continue

            channels = [
                item["target"] for item in meta_info["meta"]["sample"]["channels"]
            ]

            if core_dict == {}:
                core_dict = celltype_mapper
            else:
                assert (
                    celltype_mapper == core_dict
                ), f"celltype mapper is not the same across all files, {file.name}"

            if core_channels == []:
                core_channels = channels
            else:
                assert (
                    core_channels == channels
                ), f"channels are not the same across all files, {file.name}"

    return core_dict, core_channels


def choose_channels(channel_names, channel_mapping):
    channel_mask = []
    channel_names_updated = []
    for ch in channel_names:
        if ch in channel_mapping["channels_kept"]:
            channel_mask.append(True)
            channel_names_updated.append(channel_mapping["channels_kept"][ch])
        elif ch in channel_mapping["channels_dropped"]:
            channel_mask.append(False)
        else:
            raise ValueError(f"Channel name {ch} not found in channel_mapping.yaml")

    return channel_mask, channel_names_updated



if __name__ == "__main__":
    subdir_list = [
        "Tissue-Breast/Danenberg_BreastCancer_IMC",
        "Tissue-Breast/Jackson_BreastCancer_IMC",
        "Tissue-Breast/Keren_New_MIBI",
        "Tissue-Breast/Keren_TNBC_MIBI",
        "Tissue-Breast/Liu_Validation_MIBI",
        "Tissue-Breast/Risom_DCIS_MIBI",
        "Tissue-GI/Hartmann_CRC_MIBI",
        "Tissue-GI/Hickey_Colon_CODEX",
        "Tissue-GI/Keren_GVHD_MIBI",
        "Tissue-GI/Liu_Validation_MIBI",
        "Tissue-Immune/Liu_Validation_MIBI",
        "Tissue-Lung/McCaffrey_TB_MIBI",
        "Tissue-Lung/Sorin_Lung_IMC",
        "Tissue-Lymph_Node/Liu_Validation_MIBI",
        "Tissue-Lymph_Node/McCaffrey_TB_MIBI",
        "Tissue-Musculoskeletal/Liu_Validation_MIBI",
        "Tissue-Nervous/Karimi_Brain_IMC",
        "Tissue-Nervous/McCaffrey_TB_MIBI",
        "Tissue-Pancreas/Keren_PDAC_MIBI",
        "Tissue-Renal/Liu_Validation_MIBI",
        "Tissue-Reproductive/Greenbaum_Maternal-Fetal-Interface_MIBI",
        "Tissue-Reproductive/Liu_Validation_MIBI",
        "Tissue-Reproductive/McCaffrey_TB_MIBI",
        "Tissue-Skin/Keren_Melanoma_MIBI",
        "Tissue-Skin/Liu_Validation_MIBI",
        "Tissue-Spleen/Liu_Validation_MIBI",
        "Tissue-Thymus/Liu_Validation_MIBI",
        "Tissue-Tonsil/Liu_Validation_MIBI",
    ]
    for subdir in subdir_list:
        print(subdir)
        if subdir == "Tissue-Nervous/Karimi_Brain_IMC":
            for keyword in ["BrM", "Glioma"]:
                get_ct_ch_across_files(
                    Path("/data/data-registry/data/labels/static/2d/") / subdir,
                    keyword=keyword,
                )
        else:
            get_ct_ch_across_files(
                Path("/data/data-registry/data/labels/static/2d/") / subdir
            )
