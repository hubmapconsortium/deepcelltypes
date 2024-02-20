import os
from pathlib import Path
import yaml
import numpy as np
from .utils import flatten_nested_dict
from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt
import networkx as nx


def _add_nodes_and_edges(G, data, parent=None):
    if isinstance(data, dict):
        for key, value in data.items():
            G.add_node(key)
            if parent is not None:
                G.add_edge(parent, key)
            _add_nodes_and_edges(G, value, parent=key)
    elif isinstance(data, list):
        for item in data:
            _add_nodes_and_edges(G, item, parent=parent)
    elif isinstance(data, str):
        G.add_node(data)
        if parent is not None:
            G.add_edge(parent, data)


class DCTConfig:
    def __init__(self):
        self.SEED = 0
        self.MAX_NUM_CHANNELS = 52
        self.BATCH_SIZE = 400
        self.CHUNK_SIZE = 4096  # 4096 cells per tfrecord
        self.MAX_CHUNK_PER_CT_PER_DATASET = 25

        self.HIST_NORM_KERNEL_SIZE = 128
        self.CROP_SIZE = 64
        self.PATCH_RESIZE_SIZE = 32

        self.STANDARD_MPP_RESOLUTION = 0.5  # TODO: use this to adjust resolution

        self.data_folder = Path(os.path.dirname(__file__)) / "config"

        self._mapper_dict, self._core_tree = self._load_mapper_dict_and_core_tree()
        self._channel_mapping = self._load_channel_mapping()
        self._positivity_mapping = self._positivity_mapping()
        self._positivity_mapping_dataset_specific = (
            self._positivity_mapping_dataset_specific()
        )
        self._celltype_mapping = self._load_celltype_mapping()
        self._master_channels = self._load_master_channels()
        self._domain_mapping = self._load_domain_mapping()
        self._color_mapping = self._load_color_mapping()
        self._dataset_source_mapping = self._load_dataset_source_mapping()

    @property
    def mapper_dict(self):
        return self._mapper_dict

    @property
    def core_tree(self):
        return self._core_tree

    def _load_mapper_dict_and_core_tree(self):
        with open(self.data_folder / "core_tree.yaml", "r") as f:
            core_tree = yaml.safe_load(f)

        with open(self.data_folder / "cell_count_total.txt", "r") as f:
            cell_count_total = yaml.safe_load(f)

        master_celltype_list = flatten_nested_dict(core_tree)
        master_celltype_list_updated = []
        for celltype in master_celltype_list:
            if cell_count_total[celltype] == 0:
                continue
            master_celltype_list_updated.append(celltype)

        mapper_dict = dict(
            zip(
                master_celltype_list_updated,
                range(1, 1 + len(master_celltype_list_updated)),
            )
        )
        return mapper_dict, core_tree

    @property
    def channel_mapping(self):
        return self._channel_mapping

    def _load_channel_mapping(self):
        with open(self.data_folder / "channel_mapping.yaml", "r") as f:
            channel_mapping = yaml.safe_load(f)
        return channel_mapping

    @property
    def positivity_mapping(self):
        return self._positivity_mapping

    def _positivity_mapping(self):
        with open(
            self.data_folder / "positivity_mapping.yaml",
            "r",
        ) as f:
            positivity_mapping = yaml.safe_load(f)
        return positivity_mapping

    @property
    def positivity_mapping_dataset_specific(self):
        return self._positivity_mapping_dataset_specific

    def _positivity_mapping_dataset_specific(self):
        with open(
            self.data_folder / "positivity_mapping_dataset_specific.yaml",
            "r",
        ) as f:
            positivity_mapping_dataset_specific = yaml.safe_load(f)
        return positivity_mapping_dataset_specific

    @property
    def celltype_mapping(self):
        return self._celltype_mapping

    def _load_celltype_mapping(self):
        with open(self.data_folder / "celltype_mapping.yaml", "r") as f:
            celltype_mapping_config = yaml.load(f, Loader=yaml.FullLoader)

        return celltype_mapping_config

    # TODO: delete this file and generate master_channels from channel_mapping
    @property
    def master_channels(self):
        return self._master_channels

    def _load_master_channels(self):
        with open(self.data_folder / "master_channels.yaml", "r") as f:
            master_channels = yaml.load(f, Loader=yaml.FullLoader)
        return master_channels

    @property
    def domain_mapping(self):
        return self._domain_mapping

    def _load_domain_mapping(self):
        with open(self.data_folder / "domain_mapping.yaml", "r") as f:
            domain_mapping = yaml.load(f, Loader=yaml.FullLoader)

        new_domain_mapping = {}
        for key, value in domain_mapping.items():
            new_domain_mapping[key.replace("/", "-")] = value
        return new_domain_mapping

    @property
    def color_mapping(self):
        return self._color_mapping

    def _load_color_mapping(self):
        with open(self.data_folder / "colors.yaml", "r") as f:
            color_mapping = yaml.load(f, Loader=yaml.FullLoader)
        color_mapping = {k: to_rgb(v) for k, v in color_mapping.items()}
        return color_mapping

    def plot_celltype_tree(self):
        # Create a directed graph
        G = nx.DiGraph()

        # Parse the YAML data and build the graph
        _add_nodes_and_edges(G, self.core_tree)

        color_mapping_updated = self.color_mapping.copy()
        # Make Cell, Unknown, Background gray... black and white are not good for visualization
        color_mapping_updated["Cell"] = (0.4, 0.4, 0.4)
        color_mapping_updated["Unknown"] = (0.4, 0.4, 0.4)
        color_mapping_updated["Background"] = (0.9, 0.9, 0.9)
        node_colors = [color_mapping_updated[ct] for ct in G.nodes]

        # Visualization
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        node_labels = {node: node for node in G.nodes}
        fig, ax = plt.subplots(figsize=(12, 5))
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=1000,
            node_color=node_colors,
            font_size=10,
            font_color="black",
            # font_weight="bold",
            ax=ax,
        )
        plt.title("Hierarchical Graph of Cell Types")
        plt.axis("off")
        plt.show()

    def plot_celltype_mapping(self):
        # drop the duplicated dataset (they were splitted into different tissues)
        celltype_mapping_merged = {
            key.split("/")[1]: val for key, val in self.celltype_mapping.items()
        }

        celltype_counts = {}
        for dataset_celltypes in celltype_mapping_merged.values():
            for celltype in dataset_celltypes.values():
                celltype_counts[celltype] = celltype_counts.get(celltype, 0) + 1

        # Sort celltypes by counts in descending order
        sorted_celltypes = sorted(
            celltype_counts.keys(), key=lambda x: celltype_counts[x], reverse=True
        )

        # Extract the unique datasets and celltypes
        unique_datasets = list(celltype_mapping_merged.keys())
        unique_celltypes = sorted_celltypes

        # Create a binary matrix to represent the heatmap
        binary_heatmap = np.zeros(
            (len(unique_datasets), len(unique_celltypes)), dtype=int
        )

        for i, dataset in enumerate(unique_datasets):
            for j, celltype in enumerate(unique_celltypes):
                if celltype in celltype_mapping_merged[dataset].values():
                    binary_heatmap[i, j] = 1

        # Create the heatmap
        plt.figure(figsize=(10, 6))
        plt.imshow(binary_heatmap, cmap="binary", aspect="auto", interpolation="none")

        # Customize the plot
        plt.xticks(np.arange(len(unique_celltypes)), unique_celltypes, rotation=90)
        plt.yticks(np.arange(len(unique_datasets)), unique_datasets)
        plt.xlabel("Celltypes")
        plt.ylabel("Datasets")
        plt.title("Binary Heatmap of Datasets and Celltypes")

        # Show the plot
        plt.tight_layout()
        plt.grid()
        plt.show()

    def plot_channel_mapping(self):
        # drop the duplicated dataset (they were splitted into different tissues)
        channel_mapping_merged = {
            key.split("/")[1]: val["channels_kept"]
            for key, val in self.channel_mapping.items()
        }

        channel_counts = {}
        for dataset_channels in channel_mapping_merged.values():
            for channel in dataset_channels.values():
                channel_counts[channel] = channel_counts.get(channel, 0) + 1

        # Sort channels by counts in descending order
        sorted_channels = sorted(
            channel_counts.keys(), key=lambda x: channel_counts[x], reverse=True
        )

        # Extract the unique datasets and channels
        unique_datasets = list(channel_mapping_merged.keys())
        unique_channels = sorted_channels

        # Create a binary matrix to represent the heatmap
        binary_heatmap = np.zeros(
            (len(unique_datasets), len(unique_channels)), dtype=int
        )

        for i, dataset in enumerate(unique_datasets):
            for j, channel in enumerate(unique_channels):
                if channel in channel_mapping_merged[dataset].values():
                    binary_heatmap[i, j] = 1

        # Create the heatmap
        plt.figure(figsize=(10, 6))
        plt.imshow(binary_heatmap, cmap="binary", aspect="auto", interpolation="none")

        # Customize the plot
        plt.xticks(np.arange(len(unique_channels)), unique_channels, rotation=90)
        plt.yticks(np.arange(len(unique_datasets)), unique_datasets)
        plt.xlabel("Channels")
        plt.ylabel("Datasets")
        plt.title("Binary Heatmap of Datasets and Channels")

        # Show the plot
        plt.tight_layout()
        plt.grid()
        plt.show()

    @property
    def dataset_source_mapping(self):
        return self._dataset_source_mapping
    
    def _load_dataset_source_mapping(self):
        mapping = {}
        for key, val in self.domain_mapping.items():
            mapping[key] = "-".join(key.split("-")[2:])
        return mapping


if __name__ == "__main__":
    dct_config = DCTConfig()

    print(dct_config.__dict__)
    print(dct_config.dataset_source_mapping)
