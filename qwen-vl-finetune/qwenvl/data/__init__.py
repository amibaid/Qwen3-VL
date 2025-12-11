import re

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": f"",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

HD_EPIC_MINI = {
    "annotation_path": "/home/aryan/ami/381V-final/381V-final-project/HD-EPIC/10_annotations.json",
    "data_path": "/home/aryan/ami/381V-final/data/trimmed_clips_2",
}
HD_EPIC_BOTH_256 = {
    "annotation_path": "/home/aryan/ami/381V-final/381V-final-project/HD-EPIC/p01_annotations_verified.json",
    "data_path": "/home/aryan/ami/381V-final/data/gaze_crops_256",
}
HD_EPIC_OG_512 = {
    "annotation_path": "/home/aryan/ami/381V-final/381V-final-project/HD-EPIC/p01_annotations_verified.json",
    "data_path": "/home/aryan/ami/381V-final/data/original_crops_512",
}
HD_EPIC_GAZE_512 = {
    "annotation_path": "/home/aryan/ami/381V-final/381V-final-project/HD-EPIC/p01_annotations_verified.json",
    "data_path": "/home/aryan/ami/381V-final/data/gaze_only_512",
}

data_dict = {
    "hd_epic_mini": HD_EPIC_MINI,
    "hd_epic_both_256": HD_EPIC_BOTH_256,
    "hd_epic_gaze_512": HD_EPIC_GAZE_512,
    "hd_epic_og_512": HD_EPIC_OG_512,
    "cambrian_737k": CAMBRIAN_737K,
    "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
