from detectron2.data.datasets.register_coco import register_coco_instances
import os

from .v3det_categories import categories
def _get_builtin_metadata(categories):
    id_to_name = {x['id']: x['name'] for x in categories}
    thing_dataset_id_to_contiguous_id = {i + 1: i for i in range(len(categories))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]

    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}

def _get_builtin_metadata():
    id_to_name = {x['id']: x['name'] for x in categories}
    thing_dataset_id_to_contiguous_id = {i + 1: i for i in range(len(categories))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}


_PREDEFINED_SPLITS_V3DET = {
    "v3det_train": ("v3det/V3Det/", "v3det/v3det_2023_v1_train.json"),
    "v3det_val": ("v3det/V3Det/", "v3det/v3det_2023_v1_val.json"),
}

for key, (image_root, json_file) in _PREDEFINED_SPLITS_V3DET.items():
    register_coco_instances(
        key,
        _get_builtin_metadata(),
        os.path.join("datasets", json_file) if "://" not in json_file else json_file,
        os.path.join("datasets", image_root),
    )