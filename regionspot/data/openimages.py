from detectron2.data.datasets.register_coco import register_coco_instances
import os
from .openimages_categories import categories

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


_PREDEFINED_SPLITS_OPENIMAGES = {
    "openimages_train": ("openimages/detection/", "re_openimages_v6_train_bbox_splitdir_int_ids.json"),
    "openimages_val": ("openimages/detection/", "re_openimages_v6_train_bbox_splitdir_int_ids.json"),
}

for key, (image_root, json_file) in _PREDEFINED_SPLITS_OPENIMAGES.items():
    register_coco_instances(
        key,
        _get_builtin_metadata(),
        os.path.join("datasets", json_file) if "://" not in json_file else json_file,
        os.path.join("datasets", image_root),
    )