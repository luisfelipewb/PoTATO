from label_studio_converter.brush import image2annotation
import os
import json

masks_dir = "./runs/or_masks"
masks_files = os.listdir(masks_dir)
masks_files.sort()

tasks = []

# count = 0
for mask_file in masks_files:
    print(f"Processing {mask_file}")
    test_image_path = os.path.join(masks_dir, mask_file)

    annotation = image2annotation(
        test_image_path,
        label_name='bottle',
        from_name='tag',
        to_name='image',
        ground_truth=False, 
        model_version=None, 
        score=None
    )

    image_name = mask_file.replace("_mask.png", "_rgb.png")
    base_url = "http://morgoth:8081/"
    url = base_url + image_name

    task = {
        'data': {'image': url},
        'predictions': [annotation],
    }

    tasks.append(task)
    # count += 1
    # if count > 9:
    #     break

    
# json.dump(tasks, open("label_studio_tasks.json", "w"), indent=4)
json.dump(tasks, open("label_studio_tasks.json", "w"))
