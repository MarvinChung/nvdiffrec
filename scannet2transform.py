import os
import json
import numpy as np
import argparse
import shutil
from PIL import Image
import tqdm
import random
from natsort import natsorted

def generate_transforms_json(scannet_scene_path, output_path):
    """
    Generates transforms.json for NeRF training from a ScanNet scene with resized and cropped images.
    Also generates transforms_validate.json by randomly sampling 10% of the frames.

    Args:
        scannet_scene_path (str): Path to the ScanNet scene folder.
        output_path (str): Path to save the generated transforms.json file.
    """
    # Paths to the data
    color_folder = os.path.join(scannet_scene_path, 'color')
    pose_folder = os.path.join(scannet_scene_path, 'pose')
    intrinsic_file = os.path.join(scannet_scene_path, 'intrinsic', 'intrinsic_color.txt')

    # Check if paths exist
    if not os.path.exists(color_folder):
        print(f"Color folder not found at {color_folder}")
        return
    if not os.path.exists(pose_folder):
        print(f"Pose folder not found at {pose_folder}")
        return
    if not os.path.exists(intrinsic_file):
        print(f"Intrinsic file not found at {intrinsic_file}")
        return

    # Load camera intrinsics
    intrinsics = np.loadtxt(intrinsic_file)
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    # Desired image dimensions
    resized_width, resized_height = 640, 480
    crop_width, crop_height = 624, 468

    # Get original image dimensions
    image_files = natsorted([f for f in os.listdir(color_folder) if f.endswith('.jpg') or f.endswith('.png')])
    if not image_files:
        print(f"No images found in {color_folder}")
        return

    sample_image_path = os.path.join(color_folder, image_files[0])
    with Image.open(sample_image_path) as sample_image:
        original_width, original_height = sample_image.size

    # Calculate scaling factors based on original image size
    scale_x = resized_width / original_width
    scale_y = resized_height / original_height

    # Adjust intrinsics due to resizing
    fx *= scale_x
    fy *= scale_y
    cx *= scale_x
    cy *= scale_y

    # Adjust intrinsics due to cropping (central crop)
    left = (resized_width - crop_width) // 2
    top = (resized_height - crop_height) // 2
    cx -= left
    cy -= top

    # Prepare the transforms.json structure
    transforms = {
        "camera_angle_x": 2 * np.arctan(crop_width / (2 * fx)),
        "camera_angle_y": 2 * np.arctan(crop_height / (2 * fy)),
        "fl_x": fx,
        "fl_y": fy,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "cx": cx,
        "cy": cy,
        "w": crop_width,
        "h": crop_height,
        "aabb_scale": 1,
        "frames": []
    }

    # Output directories
    resized_folder = os.path.join(scannet_scene_path, 'resized')
    os.makedirs(resized_folder, exist_ok=True)

    # Iterate over images and poses
    ct = 0
    for image_filename in tqdm.tqdm(image_files, desc='Processing images'):
        src_image_path = os.path.join(color_folder, image_filename)
        idx = os.path.splitext(image_filename)[0]
        pose_file = os.path.join(pose_folder, f'{idx}.txt')
        # Load pose matrix
        if not os.path.exists(pose_file):
            print(f"Pose file not found for image {image_filename}, skipping.")
            breakpoint()
            continue

        # Load the pose matrix
        pose = np.loadtxt(pose_file)  # 4x4 matrix

        # Check if the pose matrix is valid
        if pose.shape != (4, 4):
            raise ValueError("Pose matrix must be a 4x4 matrix")
        
        # Scannet is c2w
        c2w = pose #np.linalg.inv(pose)

        # # colmap transform (not for our mesh)
        # c2w[0:3,2] *= -1 # flip the y and z axis
        # c2w[0:3,1] *= -1
        # c2w = c2w[[1,0,2,3],:]
        # c2w[2,:] *= -1 # flip whole world upside down

        # My transform 
        # c2w[0:3,0] *= -1
        # c2w[0:3,2] *= -1 # flip the y and z axis
        # c2w[0:3,1] *= -1
        # c2w = c2w[[1,0,2,3],:]

        # For 2d gaussain
        # It change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        # but we don't need it, so we cancel that operation
        c2w[:3, 1:3] *= -1

        # Convert to list for JSON serialization
        transform_matrix = c2w.tolist()

        # # Process and save the image
        # img = Image.open(src_image_path).convert('RGBA')  # Convert to RGBA to add alpha channel
        # img = img.resize((resized_width, resized_height), Image.BILINEAR)
        # img = img.crop((left, top, left + crop_width, top + crop_height))

        # # Save the processed image to the resized folder as PNG with alpha channel
        # dst_image_filename = os.path.splitext(image_filename)[0] + '.png'
        # dst_image_path = os.path.join(resized_folder, dst_image_filename)
        # img.save(dst_image_path, format='PNG')

        # Add frame to transforms
        frame = {
            "file_path": os.path.join('resized', os.path.splitext(image_filename)[0]),
            "transform_matrix": transform_matrix
        }
        transforms["frames"].append(frame)
        ct += 1
        # if ct > 1:
        #     break

    # Save transforms.json
    with open(output_path, 'w') as f:
        json.dump(transforms, f, indent=4)

    new_name = 'transforms_train.json'
    dst = os.path.join(os.path.dirname(output_path), new_name)

    # Copy and rename the file
    shutil.copy(args.output_path, dst)

    print(f"transforms.json saved to {output_path} and {dst}")
    print(f"Total frames: {len(transforms['frames'])}")

    # Generate transforms_validate.json by sampling 10% of the frames
    num_frames = len(transforms['frames'])
    num_val_frames = max(1, int(num_frames * 0.1))  # Ensure at least one frame is selected
    random.seed(42)  # Set a seed for reproducibility
    val_frames = random.sample(transforms['frames'], num_val_frames)

    # Create a set of validation frame file paths for quick lookup
    val_frame_paths = set(frame['file_path'] for frame in val_frames)

    # Get the remaining frames for training
    train_frames = [frame for frame in transforms['frames'] if frame['file_path'] not in val_frame_paths]

    # Create transforms_validate.json
    transforms_validate = {
        # Copy necessary fields from transforms
        "camera_angle_x": transforms["camera_angle_x"],
        "camera_angle_y": transforms.get("camera_angle_y"),
        "fl_x": transforms.get("fl_x"),
        "fl_y": transforms.get("fl_y"),
        "k1": transforms.get("k1", 0.0),
        "k2": transforms.get("k2", 0.0),
        "p1": transforms.get("p1", 0.0),
        "p2": transforms.get("p2", 0.0),
        "cx": transforms.get("cx"),
        "cy": transforms.get("cy"),
        "w": transforms.get("w"),
        "h": transforms.get("h"),
        "aabb_scale": transforms.get("aabb_scale", 1),
        "frames": val_frames
    }

    # Save transforms_validate.json
    validate_output_path = os.path.splitext(output_path)[0] + '_test.json'
    with open(validate_output_path, 'w') as f:
        json.dump(transforms_validate, f, indent=4)

    print(f"transforms_validate.json saved to {validate_output_path}")
    print(f"Validation frames: {len(transforms_validate['frames'])}")

    # Optionally, create transforms_train.json with the remaining frames
    transforms_train = {
        # Copy necessary fields from transforms
        # "camera_angle_x": transforms["camera_angle_x"],
        # "camera_angle_y": transforms.get("camera_angle_y"),
        "fl_x": transforms.get("fl_x"),
        "fl_y": transforms.get("fl_y"),
        "k1": transforms.get("k1", 0.0),
        "k2": transforms.get("k2", 0.0),
        "p1": transforms.get("p1", 0.0),
        "p2": transforms.get("p2", 0.0),
        "cx": transforms.get("cx"),
        "cy": transforms.get("cy"),
        "w": transforms.get("w"),
        "h": transforms.get("h"),
        "aabb_scale": transforms.get("aabb_scale", 1),
        "frames": train_frames
    }

    # Save transforms_train.json
    train_output_path = os.path.splitext(output_path)[0] + '_train_split.json'
    with open(train_output_path, 'w') as f:
        json.dump(transforms_train, f, indent=4)

    print(f"transforms_train.json saved to {train_output_path}")
    print(f"Training frames: {len(transforms_train['frames'])}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate transforms.json from ScanNet scene for NeRF training.')
    parser.add_argument('--scannet_scene_path', type=str, required=True, help='Path to the ScanNet scene folder.')
    parser.add_argument('--output_path', type=str, default='transforms.json', help='Output path for transforms.json.')

    args = parser.parse_args()

    generate_transforms_json(args.scannet_scene_path, args.output_path)
