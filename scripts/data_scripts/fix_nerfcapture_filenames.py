import json
import tyro
from pathlib import Path


def fix_nerfcapture_filenames(
    data_path: Path,
):
    """
    Load transforms.json and append .png to each filename in the frames list.

    Args:
        transforms_json_path (Path): The path to the transforms.json file.
 
    """
    transforms_json_path = data_path / "transforms.json"
    with open(transforms_json_path, "r") as f:
        capture_data = json.load(f)

    frame_data = capture_data["frames"]

    print(f"Found {len(frame_data)} frames in {transforms_json_path}.")
    print("Adding .png to each filename in the frames list...")
    for frame in frame_data:
        file_path = Path(frame["file_path"])
        if not file_path.suffix:
            new_file_path = file_path.with_suffix(".png")
            frame["file_path"] = str(new_file_path)

    with open(transforms_json_path, "w") as f:
        json.dump(capture_data, f, indent=4)

    print("Done!")


if __name__ == "__main__":
    tyro.cli(fix_nerfcapture_filenames)