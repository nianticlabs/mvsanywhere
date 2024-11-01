import argparse

def count_high_res_frames(training_file, base_path):
    with open(training_file, 'r') as file:
        scan_names = file.read().splitlines()

    high_res_count = {}

    for scan_name in scan_names:
        valid_frames_file = f"{base_path}/{scan_name}/valid_frames_mixed.txt"
        try:
            with open(valid_frames_file, 'r') as vf_file:
                frames = vf_file.read().splitlines()
                count = sum(1 for frame in frames if frame.split()[-1] == '1')
                high_res_count[scan_name] = count
        except FileNotFoundError:
            print(f"File not found: {valid_frames_file}")

    return high_res_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Count high resolution frames.')
    parser.add_argument('--training_file', type=str, help='Path to the training file', default='/mnt/disks/arkitscenes/Training_landscape.txt')
    parser.add_argument('--base_path', type=str, help='Base path to the scans', default='/mnt/disks/arkitscenes/raw/Training')

    args = parser.parse_args()
    training_file = args.training_file
    base_path = args.base_path
    
    high_res_count = count_high_res_frames(training_file, base_path)
    
    total = 0
    for scan_name, count in high_res_count.items():
        print(f"{scan_name}: {count} high resolution frames")
        total+=count
        
    print(f"Total high resolution frames: {total}")
        