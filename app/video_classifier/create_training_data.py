import argparse
import csv
import json
import math
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Constants
MAX_FILES_PER_SUBDIR = 1000
VIDEO_EXTENSIONS = (".mp4", ".webm")


class GlobalSegmentIndex:
    """Thread-safe counter for segment indexing, ensuring unique filenames."""

    def __init__(self):
        self._index = -1  # Start from -1 so first call to get_next_index makes it 0
        self._lock = threading.Lock()

    def get_next_index(self):
        with self._lock:
            self._index += 1
            return self._index


def get_video_duration(video_path):
    """
    Gets the duration of a video in seconds using ffprobe.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"Error getting duration for {video_path}: {e}")
        return None


def split_video(
    input_path,
    output_dir_base,
    segment_seconds,
    label_id,
    global_segment_index_manager,
    csv_writer_func,
    output_subdir_state,
):
    """
    Splits a single video into segments and writes entries to CSV.

    Args:
        input_path (str): Absolute path to the input video file.
        output_dir_base (str): Base path for the output directory.
        segment_seconds (int): Maximum length in seconds for each video segment.
        label_id (int): The label ID for this video's segments.
        global_segment_index_manager (GlobalSegmentIndex): Thread-safe manager for global segment index.
        csv_writer_func (callable): Function to write a row to the corpus CSV (filepath, label_id).
        output_subdir_state (dict): Thread-safe dictionary managing the current output subdirectory index and file count.

    Returns:
        list: A list of (output_filepath, label_id) for generated segments.
    """
    duration = get_video_duration(input_path)
    if duration is None:
        return []

    num_segments = math.ceil(duration / segment_seconds)
    generated_segments_info = []

    print(
        f"Processing video: {input_path} (Duration: {duration:.2f}s, Segments: {num_segments})"
    )

    for i in range(num_segments):
        start_time = i * segment_seconds

        # Manage output subdirectory and filename with thread-safe lock
        with output_subdir_state["lock"]:
            subdir_index = output_subdir_state["index"]

            target_subdir = os.path.join(output_dir_base, f"{subdir_index:04d}")
            os.makedirs(target_subdir, exist_ok=True)

            segment_global_index = global_segment_index_manager.get_next_index()
            segment_filename = f"segment_{segment_global_index:08d}.mp4"
            output_filepath = os.path.join(target_subdir, segment_filename)

            output_subdir_state["file_count"] += 1
            if output_subdir_state["file_count"] >= MAX_FILES_PER_SUBDIR:
                output_subdir_state["index"] += 1
                output_subdir_state["file_count"] = 0

        # FFmpeg command for splitting and re-encoding
        # Using H.264 (libx264) for video and AAC for audio with reasonable quality settings.
        # -map 0:v:0 and -map 0:a:0? select the first video and optional first audio stream.
        cmd = [
            "ffmpeg",
            "-i",
            input_path,
            "-ss",
            str(start_time),
            "-t",
            str(segment_seconds),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",  # Faster encoding preset
            "-crf",
            "23",  # Constant Rate Factor (0-51, lower is better quality/larger file)
            "-c:a",
            "aac",
            "-b:a",
            "128k",  # Audio bitrate
            "-map",
            "0:v:0",  # Map first video stream
            "-map",
            "0:a:0?",  # Map first audio stream if it exists (optional)
            "-y",  # Overwrite output files without asking
            output_filepath,
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            csv_writer_func(
                os.path.abspath(output_filepath), label_id
            )  # Write absolute path to CSV
            generated_segments_info.append((output_filepath, label_id))
            print(f"  Generated: {output_filepath} (Label ID: {label_id})")
        except subprocess.CalledProcessError as e:
            print(
                f"  Error splitting {input_path} segment {i} ({start_time}s): {e.stderr.decode().strip()}"
            )
        except Exception as e:
            print(
                f"  An unexpected error occurred during splitting {input_path} segment {i}: {e}"
            )

    return generated_segments_info


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data from video directories."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Input directory containing subdirectories of labeled videos.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory to save processed data.",
    )
    parser.add_argument(
        "--num_seconds",
        type=int,
        default=10,
        help="Maximum length in seconds for each output video segment.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of worker processes/threads for video processing. Defaults to CPU count.",
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    num_seconds = args.num_seconds
    num_workers = args.workers

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory '{output_dir}' ensured.")

    # 1. Label ID Assignment
    labels_map = {}
    label_id_counter = 0

    # Get all subdirectories (labels) from the input_dir
    label_names = sorted(
        [
            d
            for d in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, d)) and not d.startswith(".")
        ]
    )

    if not label_names:
        print(f"No label subdirectories found in '{input_dir}'. Exiting.")
        return

    for label_name in label_names:
        labels_map[label_name] = label_id_counter
        label_id_counter += 1

    labels_json_path = os.path.join(output_dir, "labels.json")
    with open(labels_json_path, "w", encoding="utf-8") as f:
        json.dump(
            labels_map, f, indent=4, ensure_ascii=False
        )  # ensure_ascii=False for proper display of non-ASCII chars
    print(f"Labels mapping saved to '{labels_json_path}'")
    print(f"Labels: {labels_map}")

    # 2. Prepare corpus.csv
    corpus_csv_path = os.path.join(output_dir, "corpus.csv")
    csv_file = open(corpus_csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["filepath", "label_id"])  # Write header

    # Protect CSV writing with a lock for multi-threading
    csv_write_lock = threading.Lock()

    def write_to_csv_thread_safe(filepath, label_id):
        with csv_write_lock:
            csv_writer.writerow([filepath, label_id])
            csv_file.flush()  # Ensure data is written to disk immediately

    # State for managing output subdirectories (index and file count)
    # Protected by a lock for thread-safe access and modification.
    output_subdir_state = {"index": 0, "file_count": 0, "lock": threading.Lock()}

    # Manager for global segment index, ensuring unique filenames across all segments
    global_segment_index_manager = GlobalSegmentIndex()

    # 3. Collect all video paths to process
    video_paths_to_process = []
    for label_name, label_id in labels_map.items():
        label_dir = os.path.join(input_dir, label_name)
        if not os.path.isdir(label_dir):
            print(f"Warning: Label directory '{label_dir}' not found, skipping.")
            continue
        for filename in os.listdir(label_dir):
            if filename.lower().endswith(VIDEO_EXTENSIONS) and not filename.startswith(
                "."
            ):
                video_path = os.path.join(label_dir, filename)
                video_paths_to_process.append((video_path, label_id))

    if not video_paths_to_process:
        print(
            f"No video files found in '{input_dir}' subdirectories matching {VIDEO_EXTENSIONS}. Exiting."
        )
        csv_file.close()
        return

    print(
        f"Found {len(video_paths_to_process)} video files to process using {num_workers} workers."
    )

    # Use ThreadPoolExecutor for concurrent video processing.
    # Python's GIL is released when subprocess.run (which calls ffmpeg) is executed,
    # so threads are effective here for I/O-bound ffmpeg tasks.
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit each video splitting task to the executor
        futures = [
            executor.submit(
                split_video,
                video_path,
                output_dir,
                num_seconds,
                label_id,
                global_segment_index_manager,
                write_to_csv_thread_safe,
                output_subdir_state,
            )
            for video_path, label_id in video_paths_to_process
        ]

        # Monitor progress of completed tasks
        for i, future in enumerate(as_completed(futures)):
            try:
                # The result of split_video is a list of generated segments info,
                # but we've already written to CSV as a side effect.
                future.result()
                print(f"Completed {i + 1}/{len(futures)} video files.")
            except Exception as exc:
                print(f"One video processing task generated an exception: {exc}")

    csv_file.close()
    print(f"Training data generation complete. Corpus saved to '{corpus_csv_path}'")


if __name__ == "__main__":
    main()
