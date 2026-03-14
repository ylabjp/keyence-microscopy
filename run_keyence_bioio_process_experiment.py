"""
Simple runner for BioIO; Can move with CLI also 

Usage:

uv run python bioio_run_process_experiment.py \
    --tiff-dir ./data \
    --xml ./Experiment.xml \
    --output-dir ./out
    ........................
"""

from pathlib import Path
import argparse
import shutil

from keyence_loader.bioio_keyence_builder import KeyenceBioioBuilder
from ylabcommon.utils.utils import get_theme, style_print, progress_bar
from ylabcommon.io.output_build_dir import build_output_dir_name
from ylabcommon.utils.infile_experiment_loader import extract_zip_and_find_tiffs

##Before used
#from thorlab_loader.backends.bioio_thorlab_builder import ThorlabBioioBuilder
#from thorlab_loader.utils import get_theme, style_print

# =============================================================================
# Argument Parser
# =============================================================================

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        prog="thorlab-bioio",
        description="Convert Keyence TIFF + validated OME datasets using BioIO",
    )

    parser.add_argument(
        "--tiff-dir",
        type=Path,
        help="Folder containing raw Thorlabs TIFF files",
    )

    parser.add_argument(
        "--xml",
        required=False,
        type=Path,
        help="Experiment.xml path (optional if validation disabled)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default="Output",
        help="Directory to write OME outputs rather than Tiff directory",
    )

    parser.add_argument(
        "--diff_outdirpath",
        type=str,
        default=None,
        help="Optional different base output directory"
    )

    parser.add_argument(
        "--no_validate",
        action="store_true",
        help="Disable BioIO metadata validation in Keyence",
    )

    parser.add_argument(
        "--base_path",
        type=str,
        default=None,
        help="Base path of dataset (auto-detected provide only for test run)"
    )

    parser.add_argument(
    "--infile_yaml",
    help="YAML file containing list of dataset zip paths"
    )

    parser.add_argument(
    "--singlefilerun",
    action="store_true",
    help="Run pipeline for a single dataset (no YAML input)"
    )

    parser.add_argument(
        "--compression",
        default="zlib",
        help="Compression type (zlib, lzw, etc.)",
    )

    parser.add_argument(
        "--compression-level",
        type=int,
        default=6,
        help="Compression level (0–9)",
    )

    parser.add_argument(
        "--dry_run", 
        action="store_true", 
        help="Run full pipeline without writing any output files"
    )

    parser.add_argument("--verbose", action="store_true")

    return parser 

# =============================================================================
# Main Execution
# =============================================================================

def main() -> None:

    parser = build_parser()
    args = parser.parse_args()
    
    dataset_name = args.base_path

    if args.diff_outdirpath:
        change_output_dir_path = args.diff_outdirpath
    else:
        change_output_dir_path = None

    if args.singlefilerun and not args.tiff_dir:
        parser.error("--tiff-dir is required when using --singlefilerun")
    
    theme = get_theme()

    style_print("\n========== Keyence BioIO Processing ======================\n", "header")
    style_print(f"Started at: {theme['timestamp']}", "info")
    style_print(f"TIFF directory  : {args.tiff_dir}", "info")
    style_print(f"Output directory: {args.output_dir}", "info")
    style_print("\n==========================================================", "header")
   
    # -----------------------------
    # Run Over Single file
    # -----------------------------

    if(args.singlefilerun):
        output_dir = build_output_dir_name("Keyence", args.output_dir, f"{dataset_name}", change_output_dir_path)

        builder = KeyenceBioioBuilder(
            tiff_dir=args.tiff_dir,
            output_dir=output_dir,
            compression=args.compression,
            compression_level=args.compression_level,
            validate_metadata=True if args.no_validate else False,
            dry_run=args.dry_run,
        )

        builder.build()

    # ----------------------------------------------------------
    # Run for big chunk data, input zip path's as yamal file 
    # ----------------------------------------------------------

    else : 
        zip_folders = args.infile_yaml
        dataset_dirs, top_dir = extract_zip_and_find_tiffs(zip_folders)
 
        success = []    
        skip = []
        total = len(dataset_dirs) 

        for i, d in enumerate(dataset_dirs):
            try:
                print(f"[{i+1}/{total}] Processing: {d}")
                output_dir = build_output_dir_name("Keyence", args.output_dir, f"{top_dir[i]}", change_output_dir_path)

                builder = KeyenceBioioBuilder(
                    tiff_dir=d,
                    output_dir=output_dir,
                    compression=args.compression,
                    compression_level=args.compression_level,
                    validate_metadata=True if args.no_validate else False,
                    dry_run=args.dry_run,
                )

                builder.build() 
                success.append(d)
                print("✔ Completed\n")

            except Exception as e:
                print(f"!!! Skipping {d}")
                print(e, "\n")
                if output_dir.exists():
                    shutil.rmtree(output_dir)
                    skip.append(d)

    # -----------------------------
    # Summary
    # -----------------------------

    if not args.singlefilerun:
        print("\n==============================")
        print("      PIPELINE SUMMARY")
        print("==============================")

        print(f"Total folders : {len(dataset_dirs)}")
        print(f"Successful    : {len(success)}")
        print(f"Skipped       : {len(skip)}")
        print("==============================")

        if skip:
             print("\nSkipped folders:")
             for f in skip:
                 print(f"   {f}")      

    print("=============================================================================")
    style_print("[Builder] DONE. Processing completed successfully : success")
    style_print("Check Summary and validation Report.\n", "info")
    print("================================================================================")

# =============================================================================

if __name__ == "__main__":
    main()

