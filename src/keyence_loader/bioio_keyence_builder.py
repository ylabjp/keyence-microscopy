from pathlib import Path
from typing import Optional
import json
import numpy as np
from datetime import datetime, UTC, timezone
#from bioio_base.types import PhysicalPixelSizes

from ylabcommon.utils.file_selection import collect_valid_tiffs
from ylabcommon.utils.outfile_name import build_output_name, extract_dimensions, build_stack_filename
from ylabcommon.utils.summary_metadata_helper import get_enhanced_metadata, generate_file_sha256
from ylabcommon.utils.utils import hybrid, style_print
from ylabcommon.utils.report_builder import ReportBuilder

from ylabcommon.bioio.core.bioio_reader import BioIOReader
from ylabcommon.bioio.core.bioio_writer import BioIOWriter

from ylabcommon.bioio.keyence.keyence_bioio_stack_builder import stack_keyence_with_bioio_calibrated
from ylabcommon.bioio.keyence.keyence_metadata_extractor import KeyenceMetadataExtractor




class KeyenceBioioBuilder:
    """
    Full reconstruction pipeline:

    TIFF discovery → Ultra stacking → BioIOReader →
    Metadata extraction → XML validation → Output naming → Write OME
    """

    def __init__(
        self,
        tiff_dir: Path,
        #xml_file: Optional[Path],
        output_dir: Path,
        *,
        compression: str = "zlib",
        compression_level: int = 6,
        validate_metadata: bool = True,
        dry_run: str = False,
    ):

        self.tiff_dir = Path(tiff_dir)
        #self.xml_file = Path(xml_file) if xml_file else None
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run

        self.compression = compression
        self.compression_level = compression_level
        self.validate_metadata = validate_metadata

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # TIFF DISCOVERY + STACK
    # -------------------------------------------------
    """
    def _get_params(self):
        params_adapter = ThorlabParamsAdapter(self.xml_file)
        get_keyences_params = params_adapter.extract()
        return get_keyences_params
    """
    def _discover_and_stack(self):

        print("[Builder] Discovering valid TIFF files...")

        tiff_files = collect_valid_tiffs(self.tiff_dir)

        if not tiff_files:
            raise RuntimeError("No valid TIFF files found.")

        print(f"[Builder] Found {len(tiff_files)} usable TIFF files")
        print("[Builder] Ultra stacking images...")

        ##get_keyences_params = self._get_params()
        stacked_data, tiff_files, channel_names, z_max_min_re  = stack_keyence_with_bioio_calibrated(tiff_files)


        #total_depth_um = stacked_data.Z.max().values
        #print(f"Total volume depth: {total_depth_um} microns")

        #data_to_process = stacked_data.data
        return stacked_data, tiff_files, channel_names, z_max_min_re


    def _validate_keyence_stack(self, image_meta):
        
        report = {"status": "VALIDATED", "checks": []}

        def record(name, ok, msg):
            report["checks"].append({"name": name, "ok": ok, "msg": msg})
            if not ok:
                report["status"] = "NOT VALIDATED"

        # -----------------------------
        # Pixel size validation
        # -----------------------------

        pixel_sizes = image_meta.pixel_sizes
        unique_pixel_sizes = set(pixel_sizes)
        validated = len(unique_pixel_sizes) == 1

        if not validated:
             raise ValueError(
                 f"Inconsistent pixel size detected: {pixel_sizes}"
             )

        record("pixel_sizes",
            validated,
            f"meta_info={pixel_sizes} {pixel_sizes.pop()}"
            )

        # -----------------------------
        # Image dimension validation
        # -----------------------------

        dims = image_meta.dimensions
        unique_dims = set(dims)
        validated = len(unique_dims) == 1

        if not validated:
            raise ValueError(
                f"Inconsistent image dimensions detected: {dims}"
            )

        record("Dimensions",
            validated,
            f"meta_info={len(dims)} {dims.pop()}"
            )  

        # -----------------------------
        # Lens consistency
        # -----------------------------

        lenses = image_meta.lens_names 
        unique_lenses = set(lenses)
        validated = len(unique_lenses) == 1

        if not validated:
            raise ValueError(
                f"Multiple lenses detected: {lenses}" 
            )

        record("Lenses",
            validated,
            f"meta_info={len(lenses)} {lenses.pop()}" 
            )

        # -----------------------------
        # Z spacing validation
        # -----------------------------

        z_positions  = image_meta.z_positions 

        if len(z_positions) > 1:
            diffs = np.diff(z_positions)

            uniform = np.allclose(diffs, diffs[0], atol=1e-3) #Note: Need to confirm tolerance may be set

            if not uniform :
                print("Warning: Z spacing not perfectly uniform")
                z_step = np.median(diffs) / 1000  # nm → µm
            else:
                 z_step = diffs[0] / 1000 # nm → µms
        else:
            diffs = []
            uniform = True
            z_step = None

        record("Z_spacing",
           uniform,
           f"meta_info={z_positions} {z_step} {diffs}"       
           )

        style_print("\n========== Validation Results ================", "header")

        for check in report["checks"]:
            status = "PASS" if check["ok"] else "Some Parameters Not validated"
            print(f"[{status}] {check['name']}: {check['msg']}")
        print(f"Final Status: {report['status']}")
        print("=============================================\n")

        return report

    # -------------------------------------------------
    # BioIO Processing Reader
    # -------------------------------------------------
    def _load_with_meta_info(self, stacked_data, tiff_files, channel_names):

        print("[Builder] Extracting image meta data ...")
            
        extractor = KeyenceMetadataExtractor(
                tiff_files,
                stacked_data.shape,
                channel_names
        )

        image_meta = extractor.extract()

        #image_meta.dim_order = "TCZYX"

        print(f"Metadata summary:")
        print(f"Pixel size:, {image_meta.pixel_size}")
        print(f"Channels:, {channel_names}")
        print(f"Data shape:, {stacked_data.shape}")

        #return data, image_meta, hybrid_channel_name
        return image_meta
    # -------------------------------------------------
    # WRITE OUTPUT
    # -------------------------------------------------

    def _write(self, data, image_meta, output_path):

        print("[Builder] Writing OME output...")

        writer = BioIOWriter(
            output_path,
            compression=self.compression,
            compression_level=self.compression_level,
        )

        # convert xarray → numpy
        if hasattr(data, "values"):
            data = data.values

        writer.write(
            data,
            dim_order=image_meta.dim_order,
            channel_names=None,
            #physical_pixel_sizes=phys_sizes,
            physical_pixel_sizes=image_meta.pixel_size,
        )

    # -------------------------------------------------
    # Validation report
    # -------------------------------------------------

    def _write_report(self, report, image_meta, output_path, hybrid_channel_name, tiff_files):

        report_path = output_path.with_suffix(".validation.json")
        extra_meta_summary = get_enhanced_metadata(image_meta, tiff_files)

        payload = {
            #"timestamp": datetime.datetime.now(datetime.UTC),
            #"timestamp": datetime.utcnow().isoformat(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **extra_meta_summary,
            "source_tiff_dir": str(self.tiff_dir),
            #"experiment_xml": str(self.xml_file) if self.xml_file else None,
            #"Channel_name_hybrid_index_str": hybrid_channel_name,
            #"image_metadata": image_meta.to_dict(),
            "validation": report,
            "software": "ylabcommon + thorlab_loader + bioio backend",
        }

        #payload.update(extra_meta_summary)

        for i in (tiff_files):
            if i == 0:  # Only do this for the first file to save time
                first_file_hash = generate_file_sha256(output_path)
                payload["integrity_check"] = {"first_file_sha256": first_file_hash}

        with open(report_path, "w") as f:
            json.dump(payload, f, indent=2)

        print(f"[Builder] Validation report → {report_path}")

    # -------------------------------------------------
    # MAIN PIPELINE
    # -------------------------------------------------

    def build(self):
        print("=============================================================================")
        print("[Builder] Starting BioIO reconstruction pipeline")

        stacked_data, tiff_files, channel_names, z_mx_min_re  = self._discover_and_stack()

        image_meta = self._load_with_meta_info(stacked_data, tiff_files, channel_names)

        report = self._validate_keyence_stack(image_meta)
        
        #output_path = build_output_name(self.output_dir, tiff_files, Z_stack_val, T_stack_val)

        image_name, dims = extract_dimensions(tiff_files)

        output_filename = build_stack_filename(self.output_dir, image_name, dims, z_mx_min_re)
        print(f"[DEBUG output_filename:] {output_filename}")

        if self.dry_run:
            style_print("[DRY RUN ENABLED]", "info")
            print("[Validating] TIFF discovery successful")
            print("[Validating] BioIO stacking successful")
            print("[Validating]  Metadata extraction successful")
            print(f"[Validating] Validation status: {report['status']}")
            print("[Skipping] file writing")
            print("[Skipping] summary JSON writing")
            print("\n    EXECUTION SUMMARY    \n")
            print(f"Input TIFF count : {len(tiff_files)}")
            print(f"Stack shape      : {stacked_data.shape}")
            print(f"Pixel size (µm)  : {image_meta.pixel_size}")
            print(f"Output name      : {output_path.name}.ome.tif")
            print("\nDry run completed successfully.\n")
            return

        #===============================================================
        #Write Skimmed/Stacked/Validated or not ome.tif file 
        #===============================================================

        if self.validate_metadata:
            style_print("Skipping Validation Run time set args.no_validate", "info")  
            self._write(stacked_data, image_meta, output_filename)
            
        else: 
            if report["status"] == "VALIDATED":
                self._write(stacked_data, image_meta, output_filename)

        #===============================================================
        #Write summary report 
        #===============================================================

        summary_report = ReportBuilder()

        summary_report.collect_dataset(
                str(self.tiff_dir), 
                "Keyence", 
                len(tiff_files)
            )

        summary_report.collect_metadata(image_meta, stacked_data)

        summary_report.set_dimensions(dims)

        summary_report.set_output(self.output_dir, output_filename)

        summary_report.finalize_validation()

        summary_report.write(self.output_dir, output_filename)

        #self._write_report(report, image_meta, output_filename, None, tiff_files) 
       
        print("[Builder] DONE.")
