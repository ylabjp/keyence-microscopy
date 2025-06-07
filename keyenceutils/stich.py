import glob
import re
import tifffile as tiff
import numpy as np
import os
import pandas as pd
from keyenceutils.metainfo import ImageMetadata

class StichedImage:
    """
    StichedImage is a utility class for processing and stitching microscopy images 
    stored in a folder. It handles multiple channels, extracts metadata, and 
    creates a stitched image canvas. The class also provides functionality to save 
    the stitched image with ImageJ-compatible metadata.
    Attributes:
        __canvas_array (np.ndarray): A 3D array representing the stitched image 
            canvas, where each channel is stored as a separate layer.
        __meta_info (pd.DataFrame): A DataFrame containing metadata for all images, 
            including their positions, dimensions, and channels.
        __channels (list): A list of unique channel identifiers found in the images.
        __nm_per_pixel_values (float): The nanometers per pixel value extracted 
            from the metadata of the first image.
    Methods:
        __init__(folder_path):
            Initializes the StichedImage object by processing the images in the 
            specified folder (e.g. XY01), extracting metadata, and creating the stitched canvas.
        get_meta_info():
            Returns the metadata DataFrame containing information about the images.
        get_channels():
            Returns the list of unique channel identifiers.
        save(output_path):
            Saves the stitched image as a TIFF file with ImageJ-compatible metadata.
    """

    def __init__(self, folder_path):
        self.__canvas_array = None  # type: np.ndarray
        self.__meta_info = None  # type: pd.DataFrame
        self.__channels = None  # type: list
        print(f"Processing folder: {folder_path}")

        # find all TIFF files in specified folder
        all_tif_files = sorted(
            glob.glob(os.path.join(folder_path, "Image_*.tif"))
        )
        # Filter out TIFF files that contain "Overlay" in their filename
        all_tif_files = [os.path.basename(
            f) for f in all_tif_files if "Overlay" not in os.path.basename(f)]

        # Identify unique channels
        self.__channels = sorted(
            {re.search(r'CH\d+', f).group()
             for f in all_tif_files if re.search(r'CH\d+', f)}
        )

        assert all_tif_files is not None, "No TIFF files found in the specified folder."
        assert len(
            self.__channels) > 0, "No channels found in the specified folder."
        assert len(
            all_tif_files) > 0, "No TIFF files found in the specified folder."

        # check for Z stack images
        zstack_mode = all(
            [re.search(r'_Z\d+_', f) is not None for f in all_tif_files])

        meta_info = []
        z_max = 0

        for c_idx, channel in enumerate(self.__channels):
            print(f"Processing channel: {channel} in folder {folder_path}")
            tif_files = sorted(
                [f for f in all_tif_files if f.endswith(f"{channel}.tif")])

            d = pd.DataFrame(
                list(map(lambda tif: ImageMetadata(
                    os.path.join(
                        folder_path, tif)).get_dict(), tif_files)
                     )
            )
            d["CH_idx"] = c_idx
            d["CH"] = channel
            d["fname"] = tif_files
            meta_info.append(d)

            if zstack_mode:
                d["Z"] = d["fname"].apply(lambda x: int(re.search(r'_Z(\d+)_', x).group(1))
                                          if re.search(r'_Z(\d+)_', x) else 0)
                z_max = max(z_max, d["Z"].max()+1)

            else:
                d["Z"] = 0
                z_max = 1

            meta_info.append(d)



        meta_info = pd.concat(meta_info)
        meta_info["X_relative"] = meta_info["X"]-meta_info["X"].min()
        meta_info["Y_relative"] = meta_info["Y"]-meta_info["Y"].min()

        # Calculate the position relative to the bottom-right corner of the canvas
        canvas_width = meta_info["X_relative"].max() + meta_info["W"].max()
        canvas_height = meta_info["Y_relative"].max() + meta_info["H"].max()

        # create a canvas with czyx shape
        canvas_array = np.zeros((meta_info["CH_idx"].nunique(), z_max, canvas_height, canvas_width),
                                dtype=np.uint16
                                )  # Merged array for the canvas

        for idx, row in meta_info.iterrows():
            print(f"Processing {row['fname']}")
            # print(f"X: {row["X_relative"]}, Y: {row["Y_relative"]}")
            # Open the image and handle different modes
            img = tiff.imread(os.path.join(folder_path, row["fname"]))
            if img.ndim > 2:  # Single channel image
                raise ValueError("ERROR: multi channel image found, please check the folder")
            if img.dtype != np.uint16:
                raise ValueError("WARNING: not 16 bit image")
            
            canvas_array[
                    row["CH_idx"],
                    row["Z"],
                    row["Y_relative"]: row["Y_relative"] + row["H"],
                    row["X_relative"]: row["X_relative"] + row["W"],
                ] = np.flipud(np.fliplr(img))
                
            del img  # Free memory

        self.__canvas_array = canvas_array
        self.__meta_info = meta_info

    def get_meta_info(self):
        return self.__meta_info

    def get_channels(self):
        return self.__channels

    def save(self, output_path):
        # Save the stitched image as a TIFF file
        # Save the stitched image as a TIFF file with ImageJ compatible metadata
        metadata = {
            'axes': 'CZYX',
            'spacing': self.__meta_info["um_per_pixel"].values[0], 
            'unit': 'um',
            'finterval': 1.0,
            'finterval_unit': 's',
            'hyperstack': True,
            'mode': 'composite',
            'Info': self.__meta_info.to_json(orient='records')
        }
        tiff.imwrite(
            output_path,
            self.__canvas_array,
            photometric='minisblack',
            metadata=metadata
        )
        pass
