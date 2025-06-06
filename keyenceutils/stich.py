import subprocess
import glob
import xml.etree.ElementTree as ET
import re
import tifffile as tiff
import numpy as np
import os
import pandas as pd

SAVE_XML = True


class ImageMetadata:
    """
    A class to extract XML data from a TIFF file and parse coordinates (X and Y) and dimensions (width & height).
    Attributes:
        image_positions (tuple): A tuple containing the X and Y coordinates.
        dimensions (tuple): A tuple containing the width and height of the image.
        nm_per_pixel_values (float): The conversion factor from nanometers to pixels.
    Methods:
        __init__(tif):
            Initializes the ImageMetadata object by extracting and parsing XML data from the given TIFF file.
    """

    def __init__(self, tif: str):
        self.__xml_file = tif.replace('.tif', '.xml')
        self.image_positions: tuple[float] = None
        self.dimensions: tuple[int] = None
        self.nm_per_pixel_values: float = None
        self.lens_name = None
        self.exposure_time = None

        # Read TIFF file as binary
        with open(tif, "rb") as file:
            content = file.read().decode(errors="ignore")   # decode as string
        print(content)
        # Extract XML content from TIFF file (file starts with <Data> and ends with <\Data>)
        match = re.search(r"<Data>.*?</Data>", content, re.DOTALL)

        if not match:
            raise ValueError(f"Could not extract XML data from {tif}")
        xml_content = match.group(0)  # extract matched xml content

        if SAVE_XML:
            with open(self.__xml_file, "w", encoding="utf-8") as xml_out:
                xml_out.write(xml_content)

        else:
            print(f"XML not saved for {self._xml_file}")

        # Parse the XML content directly and extract coordinates and dimensions
        tree = ET.ElementTree(ET.fromstring(xml_content))
        region = tree.find('.//XyStageRegion')

        if region is None:
            raise ValueError(f"File: {self.__xml_file} | Attributes not found")
        # Extract X and Y coordinates
        self.image_positions = (
            int(region.find('X').text), int(region.find('Y').text))

        # Extract width and height from the SavingImageSize section
        self.dimensions = (
            int(tree.find('.//SavingImageSize/Width').text),
            int(tree.find('.//SavingImageSize/Height').text)
        )

        # Extract width from XyStageRegion and SavingImageSize to calculate nm_per_pixel
        # Width in nm from XyStageRegion
        # Width in pixels from SavingImageSize
        self.nm_per_pixel_values = int(region.find(
            'Width').text) / self.dimensions[0]  # Conversion factor

        # Extract LensName
        # <Lens Type="Keyence.Micro.Bio.Common.Data.Metadata.Conditions.LensCondition, Keyence.Micro.Bio.Common.Data.Metadata, Version=1.1.2.14, Culture=neutral, PublicKeyToken=null">
        # <LensName Type="System.String">PlanApo 4x 0.20/20.00mm :Default</LensName>
        lens_name = tree.find('.//LensName')
        if lens_name is not None:
            self.lens_name = lens_name.text
        else:
            self.lens_name = "LensName not found"

        # Extract ExposureTime from the XML file
        # <ExposureTime Type="Keyence.Micro.Bio.Common.Utility.KeyValueContainer.ExposureTimeKeyValueContainer, Keyence.Micro.Bio.Common.Utility.KeyValueContainer, Version=1.1.2.14, Culture=neutral, PublicKeyToken=null">
        #<Numerator Type="System.Int32">1</Numerator>
        #<Denominator Type="System.Int32">30</Denominator>
        #<Line Type="System.Int32">761</Line>
         #</ExposureTime>
        exposure_time = tree.find('.//ExposureTime')
        if exposure_time is not None:
            numerator = exposure_time.find('Numerator')
            denominator = exposure_time.find('Denominator')
            if numerator is not None and denominator is not None:
                self.exposure_time = float(numerator.text) / float(denominator.text)


    def __str__(self):
        return f"X: {self.image_positions[0]}, Y: {self.image_positions[1]}, Width: {self.dimensions[0]}, Height: {self.dimensions[1]}, nm_per_pixel: {self.nm_per_pixel_values}, lens_name: {self.lens_name}, Exposure_Time: {self.exposure_time}"

    def get_dict(self):
        return {
            "X": int(self.image_positions[0]/self.nm_per_pixel_values),
            "Y": int(self.image_positions[1]/self.nm_per_pixel_values),
            "W": self.dimensions[0],
            "H": self.dimensions[1],
            "LensName": self.lens_name,
            "ExposureTime": self.exposure_time
        }


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
        self.__nm_per_pixel_values = 0
        # find all TIFF files in specified folder
        all_tif_files = sorted(
            glob.glob(os.path.join(folder_path, "Image_*.tif"))
        )
        # Filter out TIFF files that contain "Overlay" in their filename
        all_tif_files = [os.path.basename(
            f) for f in all_tif_files if "Overlay" not in os.path.basename(f)]

        # Identify unique channels
        self.__channels = sorted(
            {re.search(r'CH\d+', f).group() for f in all_tif_files if re.search(r'CH\d+', f)}
        )

        assert all_tif_files is not None, "No TIFF files found in the specified folder."
        assert len(self.__channels) > 0, "No channels found in the specified folder."
        assert len(all_tif_files) > 0, "No TIFF files found in the specified folder."

        # check for Z stack images
        zstack_mode = all([re.search(r'_Z\d+_', f) is not None for f in all_tif_files])
            
        meta_info = []
        z_max = 0
       
        for c_idx, channel in enumerate(self.__channels):
            print(f"Processing channel: {channel} in folder {folder_path}")
            tif_files = sorted([f for f in all_tif_files if f.endswith(f"{channel}.tif")])
            
            d = pd.DataFrame(
                list(map(lambda tif: ImageMetadata(os.path.join(
                    folder_path, tif)).get_dict(), tif_files))
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
           
            if c_idx == 0:
                self.__nm_per_pixel_values = ImageMetadata(os.path.join( folder_path, all_tif_files[0])).nm_per_pixel_values

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
           
            if img.dtype == np.uint16:               
                canvas_array[
                    row["CH_idx"],
                    row["Z"],
                    row["Y_relative"]: row["Y_relative"] + row["H"],
                    row["X_relative"]: row["X_relative"] + row["W"],
                ] = np.flipud(np.fliplr(img))
            else:
                raise ValueError("WARNING: not 16 bit image")

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
            'axes': 'CZYX',  #will be CZYX for Z stack
            'spacing': self.__nm_per_pixel_values / 1000,  # Convert nm to um
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
