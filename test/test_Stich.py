import os
import pytest
import numpy as np
import pandas as pd
from tifffile import imwrite
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from keyenceutils.Stich import ImageMetadata, StichedImage

@pytest.fixture
def create_test_tiff(tmp_path):
    """Fixture to create a test TIFF file with XML metadata."""
    tiff_path = tmp_path / "Image_001_CH1.tif"
    xml_content = """
    <Data>
        <XyStageRegion>
            <X>1000</X>
            <Y>2000</Y>
            <Width>50000</Width>
        </XyStageRegion>
        <SavingImageSize>
            <Width>512</Width>
            <Height>512</Height>
        </SavingImageSize>
        <LensName>TestLens 10x</LensName>
    </Data>
    """
    # Create a dummy TIFF file with embedded XML metadata
    with open(tiff_path, "wb") as f:
        f.write(b"TIFF_HEADER")
        f.write(xml_content.encode("utf-8"))
    return tiff_path

@pytest.fixture
def create_test_folder(tmp_path):
    """Fixture to create a test folder with multiple TIFF files."""
    folder_path = tmp_path / "test_folder"
    folder_path.mkdir()

    # Create dummy TIFF files
    for i in range(3):
        for ch in range(2):
            tiff_path = folder_path / f"Image_{i:03d}_CH{ch+1}.tif"
            img_data = np.random.randint(0, 65535, (512, 512), dtype=np.uint16)
            imwrite(tiff_path, img_data)

    return folder_path

def test_image_metadata_extraction(create_test_tiff):
    """Test the extraction of metadata from a TIFF file."""
    tiff_path = create_test_tiff
    metadata = ImageMetadata(str(tiff_path))

    assert metadata.image_positions == (1000, 2000)
    assert metadata.dimensions == (512, 512)
    assert metadata.nm_per_pixel_values == pytest.approx(50000 / 512)
    assert metadata.lens_name == "TestLens 10x"

def test_image_metadata_to_dict(create_test_tiff):
    """Test the conversion of metadata to a dictionary."""
    tiff_path = create_test_tiff
    metadata = ImageMetadata(str(tiff_path))
    metadata_dict = metadata.get_dict()

    assert metadata_dict["X"] == int(1000 / metadata.nm_per_pixel_values)
    assert metadata_dict["Y"] == int(2000 / metadata.nm_per_pixel_values)
    assert metadata_dict["W"] == 512
    assert metadata_dict["H"] == 512
    assert metadata_dict["LensName"] == "TestLens 10x"

def test_stitched_image_initialization(create_test_folder):
    """Test the initialization of the StichedImage class."""
    folder_path = create_test_folder
    stitched_image = StichedImage(str(folder_path))

    meta_info = stitched_image.get_meta_info()
    assert isinstance(meta_info, pd.DataFrame)
    assert not meta_info.empty
    assert meta_info["CH"].nunique() == 2  # Two channels
    assert meta_info["fname"].nunique() == 6  # Three images per channel

def test_stitched_image_save(create_test_folder, tmp_path):
    """Test saving the stitched image."""
    folder_path = create_test_folder
    stitched_image = StichedImage(str(folder_path))

    output_path = tmp_path / "stitched_output.tif"
    stitched_image.save(str(output_path))

    assert os.path.exists(output_path)