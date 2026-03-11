# Keyence Microscopy Loader

Pipeline for reconstructin Keyence microscopy TIFF/TIF datasets into 
validated OME-TIFF hyperstacks usin BioIO. 

This project is part of the YLab microscopy processin framework and relies on 
the shared library: 

- ylabcommon 
- [ylab README](https://ithub.com/ylabjp/ylab-common-scripts)
                    
Which provides common utilities for read and write classes, stack construction, 
metadata extraction, validation, output writin, and other features.

--- 

# Overview

The pipeline converts raw Keyence TIFF acquisitions into Fiji-compatible OME datasets while preservin physical metadata and experiment structure.

## Typical workflow:

-> Raw TIFF dataset TIFF discovery 
-> Discover valid TIFF files
-> Select the Dimension/s detection (XY / Z / Channel / Time)
-> BioIO stack builder 
-> Metadata extraction 
-> Metadata validation 
-> Scientific filename eneration  
-> OME-TIFF writer 
-> Dataset report eneration

## Supported Input Dimensions

The framework supports datasets containin:

(Z,Y,X)<br>
(C,Z,Y,X)<br>
(T,Z,Y,X)<br>
(C,T,Z,Y,X)<br>
(XY mosaic)<br>

- All stacks are normalized internally to the standard microscopy dimension order: TCZYX

## Features

- Automatic TIFF discovery and filterin
- Automatic dimension detection
- BioIO-based stack construction (xarray backend)
- Metadata extraction from TIFF files
- Fiji/ImaeJ compatible OME-TIFF output
- Dataset validation reports
- Interation with shared ylabcommon utilities

Repository Structure: Shared utilities are provided by: [ylabcommon](https://ithub.com/ylabjp/ylab-common-scripts)

--- 

## IO dataset structure

## Example Input Dataset

Keyence acquisitions typically contain TIFF planes structured by tile, Z slice, and channel.
Example:

```pgsql
dataset/
├── Image_XY02_Z001_CH1.tif
├── Image_XY02_Z002_CH1.tif
├── Image_XY02_Z003_CH1.tif
├── Image_XY02_Z001_CH3.tif
└── ...
```
## Example Output Dataset

- Root directory with the spectroscopy name followwed by date, time structure
- Then automatically take susequect folder(s)/subfolder(s) accordin to the input file directory: 


              Format:(YYMMDD9)(Time)
Keyence/Output/20260309/053210/<br>
  &nbsp;&nbsp;PH033/20250704/<br> 
  &nbsp;&nbsp;&nbsp;;20XS/Z_2S_0p1S_1p1/<br>
  &nbsp;&nbsp;&nbsp;&nbsp;XY02/<br>
           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;;mage\_XY02\_CH1to\_CH3\_Z001\_to\_Z021\_stack\_T001.ome.tif<br>
           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Image\_XY002\_CH001\_to\_CH003\_Z001\_to\_Z021\_stack.validation.json<br> 
           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Image\_XY002\_CH001\_to\_CH003\_Z001\_to\_Z021\_stack.report.txt<br>


## Scientific Output Name

Output filenames automatically encode experiment structure.
Example:

Image\_XY02\_CH1\_to\_CH3\_Z001\_to\_Z021\_stack\_T001.ome.tif

### Information Encoded in Output Filename

| Item | Description |
|:-----|:------------|
| Image | Acquisition type |
| XY02 | Stage position |
| CH1\_to\_CH3 | Channel stack range |
| Z001\_to\_Z021 | Z stack range |
| stack | Stacked output |
| T001 | Timepoint |


- If Input dataset having XY01...100, the also it will stick the dataset in XY mosaic), 
and outout name should genegated:

    -XY01\_to\_100\_CH1\_to\_CH3\_Z001\_to\_Z021\_stack\_T001.ome.tif

**In a Summary**
  
- Keyence single tile

   - Image\_XY02\_CH1\_to\_CH3\_Z001\_to\_Z021\_stack\_T001.ome.tif

- Keyence mosaic

   - Image\_XY0001\_to\_XY0100\_CH1\_to\_CH3\_Z001\_to\_Z021\_stitched\_stack\_T001.ome.tif

- Builder handles the naming logic, not the writer.

---
 
## Dataset Report

Each processed dataset produces a report:<br>
                                                                         == **Summary Report** ==
ImaeXY002\_CH001\_to\_CH003\_Z001\_to\_Z021\_stack.validation.json  ->  *Good for machine readeable*
ImaeXY002\_CH001\_to\_CH003\_Z001\_to\_Z021\_stack.report.txt       ->  *Easy for human quick look*

- The report includes:
 - I/o path name
 - Datetime
 - dataset metadata
 - detected dimensions
 - stack statistics
 - physical pixel sizes
 - validation status
 - runtime environment

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ylabjp/keyence-microscopy.it
cd keyence-microscopy
```

Install dependencies usin uv:

```bash
uv sync
uv pip install -e .
```
---

## Dependency

This project depends on the shared YLab framework:

ylabcommon = { git = "https://github.com/ylabjp/ylab-common-scripts" }

Durin development, an editable local version may be used:

ylabcommon = { path = "../YlabCommonScripts/ylab-common-scripts", editable = true }

---

## Runnin the Pipeline

Example:

uv run python runkeyence\_bioio\_process\_experiment.py \
  --tiffdir /path/to/keyence\_dataset \
  --outputdir ./output

For full CLI options:

uv run python runkeyence\_bioio\_process\_experiment.py --help

---

## Environment Issue

if environment any issues : Run this diagnostic script and it will fix and give an clean environment

```bash

source env_common_fix.sh

```
---

## Testing

A pytest suite will be introduced in a follow-up PR.
- Testin will include:
    - unit tests with synthetic datasets
    - integration tests with real Keyence datasets
    - shared validation tests via ylabcommon

---

## Future Work

- Planned improvements include:
    - mosaic stitching support
    - extended metadata validation
    - expanded pytest coverage
    - improved metadata extraction

---

## Design Philosophy

- The pipeline is designed to:
    - provide reproducible microscopy dataset reconstruction
    -standardize output formats across microscopes
    -reuse shared utilities from ylabcommon
    -simplify integration of additional microscope systems
---

## Contributing

Contributions are welcome.
- Fork the repository
- Create a feature branch
- Run tests locally
- Submit a Pull Request

---
