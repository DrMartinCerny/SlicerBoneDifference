cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(SlicerBoneDifference)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/SlicerBoneDifference")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Martin Cerny, MD, PhD (Dpt. of Neurosurgery, Military University Hospital, Prague, Czech Republic)")
set(EXTENSION_DESCRIPTION "Detects bone defects by subtracting preoperative and postoperative CT volumes and thresholding the difference to generate segmentation masks. Intended for automated dataset creation and self-supervised learning.")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/SlicerBoneDifference.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/SlicerBoneDifference/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(BoneDifference)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
