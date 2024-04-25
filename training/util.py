import enum

import nibabel as nib


def load_nibabel_image_with_axcodes(
    image: nib.nifti1.Nifti1Image, axcodes: str = "RAS"
) -> nib.nifti1.Nifti1Image:
    input_axcodes = "".join(nib.aff2axcodes(image.affine))
    axcodes = "".join(axcodes)
    if input_axcodes != axcodes:
        input_ornt = nib.orientations.axcodes2ornt(input_axcodes)
        expected_ornt = nib.orientations.axcodes2ornt(axcodes)
        transform = nib.orientations.ornt_transform(input_ornt, expected_ornt)
        return image.as_reoriented(transform)
    return image


def convert_nibabel_to_original_with_axcodes(
    image_transformed: nib.nifti1.Nifti1Image,
    image_original: nib.nifti1.Nifti1Image,
    transformed_axcodes: str = "RAS",
) -> nib.nifti1.Nifti1Image:
    img_ornt = nib.orientations.io_orientation(image_original.affine)
    ras_ornt = nib.orientations.axcodes2ornt(transformed_axcodes)
    from_canonical = nib.orientations.ornt_transform(ras_ornt, img_ornt)
    return image_transformed.as_reoriented(from_canonical)


class BodyRegions(enum.IntEnum):
    BACKGROUND = 0
    SUBCUTANEOUS_TISSUE = 1
    MUSCLE = 2
    ABDOMINAL_CAVITY = 3
    THORACIC_CAVITY = 4
    BONE = 5
    PAROTID_GLANDS = 6
    PERICARDIUM = 7
    BREAST_IMPLANT = 8
    MEDIASTINUM = 9
    BRAIN = 10
    SPINAL_CORD = 11
    THYROID_GLANDS = 12
    SUBMANDIBULAR_GLANDS = 13


class BodyParts(enum.IntEnum):
    BACKGROUND = 0
    TORSO = 1
    HEAD = 2
    RIGHT_LEG = 3
    LEFT_LEG = 4
    RIGHT_ARM = 5
    LEFT_ARM = 6
