import argparse
import datetime
import functools
import getpass
import multiprocessing as mp
import multiprocessing.synchronize as mps
import pathlib
import shutil
import signal
import tempfile
import traceback
import zipfile
from typing import Hashable, Optional, Tuple

import numpy as np
import pandas as pd
import pydicom
import pydicom_seg
import requests
import SimpleITK as sitk
from tqdm import tqdm

NBIA_API_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v2"
NBIA_LOGIN_URL = "https://services.cancerimagingarchive.net/nbia-api/oauth/token"


def _set_ignore_label(seg: sitk.Image, label: int) -> sitk.Image:
    arr = sitk.GetArrayFromImage(seg)
    arr[np.nonzero(np.all(arr == 0, axis=1))] = label

    ignore_seg = sitk.GetImageFromArray(arr)
    ignore_seg.CopyInformation(seg)
    return ignore_seg


def handle_case(
    pbar: tqdm,
    working_dir: pathlib.Path,
    target_dir: pathlib.Path,
    series_instance_uid: str,
    regions_series_instance_uid: str,
    parts_series_instance_uid: str,
    authentication_token: str,
    save_original_image: bool = True,
    save_meta_dicoms: bool = True,
    save_dicoms: bool = True,
    set_ignore: Optional[int] = None,
    force: bool = False,
) -> None:
    if (
        not force
        and (target_dir / "image.nii.gz").exists()
        and (not save_original_image or (target_dir / "image_original.nii.gz").exists())
        and (not save_dicoms or (target_dir / "dicom").exists())
        and (
            not save_meta_dicoms
            or (
                (target_dir / "meta_first.dcm").exists()
                and (target_dir / "meta_last.dcm").exists()
            )
        )
    ):
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    working_dir.mkdir(parents=True, exist_ok=True)

    downloads = [
        ("image", series_instance_uid),
        ("body-regions", regions_series_instance_uid),
        ("body-parts", parts_series_instance_uid),
    ]
    for name, sid in downloads:
        _download_series(working_dir / f"{name}.zip", sid, authentication_token)
    pbar.update()
    for name, _ in downloads:
        _extract_series(working_dir, working_dir / f"{name}.zip")
    pbar.update()

    image = _load_series(
        target_dir=target_dir,
        dicom_dir=working_dir,
        series_instance_uid=series_instance_uid,
        save_original_image=save_original_image,
        save_meta_dicoms=save_meta_dicoms,
        save_dicoms=save_dicoms,
    )
    image = _resample_image_to_thickness(image, thickness=5)
    body_regions = _load_segmentation(
        target_dir,
        working_dir / "body-regions.dcm",
        save_original_image,
        save_meta_dicoms,
        save_dicoms,
    )
    body_parts = _load_segmentation(
        target_dir,
        working_dir / "body-parts.dcm",
        save_original_image,
        save_meta_dicoms,
        save_dicoms,
    )

    body_regions = sitk.Resample(
        body_regions, image, sitk.Transform(), sitk.sitkNearestNeighbor
    )

    body_parts = sitk.Resample(
        body_parts, image, sitk.Transform(), sitk.sitkNearestNeighbor
    )

    for seg in [body_regions, body_parts]:
        assert seg.GetSize() == image.GetSize(), (
            f"{target_dir.name}: Unexpected shape after resampling - "
            f"{seg.GetSize()} vs {image.GetSize()}"
        )
        assert np.isclose(seg.GetSpacing(), image.GetSpacing()).all(), (
            f"{target_dir.name}: Image and segmentation do not have the same spacing -"
            f"{seg.GetSpacing()} vs. {image.GetSpacing()}"  # type: ignore
        )

    if set_ignore is not None:
        body_regions = _set_ignore_label(body_regions, set_ignore)
        body_parts = _set_ignore_label(body_parts, set_ignore)

    sitk.WriteImage(image, str(target_dir / "image.nii.gz"), True)
    sitk.WriteImage(body_regions, str(target_dir / "body-regions.nii.gz"), True)
    sitk.WriteImage(body_parts, str(target_dir / "body-parts.nii.gz"), True)

    pbar.update()


def _download_series(
    target_path: pathlib.Path,
    series_instance_uid: str,
    authentication_token: str,
) -> None:
    with requests.get(
        NBIA_API_URL + "/getImage",
        params={"SeriesInstanceUID": series_instance_uid},
        headers={"Authorization": "Bearer " + authentication_token},
        stream=True,
    ) as response:
        response.raise_for_status()
        with target_path.open("wb") as ofile:
            for chunk in response.iter_content(8096):
                if chunk:
                    ofile.write(chunk)


def _extract_series(target_dir: pathlib.Path, archive_path: pathlib.Path) -> None:
    target_dir.mkdir(exist_ok=True, parents=True)
    archive = zipfile.ZipFile(archive_path)
    archive.extractall(target_dir)


def _load_series(
    target_dir: pathlib.Path,
    dicom_dir: pathlib.Path,
    series_instance_uid: str,
    save_original_image: bool,
    save_meta_dicoms: bool,
    save_dicoms: bool,
) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    dcm_files = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_instance_uid)
    reader.SetFileNames(dcm_files)
    image = reader.Execute()
    if save_original_image:
        sitk.WriteImage(image, str(target_dir / "image_original.nii.gz"), True)
    if save_dicoms:
        (target_dir / "dicom").mkdir(parents=True, exist_ok=True)
        for idx, dcm_file in enumerate(dcm_files):
            path = pathlib.Path(dcm_file)
            if save_meta_dicoms:
                if idx == 0:
                    shutil.copy2(path, target_dir / "meta_first.dcm")
                elif idx == len(dcm_files) - 1:
                    shutil.copy2(path, target_dir / "meta_last.dcm")
            shutil.move(path, target_dir / "dicom" / path.name)
    elif save_meta_dicoms:
        shutil.move(dcm_files[0], target_dir / "meta_first.dcm")
        shutil.move(dcm_files[-1], target_dir / "meta_last.dcm")
    return image


def _load_segmentation(
    target_dir: pathlib.Path,
    dicom_file: pathlib.Path,
    save_original_image: bool,
    save_meta_dicoms: bool,
    save_dicoms: bool,
) -> sitk.Image:
    dcm = pydicom.dcmread(dicom_file)
    segmentation_name = dicom_file.name.replace(".dcm", "")
    reader = pydicom_seg.MultiClassReader()
    result = reader.read(dcm)
    image = result.image
    if save_original_image:
        sitk.WriteImage(
            image, str(target_dir / f"{segmentation_name}_original.nii.gz"), True
        )
    if save_meta_dicoms:
        shutil.copy2(dicom_file, target_dir / f"{segmentation_name}_meta_first.dcm")
    if save_dicoms:
        (target_dir / "dicom").mkdir(parents=True, exist_ok=True)
        shutil.move(dicom_file, target_dir / "dicom" / dicom_file.name)

    return image


def _resample_image_to_thickness(image: sitk.Image, thickness: float = 5) -> sitk.Image:
    input_size = image.GetSize()
    input_spacing = image.GetSpacing()
    output_spacing = (input_spacing[0], input_spacing[1], thickness)
    output_direction = image.GetDirection()
    output_origin = image.GetOrigin()
    output_size = (
        input_size[0],
        input_size[1],
        # TODO This should be an int cast, but for annotation the imaging was exported
        #   with slice rounding
        round(input_size[2] * input_spacing[2] / output_spacing[2]),
    )
    return sitk.Resample(
        image,
        output_size,
        sitk.Transform(),
        sitk.sitkLinear,
        output_origin,
        output_spacing,
        output_direction,
    )


def _worker(
    row_with_index: Tuple[Hashable, pd.Series], args: argparse.Namespace
) -> None:
    if stop_event.is_set():
        return

    _, row = row_with_index
    with tempfile.TemporaryDirectory() as working_dir, tqdm(
        position=mp.current_process()._identity[0],
        total=5,
        desc=row.id,
        leave=False,
    ) as pbar:
        try:
            handle_case(
                pbar,
                working_dir=pathlib.Path(working_dir) / row.id,
                target_dir=args.target_dir / row.id,
                series_instance_uid=row.tcia_series_instance_uid,
                regions_series_instance_uid=row.regions_series_instance_uid,
                parts_series_instance_uid=row.parts_series_instance_uid,
                save_original_image=args.save_original_image,
                save_meta_dicoms=args.save_meta_dicoms,
                save_dicoms=args.save_dicoms,
                set_ignore=args.set_ignore,
                force=args.force_download,
                authentication_token=shared_authentication_token.value.decode(),  # type: ignore
            )
        except Exception:
            traceback.print_exc()


def _worker_init(
    lock: mps.RLock,
    event: mps.Event,
    token: mp.Array,  # type: ignore
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    tqdm.set_lock(lock)
    global stop_event
    stop_event = event
    global shared_authentication_token
    shared_authentication_token = token


def get_authentication_token(username: str, password: str) -> Tuple[str, str, int]:
    response = requests.get(
        NBIA_LOGIN_URL,
        params={
            "username": username,
            "password": password,
            "grant_type": "password",
            "client_id": "nbiaRestAPIClient",
            "client_secret": "ItsBetweenUAndMe",
        },
    )
    response.raise_for_status()
    json_data = response.json()
    auth_token: str = json_data["access_token"]
    refresh_token: str = json_data["refresh_token"]
    expires_in: int = json_data["expires_in"]
    return auth_token, refresh_token, expires_in


def refresh_authentication_token(refresh_token: str) -> Tuple[str, str, int]:
    response = requests.get(
        NBIA_LOGIN_URL,
        params={
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
            "client_id": "nbiaRestAPIClient",
            "client_secret": "ItsBetweenUAndMe",
        },
    )
    response.raise_for_status()
    json_data = response.json()
    auth_token: str = json_data["access_token"]
    refresh_token = json_data["refresh_token"]
    expires_in: int = json_data["expires_in"]
    return auth_token, refresh_token, expires_in


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--info-csv",
        default=pathlib.Path("info.csv"),
        type=pathlib.Path,
        help="Path to the file with the information about the cases.",
    )
    parser.add_argument(
        "--target-dir",
        default=pathlib.Path("data"),
        type=pathlib.Path,
        help="Directory path where the downloaded files should be stored. "
        "Defaults to data.",
    )
    parser.add_argument(
        "--set-ignore",
        default=None,
        type=int,
        help="Set all the non-segmented slices to the desired value in the segmentation mask.",
    )
    parser.add_argument(
        "--save-original-image",
        default=False,
        action="store_true",
        help="Save the original ITK image without resampling as an additional file.",
    )
    parser.add_argument(
        "--save-meta-dicoms",
        default=False,
        action="store_true",
        help="Save the first and last DICOM file of the image series.",
    )
    parser.add_argument(
        "--save-dicoms",
        default=False,
        action="store_true",
        help="Save all DICOM files of the image series.",
    )
    parser.add_argument(
        "--force-download",
        default=False,
        action="store_true",
        help="Force download from TCIA even if the data already exists.",
    )
    parser.add_argument(
        "--no-login",
        default=False,
        action="store_true",
        help="Download only publicly available cases",
    )
    parser.add_argument(
        "--parallel-downloads",
        default=2,
        type=int,
        help="Number of parallel downloads. Please use carefully in order to prevent "
        "overloading the TCIA servers.",
    )
    args = parser.parse_args()

    if args.no_login:
        print(
            "Using the --no-login options downloads only publically available images. "
            "For full download please create a TCIA account and apply for the "
            "restricted collections used in SAROS."
        )
        username = "nbia_guest"
        password = ""
    else:
        print(
            "Login with a TCIA user account is required in order to download all "
            "cases. Some collections have restricted access and require signing a "
            "data usage agreement. Please ensure your account has all appropriate "
            "rights before proceeding or use the --no-login option to only download "
            "publically available data."
        )
        username = input("Username: ")
        password = getpass.getpass("Password:")

    authentication_token, refresh_token, token_expires_in = get_authentication_token(
        username, password
    )
    shared_authentication_token = mp.Array("c", authentication_token.encode())
    authentication_timestamp = datetime.datetime.now()

    info_df = pd.read_csv(args.info_csv)

    stop_event = mp.Event()
    tqdm.set_lock(mp.RLock())
    with mp.Pool(
        processes=args.parallel_downloads,
        initializer=_worker_init,
        initargs=(tqdm.get_lock(), stop_event, shared_authentication_token),
    ) as pool:
        try:
            for _ in tqdm(
                pool.imap_unordered(
                    functools.partial(_worker, args=args), info_df.iterrows()
                ),
                total=len(info_df),
            ):
                if (
                    datetime.datetime.now() - authentication_timestamp
                ) > datetime.timedelta(seconds=int(token_expires_in * 0.75)):
                    tqdm.write("Refreshing authentication token...")
                    (
                        authentication_token,
                        refresh_token,
                        token_expires_in,
                    ) = refresh_authentication_token(refresh_token)
                    shared_authentication_token.value = authentication_token.encode()  # type: ignore
                    authentication_timestamp = datetime.datetime.now()
        except KeyboardInterrupt:
            while True:
                try:
                    tqdm.write(
                        "Received a keyboard interrupt. Processes will stop after active jobs are finished."
                    )
                    stop_event.set()
                    pool.close()
                    pool.join()
                    break
                except KeyboardInterrupt:
                    pass
