from typing import List, Union
import cv2
import numpy as np


def demosaicing(img_raw: np.ndarray) -> List[np.ndarray]:
    """Polarization demosaicing
    Parameters
    ----------
    img_raw : np.ndarray
        Polarization image taken with polarizatin sensor
    Returns
    -------
    img_demosaiced_list : List[np.ndarray]
        List of demosaiced images. The shape of each image is (height, width, 3).
    """
    # split
    # (0, 0):90,  (0, 1):45 (1, 0):135, (1, 1):0
    img_bayer_090 = img_raw[0::2, 0::2]
    img_bayer_045 = img_raw[0::2, 1::2]
    img_bayer_135 = img_raw[1::2, 0::2]
    img_bayer_000 = img_raw[1::2, 1::2]

    # debayer
    img_bgr_090 = cv2.cvtColor(img_bayer_090, cv2.COLOR_BayerBG2BGR)
    img_bgr_045 = cv2.cvtColor(img_bayer_045, cv2.COLOR_BayerBG2BGR)
    img_bgr_135 = cv2.cvtColor(img_bayer_135, cv2.COLOR_BayerBG2BGR)
    img_bgr_000 = cv2.cvtColor(img_bayer_000, cv2.COLOR_BayerBG2BGR)

    return [img_bgr_000, img_bgr_045, img_bgr_090, img_bgr_135]


def rgb(demosaiced_list: List[np.ndarray]) -> np.ndarray:
    """Extract rgb image
    Parameters
    ----------
    demosaiced_list : List of demosaiced images.
    Returns
    -------
    img_bgr : Collored image
    """
    img_a = cv2.addWeighted(demosaiced_list[0], 0.5, demosaiced_list[2], 0.5, 0.0)
    img_b = cv2.addWeighted(demosaiced_list[1], 0.5, demosaiced_list[3], 0.5, 0.0)
    img_bgr = cv2.addWeighted(img_a, 0.5, img_b, 0.5, 0.0)

    return img_bgr


def calcStokes(demosaiced_list: List[np.ndarray]) -> np.ndarray:
    """ Compute stokes vector
    Parameters
    ----------
    demosaiced_list : List of demosaiced images. (assuming order of 0, 45, 90, 135)
    Returns
    -------
    stokes : stokes vector
    """
    s0 = np.sum(demosaiced_list, axis=(0), dtype=np.float64)/2
    s1 = demosaiced_list[0].astype(np.float64) - demosaiced_list[2].astype(np.float64) #0-90
    s2 = demosaiced_list[1].astype(np.float64) - demosaiced_list[3].astype(np.float64) #45-135
    return np.stack((s0, s1, s2), axis=-1)


def calcDoLP(stokes):
    """Compute the degree of linear polarization
    Parameters
    ----------
    stokes : np.ndarray
        3 channel array representing the stokes parameters
    Returns
    -------
    dolp : List[np.ndarray]
        Single channel image representing the degree of lienar polarization.
    """
    # return np.sqrt(s1**2 + s2**2) / s0
    dolp = np.sqrt(stokes[...,1]**2 + stokes[...,2]**2) / stokes[...,0]
    return dolp


def calcAoLP(stokes):
    """Comput the angle of linear polarization
    Parameters
    ----------
    stokes : np.ndarray
        3 channel array representing the stokes parameters
    Returns
    -------
    dolp : List[np.ndarray]
        Single channel image representing the degree of lienar polarization.
    """
    aolp = np.mod(0.5 * np.arctan2(stokes[...,2], stokes[...,1]), np.pi)
    return aolp


def falseColoring(aolp: np.ndarray, value: Union[float, np.ndarray] = 1.0) -> np.ndarray:
    """False colloring to AoLP. Possible to use DoLP as value

    Parameters
    ----------
    AoLP : np.ndarray
        AoLP values ranging from 0.0 to pi
    value : Union[float, np.ndarray], optional
        Value value(s), by default 1.0

    Returns
    -------
    colored : np.ndarray
        False colored image (in BGR format)
    """
    ones = np.ones_like(aolp)

    hue = (np.mod(aolp, np.pi) / np.pi * 179).astype(np.uint8)  # [0, pi] to [0, 179]
    saturation = (ones*255).astype(np.uint8)
    value = np.clip(ones * value * 255, 0, 255).astype(np.uint8)

    hsv = cv2.merge([hue, saturation, value])
    colored = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return colored


def calcDiffuse(stokes: np.ndarray) -> np.ndarray:
    """Convert stokes parameters to diffuse

    Parameters
    ----------
    stokes : np.ndarray
        Stokes parameters
    Returns
    -------
    diffuse : np.ndarray
        Diffuse
    """
    diffuse = (stokes[..., 0] - np.sqrt(stokes[..., 1]**2 + stokes[..., 2]**2)) * 0.5
    return diffuse.astype(np.uint8)


def calcSpecular(stokes: np.ndarray) -> np.ndarray:
    """Convert stokes parameters to specular reflection

    Parameters
    ----------
    stokes : np.ndarray
        Stokes parameters
    Returns
    -------
    specular : np.ndarray
        Specular
    """
    specular = np.sqrt(stokes[..., 1]**2 + stokes[..., 2]**2)  # same as Imax - Imin
    return specular.astype(np.uint8)


def extractColorAndPol(img_raw: np.ndarray):
    """Extrac RGB and Colored images from the raw image

    Parameters
    ----------
    img_raw : np.ndarray
    Returns
    -------
    img_rgb : np.ndarray
    img_pol : np.ndarray
    """
    demosaiced_color = demosaicing(img_raw)

    demosaiced_mono = []
    for i in range(4):
        demosaiced_mono.append(cv2.cvtColor(demosaiced_color[i], cv2.COLOR_BGR2GRAY))

    img_rgb = rgb(demosaiced_color)

    stokes_mono = calcStokes(demosaiced_mono)

    val_dolp_mono  = calcDoLP(stokes_mono)

    val_aolp_mono = calcAoLP(stokes_mono)

    img_pol_mono = falseColoring(val_aolp_mono, value=val_dolp_mono)

    return img_rgb, img_pol_mono


def extractLake(img_raw: np.ndarray):
    """Extrac Six images commonly used in the lake from the raw image

    Parameters
    ----------
    img_raw : np.ndarray
    Returns
    -------
    img_rgb : np.ndarray
    img_rgb_90 : np.ndarray
    img_rgb_dif : np.ndarray
    img_mono : np.ndarray
    img_dolp_mono : np.ndarray
    img_pol_mono : np.ndarray
    """
    demosaiced_color = demosaicing(img_raw)

    demosaiced_mono = []
    for i in range(4):
        demosaiced_mono.append(cv2.cvtColor(demosaiced_color[i], cv2.COLOR_BGR2GRAY))

    img_rgb = rgb(demosaiced_color)

    img_rgb_90 = demosaiced_color[2]

    img_mono = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    stokes_color = calcStokes(demosaiced_color)

    img_rgb_dif = calcDiffuse(stokes_color)

    stokes_mono = calcStokes(demosaiced_mono)

    val_DoLP_mono  = calcDoLP(stokes_mono) # 0~1

    img_dolp_mono = (val_DoLP_mono * 255).round().astype(np.uint8)

    val_aolp_mono = calcAoLP(stokes_mono)

    img_pol_mono = falseColoring(val_aolp_mono, value=val_DoLP_mono)

    return img_rgb, img_rgb_90, img_rgb_dif, img_mono, img_dolp_mono, img_pol_mono
