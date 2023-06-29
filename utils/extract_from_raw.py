import os
import cv2
import numpy as np
import argparse
from pimage_lib import pimage as pi
import polanalyser as pa
import threading
import time

THREADS = 7

def extractLake(img_raw):
    """ Using polanalyser to extract the lake images
    """

    # Extract demosaiced rgb images: 4x(2048, 2448, 3) uint8
    demosaiced_color = pa.demosaicing(img_raw, pa.COLOR_PolarRGB)

    # Extract monocolor polarization channels
    demosaiced_mono = []
    for i in range(4):
        demosaiced_mono.append(cv2.cvtColor(demosaiced_color[i], cv2.COLOR_BGR2GRAY))

    # Extract regular RGB image (I_0 + I_90)
    img_rgb = np.empty((2048, 2448, 3), demosaiced_color[0].dtype)
    for i in range(3):
        img_0 = demosaiced_color[0][...,i] 
        img_90 = demosaiced_color[2][...,i]
        img_rgb[...,i] = cv2.addWeighted(img_0, 0.5, img_90, 0.5, 0.0)

    # Same as regular filter
    img_rgb_90 = demosaiced_color[2]

    # Monochrome
    img_mono = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    # Compute stokes parameters for each color: (2048, 2448, 3, 3) float64
    radians = np.array([0, np.pi/4, np.pi/2, np.pi*3/4])
    stokes_color = pa.calcStokes(demosaiced_color, radians)

    # Compute stokes parameters for monochrome: (2048, 2448, 3) float64
    radians = np.array([0, np.pi/4, np.pi/2, np.pi*3/4])
    stokes_mono = pa.calcStokes(demosaiced_mono, radians)

    # Compute DoLP values for each color: (2048, 2448, 3) float64
    # val_DoLP_color  = pa.cvtStokesToDoLP(stokes_color) # 0~1

    # Compute DoLP values for monochrome: (2048, 2448, 1) float64
    val_DoLP_mono  = pa.cvtStokesToDoLP(stokes_mono) # 0~1
    img_DoLP_mono = (val_DoLP_mono * 255).round().astype(np.uint8)

    # Compute diffuse images
    img_rgb_dif = pa.cvtStokesToDiffuse(stokes_color).astype(np.uint8)

    # Compute AoLP values: (2048, 2448, 1) float64
    val_AoLP_mono = pa.cvtStokesToAoLP(stokes_mono)

    # Generate False colored AoLP_DoLP representation for monochrome
    img_AoLP_DoLP_mono = pa.applyColorToAoLP(val_AoLP_mono, saturation=1.0, value=val_DoLP_mono)

    return img_rgb, img_rgb_90, img_rgb_dif, img_mono, img_DoLP_mono, img_AoLP_DoLP_mono


def process_image(img_raw, dir, token):
    """ Convert raw images and stores into files
    """
    img_rgb, img_rgb_90, img_rgb_dif, img_mono, img_dolp_mono, img_pol_mono = extractLake(img_raw)
    # img_rgb, img_rgb_90, img_rgb_dif, img_mono, img_dolp_mono, img_pol_mono = pi.extractLake(img_raw)

    # Save image outputs        
    cv2.imwrite(os.path.join(dir,token+"_dolp.png"), img_dolp_mono)
    cv2.imwrite(os.path.join(dir,token+"_rgb.png"), img_rgb)
    cv2.imwrite(os.path.join(dir,token+"_mono.png"), img_mono)
    cv2.imwrite(os.path.join(dir,token+"_rgb90.png"), img_rgb_90)
    cv2.imwrite(os.path.join(dir,token+"_rgbdif.png"), img_rgb_dif)
    cv2.imwrite(os.path.join(dir,token+"_pol.png"), img_pol_mono)

    # # # Full tile 
    # # Expand images with a single channel
    # img_mono3 = cv2.cvtColor(img_mono, cv2.COLOR_GRAY2BGR)
    # img_dolp3 = cv2.cvtColor(img_dolp_mono, cv2.COLOR_GRAY2BGR)
    
    # # Label images
    # cv2.putText(img_rgb, "RGB", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)
    # cv2.putText(img_rgb_90, "RGB90", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)
    # cv2.putText(img_rgb_dif, "Diffuse", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)
    # cv2.putText(img_mono3, "Gray", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)
    # cv2.putText(img_dolp3, "DoLP", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)
    # cv2.putText(img_pol_mono, "DoLP and AoLP", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 4)

    # # Small tile (2 images)
    # stile = cv2.hconcat([img_rgb, img_pol_mono])
    # cv2.imwrite(os.path.join(dir, token+"_stile.jpg"), stile)

    # # Large tile (6 images)
    # top_tile = cv2.hconcat([img_rgb, img_rgb_90, img_rgb_dif])
    # bot_tile = cv2.hconcat([img_mono3, img_dolp3, img_pol_mono])
    # ltile = cv2.vconcat([top_tile, bot_tile])
    # cv2.imwrite(os.path.join(dir, token+"_ltile.jpg"), ltile)


def producer(filenames, img_dir, buffer, mutex, finished):
    """ Load files into the image buffer
    """
    for raw_img_name in filenames:
        token = raw_img_name[:-8] # remove _raw.png ending

        # Read image
        img_raw = cv2.imread(os.path.join(img_dir, raw_img_name), cv2.IMREAD_GRAYSCALE)

        # TODO: Add upper limit on buffern lengh
        mutex.acquire(blocking=False)
        buffer.append({'img':img_raw, 'token':token, 'dir':img_dir})
        mutex.release()

    finished.set()
    print("Finished loading all images")
    
    return


def consumer(buffer, mutex, finished):
    """ Process images from the image buffer
    """
    completed = False
    while True:
        if finished.is_set():
            completed = True

        mutex.acquire(blocking=False)
        buffer_len = len(buffer)
        print("Buffer length:",buffer_len, end='\r')

        if len(buffer) == 0:
            mutex.release
            if completed:
                # print("C: Finished")
                break
            else:
                # print("C: Slow producer")
                time.sleep(0.1)
        else:
            item = buffer.pop()
            # print("C: release loaded image")
            mutex.release
            process_image(item['img'], item['dir'], item['token'])

    return


def run(args):
    """ Get file list and manage threads.
    """
    img_dir = args.img_dir

    # Get a list of files containing the "_raw" tag
    filenames = []
    for file in os.listdir(img_dir):
        if "_raw.png" in file:
            filenames.append(file)
    filenames.sort()

    buffer = []
    mutex = threading.Lock()
    finished = threading.Event()
    
    prod = threading.Thread(target=producer, args=(filenames, img_dir, buffer, mutex, finished))

    # Start all threads
    prod.start()
    consumers = []
    for i in range(THREADS):
        consumers.append(threading.Thread(target=consumer, args=(buffer, mutex, finished)))
    for i in range(THREADS):
        consumers[i].start()
    print(f"Started {THREADS+1} threads")

    # Wait all threads to finish
    prod.join()
    for i in range(THREADS):
        consumers[i].join()
    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Extract data from folder with raw images.")

    parser.add_argument("img_dir", help="Image directory.")

    args = parser.parse_args()

    if os.path.isdir(args.img_dir) is False:
        print("Directory", args.dir,"does not exists")
        quit()

    run(args)
    print("\nFinished")
    quit()