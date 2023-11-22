import os
import cv2
import numpy as np
import argparse
from pimage_lib import pimage as pi
import threading
import time
import matplotlib.pyplot as plt


THREADS = 24

def process_image(img_raw, dir, token):
    """ Convert raw images and stores into files
    """
    img_mono, img_rgb, img_rgb_dif, img_dolp_mono, img_pol_mono, img_pauli = pi.extractPotato(img_raw)
    # val_stokes_mono, img_stokes, val_dolp_mono, img_dolp_mono, val_aolp_mono, img_aolp = pi.extractCoreModalities(img_raw)

    # Save image outputs
    cv2.imwrite(os.path.join(dir,token+"_mono.png"), img_mono)
    cv2.imwrite(os.path.join(dir,token+"_rgb.png"), img_rgb)
    cv2.imwrite(os.path.join(dir,token+"_rgbdif.png"), img_rgb_dif)
    cv2.imwrite(os.path.join(dir,token+"_dolp.png"), img_dolp_mono)
    cv2.imwrite(os.path.join(dir,token+"_pol.png"), img_pol_mono)
    cv2.imwrite(os.path.join(dir,token+"_pauli.png"), img_pauli)

    # Show image using cv2
    # cv2.imshow("MONO", img_mono)
    # cv2.imshow("RGB", img_rgb)
    # cv2.imshow("RGBDIF", img_rgb_dif)
    # cv2.imshow("DoLP", img_dolp_mono)
    # cv2.imshow("POL", img_pol_mono)
    # cv2.imshow("PAULI", img_pauli)
    # cv2.waitKey(0)

    # convert the histogram to plot in matplotlib and save to a file
    # plt.plot(dolp_hist)
    # plt.savefig(os.path.join(dir,token+"_dolp_hist.png"))

    # dolp_filter = (np.ones_like(img_rgb).astype(np.float32))
    # dolp_filter[:,:,0] -= val_dolp_mono
    # dolp_filter[:,:,1] -= val_dolp_mono
    # dolp_filter[:,:,2] -= val_dolp_mono

    # img_noref =  (dolp_filter * img_rgb).round().astype(np.uint8)
    # cv2.imwrite(os.path.join(dir,token+"_noref.png"), img_noref)

    # # # Full tile
    # # Expand images with a single channel
    img_mono3 = cv2.cvtColor(img_mono, cv2.COLOR_GRAY2BGR)
    # img_dolp3 = cv2.cvtColor(img_dolp_mono, cv2.COLOR_GRAY2BGR)

    # # Label images
    cv2.putText(img_mono3, "MONO", (100,150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 5)
    cv2.putText(img_rgb, "RGB", (100,150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 5)
    cv2.putText(img_rgb_dif, "DIF", (100,150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 5)
    cv2.putText(img_dolp_mono, "DOLP", (100,150), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 5)
    cv2.putText(img_pol_mono, "POL", (100,150), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 5)
    cv2.putText(img_pauli, "PAULI", (100,150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 5)

    # # Small tile (2 images)
    # stile = cv2.hconcat([img_rgb, img_pol_mono])
    # cv2.imwrite(os.path.join(dir, token+"_stile.jpg"), stile)

    # # Large tile (6 images)
    # single_tile = cv2.hconcat([img_rgb, img_dolp3, img_noref])
    top_tile = cv2.hconcat([img_mono3, img_rgb, img_rgb_dif])
    bot_tile = cv2.hconcat([img_dolp_mono, img_pol_mono, img_pauli])
    ltile = cv2.vconcat([top_tile, bot_tile])
    cv2.imwrite(os.path.join(dir, token+"_ltile.jpg"), ltile)
    # cv2.imwrite(os.path.join(dir, token+"_stile.jpg"), single_tile)



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
    img_dir = args.directory

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

    #option argument
    parser.add_argument("-d", "--directory", help="Image directory.", type=str, default="./utils/input/small_test/")

    args = parser.parse_args()

    if os.path.isdir(args.directory) is False:
        print("Directory", args.directory,"does not exists")
        quit()

    run(args)
    print("\nFinished")
    quit()
