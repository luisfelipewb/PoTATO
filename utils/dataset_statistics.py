import os
import matplotlib.pyplot as plt
import json
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Dataset analysis')
    parser.add_argument('-i','--input_dir', help='Path to the directory containing annotations', default='../datasets/potato/split_seq/')
    parser.add_argument('-a','--ann_file', help='JSON format annotation file', default='result.json')
    parser.add_argument('-o','--ouput_dir', help='Path to the directory where output images are saved', default='./output/')

    args = parser.parse_args()

    return args

def get_bbox_sizes(coco_ann):
    a_bbox_sizes = []
    s_bbox_sizes = []
    m_bbox_sizes = []
    l_bbox_sizes = []
    super_small = []
    super_small_imgs = {}

    for ann in coco_ann['annotations']:
        size = bbox = ann['bbox'][2] * ann['bbox'][3]
        a_bbox_sizes.append(size)

        if size <= (32**2):
            s_bbox_sizes.append(size)
            if size <= (34):
                super_small.append(size)
                super_small_imgs[ann["image_id"]] = size

                if size < (14):
                    print("tiny bbox. Probably error")
                    print(ann)
        elif size <= (96**2):
            m_bbox_sizes.append(size)
        else:
            l_bbox_sizes.append(size)

    print(f"Found {len(super_small)} super small bounding boxes")
    print(f"In {len(super_small_imgs)} different images")

    for key in super_small_imgs.keys():
        print(f"key: {key} index:{coco_ann['images'][key]['file_name']} size: {super_small_imgs[key]}")


    return a_bbox_sizes, s_bbox_sizes, m_bbox_sizes, l_bbox_sizes

def count_bbox_per_image(coco_ann):
    bbox_per_image = {}
    for ann in coco_ann['annotations']:
        image_id = ann['image_id']
        if image_id not in bbox_per_image:
            bbox_per_image[image_id] = 0
        bbox_per_image[image_id] += 1

    return list(bbox_per_image.values())


def plot_bboxes(a_bbox_sizes, s_bbox_sizes, m_bbox_sizes, l_bbox_sizes, title="combined"):
    # bbox_stack = [tst_bbox_count, val_bbox_count, trn_bbox_count]

    fig, axs = plt.subplots(4)
    fig.suptitle(title)
    fig.tight_layout()
    fig.set_figheight(15)
    fig.set_figwidth(15)

    axs[0].hist(a_bbox_sizes, bins=300)
    axs[0].set_xlabel('Bounding box size ')
    axs[0].set_ylabel('Count')
    axs[0].set_title('All bboxes')
    axs[0].set_xlim((0,80000))
    axs[0].set_ylim((0,200))
    # plt.savefig(os.path.join(args.ouput_dir,"bb_size.png"))

    axs[1].hist(s_bbox_sizes, bins=100)
    axs[1].set_xlabel('Bounding box size ')
    axs[1].set_ylabel('Count')
    axs[1].set_title('Small bboxes')
    # plt.savefig(os.path.join(args.ouput_dir,"bb_size_small.png"))

    axs[2].hist(m_bbox_sizes, bins=100)
    axs[2].set_xlabel('Bounding box size ')
    axs[2].set_ylabel('Count')
    axs[2].set_title('Medium bboxes')
    # plt.savefig(os.path.join(args.ouput_dir,"bb_size_med.png"))

    axs[3].hist(l_bbox_sizes, bins=100)
    axs[3].set_xlabel('Bounding box size ')
    axs[3].set_ylabel('Count')
    axs[3].set_title('Large bboxes')
    # plt.savefig(os.path.join(args.ouput_dir,"bb_size_large.png"))

    plt.savefig(os.path.join(args.ouput_dir,title+".png"))

def plot_log(trn_bboxes, val_bboxes, tst_bboxes, title="log"):

    bboxes = [trn_bboxes, val_bboxes, tst_bboxes]
    # print(np.shape(bboxes))

    plt.figure(figsize=(6,2))
    logbins = np.logspace(start=np.log2(2), stop=np.log2(32768), num=70, base=2)

    plt.hist(bboxes, bins=logbins, stacked=True)
    plt.xscale('log')
    plt.minorticks_off()
    plt.xlabel('Bounding box size')
    plt.ylabel('Count')
    plt.title("")
    plt.annotate("",
            xy=(96*96, 0), xycoords='data',
            xytext=(96*96, 600), textcoords='data',
            arrowprops=dict(arrowstyle="-", connectionstyle="arc3"))
    plt.annotate("",
            xy=(32*32, 0), xycoords='data',
            xytext=(32*32, 600), textcoords='data',
            arrowprops=dict(arrowstyle="-", connectionstyle="arc3"))

    plt.annotate("Small\n$(area < 32^2)$\n$1234 (81\%)$",
                xy=(20, 400), xycoords='data',
                xytext=(2, 0), textcoords='offset points')

    plt.annotate("Medium\n$(32^2 < area < 96^2)$\n$234 (18\%)$",
                xy=(32*32, 400), xycoords='data',
                xytext=(2, 0), textcoords='offset points')

    plt.annotate("Large\n$(96^2 < area)$\n$34 (2\%)$",
                xy=(96*96, 400), xycoords='data',
                xytext=(2, 0), textcoords='offset points')

    plt.xlim((20,45000))
    plt.xticks([6*6, 14*14, 32*32, 96*96, 192*192], labels=["$6^2$", "$14^2$", "$32^2$", "$96^2$", "$192^2$"])


    # axs[1].set_ylim((0,200))

    plt.savefig(os.path.join(args.ouput_dir,title+".png"), bbox_inches='tight')


def plot_log2(trn_bboxes, val_bboxes, tst_bboxes):

    bboxes = [trn_bboxes, val_bboxes, tst_bboxes]
    labels = ['Train', 'Validation', 'Test']


    fig, ax = plt.subplots(1)
    # fig.suptitle(title)
    fig.tight_layout()
    fig.set_figheight(2)
    fig.set_figwidth(6)

    logbins = np.logspace(start=np.log2(2), stop=np.log2(32768), num=70, base=2)

    ax.hist(bboxes, bins=logbins, stacked=True)
    ax.set_title('')
    ax.set_xscale('log')
    ax.set_xlabel('Bounding box size')
    ax.set_ylabel('Count')
    ax.legend(labels, loc='lower right', bbox_to_anchor=(1,0.1), prop={"size":9})
    ax.annotate("",
            xy=(96*96, 0), xycoords='data',
            xytext=(96*96, 600), textcoords='data',
            arrowprops=dict(arrowstyle="-", connectionstyle="arc3"))
    ax.annotate("",
            xy=(32*32, 0), xycoords='data',
            xytext=(32*32, 600), textcoords='data',
            arrowprops=dict(arrowstyle="-", connectionstyle="arc3"))

    ax.annotate("Small\n$(area < 32^2)$\n$10052 (81.2\%)$",
                xy=(20, 400), xycoords='data',
                xytext=(2, 0), textcoords='offset points')

    ax.annotate("Medium\n$(32^2 < area < 96^2)$\n$2079 (16.8\%)$",
                xy=(32*32, 400), xycoords='data',
                xytext=(2, 0), textcoords='offset points')

    ax.annotate("Large\n$(96^2 < area)$\n$74 (2.0\%)$",
                xy=(96*96, 400), xycoords='data',
                xytext=(2, 0), textcoords='offset points')

    ax.set_xlim((20,60000))
    ax.set_xticks([6*6, 14*14, 32*32, 96*96, 192*192], labels=["$6^2$", "$14^2$", "$32^2$", "$96^2$", "$192^2$"])

    plt.minorticks_off()
    plt.title("")
    plt.savefig(os.path.join(args.ouput_dir,"dataset_statistics.png"), bbox_inches='tight')

def print_stats(a_bbox_sizes, s_bbox_sizes, m_bbox_sizes, l_bbox_sizes, input_file):
    print(input_file)
    print(f"Smallest bbox area: {np.min(a_bbox_sizes)}, {np.min(s_bbox_sizes)}")
    print(f"Largest bbox area: {np.max(a_bbox_sizes)}, {np.max(l_bbox_sizes)}")

    # trn_bbox_count = count_bbox_per_image(trn_ann)
    # val_bbox_count = count_bbox_per_image(val_ann)
    # tst_bbox_count = count_bbox_per_image(tst_ann)

    nbb_all = len(a_bbox_sizes)
    nbb_s = len(s_bbox_sizes)
    nbb_m = len(m_bbox_sizes)
    nbb_l = len(l_bbox_sizes)
    print(f"total bbs #     {nbb_all}, or {nbb_s+nbb_m+nbb_l}")
    print(f"Small   (32*82) {nbb_s:04d}/{nbb_all} - {nbb_s/nbb_all:.3f}")
    print(f"Medium: (96**2) {nbb_m:04d}/{nbb_all} - {nbb_m/nbb_all:.3f}")
    print(f"Large:          {nbb_l:04d}/{nbb_all} - {nbb_l/nbb_all:.3f}")
    print("\n\n\n")


def get_size_and_position(ann):

    sizes = []
    position = []
    for bbox in ann['annotations']:
        size = bbox['bbox'][2] * bbox['bbox'][3]
        # get y center position
        yc = bbox['bbox'][1] + bbox['bbox'][3]/2
        y = bbox['bbox'][1]

        sizes.append(size)
        position.append(yc)

    return sizes, position

def plot_size_and_position(data):

    # Scatter plot of size and position
    fig, ax = plt.subplots(1)
    fig.tight_layout()
    fig.set_figheight(2)
    fig.set_figwidth(6)


    # stack the data vertically
    # stack the first axis of the data
    # data = np.stack(data,axis=1)
    # ax.scatter(data[0], data[1], s=1, c=data[0], alpha=1.0, cmap='turbo')
    ax.scatter(data[0][0], data[0][1], s=1, c=data[0][0], alpha=1.0, cmap='turbo')
    ax.scatter(data[1][0], data[1][1], s=1, c=data[1][0], alpha=1.0, cmap='turbo')
    ax.scatter(data[2][0], data[2][1], s=1, c=data[2][0], alpha=1.0, cmap='turbo')
    ax.set_title('')
    ax.set_xscale('log')
    ax.set_xlabel('Bounding box size')
    ax.set_ylabel('Y position')
    ax.annotate("",
            xy=(96*96, 0), xycoords='data',
            xytext=(96*96, 1024), textcoords='data',
            arrowprops=dict(arrowstyle="-", connectionstyle="arc3"))
    ax.annotate("",
            xy=(32*32, 0), xycoords='data',
            xytext=(32*32, 1024), textcoords='data',
            arrowprops=dict(arrowstyle="-", connectionstyle="arc3"))

    ax.annotate("Small\n$(area < 32^2)$\n$10052 (81.2\%)$",
                xy=(20, 330), xycoords='data',
                xytext=(2, 0), textcoords='offset points')

    ax.annotate("Medium\n$(32^2 < area < 96^2)$\n$2079 (16.8\%)$",
                xy=(32*32, 330), xycoords='data',
                xytext=(2, 0), textcoords='offset points')

    ax.annotate("Large\n$(96^2 < area)$\n$74 (2.0\%)$",
                xy=(96*96, 330), xycoords='data',
                xytext=(2, 0), textcoords='offset points')

    ax.set_xlim((20,60000))
    ax.set_xticks([6*6, 14*14, 32*32, 96*96, 192*192], labels=["$6^2$", "$14^2$", "$32^2$", "$96^2$", "$192^2$"])
    # set y range to 0-1024
    ax.set_ylim((1024,0))

    plt.minorticks_off()
    plt.title("")
    plt.savefig(os.path.join(args.ouput_dir,"size.png"), bbox_inches='tight')




if __name__ == '__main__':

    args = parse_args()

    os.makedirs(args.ouput_dir, exist_ok=True)

    input_files = ["result.json", "train_rgb.json", "val_rgb.json", "test_rgb.json"]

    stats = []
    spos = []

    for input_file in input_files:
        ann_path = os.path.join(args.input_dir, input_file)

        with open(ann_path, 'r') as f:
            ann = json.load(f)

        sizes = get_bbox_sizes(ann)
        stats.append(sizes)
        a_bbox_sizes, s_bbox_sizes, m_bbox_sizes, l_bbox_sizes = sizes
        print_stats(a_bbox_sizes, s_bbox_sizes, m_bbox_sizes, l_bbox_sizes, input_file)
        # plot_bboxes(a_bbox_sizes, s_bbox_sizes, m_bbox_sizes, l_bbox_sizes, input_file)
        size, position = get_size_and_position(ann)
        spos.append([size,position])

    plot_size_and_position(spos)

    plot_log2(stats[1][0], stats[2][0], stats[3][0])



