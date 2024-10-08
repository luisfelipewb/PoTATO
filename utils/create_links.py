import os
import argparse



def parse_args():
    parser = argparse.ArgumentParser(description='SFOD implementation')
    parser.add_argument('-i','--input_file', help='Path to the file with image names', default='../datasets/potato/test.txt')
    parser.add_argument('-o','--output_folder', help='Path to the folder where the links will be created', default='./input/test_images')
    parser.add_argument('-g','--image_folder', help='Path to the folder containing the images', default='../datasets/potato/images')
    parser.add_argument('-l','--label_folder', help='Path to the folder containing the labels', default='../datasets/potato/labels')

    # parser.add_argument('-c','--cs_thresh', help='Confidence Score Threshold', default=0.5, type=float)

    args = parser.parse_args()

    return args


def create_symbolic_link(target, link_name):
    print(target, " <-- ", link_name)
    # link_name = filename  # The name for the symbolic link will be the same as the filename
    # os.symlink(path, link_name)
    # print(f"Symbolic link created: {link_name} -> {path}")


def parse_file(args, tags):

    with open(args.input_file, 'r') as file:
        for token in file:
            for tag in tags:
                token = token.strip()
                img_name = f"{token}_{tag}.png"
                target = os.path.join(args.image_folder, img_name)
                link_name = os.path.join(args.output_folder, img_name)
                os.symlink(target, link_name)

def create_label_links(args, tags):

    with open(args.input_file, 'r') as file:
        for token in file:
            for tag in tags:
                token = token.strip()
                img_name = f"{token}_{tag}.txt"
                target = f"{token}_rgb.txt"
                link_name = os.path.join(args.label_folder, img_name)
                os.symlink(target, link_name)
                print(target, " <-- ", link_name)

if __name__ == '__main__':

    # Get parameters including cs_thresh, nms_thresh, and sigma.
    args = parse_args()

    tags = ["mono", "rgb", "rgbdif", "dolp", "pol", "pauli"]

    create_label_links(args, tags)




    

