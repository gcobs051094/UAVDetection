from utils.general import download, os, Path
from PIL import Image
from tqdm import tqdm
import glob

def visdrone2yolo(path):
    def convert_box(size, box):
        # Convert VisDrone box to YOLO xywh box
        dw = 1. / size[0]
        dh = 1. / size[1]
        return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh

    #if not os.path.exists(path + '/labels'):
        #os.makedirs(path + '/labels')
        #print('Create dir!')
    pbar = tqdm(glob.glob(os.path.join(path + '/annotations', '*.txt')))
    for f in pbar:
        img_size = Image.open((path + '/images/' + f.split('\\')[1].split('.')[0] + '.jpg')).size
        lines = []
        with open(f, 'r') as file:  # read annotation.txt
            for row in [x.split(',') for x in file.read().strip().splitlines()]:
                if row[4] == '0':  # VisDrone 'ignored regions' class 0
                    continue
                cls = int(row[5]) - 1
                box = convert_box(img_size, tuple(map(int, row[:4])))
                lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
                with open(str(f).replace(os.sep + 'annotations' + os.sep, os.sep + 'labels' + os.sep), 'w') as fl:
                    fl.writelines(lines)  # write label.txt
                    
if __name__ == '__main__':
    # Convert
    path = './datasets/VisDrone/'
    for d in 'VisDrone2019-DET-train', 'VisDrone2019-DET-val', 'VisDrone2019-DET-test-dev':
        visdrone2yolo(path + d)  # convert VisDrone annotations to YOLO labels
