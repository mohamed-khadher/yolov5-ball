import torch
import cv2
from tools import *

camera = cv2.VideoCapture(0)

ckpt = torch.load('best.pt', map_location='cpu')
ckpt = (ckpt.get('ema')).to('cuda').float()
ckpt.stride = torch.tensor([32.])
ckpt.names = dict(enumerate(['blue', 'red', 'yellow', 'green']))


model = ckpt.fuse().eval()
classes = ['Blue Ball', 'Red Ball', 'Yellow Ball', 'Green Ball']


while True:
    ret, frame = camera.read()
    
    im,r,_ = letterbox(frame, 416, 32, True)
    im = im.transpose((2, 0, 1))[::-1] 
    im = np.ascontiguousarray(im) 
    im = torch.tensor(im/255,dtype=torch.float, device='cuda').unsqueeze(0)

    preds = model(im)
    nms = non_max_suppression(preds, 0.25, 0.45)[0]
    image = draw_boxes(frame, r[0], nms, classes)
    cv2.imshow('RT Balls Detection', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


camera.release()
cv2.destroyAllWindows()