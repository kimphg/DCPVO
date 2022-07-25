
from libs.deep_models.flow.lite_flow_net.lite_flow import LiteFlow
import torch
import numpy as np
import cv2
def initialize_deep_flow_model(h, w, weight):
    """Initialize optical flow network

    Args:
        h (int): image height
        w (int): image width
    
    Returns:
        flow_net (nn.Module): optical flow network
    """
    flow_net = LiteFlow(h, w)
    flow_net.initialize_network_model(
            weight_path=weight,
            finetune=False
            )
    return flow_net
ref_h = 200
ref_w = 300
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
flow_net = initialize_deep_flow_model(ref_h, ref_w, "network-default.pytorch")
# resize image
path = "D:/DATA/"
filename = "Cessna_1.avi"
cap = cv2.VideoCapture(path+filename)
ret, prev_img = cap.read()
output = cv2.VideoWriter(path+"_"+filename,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (ref_w,ref_h))
prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2RGB)
prev_img = cv2.resize(prev_img, (ref_w, ref_h))
mask = np.zeros_like(prev_img)
while(cap.isOpened()):
    ret, new_img = cap.read()
    
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    new_img = cv2.resize(new_img, (ref_w, ref_h))
    # resize image
    
    ''' prediction '''
    cur_imgs = [np.transpose((new_img)/255, (2, 0, 1))]
    ref_imgs = [np.transpose((prev_img)/255, (2, 0, 1))]
    ref_imgs = torch.from_numpy(np.asarray(ref_imgs)).float().to(device)
    cur_imgs = torch.from_numpy(np.asarray(cur_imgs)).float().to(device)

    flows = {}
    # Flow inference
    batch_flows = flow_net.inference_flow(
                            img1=cur_imgs[0:1],
                            img2=ref_imgs[0:1],
                            forward_backward=False,
                            dataset="kitti")
        
    flows = batch_flows['forward']

    # resie flows back to original size
    ''' Save result '''
    _, _, h, w = flows.shape
    flows = flow_net.resize_dense_flow(flows, h, w).cpu()
 
    _, _, h, w = flows.shape
    flows3 = np.ones((h, w, 1))
    
    # if args.flow_mask_thre is not None:
    #     resized_mask = cv2.resize(batch_flows['flow_diff'][0,:,:,0], (w, h))
    #     flow_mask = (resized_mask < args.flow_mask_thre) * 1
    #     flows3[:, :, 0] = flow_mask
    #flows3[:, :, 2] = flows[0,0] * 128 + 2**15
    flows3[:, :, 0] = flows[0,1] * 128 # + 2**15
    output.write(flows3)
    flows3 = flows3.astype(np.uint16)

    cv2.imshow("out_put", flows3)
    
    cv2.imshow("new_img", new_img)
    cv2.waitKey(30)
    prev_img  = new_img