import torch
import cv2
import threading
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import torch
import ctypes

CONF_THRESH = 0.25
IOU_THRESHOLD =  0.45
MAX_OUTPUT_BBOX_COUNT = 1000
KEY_POINTS_NUM = 17
PILGIN_LIB = '/yolov7-pose/YoloLayer_TRT_v7.0/build/libyolo.so'
ENGINE = 'yolov7-w6-pose.engine'


class yolopose(object):
    def __init__(self, engine_file_path):
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []
        context.set_binding_shape(0, (1, 3,  960, 960))

        for binding in engine:
            print('bingding:', binding, engine.get_binding_shape(binding))
            binding_idx = engine.get_binding_index(binding)
            size = trt.volume(context.get_binding_shape(binding_idx)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = 1

    def cxcywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [cx, cy, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        """
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = (x[:, 0] - (x[:, 2] / 2)) * (origin_w / 960)
        y[:, 2] = (x[:, 0] + (x[:, 2] / 2)) * (origin_w / 960)
        y[:, 1] = (x[:, 1] - (x[:, 3] / 2)) * (origin_h / 960)
        y[:, 3] = (x[:, 1] + (x[:, 3] / 2)) * (origin_h / 960)
        return y
    
    
    def nms(self, prediction, origin_h, origin_w, conf_thres=0.75, nms_thres=0.65):
        # Get the boxes that score > CONF_THRESH
        boxes = prediction[prediction[:, 4] >= conf_thres]
        # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes[:, :4] = self.cxcywh2xyxy(origin_h, origin_w, boxes[:, :4])
        # clip the coordinates
        boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w -1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w -1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h -1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h -1)
        # Object confidence
        confs = boxes[:, 4]
        # Sort by the confs
        boxes = boxes[np.argsort(-confs)]
        # Perform non-maximum suppression
        keep_boxes = []

        while boxes.shape[0]:
            large_overlap = self.bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > nms_thres
            label_match = boxes[0, 5] == boxes[:, 5]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            keep_boxes += [boxes[0]]
            boxes = boxes[~invalid]
        boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
        return boxes

    def infer_kpts(self, image_raw):
        threading.Thread.__init__(self)
        self.ctx.push()
        # Restore
        stream = self.stream
        context = self.context
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # Do image preprocess
        blob = cv2.dnn.blobFromImage(image_raw,scalefactor=1/255,size=(960,960),swapRB=True) ## TODO : DO this with pytorch       
        # Copy input image to host buffer
        np.copyto(host_inputs[0], blob.ravel())
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)

        # Synchronize the stream
        stream.synchronize()
        self.ctx.pop()

        detections = host_outputs[0]
        bboxes,scores,kpts = [],[],[],[],[]
        num = int(detections[0])
        # Reshape to a two dimentional ndarray
        pred = np.reshape(detections[1:], (-1, 57))[:min(num, MAX_OUTPUT_BBOX_COUNT), :]
        H,W,C = image_raw.shape
        boxes = self.nms(pred, H, W, conf_thres=CONF_THRESH, nms_thres=IOU_THRESHOLD)
        for i in range(boxes.shape[0]):
            bboxes.append([int(boxes[i, 0]), int(boxes[i, 1]), int(boxes[i, 2] - boxes[i, 0]), int(boxes[i, 3] - boxes[i, 1])])
            scores.append(boxes[i, 4])
            kpts.append(boxes[i, 6:])
        return bboxes, scores, kpts

    def post_process(self, bboxes, kpts, image_raw):
        H,W,C = image_raw.shape
        for box in bboxes:
            image_raw = cv2.rectangle(image_raw, (box[0],box[1]), (box[0]+box[2],box[1]+box[3]), (0,0,255), 1)
        
        for k in kpts: 
            for i in range(KEY_POINTS_NUM-1):
                kpt_x = int(k[i*3] * (W / 960))
                kpt_y = int(k[(i*3)+1] * (H / 960))
                image_raw = cv2.circle(image_raw, (kpt_x, kpt_y), 2, (0,255,0), -1)

        return image_raw

    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
            
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # Get the coordinates of the intersection rectangle
        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)

        # Intersection area
        inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
        return iou


        
class infer_video_pose:
    def __init__(self):
        ctypes.CDLL(PILGIN_LIB)
        self.detector = yolopose(ENGINE)
     
    def run(self, image):
        return self.detector.infer_kpts(image)
