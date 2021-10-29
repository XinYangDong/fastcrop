import csv
import sys

sys.path.insert(0, './yolov5')

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn


def on_mouse(event, x, y, flags, userdata):
    global state, p1, p2, line
    # Left click
    if event == cv2.EVENT_LBUTTONUP:
        # Select first point
        if state == 0:
            p1 = (x, y)
            state += 1
        # Select second point
        elif state == 1:
            p2 = (x, y)
            state += 1
            print("设置车道编号如 01")
            line = input()
        # 设置车道数

    # Right click (erase current ROI)
    if event == cv2.EVENT_RBUTTONUP:
        p1, p2 = (0, 0), (0, 0)
        state = 0


p1 = (0, 0)
p2 = (0, 0)
state = 0
line = "01"
parent_path = ""
file_name = ""


def inDetArea(center, roi):
    if (center[0] > roi[0]) and (center[0] < roi[2]) and (center[1] > roi[1]) and (
            center[1] < roi[3]):
        return True
    else:
        return False


# 新创建的csv文件的存放路径，也可以创建一个.doc的word文档
def text_create(full_path):
    file = open(full_path, 'a', encoding='utf-8', newline='')
    file.close()


def createCsv(file_name, line, select, bbox, classname):
    print(file_name.split('.')[0])
    if (classname == "person"):
        classname = "e"
    else:
        classname = classname[0]
    csvfile = open(csvpath, 'a', newline='')
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(
        [str(file_name.split('.')[0]), str(line), select, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
         classname])


global csvpath
tracks = []


def detect(opt):
    out, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate = \
        opt.output, opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.img_size, opt.evaluate
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    save_path = str(Path(out))
    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(out)) + '/' + txt_file_name + '.txt'

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_sync()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_sync()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)
            raw_frame = im0.copy()
            selectImgpath = parent_path + "/frames_original/" + str(frame_idx).zfill(5) + ".jpg"
            dirs = os.path.dirname(selectImgpath)
            if not os.path.exists(dirs):
                os.makedirs(dirs)
            cv2.imwrite(selectImgpath, raw_frame)  # 保裁剪的车

            annotator = Annotator(im0, line_width=2, pil=not ascii)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)

                # draw boxes for visualization
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)):

                        bboxes = output[0:4]
                        if bboxes[0] > 10:
                            bboxes[0] -= 10
                        if bboxes[1] > 10:
                            bboxes[1] -= 10
                        if bboxes[2] < im0.shape[1] - 10:
                            bboxes[2] += 10
                        if bboxes[3] < im0.shape[0] - 10:
                            bboxes[3] += 10
                        id = output[4]
                        center = (int(((bboxes[0]) + (bboxes[2])) / 2), int(((bboxes[1]) + (bboxes[3])) / 2))
                        roi = (p1[0], p1[1], p2[0], p2[1])
                        if tracks.__contains__(id) and not inDetArea(center, roi):
                            continue
                        elif not tracks.__contains__(id) and inDetArea(center, roi):
                            tracks.append(id)

                            cls = output[5]
                            c = int(cls)  # integer class
                            class_name = names[c]
                            # print(class_name)
                            label = f'{id} {names[c]} {conf:.2f}'
                            annotator.box_label(bboxes, label, color=colors(c, True))

                            # 裁切
                            #bboxes_copy = bboxes.copy()
                            #print(bboxes_copy)
                            print(im0.shape)


                            cropImg = raw_frame[int(bboxes[1]):int(bboxes[3]), int(bboxes[0]): int(
                                bboxes[2])]  # 获取感兴趣区域
                            # cv2.imwrite('./cropImg/cropImg'+str(track.track_id)+'.png', cropImg)




                            # text = carplateidentity("./cropImg/cropImg0.png")
                            # colorname = vcrnet("./cropImg/cropImg0.png")
                            # cv2.putText(cropImg, colorname, (int(20), int(40)), 0, 20e-3 * 100, (0, 255, 0), 2)
                            cv2.namedWindow(str(class_name), 0)
                            cv2.resizeWindow(str(class_name), 200, 200)
                            cv2.imshow(str(class_name), cropImg)
                            print("如果需要保存请按s ,校正车辆类别按c ,其他按键esc取消保存")
                            k = cv2.waitKey(0)
                            if k == ord('s'):  # wait for 's' key to save and exit
                                croppath = parent_path + "/cut/" + line + "/" + str(frame_idx).zfill(5) + ".jpg"
                                dirs = os.path.dirname(croppath)
                                if not os.path.exists(dirs):
                                    os.makedirs(dirs)
                                cv2.imwrite(croppath, cropImg)  # 保裁剪的车

                                selectImg = raw_frame.copy()
                                selectImgpath = parent_path + "/select/" + line + "/" + str(frame_idx).zfill(5) + ".jpg"
                                dirs = os.path.dirname(selectImgpath)
                                if not os.path.exists(dirs):
                                    os.makedirs(dirs)
                                cv2.imwrite(selectImgpath, selectImg)  # 完整图片保存

                                createCsv(file_name, line, os.path.split(selectImgpath)[-1], bboxes, str(class_name))
                            elif k == ord('c'):
                                print("输入校正后的类别: e 电瓶车 electric bicycle m 摩托车 motorcycle c 轿车 car v 面包车 van b 公共汽车 bus t 卡车 truck ")
                                class_name = input()
                                croppath = parent_path + "/cut/" + line + "/" + str(frame_idx).zfill(5) + ".jpg"
                                dirs = os.path.dirname(croppath)
                                if not os.path.exists(dirs):
                                    os.makedirs(dirs)
                                cv2.imwrite(croppath, cropImg)  # 保裁剪的车

                                selectImg = raw_frame.copy()
                                selectImgpath = parent_path + "/select/" + line + "/" + str(frame_idx).zfill(5) + ".jpg"
                                dirs = os.path.dirname(selectImgpath)
                                if not os.path.exists(dirs):
                                    os.makedirs(dirs)
                                cv2.imwrite(selectImgpath, selectImg)  # 完整图片保存

                                createCsv(file_name, line, os.path.split(selectImgpath)[-1], bboxes, str(class_name))

                            if save_txt:
                                # to MOT format
                                bbox_left = output[0]
                                bbox_top = output[1]
                                bbox_w = output[2] - output[0]
                                bbox_h = output[3] - output[1]
                                # Write MOT compliant results to file
                                with open(txt_path, 'a') as f:
                                    f.write(('%g ' * 10 + '\n') % (frame_idx, id, bbox_left,
                                                                   bbox_top, bbox_w, bbox_h, -1, -1, -1,
                                                                   -1))  # label format

            else:
                deepsort.increment_ages()

            # Print time (inference + NMS)

            # Stream results
            im0 = annotator.result()
            if show_vid:
                cv2.namedWindow(p, 0)
                cv2.resizeWindow(p, 980, 480)

                # Register the mouse callback
                cv2.setMouseCallback(p, on_mouse)

                print('%sDone. (%.3fs)' % (s, t2 - t1))
                if state > 1:
                    cv2.rectangle(im0, p1, p2, (255, 0, 0), 10)
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

    if save_txt or save_vid:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', nargs='+', type=str, default='yolov5/weights/yolov5x.pt',
                        help='model.pt path(s)')
    parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7',
                        help='ckpt.t7 path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', default="True", action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', default=[0, 2, 5, 7], nargs='+', type=int,
                        help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)

    print("输入视频地址；如X:/gaodata/0001/0001.mp4")
    args.source = input()
    file_name = args.source
    parent_path = os.path.dirname(args.source)
    file_name = os.path.split(file_name)[-1]
    csvpath = os.path.join(parent_path, file_name.split('.')[0] + '.csv')  # CSV文档存点
    print(csvpath)
    text_create(csvpath)
    with torch.no_grad():
        detect(args)
