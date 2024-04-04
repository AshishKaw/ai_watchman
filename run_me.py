# This File

import argparse
import glob
import json
import os
import sys
import time
import urllib

import cv2 as cv
import numpy as np
import requests
from PIL import Image
from gtts import gTTS
from playsound import playsound

from face_detection_yunet.yunet import YuNet
from sface import SFace

# Check OpenCV version
assert cv.__version__ >= "4.9.0", \
    "Please install latest opencv-python to try this demo: python3 -m pip install --upgrade opencv-python"

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA, cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA, cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX, cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN, cv.dnn.DNN_TARGET_NPU]
]

globalWaitime = 5

parser = argparse.ArgumentParser(
    description="SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition (https://ieeexplore.ieee.org/document/9318547)")
parser.add_argument('--base_path', type=str,
                    default="C:/SHIV/ai_watchman/ai_watchman/",
                    help="Base path or project location")
parser.add_argument('--video_path', type=str,
                    default="D:/NAS/KAWS_HUB/RHYTHM/M30s_Pictures/Camera/20230605_190632.mp4",
                    help="Path to the video stream or a clip")
parser.add_argument('--base_url_alaxa', '-t', type=str,
                    default="https://api-v2.voicemonkey.io/announcement?token=4e5f9535115a8c16298972c2692cf648_4131930f544cda5db63e431eadbc0bfa&device=kaws-home&text=",
                    help="Alexa voicemonkey.io API link to play sounds on your alexa devices")
parser.add_argument('--pass_score', type=float, default=0.5,
                    help="pass score to match face ")
parser.add_argument('--faceArea', type=int, default=500,
                    help="Minimum Face Size to be processed 22500(150px x 150px) so that far faces are ignored")
parser.add_argument('--max_matched_results_capture', type=int, default=20,
                    help="Maximum Number of latest matched results captures to saved")
parser.add_argument('--max_unknown_results_capture', type=int, default=100,
                    help="Maximum Number of latest matched results captures to saved")
parser.add_argument('--model', '-m', type=str, default='face_recognition_sface_2021dec.onnx',
                    help='Usage: Set model path, defaults to face_recognition_sface_2021dec.onnx.')
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pair to run this demo:
                        {:d}: (default) OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
parser.add_argument('--dis_type', type=int, choices=[0, 1], default=0,
                    help='Usage: Distance type. \'0\': cosine, \'1\': norm_l1. Defaults to \'0\'')
parser.add_argument('--save', '-s', action='store_true', default='true',
                    help='Usage: Specify to save file with results (i.e. bounding box, confidence level). Invalid in case of camera input.')
parser.add_argument('--vis', '-v', action='store_true', default='true',
                    help='Usage: Specify to open a new window to show results. Invalid in case of camera input.')

args = parser.parse_args()


def visualize(img1, faces1, details, matches, scores):
    out1 = img1.copy()
    pass_score = args.pass_score

    matched_box_color = (0, 255, 0)  # BGR
    mismatched_box_color = (0, 0, 255)  # BGR

    h1, w1, _ = out1.shape
    top = 0
    left = 0

    # Draw bbox
    assert faces1.shape[0] == len(matches), "number of faces2 needs to match matches"
    assert len(matches) == len(scores), "number of matches needs to match number of scores"
    for index, match in enumerate(matches):
        score = scores[index]
        bbox2 = faces1[index][:4]
        x, y, w, h = bbox2.astype(np.int32)
        text_color = matched_box_color if match and score > pass_score else mismatched_box_color

        if match == 0:
            color_converted = cv.cvtColor(img1.copy(), cv.COLOR_BGR2RGB)
            img11 = Image.fromarray(color_converted)
            img22 = img11.crop((x - w, y - h, x + (w * 2), y + (h * 2)))  # left,top,right,bottom
            img_detail = {}
            img_detail['name'] = 'unknown_' + str(index)
            img_detail['img_path'] = args.base_path + '/unknown_faces/unkwon_' + \
                                     img_detail['name'] + '_' + time.strftime(
                "%Y%m%d-%H%M") + '.jpg'
            img22.save(img_detail['img_path'])
        box_color = matched_box_color if match and score > pass_score else mismatched_box_color
        cv.rectangle(out1, (x + left, y + top), (x + left + w, y + top + h), box_color, 2)
        cv.putText(out1, "{:.2f}".format(score) + '_' + details[index]['name'], (x + left, y + top - 5),
                   cv.FONT_HERSHEY_DUPLEX, 0.4, text_color)

    delete_older_matches(args.base_path + '/unknown_faces/', args.max_unknown_results_capture)
    delete_older_matches(args.base_path + '/unknown_faces/', args.max_matched_results_capture)
    return np.concatenate([out1], axis=1)


def delete_older_matches(directory, max_store):
    files_path = os.path.join(directory, '*')
    files = sorted(glob.iglob(files_path), key=os.path.getctime, reverse=True)
    if len(files) > max_store:
        for x in range(max_store, len(files)):
            os.remove(files[x])


def cam_scanner():
    try:
        pass_score = args.pass_score
        backend_id = backend_target_pairs[args.backend_target][0]
        target_id = backend_target_pairs[args.backend_target][1]

        jsonFile = open(args.base_path + '/input_faces.json', 'r')
        data = json.load(jsonFile)
        jsonFile.close()
        # Instantiate SFace for face recognition
        recognizer = SFace(modelPath=args.model,
                           disType=args.dis_type,
                           backendId=backend_id,
                           targetId=target_id)
        # Instantiate YuNet for face detection
        detector = YuNet(
            modelPath=args.base_path + '/face_detection_yunet/face_detection_yunet_2023mar.onnx',
            inputSize=[320, 320],
            confThreshold=0.7,
            nmsThreshold=0.3,
            topK=5000,
            backendId=backend_id,
            targetId=target_id)
        cap = cv.VideoCapture()
        cap.open(args.video_path)

        loop = 0
        max_loop = 50
        while cv.waitKey(0) < 0 and loop < max_loop:
            hasFrame, frame = cap.read()
            loop += 1
            if not hasFrame:
                print('No frames grabbed!')
                return 'NO FRAMES'

            height0, width0, channels = frame.shape
            detector.setInputSize([width0, height0])
            results = detector.infer(frame)  # results is a tuple

            unknownFaces = detector.infer(frame)

            if results.shape[0] > 0:
                scores = []
                matches = []
                details = []
                index = 0
                matched = 0
                detected = 0

                for face in unknownFaces:
                    matched_img = {"name": "unknown_" + str(index)}
                    face_box = face[:4]
                    x, y, w, h = face_box.astype(np.int32)
                    face_area = w * h
                    if face_area > args.faceArea:
                        for img in data['images']:
                            if not img['name'].startswith(
                                    'unknown'):  # rename images when moving to input faces and entry needs to done in
                                img_path = img['img_path']
                                img2 = cv.imread(img_path)
                                height, width, channels = img2.shape
                                detector.setInputSize([width, height])
                                faces2 = detector.infer(img2)
                                result = recognizer.match(frame, face[:-1], img2, faces2[:-1])
                                if result[1] > 0 and result[0] > pass_score:
                                    matched += 1
                                    matched_img = img
                                    max_loop = 1
                                    break
                        scores.append(result[0])
                        matches.append(result[1])
                        detected += 1
                        index += 1
                        details.append(matched_img)
                    else:
                        print("? Found small face(s) / person standing far! decrease --faceArea to recognize far faces")
                if matched > 0:
                    image = visualize(frame, unknownFaces, details, matches, scores)
                    send_results(details)
                    time.sleep(2)
                    print("Total matches:: [" + str(matched) + '] out of persons [' + str(detected) + '] detected')
                    # Save results if save is true
                    if args.save:
                        cv.imwrite(
                            args.base_path + '/matched_facesmatched_faces/captureMatch' + time.strftime(
                                "%Y%m%d-%H%M") + '.jpg', image)

                else:
                    max_loop = 1
                    if len(details) > 0:
                        print('Un MatchesMatches saved to input_faces\n')
                        send_results(details)
                        time.sleep(2)
                        img_detail = {}
                        img_detail['name'] = 'unknown_' + str(detected)
                        visualize(frame, unknownFaces, details, matches, scores)

        cap.release()
        return 'RUNNING'
    except Exception as e:
        print('Unexpected error:', sys.exc_info()[0])
        print(e)
        return 'ERROR'


def send_results(result_details):
    print("Matched Persons")
    print(result_details)
    language = 'en'
    for i in range(len(result_details)):
        if not result_details[i]['name'].startswith('unknown'):
            # Use https://voicemonkey.io/ to set  Voice Routine for playing sounds in alexa devices
            url = args.base_url_alaxa + urllib.parse.quote(result_details[i]['speech'].encode('UTF-8'))
            requests.get(url)
            # Alternatively sounds can be played using text to speech conversion
            myobj = gTTS(text=result_details[i]['speech'], lang=language, slow=True)
            myobj.save(args.base_path + "/matched_faces/audio/_" + result_details[i]['name'] + ".mp3")
            playsound(args.base_path + '/matched_faces/audio/_' + result_details[i]['name'] + '.mp3')
        else:
            # Use https://voicemonkey.io/ to set  Voice Routine for playing sounds in alexa devices
            url = args.base_url_alaxa + urllib.parse.quote('Unknown visitor at the door'.encode('UTF-8'))
            requests.get(url)
            myobj1 = gTTS(text='Unknown visitor at the door', lang=language, slow=True)
            myobj1.save(args.base_path + "/unknown_faces/audio/unknown.mp3")
            playsound(args.base_path + "/unknown_faces/audio/unknown.mp3")


if __name__ == '__main__':

    while True:
        print("GlobalTime::" + str(globalWaitime))
        print("in Sleep Time [" + str(globalWaitime) + "] seconds")
        time.sleep(globalWaitime)
        print("Start Time :: " + time.strftime("%Y.%m.%d-%H:%M:%S"))
        out = cam_scanner()
        print("End Time :: " + time.strftime("%Y.%m.%d-%H:%M:%S"))
        if out == 'ERROR' or out == 'NO FRAMES':
            print('Sleeping for 30 seconds...')
            time.sleep(60)
