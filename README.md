AI Watchman!
-
- This project is to detect and recognize visitors at your door though voice alerts or notification through alexa devices.
this can work with existing installed cameras at your door all needs to be done is to get RSTP [https://en.wikipedia.org/wiki/Real-Time_Streaming_Protocol] data stream from you camera.
Video files will also be processed
- AI Watchman will match visitor faces with the faces provided in input_faces folder along with their identity details in input_faces.json and in realworld this can be done using a service or a DB
- If AI watchman matches visitor among the input faces then it will send notification Alexa with visitor's identity details and alexa will play audio message like "You a new visitor Ashish at the door"
- Matched faces results will be captured and saved to output_matches
- If AI watchman doesn't recognize visitor from input faces, then I will notify with audio message "Unknown Visitor at the door"
- Later on it will capture image of Unknown visitor(s) and save it in a folder (un_known_faces) and in realworld this can be taken care by a DB or a service
- User can tag unknown faces to an identity and move them to input faces later on and this also can be taken care by a service or application

Face Recognition is based on SFace.
SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition

Note:

- SFace is contributed by [Yaoyao Zhong](https://github.com/zhongyy).
- Sources for Face Detection and Recognition Models
  - https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
  - https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface

| Models      | Accuracy |
| ----------- | -------- |
| SFace       | 0.9940   |
| SFace quant | 0.9932   |

\*: 'quant' stands for 'quantized'.

## Demo

***NOTE***: This demo uses [face_detection_yunet](../face_detection_yunet) as face detector, which supports 5-landmark detection for now (2021sep).

Run the following command to try the runme:

```shell
# recognize on images
python run_me.py --video_path /path/or/rstp uri to video --base_path /path/to/project folder --base_url_alaxa /alexa voicemonkey api url

# get help regarding various parameters
python run_me.py --help
```

## License

All files in this directory are licensed under [Apache 2.0 License](./LICENSE).

## Reference

- https://ieeexplore.ieee.org/document/9318547
- https://github.com/zhongyy/SFace
