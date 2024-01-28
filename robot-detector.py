import cv2 as cv
import numpy as np
import cscore
import ntcore
import sys

import detector.detect as detect


# sinks = cscore.VideoSink.enumerateSinks()
# for sink in sinks:
#     print(sink.getName())
# sys.exit(0)

window_name = "Robot Feed"
cv.namedWindow(window_name, cv.WINDOW_NORMAL)

cam = cscore.HttpCamera("ring-feed", "http://10.21.65.11:1184/?action=stream")
# cam = cscore.CameraServer.startAutomaticCapture()
feed = cscore.CameraServer.getVideo(cam)
# feed = cscore.CameraServer.getVideo("photonvision_Port_1184_MJPEG_Server")

lower = np.array([0, 85, 100])
upper = np.array([32, 216, 255])

frame = np.ndarray((480, 640, 3))
while cv.waitKey(20) != 27:
    err, frame = feed.grabFrame(frame)
    if err == 0:
        # print("something went wrong")
        continue

    hsl = cv.cvtColor(frame, cv.COLOR_BGR2HLS_FULL)
    matches = detect.ring(hsl, lower, upper)
    out = cv.cvtColor(hsl, cv.COLOR_HLS2BGR_FULL)
    for m in matches:
        out = m.show(out, line_weight=2)
    
    cv.imshow(window_name, out)

    # ok, frame = video.read()
    # now = time.time()
    # if not ok:
    #     break

    # frame = cv.resize(frame, dims)
    
    # hsl = cv.cvtColor(frame, cv.COLOR_BGR2HLS_FULL)
    # matches = detect.ring(hsl, lower, upper)
    # #out = frame
    # out = cv.cvtColor(hsl, cv.COLOR_HLS2BGR_FULL)
    # for m in matches:
    #     out = m.show(out, line_weight=2)

    # steps.append(time.time() - now)
    # match_count.append(len(matches))
    # targeting_video.write(out)

cv.destroyWindow(window_name)



    