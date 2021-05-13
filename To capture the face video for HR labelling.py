from imutils import face_utils
import imutils
import dlib
import cv2 as cv
import numpy as np
import uuid
import math
import tensorflow
import glob

#spatial temporal map function => 5x5 pixel image
def stmap(src):
    src = cv.resize(src, dsize=(200,150), interpolation=cv.INTER_AREA)
    Y_o, U_o, V_o = cv.split(src)

    row = len(Y_o)
    col = len(Y_o[1, :])

    row_index = int((row / 5))
    col_index = int((col / 5))
    x = row % 5
    y = col % 5

    array = np.zeros((5, 5, 3))
    Y, U, V  = cv.split(array)

    for i in range(0, row, row_index):
        for j in range(0, col, col_index):
            for a in range(i, i + row_index):
                for b in range(j, j + col_index):
                    Y[int(i / row_index), int(j / col_index)] += Y_o[a, b]
                    U[int(i / row_index), int(j / col_index)] += U_o[a, b]
                    V[int(i / row_index), int(j / col_index)] += V_o[a, b]
            Y[int(i / row_index), int(j / col_index)] = Y[int(i / row_index), int(j / col_index)] / (col_index * row_index)
            U[int(i / row_index), int(j / col_index)] = U[int(i / row_index), int(j / col_index)] / (col_index * row_index)
            V[int(i / row_index), int(j / col_index)] = V[int(i / row_index), int(j / col_index)] / (col_index * row_index)
    #Y=Y.reshape((25,1))
    #U=U.reshape((25,1))
    #V=V.reshape((25,1))

    return Y,U,V


def face_remap(shape):
    remapped_image = cv.convexHull(shape)
    return remapped_image

path_video = glob.glob(r'*.mp4')
path_txt = glob.glob(r'*.txt')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_81_face_landmarks.dat')
c=0;
for i in path_video :
    print(i)
    c=c+1;
    cap = cv.VideoCapture(i)
    length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    #width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    #height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps=int(cap.get(cv.CAP_PROP_FPS))
    print(fps)
    print(length)
    #print(length)
    #print(width)
    #print(height)
    T=fps
    nocount=0;
    count=0
    count1=1;
    while True:

        ret, img_frame_o = cap.read()
        # resize the video
        if(ret==False) :
            break
        img_frame = cv.resize(img_frame_o, dsize=(720, 480), interpolation=cv.INTER_AREA)
        cv.cvtColor(img_frame,cv.COLOR_BGR2YUV)
        img_gray = cv.cvtColor(img_frame, cv.COLOR_BGR2GRAY)
        out_face = np.zeros_like(img_frame)

        dets = detector(img_gray, 1)
        if not dets :
            nocount = nocount+1
            print("no face sensing") # 얼굴인식이 안될 경우 --> 처리 x
            break
        print(dets)
        count=count+1
        for face in dets:

            shape = predictor(img_frame, face)  # 얼굴에서 81개 점 찾기
            shape = face_utils.shape_to_np(shape)

            # initialize mask array
            remapped_shape = np.zeros_like(shape)
            feature_mask = np.zeros((img_frame.shape[0], img_frame.shape[1]))

            # we extract the face
            remapped_shape = face_remap(shape)
            cv.fillConvexPoly(feature_mask, remapped_shape[0:27], 1)
            feature_mask = feature_mask.astype(np.bool)
            out_face[feature_mask] = img_frame[feature_mask]

            #crop face image
        img_ROI = out_face[(int)((1 / 1.2) * face.top()):face.bottom(), face.left():face.right()]
        #cv.imshow('result', img_ROI)
        FileName="DOWN/face"+".png"
        cv.imwrite(FileName,img_ROI)
        height, width, channel = img_ROI.shape
        print(height, width, channel)
        img_5Y,img_5U,img_5V=stmap(img_ROI)
        img_picture=cv.merge((img_5Y,img_5U,img_5V))
        #if count>1 :
         #   Y=np.hstack((Y,img_5Y)); U=np.hstack((U,img_5U)); V=np.hstack((V,img_5V));
        #elif count==1 :
         #   Y=img_5Y; U=img_5U; V=img_5V;
        #if (count == T):  ##T==fps 일 때
         #   YUV = cv.merge((Y, U, V));
        img = cv. merge((img_5Y,img_5U,img_5V))
        print
        FaceFileName = "DOWN/face_" +str(c)+"video_" + str(count1) + "_" + str(count) + ".png"

        cv.imwrite(FaceFileName, img);
        count = 0
        count1 = count1 + 1



        #print(Y.shape)
        #YUV=np.zeros((5,5,3));
        #Y=np.zeros((5,5)); U=np.zeros((5,5)); V=np.zeros((5,5));
        #Y=np.append(img_5Y); U=np.append(img_5U); V=np.append(img_5V);
        #Y = np.array(Y, dtype=np.uint8)
        #U = np.array(U, dtype=np.uint8)
        #V = np.array(V, dtype=np.uint8)
        #YUV=cv.merge((Y,U,V))
        #YUV = np.array(YUV, dtype=np.uint8)

        #save ROI image
        #FaceFileName = "DOWN/face_" + str(uuid.uuid4()) + ".png"

        print(count)
        print(nocount)
        if cv.waitKey(1) & 0xFF == 27: break

cap.release()
#writer.release()
cap.destroyAllWindows()