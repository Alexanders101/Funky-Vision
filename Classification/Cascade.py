__author__ = 'Alex'

import os
import cv2
from time import time
from numpy import min as np_min
from time import sleep
from copy import copy


CLASSIFIER = None
DETECT = None


def get_centered_box(frame, size=1):
    """size can be:
    1: small
    2: medium
    3: large"""

    shape = frame.shape
    width = shape[1]
    length = shape[0]
    smallest_dim = np_min([width, length])
    offset = int(((size + 1) / 4.0) * (smallest_dim / 2.0))
    x = (width / 2) - offset
    y = (length / 2) - offset
    w = offset * 2
    h = offset * 2

    return (x, y, w, h)


def train(image_get_function, HOME_PATH, FRAME_SKIP=5,
          SAMPLE_DIM=25, NUM_STAGES=15, NUM_POS_SAMPLES=1500):
    """get functions has to return: (Success, frame)"""

    # Folder Setup
    os.system('mkdir {}'.format(HOME_PATH))
    HOME_PATH += '/'
    os.system('mkdir {}'.format((HOME_PATH + 'front')))
    os.system('mkdir {}'.format((HOME_PATH + 'back')))
    # Change to working directory
    os.chdir(HOME_PATH)

    # Path setup
    b_file_path = HOME_PATH + 'back.dat'
    b_folder_path = HOME_PATH + 'back/{}.jpg'
    b_dest_path = './back/{}.jpg\n'
    f_file_path = HOME_PATH + 'front.dat'
    f_folder_path = HOME_PATH + 'front/{}.jpg'
    f_dest_path = './front/{}.jpg\n'
    vec_file_path = HOME_PATH + 'samples.vec'

    # Format functions
    f_image = f_folder_path.format
    b_image = b_folder_path.format
    f_dest = f_dest_path.format
    b_dest = b_dest_path.format

    # File Setup
    front_file = open(f_file_path, 'w')
    back_file = open(b_file_path, 'w')

    # Loop Setup
    ret = True
    front_frames = []
    back_frames = []
    count = 0
    count_skip = 0
    font = cv2.FONT_HERSHEY_SIMPLEX


    # Optimization
    cap_read = image_get_function
    cv2_im_write = cv2.imwrite
    cv2_im_show = cv2.imshow
    cv2_putText = cv2.putText
    cv2_rectangle = cv2.rectangle
    front_file_write = front_file.write
    back_file_write = back_file.write
    add_front_frame = front_frames.append
    add_back_frame = back_frames.append

    raw_input('First Step is to provide background samples. Press ENTER to start')

    # Back frames Loop
    while ret:
        ret, frame = cap_read()
        text_loc = (10, frame.shape[0] - 10)
        cv2_putText(frame, 'prepare background, press w when ready', text_loc, font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2_im_show('video', frame)
        if cv2.waitKey(1) & 0xFF == ord('w'):
            break
    while ret:
        t0 = time()
        count_skip += 1
        ret, frame = cap_read()
        if (count_skip % FRAME_SKIP) == 0:
            add_back_frame(frame)
        t1 = time()
        cv2_im_show('video', frame)
        print('%.2f FPS' % (1 / (t1 - t0)))
        print(len(back_frames))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            print('number of background images: {}'.format(len(back_frames)))
            for frame in back_frames:
                count += 1
                cv2_im_write(b_image('back%i' % (count)), frame)
                back_file_write(b_dest('back%i' % (count)))
            break

    raw_input('Prepare Object. This will take you through 3 views: far, medium, and close view. Press ENTER to start')
    count = 0
    count_skip = 0
    text_loc = (10, frame.shape[0] - 10)

    # Front frames Loop
    for i in range(1, 4):
        x, y, w, h = get_centered_box(frame, i)
        while ret:
            ret, frame = cap_read()
            cv2_rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2_putText(frame, 'Allign object and press w when ready', text_loc, font, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2_im_show('video', frame)
            if cv2.waitKey(1) & 0xFF == ord('w'):
                break
        while ret:
            t0 = time()
            count_skip += 1
            ret, frame = cap_read()
            if (count_skip % FRAME_SKIP) == 0:
                add_front_frame(copy(frame[y:y + h, x:x + w]))
            t1 = time()
            cv2_rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2_im_show('video', frame)
            print('%.2f FPS' % (1 / (t1 - t0)))
            print(len(front_frames))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                print('number of object images: {}'.format(len(front_frames)))
                break

    for frame in front_frames:
        count += 1
        cv2_im_write(f_image('front%i' % (count)), frame)
        front_file_write(f_dest(('front%i' % (count))))

    # Clean Up
    front_file.close()
    back_file.close()
    num_back = len(back_frames)
    print('Finishing up. Please wait 3 seconds')
    sleep(3)

    # Training
    print('Creating the vec file, this could take some time.\n')
    os.system('perl /home/alex/source/opencv/utils/createtrainsamples.pl {} {} {} {} '
              '"opencv_createsamples -maxxangle 1.1 -maxyangle 1.1 maxzangle 0.5 -maxidev 40 '
              '-w {} -h {}"'.format(f_file_path, b_file_path, HOME_PATH + 'samples', NUM_POS_SAMPLES, SAMPLE_DIM,
                                    SAMPLE_DIM))
    os.system('python /home/alex/source/opencv/utils/mergevec.py -v {} -o {}'.format('samples', 'samples.vec'))

    sleep(5)

    print('\nDone creating vec file, proceeding to training\n')
    os.system('opencv_traincascade -data {} -vec {} -bg {} -w {} -h {} -numPos {} -numNeg {} '
              '-numStages {} -featureType LBP'.format(HOME_PATH, vec_file_path, b_file_path, SAMPLE_DIM, SAMPLE_DIM,
                                                      int(NUM_POS_SAMPLES * 0.8), num_back, NUM_STAGES))

    return (HOME_PATH + 'cascade.xml')


def setup(path):
    global CLASSIFIER
    global DETECT
    CLASSIFIER = cv2.CascadeClassifier(path)
    DETECT = CLASSIFIER.detectMultiScale


# def predict(frame):
#     gray = cv2_cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 5, 1)





