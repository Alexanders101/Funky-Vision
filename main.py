import freenect as fn
from copy import copy

import cv2
import scipy.stats as sp

import frame_convert
from utils import *
from Classification.Generic import prediction
from Classification.Neural_Net import Neural_Net
from Density_Isolate import isolate_depths, __pipeline, __get_bins
from Key_Points import KeyPoints


MakeFirst = False
ModelPath = "/home/alex/brains/main.pkl"
ImageSize = 307200  # 480 by 640

def get_depth(raw=False):
    frame = fn.sync_get_depth()[0]
    if not raw:
        frame = frame_convert.pretty_depth(frame)
    if raw:
        frame = make_pretty_raw(frame)
    else:
        frame = np.invert(frame)
    # frame /= 8
    # frame = frame.astype(np.uint8)
    return frame
def get_video():
    frame = fn.sync_get_video()[0]
    return frame

def remove_background_flat(frame, threshold=100):
    frame = sp.threshold(frame, threshold, None, 0)
    return frame
def remove_background_percent(frame, thresh=.5, average=None):
    if average is None:
        max = np.max(frame)
    else:
        max = n_largest(frame, average)
        max = int(np.mean(max))
    threshold = max - (thresh * max)
    frame = sp.threshold(frame, threshold, None, 0)
    return frame
def remove_background_farthest(frame, thresh, average=None):
    if average is None:
        max = np.min(frame)
    else:
        max = n_smallest(frame, average)
        max = int(np.mean(max))
    if max == 0:
        max = np.mean(frame)
    threshold = max + (thresh * max)
    frame = sp.threshold(frame, threshold, None, 0)
    return frame
def remove_back_clahe(frame):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    frame = clahe.apply(frame)
    img = cv2.GaussianBlur(frame, (5, 5), 0)
    ret, th = cv2.threshold(img, 0, 2048, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def remove_foreground(frame, thresh=.2):
    bottom = frame[-1]
    bottom = np.mean(bottom)
    threshhold = bottom - (bottom * thresh)
    frame = sp.threshold(frame, None, threshhold, 0)
    return frame


def main_vision(load):
    # inits
    fn_ctx = fn.init()
    fn_dev = fn.open_device(fn_ctx, fn.num_devices(fn_ctx) - 1)
    fn.set_tilt_degs(fn_dev, 0)
    fn.close_device(fn_dev)
    key_point = KeyPoints(150)
    predictor = prediction(ModelPath)
    preds = []

    # optimization
    t0 = 0.0
    t1 = 0.0
    fps = 0.0
    total_fps = 0.0
    frames = 0
    kp_speed = key_point._get_kp_speedup()
    draw_speed = key_point._get_draw_speedup()
    proc_speed = key_point._get_process_speedup()
    cvtColor = cv2.cvtColor
    BGR2RGB = cv2.COLOR_BGR2RGB
    get_kp = key_point.get_key_points
    draw_kp = key_point.draw_key_points
    process_image = key_point.__process_image
    show_image = cv2.imshow
    wait_for_key = cv2.waitKey
    copy_thing = copy.copy
    num_features = key_point.get_num_features()
    arr_shape = np.shape
    shape_check = (num_features, 32)
    ravel = np.ravel
    append_pred = preds.append
    get_time = time.time

    current_class = 0
    if load:
        brain = predictor.load_brain()
        pred_speed = predictor.get_pred_speed()
        predict = predictor.predict
    else:
        add_speed = predictor.get_add_speed()
        add_data = predictor.add_data
        get_length = predictor.get_data_length
    if load:
        net = Neural_Net(predictor.brain.getPoint(), np.vstack(predictor.brain.getData()), 4800 * 2, num_features)
        nl_predict = net.predict
        nl_speed = net.get_neural_speed()

    # mainLoop
    while True:
        t0 = get_time()

        # Get a fresh frame
        depth = get_depth()
        frame = get_video()
        show_image('Raw Image', cvtColor(frame, BGR2RGB))

        # Process Depth Image
        # depth = remove_background(depth, 25)
        depth = remove_background_percent(depth, .5, 50)
        depth = convert_to_bw(depth)
        mask = make_mask(depth)

        # Process Image
        frame = cvtColor(frame, BGR2RGB)
        video = copy_thing(frame)
        frame = process_image(frame, proc_speed)
        # Make Masked Frame
        masked_frame = copy_thing(frame)
        masked_frame[mask] = 0

        # Process Key Points
        kp, des = get_kp(masked_frame, kp_speed)
        video = draw_kp(video, kp, True, speedup=draw_speed)

        # Predict current
        if (load) and (des is not None) and (arr_shape(des) == shape_check):
            pred = predict(ravel(des), pred_speed)
            append_pred(pred)
            print(pred)
            print(nl_predict([ravel(des)], nl_speed))
        # Add object description to data set
        if (not load) and (des is not None) and (arr_shape(des) == shape_check):
            add_data(add_speed, np.ravel(des), current_class)
            print('Current Class and Length:\t%i\t%i' % (get_length(), current_class))

        t1 = get_time()
        fps = (1 / (t1 - t0))
        total_fps += fps
        frames += 1
        print('%.2f FPS' % fps)
        show_image('masked image', masked_frame)
        show_image('depth', depth)
        show_image('key points', video)
        # show_image('all', frame, masked_frame, depth, video)
        if wait_for_key(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            if load:
                break
            print('Current Class: %i\nn : Next Class\nr : Continue Current Class\nq : Quit' % (current_class))
            inp = raw_input()
            if inp == 'n':
                current_class += 1
            elif inp == 'q':
                break

    # print(np.mean(preds))
    cv2.destroyAllWindows()
    print('Average FPS: %.2f' % (total_fps / frames))
    fn.sync_stop()
    if not load:
        predictor.create_brain()
        main_vision(True)


def depth_view():
    import matplotlib.pyplot as plt

    fn_ctx = fn.init()
    fn_dev = fn.open_device(fn_ctx, fn.num_devices(fn_ctx) - 1)
    fn.set_tilt_degs(fn_dev, 0)
    fn.close_device(fn_dev)

    x = np.arange(0, 256, 1)
    zeros = np.zeros_like(x)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    view1, = ax.plot(x, zeros, '-k', label='black')
    view2, = ax.plot(x, zeros, '-r', label='red')

    np_array = np.array
    np_max = np.max
    view1_sety = view1.set_ydata
    view2_sety = view2.set_ydata
    ax_relim = ax.relim
    ax_autoscale_view = ax.autoscale_view
    while True:
        dep = get_depth(False)
        cv2.imshow('raw', dep)

        dep = cv2.medianBlur(dep, 3)
        bin = __get_bins(dep)
        clean = copy(bin)
        clean = __pipeline(clean)
        bin[:2] = 0
        clean *= np_max(bin)
        view1_sety(bin)
        view2_sety(clean)
        ax_relim()
        ax_autoscale_view()
        im = fig2img(fig)
        graph = np_array(im)


        # dep = remove_background(dep, 100)
        dep = isolate_depths(dep)
        # dep = convert_to_bw(dep)

        cv2.imshow('depth', dep)
        cv2.imshow('graph', graph)
        # show_image('all', frame, masked_frame, depth, video)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def density_plot():
    fn_ctx = fn.init()
    fn_dev = fn.open_device(fn_ctx, fn.num_devices(fn_ctx) - 1)
    fn.set_tilt_degs(fn_dev, 0)
    fn.close_device(fn_dev)

    show_image = cv2.imshow
    waitkey = cv2.waitKey
    ravel = np.ravel
    countbin = np.bincount

    length = 256
    nums = np.arange(0, length, 1)
    zero = np.zeros_like(nums)

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig, ax = plt.subplots()
    line, = ax.plot(nums, zero)
    ax.set_ylim(0, 10000)
    ax.set_xlim(0, 256)
    set_y_data = line.set_ydata

    def update(data):
        set_y_data(data)
        return line,

    def get_dep():
        dep = get_depth()
        dep = cv2.medianBlur(dep, 3, dep)
        dep = ravel(dep)
        # dep = medfilt(dep, 21).astype(np.uint8)
        return dep

    def data_gen():
        while True:
            yield countbin(get_dep(), minlength=length)

    ani = animation.FuncAnimation(fig, update, data_gen)
    plt.show()

    cv2.destroyAllWindows()

    fn.sync_stop()


def temp_test():
    fn_ctx = fn.init()
    fn_dev = fn.open_device(fn_ctx, fn.num_devices(fn_ctx) - 1)
    fn.set_tilt_degs(fn_dev, 0)
    fn.close_device(fn_dev)

    while True:
        dep = get_depth()
        dep *= (dep * 1.3).astype(np.uint8)
        print("{}\t,\t{}".format(np.min(dep), np.max(dep)))

        cv2.imshow('depth', dep)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            fn.sync_stop()
            break

# doloop(not MakeFirst)
depth_view()
# density_plot()
# temp_test()
