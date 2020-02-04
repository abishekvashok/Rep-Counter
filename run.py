import tensorflow as tf
import cv2
import time
import argparse

import posenet

parser = argparse.ArgumentParser()
# The best known value of variance is yet to be known, I assume it as 40.
parser.add_argument('--variance', type=int, default=40, help='The tolerance for the model in integers')
parser.add_argument('--model', type=int, default=101, help='The model to use, available versions are 101 (def.), 102, 103 etc')
parser.add_argument('--cam_id', type=int, default=0, help='The respective cam id to use (default 0)')
parser.add_argument('--cam_width', type=int, default=1280, help='The width of the webcam in pixels (def. 1280)')
parser.add_argument('--cam_height', type=int, default=720, help='The height of the webcam in pixels (def. 780)')
parser.add_argument('--scale_factor', type=float, default=0.7125, help='The scale factor to use (default: .7125)')
parser.add_argument('--file', type=str, default=None, help="Use the video file at specified path instead of live cam")
args = parser.parse_args()

keyValues = ['Nose', 'Left eye', 'Right eye', 'Left ear', 'Right ear', 'Left shoulder',
             'Right shoulder', 'Left elbow', 'Right elbow', 'Left wrist', 'Right wrist',
             'Left hip', 'Right hip', 'Left knee', 'Right knee', 'Left ankle', 'Right ankle']
variance = args.variance

def countRepetition(previous_pose, current_pose, previous_novel, flag):
    if current_pose[0][10][0] == 0 and current_pose[0][10][1] == 0:
        return 'Cannot detect any joint in the frame', previous_pose, previous_novel, flag
    else:
        string = ''
        novel_string = ''
        # Discard first 5 (0-4 indices) values, we don't need the value of nose, eye etc
        for i in range(5, 17):
            # The fancy text overlay
            string = string + keyValues[i] + ': '
            string = string + str('%.2f' % (current_pose[0][i][0])) + ' ' + str('%.2f' % current_pose[0][i][1]) + '\n'
            # If the difference is greater or lesser than variance or -variance indicate increase or decrease
            # respectively. If no such situation occurs, copy the previous value
            dx = (current_pose[0][i][0] - previous_pose[0][i][0])
            dy = (current_pose[0][i][1] - previous_pose[0][i][1])
            if(dx > variance):
                novel_string += '1'
            elif(dx < (-1 * variance)):
                novel_string += '0'
            else:
                novel_string += previous_novel[((i-5)*2)]
            if(dy > variance):
                novel_string += '1'
            elif(dy < (-1 * variance)):
                novel_string += '0'
            else:
                novel_string += previous_novel[((i-5)*2)+1]
        # Increase flag if there is a change in the slope of the graph
        if(novel_string != previous_novel):
            flag += 1 # Not flag = 1 bcz flag is initialized to -1
        return string, current_pose, novel_string, flag

def main():
    with tf.Session() as sess:
        # Load the models
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        if args.file is not None: # Frame source, speicifed file or the specified(or default) live cam
            cap = cv2.VideoCapture(args.file)
        else:
            cap = cv2.VideoCapture(args.cam_id)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)
        previous_pose = '' # '' denotes it is empty, really fast checking!
        count = 0 # Stores the count of repetitions
        # A flag denoting change in state. 0 -> previous state is continuing, 1 -> state has changed
        flag = -1
        # Novel string stores a pair of bits for each of the 12 joints denoting whether the joint is moving up or down
        # when plotted in a graph against time, 1 denotes upward and 0 denotes downward curving of the graph. Assume
        # we move upward first, all motions ideally start from the ground
        novel_string = '1' * 24
        while True:
            # Get a frame, and get the model's prediction
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)
            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )
            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.4)
            keypoint_coords *= output_scale # Normalising the output against the resolution

            if(previous_pose == ''): # if previous_pose was not inialised, assign the current keypoints to it
                previous_pose = keypoint_coords
            
            text, previous_pose, novel_string, flag = countRepetition(previous_pose, keypoint_coords, novel_string, flag)

            if(flag == 1):
                count += 1
                flag = 0

            image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.4, min_part_score=0.1)

            # OpenCV does not recognise the use of \n delimeter
            y0, dy = 20, 20
            for i, line in enumerate(text.split('\n')):
                y = y0 + i*dy
                image = cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255),1)

            image = cv2.putText(image, 'Count: ' + str(count), (10, y+20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,0,0),2)
            cv2.imshow('RepCounter', image)

            if(cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == ord('Q')):
                break # Exit on press of q or Q

if __name__ == "__main__":
    main()
