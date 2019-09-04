import typing

import cv2
import numpy
import scipy.stats

from tqdm import tqdm

from util.config import load_config
from nnet import predict
from util import visualize
from dataset.pose_dataset import data_to_input


cfg = load_config('demo/pose_cfg.yaml')
sess, inputs, outputs = predict.setup_pose_prediction(cfg)


def angle_between(
    p1: typing.Tuple[float, float],
    p2: typing.Tuple[float, float],
) -> float:
    ang1 = numpy.arctan2(*p1[::-1])
    ang2 = numpy.arctan2(*p2[::-1])
    return numpy.rad2deg((ang1 - ang2) % (2 * numpy.pi))


def get_video_frames(source) -> typing.Generator[bytes, None, None]:
    while source:
        ret, frame = source.read()
        if not ret:
            break
        yield frame
    source.release()


def get_frame_pose(frame):
    image_batch = data_to_input(frame)
    outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
    scmap, locref, _ = predict.extract_cnn_output(outputs_np, cfg)
    # Extract maximum scoring location from the heatmap, assume 1 person
    return predict.argmax_pose_predict(scmap, locref, cfg.stride)


def get_iterable_data(source) -> typing.Generator[dict, None, None]:
    for frame in get_video_frames(source):
        pose = get_frame_pose(frame)
        hip_average_position = numpy.array((
            int((pose[2][0] + pose[3][0]) / 2),
            int((pose[2][1] + pose[3][1]) / 2),
        ))
        shoulder_average_position = numpy.array((
            int((pose[8][0] + pose[9][0]) / 2),
            int((pose[8][1] + pose[9][1]) / 2),
        ))
        body_angle = 360 - angle_between(
            shoulder_average_position, hip_average_position
        )
        if body_angle > 90:
            # FIXME
            body_angle = angle_between(
                shoulder_average_position, hip_average_position
            )
        yield {
            'frame': frame,
            'pose': pose,
            'body_angle': body_angle,
            'ankle_positions': (
                pose[0], pose[5]
            ),
            'hip_average_position': hip_average_position,
            'shoulder_average_position': shoulder_average_position,
            'minimum_probability_passed': not (
                pose[0][-1] < 0.7 or pose[5][-1] < 0.7
            ),  # skip frames with ankle position prob less that 70%
        }


def get_run_direction(
    first_ankle_position: typing.Tuple[float, float],
    finish_ankle_position: typing.Tuple[float, float],
) -> str:
    run_direction = None
    if first_ankle_position[0] > finish_ankle_position[0]:  # right to left
        run_direction = 'rtl'
    else:
        run_direction = 'ltr'
    return run_direction


def main():
    video_source = cv2.VideoCapture('/mnt/source.mp4')

    width = int(video_source.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_source.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_source.get(cv2.CAP_PROP_FPS))
    total_frames_count = int(video_source.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f'Video parameters: width={width}, height={height}, fps={fps}, total_frames_count={total_frames_count}')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_output = cv2.VideoWriter(
        '/mnt/output.mp4', fourcc, fps, (width, height)
    )

    average_hip_position_line = []  # contains all average ankle points
    average_shoulder_position_line = []
    all_ankle_positions = []
    average_body_angle_data = numpy.empty((0, 1), float)
    for i, data in enumerate(tqdm(get_iterable_data(video_source))):
        frame = data['frame']
        average_hip_position = data['hip_average_position']
        average_shoulder_position = data['shoulder_average_position']
        average_hip_position_line.append(average_hip_position)
        average_shoulder_position_line.append(average_shoulder_position)
        average_body_angle_data = numpy.append(
            average_body_angle_data, data['body_angle']
        )
        all_ankle_positions.append(data['ankle_positions'])

        average_hip_height = int(numpy.average(
            numpy.array(average_hip_position_line)[-(fps // 4):, 1]
        ))
        average_shoulder_height = int(numpy.average(
            numpy.array(average_shoulder_position_line)[-(fps // 4):, 1]
        ))

        # draw joints and ankle to shoulder line
        frame = visualize.visualize_joints(frame, data['pose'])
        cv2.line(
            frame,
            (average_hip_position[0], average_hip_height),
            (average_shoulder_position[0], average_shoulder_height),
            (0, 255, 0), 5
        )

        # remove outliners from raw data via iqr range
        # https://en.wikipedia.org/wiki/Interquartile_range
        median_angle = numpy.median(average_body_angle_data)
        iqr = scipy.stats.iqr(average_body_angle_data)
        maximum_angle = median_angle + iqr
        minimum_angle = median_angle - iqr
        average_body_angle_data_without_outliners = numpy.extract(
            (minimum_angle <= average_body_angle_data) & (average_body_angle_data <= maximum_angle), average_body_angle_data
        )

        # draw statistics
        average_body_angle = round(numpy.average(
            average_body_angle_data_without_outliners
        ), 1)
        cv2.putText(
            frame,
            f'Body angle [avg]: {average_body_angle}',
            (0, 64), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0),
            3, cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f'Body angle [avg]: {average_body_angle}',
            (0, 64), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255),
            2, cv2.LINE_AA,
        )
        maximum_body_angle = round(numpy.amax(
            average_body_angle_data_without_outliners
        ), 1)
        cv2.putText(
            frame,
            f'Body angle [max]: {maximum_body_angle}',
            (0, 128), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0),
            3, cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f'Body angle [max]: {maximum_body_angle}',
            (0, 128), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255),
            2, cv2.LINE_AA,
        )

        video_output.write(frame)
    run_direction = get_run_direction(
        average_hip_position_line[0],
        average_hip_position_line[-1],
    )
    average_hip_position_line = numpy.array(average_hip_position_line)
    average_run_line_height = numpy.average(average_hip_position_line[:, 1])
    average_body_angle = numpy.median(average_body_angle_data)
    print(f'Run direction: {run_direction}')
    print(f'Average run line height: {average_run_line_height}')
    print(f'Average body angle: {average_body_angle}')

    steps_counter: typing.List[int] = []
    current_ankle_index: int = 0
    last_ankle_flip_frame_index: int = 0
    for frame, ankle_positions in enumerate(all_ankle_positions):
        reverse_ankle_index = 1 if current_ankle_index == 0 else 0
        ankle_flip = ankle_positions[current_ankle_index][1] > ankle_positions[reverse_ankle_index][1]
        if ankle_flip and frame > last_ankle_flip_frame_index + (fps / 3):
            current_ankle_index = reverse_ankle_index
            last_ankle_flip_frame_index = frame
            steps_counter.append(frame)

    total_steps_count = len(steps_counter)
    total_video_length = (total_frames_count / fps)
    total_steps_per_minute = (total_steps_count / total_video_length) * 60

    print(f'Total steps count: {total_steps_count}')
    print(f'Total steps per minute: {total_steps_per_minute}')

    return {
        'run_direction': run_direction,
        'average_body_angle': average_body_angle,
        'total_steps_count': total_steps_count,
        'steps_per_minute': total_steps_per_minute,
    }


if __name__ == '__main__':
    main()
