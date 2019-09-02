import cv2
import numpy
import typing

from fastapi import FastAPI
from starlette.responses import HTMLResponse
from starlette.websockets import WebSocket

from util.config import load_config
from nnet import predict
from dataset.pose_dataset import data_to_input

cfg = load_config('demo/pose_cfg.yaml')
sess, inputs, outputs = predict.setup_pose_prediction(cfg)

app = FastAPI()


def angle_between(
    p1: typing.Tuple[float, float],
    p2: typing.Tuple[float, float],
) -> float:
    ang1 = numpy.arctan2(*p1[::-1])
    ang2 = numpy.arctan2(*p2[::-1])
    return numpy.rad2deg((ang1 - ang2) % (2 * numpy.pi))


def get_video_frames(path: str) -> typing.Generator[bytes, None, None]:
    source = cv2.VideoCapture(path)
    while source:
        ret, frame = source.read()
        if not ret:
            break
        yield frame


def get_frame_pose(frame):
    image_batch = data_to_input(frame)
    outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
    scmap, locref, _ = predict.extract_cnn_output(outputs_np, cfg)
    # Extract maximum scoring location from the heatmap, assume 1 person
    return predict.argmax_pose_predict(scmap, locref, cfg.stride)


def get_iterable_data(path: str):
    for frame in get_video_frames(path):
        pose = get_frame_pose(frame)
        # skip frames with prob less that 70%
        # if pose[-1] < 0.7:
        #     continue
        ankle_average_position = numpy.array((
            (pose[0][0] + pose[5][0]) / 2,
            (pose[0][1] + pose[5][1]) / 2,
        ))
        shoulder_average_position = numpy.array(( 
            (pose[8][0] + pose[9][0]) / 2,
            (pose[8][1] + pose[9][1]) / 2,
        ))
        spine_angle = 360 - angle_between(
            shoulder_average_position, ankle_average_position
        )
        if spine_angle > 90:
            # FIXME
            spine_angle = angle_between(
                shoulder_average_position, ankle_average_position
            )
        yield {
            'frame': frame,
            'pose': pose,
            'spine_angle': spine_angle,
            'ankle_average_position': ankle_average_position,
            'shoulder_average_position': shoulder_average_position,
        }


def get_run_direction(
    first_ankle_position: typing.Tuple[float, float],
    finish_ankle_position: typing.Tuple[float, float],
) -> str:
    run_direction = None
    if first_ankle_position[0] > finish_ankle_position[0]: # right to left
        run_direction = 'rtl'
    else:
        run_direction = 'ltr'
    return run_direction


def main():
    video_source = '/mnt/source.mp4'
    average_ankle_position_line = []  # contains all average ankle points
    average_spine_angle = numpy.empty((0, 1), float)
    for i, data in enumerate(get_iterable_data(video_source)):
        print(f'Processing frame #{i}')
        print(f'Spine angle: {data["spine_angle"]}')
        average_ankle_position_line.append(data['ankle_average_position'])
        average_spine_angle = numpy.append(average_spine_angle, data['spine_angle'])
        print(average_spine_angle)
        if i == 3:
            break
    run_direction = get_run_direction(
        average_ankle_position_line[0],
        average_ankle_position_line[-1],
    )
    average_ankle_position_line = numpy.array(average_ankle_position_line)
    average_run_line_height = numpy.median(average_ankle_position_line[:, 1])
    print(f'Average angle position line: {average_ankle_position_line}')
    print(f'Run direction: {run_direction}')
    print(f'Average run line height: {average_run_line_height}')


main()

# html = """
# <!DOCTYPE html>
# <html>
#     <head>
#         <title>Chat</title>
#     </head>
#     <body>
#         <h1>WebSocket Chat</h1>
#         <form action="" onsubmit="sendMessage(event)">
#             <input type="text" id="messageText" autocomplete="off"/>
#             <button>Send</button>
#         </form>
#         <ul id='messages'>
#         </ul>
#         <script>
#             var ws = new WebSocket("ws://localhost:8000/ws");
#             ws.onmessage = function(event) {
#                 var messages = document.getElementById('messages')
#                 var message = document.createElement('li')
#                 var content = document.createTextNode(event.data)
#                 message.appendChild(content)
#                 messages.appendChild(message)
#             };
#             function sendMessage(event) {
#                 var input = document.getElementById("messageText")
#                 ws.send(input.value)
#                 input.value = ''
#                 event.preventDefault()
#             }
#         </script>
#     </body>
# </html>
# """


# @app.get("/")
# async def get():
#     return HTMLResponse(html)


# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     while True:
#         data = await websocket.receive_text()
#         await websocket.send_text(f"Message text was: {data}")
