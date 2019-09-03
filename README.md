# runtech


# Usage:

```
$ cd ~/dveselov/runtech
$ touch output.mp4
$ docker build -t bureau/runtech .
$ docker run --rm -ti \
    --volume /home/dveselov/runtech/source.mp4:/mnt/source.mp4 \
    --volume /home/dveselov/runtech/output.mp4:/mnt/output.mp4 \
    bureau/runtech python -W ignore main.py

...

Video parameters: width=1920, height=1080, fps=29, total_frames_count=57
57it [03:40,  3.87s/it]
Run direction: rtl
Average run line height: 734.6491228070175
Average spine angle: 6.792625998581741
Total steps count: 5
Total steps per minute: 152.6315789473684
```
