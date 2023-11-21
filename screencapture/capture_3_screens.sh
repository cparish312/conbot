#!/bin/bash

# Run a command in the background.
# _evalBg() {
#     eval "$@" &>/dev/null & disown;
# }

# _evalBg ""screencapture -D 1 -v $(date +'%Y-%m-%d_%H-%M-%S')_1_screencapture.mov""; 
# _evalBg ""screencapture -D 2 -v $(date +'%Y-%m-%d_%H-%M-%S')_2_screencapture.mov""; 
# _evalBg ""screencapture -D 3 -v $(date +'%Y-%m-%d_%H-%M-%S')_3_screencapture.mov""; 

# screencapture -D 1 -v $(date +'%Y-%m-%d_%H-%M-%S')_1_screencapture.mov &
# screencapture -D 2 -v $(date +'%Y-%m-%d_%H-%M-%S')_2_screencapture.mov &
# screencapture -D 3 -v $(date +'%Y-%m-%d_%H-%M-%S')_3_screencapture.mov &&
# fg
# sleep 10000000

(trap 'kill 0' SIGINT; echo "running screen capture" & screencapture -D 1 -v data/$(date +'%Y-%m-%d_%H-%M-%S')_1_screencapture.mov &
screencapture -D 2 -v data/$(date +'%Y-%m-%d_%H-%M-%S')_2_screencapture.mov &
screencapture -D 3 -v data/$(date +'%Y-%m-%d_%H-%M-%S')_3_screencapture.mov)