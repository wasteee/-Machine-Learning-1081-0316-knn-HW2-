"""The template of the main script of the machine learning process
"""
import pickle
import games.arkanoid.communication as comm
import numpy as np
from games.arkanoid.communication import ( \
    SceneInfo, GameInstruction, GameStatus, PlatformAction
)

def ml_loop():
    """The main loop of the machine learning process

    This loop is run in a separate process, and communicates with the game process.

    Note that the game process won't wait for the ml process to generate the
    GameInstruction. It is possible that the frame of the GameInstruction
    is behind of the current frame in the game process. Try to decrease the fps
    to avoid this situation.
    """

    # === Here is the execution order of the loop === #
    # 1. Put the initialization code here.

    # 2. Inform the game process that ml process is ready before start the loop.
    filename = "C:\\Users\\fghj8\\MLGame-master\\games\\arkanoid\\knn_example.sav"
    model = pickle.load(open(filename,'rb'))
    
    last_ball_x = 0
    last_ball_y = 0
    comm.ml_ready()
    # 3. Start an endless loop.
    scene_info = comm.get_scene_info()
    while True:
        # 3.1. Receive the scene information sent from the game process.
        
        
        last_ball_x = scene_info.ball[0]
        last_ball_y = scene_info.ball[1]
        scene_info = comm.get_scene_info()
        plat_cneter_x = scene_info.platform[0]+20
        
        if(last_ball_x - scene_info.ball[0] > 0):
            LR = 1
        else:
            LR = 0
        if(last_ball_y - scene_info.ball[1] > 0):
            UP = 0
        else:
            UP = 1
        
        inp_temp = np.array([scene_info.ball[0],scene_info.ball[1],LR,UP,scene_info.platform[0]])
        input = inp_temp[np.newaxis,:]
        
        if scene_info.status == GameStatus.GAME_OVER or \
            scene_info.status == GameStatus.GAME_PASS:
            # Do some stuff if needed
            # 3.2.1. Inform the game process that ml process is ready
            scene_info = comm.get_scene_info()
            
        
        
        move = model.predict(input)
    
        # 3.3. Put the code here to handle the scene information

        # 3.4. Send the instruction for this frame to the game process
        
        #plat_location拿到的位置是平板的最左
        #平板中心為 平板長度/2 + plat_location
        if(move<0):
            comm.send_instruction(scene_info.frame, PlatformAction.MOVE_LEFT)
        elif(move>0):
            comm.send_instruction(scene_info.frame, PlatformAction.MOVE_RIGHT)
        else:
            comm.send_instruction(scene_info.frame, PlatformAction.NONE)
    
            

