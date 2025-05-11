import msgParser
import carState
import carControl
import pandas as pd
import os
import torcs_offline_rl
class Driver(object):

    def __init__(self, stage):
        '''Constructor'''
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3
        self.stage = stage
        self.lapNum = 0
        
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()
        
        self.steer_lock = 0.785398
        self.max_speed = 300  
        self.prev_rpm = None

        self.control.setGear(1)

        self.controller = torcs_offline_rl.RacingController()
        self.log_file = "log_file.csv"

    def init(self):
        '''Return init string with rangefinder angles'''
        self.angles = [0 for x in range(19)]
        
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15
        
        for i in range(5, 9):
            self.angles[i] = -20 + (i-5) * 5
            self.angles[18 - i] = 20 - (i-5) * 5
        
        return self.parser.stringify({'init': self.angles})
    def gear(self):
        gear = self.state.getGear()
        speed = self.state.getSpeedX()

        # Generic upshift/downshift speed thresholds
        speedup = [45, 75, 120, 170, 230, 300]
        speeddown = [0, 40, 70, 110, 160, 210]

        hysteresis = 5  # prevent rapid gear shifts

        # Ensure gear is initialized
        if gear < 1:
            gear = 1

        # Upshift logic
        if gear < 6 and speed > speedup[gear - 1] + hysteresis:
            gear += 1

        # Downshift logic
        elif gear > 1 and speed < speeddown[gear - 1] - hysteresis:
            gear -= 1

        self.control.setGear(gear)


    def gear_rpm(self):
        # Refined RPM thresholds for gear up and gear down adjusted for max RPM of 8k
        gearup = [6800, 7000, 7200, 7400, 7600, 0]  # Reduced gear-up thresholds
        geardown = [0, 3300, 3700, 3900, 4000, 4200]  # Reduced gear-down thresholds

        gear = self.state.getGear()
        rpm = self.state.getRpm()

        # Ensure gear is at least 1 (no reverse or neutral)
        if gear < 1:
            gear = 1

        # Limit RPM to 8000 if necessary
        if rpm > 8000:
            rpm = 8000

        # Gear up logic: If RPM exceeds the gear-up threshold with a reduced buffer zone
        if gear < 6 and rpm >= gearup[gear - 1] and rpm < gearup[gear - 1] + 100:
            gear += 1  

        # Gear down logic: If RPM drops below the gear-down threshold with a reduced buffer zone
        elif gear > 1 and rpm <= geardown[gear - 1] and rpm > geardown[gear - 1] - 200:
            new_gear = gear - 1
            if rpm * (gear / new_gear) < 8000:  
                gear = new_gear
        
        # Return the updated gear
        self.control.setGear(gear)
    def drive(self, msg):
        self.state.setFromMsg(msg)

        # Get current state for controller with all 28 required values
        current_state = {
            'RPM': self.state.getRpm(),
            'SpeedX': self.state.getSpeedX(),
            'SpeedY': self.state.getSpeedY(),
            'SpeedZ': self.state.getSpeedZ(),
            'TrackPosition': self.state.getTrackPos(),
            'Z': self.state.getZ(),
        }
        
        # Add all 19 track sensors
        track_sensors = self.state.getTrack()
        for i in range(19):
            current_state[f'Track_{i+1}'] = track_sensors[i]

        # Get controller prediction
        controller_action = self.controller.predict(current_state)
        
        # Apply controller actions
        self.control.setAccel(controller_action.get('Acceleration', 0.0))
        self.control.setBrake(controller_action.get('Braking', 0.0))
        self.control.setSteer(controller_action.get('Steering', 0.0))
        self.gear()  # Use automatic gear if controller doesn't specify

       # self.log_data([])  # Pass empty list since we're not using manual controls
        return self.control.toMsg()

    def log_data(self, keys_pressed):
        lap_number = self.state.getLapNum()
        data = {
            'angle': self.state.getAngle(),
            'curLapTime': self.state.getCurLapTime(),
            'damage': self.state.getDamage(),
            'distFromStart': self.state.getDistFromStart(),
            'distRaced': self.state.getDistRaced(),
            'fuel': self.state.getFuel(),
            'gear': self.state.getGear(),
            'lastLapTime': self.state.getLastLapTime(),
            'opponents': str(self.state.getOpponents()), 
            'racePos': self.state.getRacePos(),
            'rpm': self.state.getRpm(),
            'speedX': self.state.getSpeedX(),
            'speedY': self.state.getSpeedY(),
            'speedZ': self.state.getSpeedZ(),
            'track': str(self.state.getTrack()),         
            'trackPos': self.state.getTrackPos(),
            'wheelSpinVel': str(self.state.getWheelSpinVel()), 
            'z': self.state.getZ(),
            'lap_number': lap_number,
            'clutch': self.control.clutch,
            'brake' : self.control.brake,
            'steer' : self.control.steer,
            'accel' : self.control.accel,
        }
        df = pd.DataFrame([data])
        write_header = not os.path.exists(self.log_file)
        df.to_csv(self.log_file, mode='a', header=write_header, index=False)
        print(f"Data logged to {self.log_file}")  # Debugging statement


    def onShutDown(self):
        pass
    
    def onRestart(self):
        pass

