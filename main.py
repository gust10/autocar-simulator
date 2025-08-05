from panda3d.core import *
from panda3d.bullet import *

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.showbase.InputStateGlobal import inputState

from panda3d.core import loadPrcFileData
loadPrcFileData('', 'show-frame-rate-meter true')
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode
import time

from panda3d.bullet import BulletDebugNode

from panda3d.core import WindowProperties, FrameBufferProperties, GraphicsOutput, Texture, GraphicsPipe, CardMaker
from panda3d.core import PNMImage
from panda3d.core import Camera, NodePath, OrthographicLens
from panda3d.core import Lens

import numpy as np
import cv2

import live_plot
from steering_control import calculate_steering_angle

# Start the live graph once
# live_plot.run_in_thread()

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def update(self, target, current, dt):
        error = target - current
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error

        return self.kp * error + self.ki * self.integral + self.kd * derivative


class VehicleSim(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.set_background_color(0.063, 0.455, 0.988) # Set background color to blue
        # self.disable_mouse()

        debugNode = BulletDebugNode('Debug')
        debugNode.showWireframe(True)

        debugNP = render.attachNewNode(debugNode)
        # debugNP.show()

        

        # Create physics world
        self.world = BulletWorld()
        self.world.setGravity(Vec3(0, 0, -9.81))
        self.world.setDebugNode(debugNP.node())

        # Set up camera
        self.setup_camera_texture()

        self.setup_environment()
        self.setup_vehicle()
        self.setup_controls()

        # Steering control
        self.steering = 0.0
        self.steeringClamp = 30.0  # degrees
        self.steeringIncrement = 100.0  # degrees per second

        self.start_time = time.time()
        

        self.taskMgr.add(self.update, 'updateWorld')

    def setup_camera_texture(self):
        self.tex = Texture()

        # Create a separate buffer for the camera
        winprops = WindowProperties.size(160, 120)
        fbprops = FrameBufferProperties()
        fbprops.setRgbColor(True)
        fbprops.setDepthBits(1)

        self.buffer = self.graphicsEngine.makeOutput(
            self.pipe, "camera buffer", -2,
            fbprops, winprops,
            GraphicsPipe.BFRefuseWindow,
            self.win.getGsg(), self.win)

        self.buffer.addRenderTexture(self.tex, GraphicsOutput.RTMCopyRam)

        # Attach a new camera to render to this buffer
        self.cam_np = self.makeCamera(self.buffer)


    def setup_environment(self):
        # Ground
        shape = BulletPlaneShape(Vec3(0, 0, 1), 0)
        ground_np = self.render.attachNewNode(BulletRigidBodyNode('Ground'))
        ground_np.node().addShape(shape)
        ground_np.setPos(0, 0, 0)
        ground_np.setCollideMask(BitMask32.allOn())
        self.world.attachRigidBody(ground_np.node())

        # Visual ground
        # cm = CardMaker("ground")
        # cm.setFrame(-50, 50, -50, 50)
        # ground_vis = self.render.attachNewNode(cm.generate())
        # ground_vis.setPos(0, 0, -1)
        # ground_vis.setHpr(0, -90, 0)
        # ground_vis.setColor(1, 1, 1)

        # Visual track
        # track_vis = loader.loadModel("models/track1.gltf")
        # track_vis.setScale(30.0, 30.0, 30.0)  # Adjust scale as needed
        # track_vis.setPos(0, 0, -1)
        # track_vis.reparentTo(self.render)

        track2_vis = loader.loadModel("models/track2.gltf")
        track2_vis.setScale(32.0, 32.0, 32.0)  # Adjust scale as needed
        track2_vis.setPos(0, 0, -1) # 250, 0, -1
        track2_vis.reparentTo(self.render)
        
        




    def setup_vehicle(self):
        # Chassis
        shape = BulletBoxShape(Vec3(0.7, 2, 0.2))
        ts = TransformState.makePos(Point3(0, 0, 0.5))

        self.chassis_np = self.render.attachNewNode(BulletRigidBodyNode('Vehicle'))
        self.chassis_np.node().addShape(shape, ts)
        self.chassis_np.setPos(0, 0, 0)
        self.chassis_np.setHpr(45, 0, 0)  # Adjust orientation if needed
        self.chassis_np.node().setMass(400.0)
        self.chassis_np.node().set_linear_damping(0.1) # added
        self.chassis_np.node().set_angular_damping(0.1) # added
        self.chassis_np.node().setDeactivationEnabled(False)
        self.world.attachRigidBody(self.chassis_np.node())

        # Chassis visual
        vis_model = loader.loadModel("models/autocar.gltf")  # Replace with better model if desired
        vis_model.setHpr(0, 0, 0)  # Adjust rotation if needed
        vis_model.setScale(1.2, 1.3, 1.0)  # Scale to match physics shape
        vis_model.setPos(0, 0, 0.2)  # Position to match physics shape
        
        vis_model.reparentTo(self.chassis_np)

        # Vehicle setup
        self.vehicle = BulletVehicle(self.world, self.chassis_np.node())
        self.vehicle.setCoordinateSystem(ZUp)
        self.world.attachVehicle(self.vehicle)

        #오른쪽, 앞, 위
        self.wheel_nps = []
        self.add_wheel(Point3( 1.4,  1.4, 0.5), True)   # Front right
        self.add_wheel(Point3(-1.4,  1.4, 0.5), True)   # Front left
        self.add_wheel(Point3( 1.4, -1.4, 0.5), False)  # Rear right
        self.add_wheel(Point3(-1.4, -1.4, 0.5), False)  # Rear left

        # Camera model
        cam_model = loader.loadModel("models/camera.egg")  # Replace with better model if desired
        
        cam_model.setHpr(0, 0, 0)  # Adjust rotation if needed
        cam_model.setScale(0.5, 0.5, 0.5)  # Scale to match physics shape
        cam_model.setPos(0, 0, 5)  # Position to match physics shape
        cam_model.lookAt(0, 10, 0)  # Look at the front of the car
        cam_model.reparentTo(self.chassis_np)
        

        
    def add_wheel(self, pos, front):
        wheel_np = loader.loadModel("models/wheels.gltf")  # Use any small sphere-like object
        wheel_np.setScale(0.20)
        wheel_np.reparentTo(self.render)

        wheel = self.vehicle.createWheel()
        wheel.setNode(wheel_np.node())
        wheel.setChassisConnectionPointCs(pos)
        wheel.setFrontWheel(front)
        wheel.setWheelDirectionCs(Vec3(0, 0, -1))
        wheel.setWheelAxleCs(Vec3(1, 0, 0))
        wheel.setWheelRadius(0.20)
        wheel.setMaxSuspensionTravelCm(40.0)

        wheel.setSuspensionStiffness(40.0)
        wheel.setWheelsDampingRelaxation(2.3)
        wheel.setWheelsDampingCompression(4.4)
        wheel.setFrictionSlip(20.0)
        wheel.setRollInfluence(0)

        self.wheel_nps.append(wheel_np)

    def setup_controls(self):
        inputState.watchWithModifiers('forward', 'w')
        inputState.watchWithModifiers('reverse', 's')
        inputState.watchWithModifiers('turnLeft', 'a')
        inputState.watchWithModifiers('turnRight', 'd')

    def update(self, task):
        dt = globalClock.getDt()
        target_angle = 0.0

        # --- Engine and Brake Logic ---
        engineForce = 0.0
        brakeForce = 0.0

        if inputState.isSet('forward'):
            engineForce = 2000.0
        elif inputState.isSet('reverse'):
            brakeForce = 100.0
        else:
            # If no input, apply some brake to slow down
            brakeForce = 20.0

        

        # Display current speed
        velocity = self.vehicle.getCurrentSpeedKmHour()
        current_speed = velocity
        # print(f"Speed: {current_speed:.2f} units/sec")

        # Trying out PID controller for speed control
        # PID setup
        speed_pid = PIDController(kp=500, ki=0, kd=0)
        target_speed = 50.0  # Target speed in km/h
        engine_force = speed_pid.update(target_speed, current_speed, dt)

        

        # Optional: Clamp force to prevent excessive values
        engine_force = max(min(engine_force, 10000), -200)

        # Apply the engine force to the wheels
        for i in [2, 3]:  # Rear wheels
            self.vehicle.applyEngineForce(engine_force, i)

        # plotting a live graph
        # # Feed data to live plot
        # elapsed = time.time() - self.start_time
        # live_plot.add_data(elapsed, current_speed, target_speed)

       

        # Apply force (rear wheels)
        # for i in [2, 3]:
        #     self.vehicle.applyEngineForce(engineForce, i)
        #     self.vehicle.setBrake(brakeForce, i)

        # Attach camera to car
        self.camera.reparentTo(self.chassis_np)
        self.camera.setPos(10, -30, 30) # 0, -20, 5
        self.camera.lookAt(self.chassis_np)

        self.cam_np.reparentTo(self.chassis_np)
        self.cam_np.setPos(0, 0, 5) 
        self.cam_np.lookAt(0, 10, 0)
        self.cam_np.node().getLens().setFov(90)  # Set camera field of view

        

        # Grab image from render-to-texture
        if self.tex.hasRamImage():
            img = self.tex.getRamImageAs("RGB")
            img = np.frombuffer(img.get_data(), np.uint8)

            w, h = self.tex.getXSize(), self.tex.getYSize()
            img = img.reshape((h, w, 3))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.flip(img, 0)  # 👈 Fix upside down

            target_angle = calculate_steering_angle(img, max_angle_deg=self.steeringClamp)
            print(f"Steering Angle: {target_angle:.2f} radians")

            # --- Steering Logic ---
            if target_angle > self.steering:
                self.steering += dt * self.steeringIncrement
                self.steering = min(self.steering, self.steeringClamp)
            elif target_angle < self.steering:
                self.steering -= dt * self.steeringIncrement
                self.steering = max(self.steering, -self.steeringClamp)
            else:
                # Relax steering slowly
                if self.steering > 0:
                    self.steering -= dt * self.steeringIncrement
                    self.steering = max(self.steering, 0)
                elif self.steering < 0:
                    self.steering += dt * self.steeringIncrement
                    self.steering = min(self.steering, 0)


            
            

            # Optional OpenCV logic here:
            # cv2.imshow("Camera View", img)
            # cv2.waitKey(1)

            #------------------------------------------------------
            # OPENCV LOGIC HERE
            #------------------------------------------------------
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)            # Grayscale
            # edges = cv2.Canny(gray, 50, 150)                        # Edge detection

            # # Convert edges back to BGR to overlay
            # edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            # # Draw a red rectangle (x=50, y=50 to x=150, y=150)
            # # cv2.rectangle(edges_bgr, (50, 50), (150, 150), (0, 0, 255), 2)

            # # Show in a window
            

            #-------------------------------------------------------
            # END OF OPENCV LOGIC
            #-------------------------------------------------------

        # PID for steering angle
        angle_pid = PIDController(kp=1, ki=0, kd=0)
        # self.steering = angle_pid.update(target_angle, self.steering, dt)
        # Clamp steering to max angle
        # self.steering = max(min(self.steering, self.steeringClamp), -self.steeringClamp)

        # Apply steering (front wheels)
        self.vehicle.setSteeringValue(self.steering, 0)
        self.vehicle.setSteeringValue(self.steering, 1)

        # Step simulation
        self.world.doPhysics(dt, 10, 0.008)

        

        return Task.cont

app = VehicleSim()
app.run()
