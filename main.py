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

import numpy as np
import cv2

class VehicleSim(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.set_background_color(0.1, 0.1, 0.1)
        self.disable_mouse()

        debugNode = BulletDebugNode('Debug')
        debugNode.showWireframe(True)

        debugNP = render.attachNewNode(debugNode)
        debugNP.show()

        

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
        self.steeringIncrement = 300.0  # degrees per second


        

        self.taskMgr.add(self.update, 'updateWorld')

    def setup_camera_texture(self):
        self.tex = Texture()

        # Create a separate buffer for the camera
        winprops = WindowProperties.size(320, 240)
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
        cm = CardMaker("ground")
        cm.setFrame(-50, 50, -50, 50)
        ground_vis = self.render.attachNewNode(cm.generate())
        ground_vis.setPos(0, 0, -1)
        ground_vis.setHpr(0, -90, 0)
        ground_vis.setColor(0.5, 0.5, 0.5)



    def setup_vehicle(self):
        # Chassis
        shape = BulletBoxShape(Vec3(0.7, 1.5, 0.5))
        ts = TransformState.makePos(Point3(0, 0, 0.5))

        self.chassis_np = self.render.attachNewNode(BulletRigidBodyNode('Vehicle'))
        self.chassis_np.node().addShape(shape, ts)
        self.chassis_np.setPos(0, 0, 0)
        self.chassis_np.node().setMass(400.0)
        self.chassis_np.node().setDeactivationEnabled(False)
        self.world.attachRigidBody(self.chassis_np.node())

        # Chassis visual
        # vis_model = loader.loadModel("models/box")  # Replace with better model if desired
        # vis_model.setScale(0.7, 1.5, 0.5)
        # vis_model.reparentTo(self.chassis_np)

        # Vehicle setup
        self.vehicle = BulletVehicle(self.world, self.chassis_np.node())
        self.vehicle.setCoordinateSystem(ZUp)
        self.world.attachVehicle(self.vehicle)

        #오른쪽, 앞, 위
        self.wheel_nps = []
        self.add_wheel(Point3( 1.1,  1.1, 0.5), True)   # Front right
        self.add_wheel(Point3(-1.1,  1.1, 0.5), True)   # Front left
        self.add_wheel(Point3( 1.1, -1.1, 0.5), False)  # Rear right
        self.add_wheel(Point3(-1.1, -1.1, 0.5), False)  # Rear left

        

        
    def add_wheel(self, pos, front):
        # wheel_np = loader.loadModel("models/smiley")  # Use any small sphere-like object
        # wheel_np.setScale(0.25)
        # wheel_np.reparentTo(self.render)

        wheel = self.vehicle.createWheel()
        # wheel.setNode(wheel_np.node())
        wheel.setChassisConnectionPointCs(pos)
        wheel.setFrontWheel(front)
        wheel.setWheelDirectionCs(Vec3(0, 0, -1))
        wheel.setWheelAxleCs(Vec3(1, 0, 0))
        wheel.setWheelRadius(0.4)
        wheel.setMaxSuspensionTravelCm(40.0)

        wheel.setSuspensionStiffness(40.0)
        wheel.setWheelsDampingRelaxation(2.3)
        wheel.setWheelsDampingCompression(4.4)
        wheel.setFrictionSlip(100.0)
        wheel.setRollInfluence(0.1)

        # self.wheel_nps.append(wheel_np)

    def setup_controls(self):
        inputState.watchWithModifiers('forward', 'w')
        inputState.watchWithModifiers('reverse', 's')
        inputState.watchWithModifiers('turnLeft', 'a')
        inputState.watchWithModifiers('turnRight', 'd')

    def update(self, task):
        dt = globalClock.getDt()

        # --- Steering Logic ---
        if inputState.isSet('turnLeft'):
            self.steering += dt * self.steeringIncrement
            self.steering = min(self.steering, self.steeringClamp)
        elif inputState.isSet('turnRight'):
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

        # --- Engine and Brake Logic ---
        engineForce = 0.0
        brakeForce = 0.0

        if inputState.isSet('forward'):
            engineForce = 2000.0
        elif inputState.isSet('reverse'):
            brakeForce = 100.0

        # Apply steering (front wheels)
        self.vehicle.setSteeringValue(self.steering, 0)
        self.vehicle.setSteeringValue(self.steering, 1)

        # Apply force (rear wheels)
        for i in [2, 3]:
            self.vehicle.applyEngineForce(engineForce, i)
            self.vehicle.setBrake(brakeForce, i)

        # Attach camera to car
        self.camera.reparentTo(self.chassis_np)
        self.camera.setPos(0, -20, 5)
        self.camera.lookAt(self.chassis_np)

        self.cam_np.reparentTo(self.chassis_np)
        self.cam_np.setPos(0, -5, 10)
        self.cam_np.lookAt(0, 5, 0)
        

        # Step simulation
        self.world.doPhysics(dt, 10, 0.008)

        # Grab image from render-to-texture
        if self.tex.hasRamImage():
            img = self.tex.getRamImageAs("RGB")
            img = np.frombuffer(img.get_data(), np.uint8)

            w, h = self.tex.getXSize(), self.tex.getYSize()
            img = img.reshape((h, w, 3))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.flip(img, 0)  # 👈 Fix upside down

            # Optional OpenCV logic here:
            # cv2.imshow("Camera View", img)
            # cv2.waitKey(1)

            #------------------------------------------------------
            # OPENCV LOGIC HERE
            #------------------------------------------------------
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)            # Grayscale
            edges = cv2.Canny(gray, 50, 150)                        # Edge detection

            # Convert edges back to BGR to overlay
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            # Draw a red rectangle (x=50, y=50 to x=150, y=150)
            # cv2.rectangle(edges_bgr, (50, 50), (150, 150), (0, 0, 255), 2)

            # Show in a window
            cv2.imshow("Camera View with Edges", edges_bgr)
            cv2.waitKey(1)

            #-------------------------------------------------------
            # END OF OPENCV LOGIC
            #-------------------------------------------------------

        return Task.cont

app = VehicleSim()
app.run()
