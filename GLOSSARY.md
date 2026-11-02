# dr-os — Delta Robot Open Source

dr-os is an open-source software ecosystem for cheap, 3D-printed rotary
(Clavel-type) delta robots driven by Feetech STS3215 serial-bus servos. This
glossary is the single source of truth for the project's vocabulary: code, docs,
and commit messages use these words and no others. The `_Avoid_` lists are
binding — if you reach for one of those words, use the canonical term instead.

## Mechanism

**Arm**:
One of the three identical kinematic chains connecting the base to the
end-effector. The robot has exactly three, indexed 0–2. Each arm is made of an
upper arm and a lower arm (below), so "arm" names the whole chain and never a
single link.
_Avoid_: Limb, leg

**Upper arm**:
The rigid link of an arm driven directly by the shoulder, running shoulder →
elbow.
_Avoid_: Proximal link, bicep, crank

**Lower arm**:
The link of an arm running elbow → wrist. Physically a parallelogram of two
parallel rods — this is what keeps the end-effector level and gives the robot
its 3-DOF, translation-only motion.
_Avoid_: Distal link, forearm, rod (a single rod is one half of one lower arm)

**Base**:
The fixed platform that carries the three shoulders. Origin of the robot's
reference frame.
_Avoid_: Fixed platform, frame plate

**End-effector**:
The moving platform at the business end of the three arms. Its location is a
position (see below); it carries no tool yet.
_Avoid_: Effector, moving platform, traveling plate, TCP

**Tool / TCP**:
Reserved, not yet built. A device mounted on the end-effector — gripper, pen,
probe — together with its tool-center-point. Distinct from the end-effector;
to be defined by its own ADR when one exists.

## Joints

**Shoulder**:
The actuated joint where an upper arm meets the base. The only driven joint;
one per arm.
_Avoid_: Base joint, hip, actuator joint

**Elbow**:
The passive joint where the upper arm meets the lower arm.
_Avoid_: Knee

**Wrist**:
The passive joint where the lower arm meets the end-effector.
_Avoid_: Platform joint

## Motion & frames

**Theta (θ)**:
The angle of a shoulder, in radians. θ = 0 is the upper arm horizontal;
positive θ rotates the upper arm downward.
_Avoid_: q, joint var, servo angle

**Position**:
The location of the end-effector in the base frame: three translational degrees
of freedom (x, y, z), no orientation. A delta end-effector cannot be tilted, so
it never has a "pose".
_Avoid_: Pose, coordinate, point

**Rest position**:
The low position the end-effector moves to before torque is turned off, so it is
never dropped from height.
_Avoid_: Park, home

**Workspace**:
Every position the end-effector can reach, given the arm lengths and joint
limits.
_Avoid_: Work area, envelope

**Target cylinder**:
The upright volume on the z axis that a geometry's workspace must cover, given
by a diameter and a height.
_Avoid_: Work volume

**Placement**:
Where along z a target cylinder sits. The check searches for the best one.

**Shortfall**:
A position inside the target cylinder that the end-effector cannot reach.

**Payload**:
The mass the end-effector carries. An input to the torque analysis, not a part of
the geometry.

**Load**:
The payload and moving masses a shoulder has to carry, with the size of the
acceleration. Compared with a servo's continuous torque through the margin.

**Margin**:
A servo's continuous torque set against the worst shoulder torque required.

**Jacobian**:
How the end-effector moves per radian of each shoulder (dp/dθ, a 3×3 matrix).

**Singularity**:
A position where the lower arms are coplanar, or a lower arm lies in the plane of
its upper arm and shoulder axis. Accelerating shoulder torque grows without bound
near one, and static torque too when the lower arms are coplanar.

**Tick error**:
The largest position error at a position from rounding each shoulder to a tick.

**Forward (kinematics)**:
The map from the three shoulder angles (θ) to the end-effector position. Three
angles allow two positions; it returns the lower one, and a position that only
the upper one gives counts as out of reach.
_Avoid_: FK, direct kinematics

**Inverse (kinematics)**:
The map from an end-effector position to the three shoulder angles (θ).
_Avoid_: IK

## Control & hardware

**Joint bus**:
The hardware-agnostic interface through which shoulder angles are read and
commanded. It speaks SI and knows nothing about motors. The mock bus, the sim bus
and the Feetech bus all implement it.
_Avoid_: Driver, controller, port

**Joint bus contract**:
The shared test suite (`tests/joint_bus_contract.py`) that every joint bus must
pass: joint limits, refusal with torque off, and read-back.

**Mock bus**:
A joint bus that stores what it is told and moves instantly. It has no motion of
its own.
_Avoid_: Fake, stub

**Sim bus**:
A joint bus that imitates servo motion: limited speed and acceleration, tick
rounding, and torque on or off. It has no physics; forces, inertia and gravity
belong to the digital twin.
_Avoid_: Simulator, emulator, digital twin

**Feetech bus**:
The joint bus for real Feetech UART-bus servos. It covers every Feetech model the
project uses.
_Avoid_: STS3215 bus, servo driver

**Digital twin**:
Not yet built. A physics model of the whole robot, including its closed chains,
inertia and gravity.
_Avoid_: Sim, simulator

**Ticks**:
Raw servo encoder counts, 4096 per revolution on Feetech servos. Converting SI
radians to ticks happens only in the servo layer.
_Avoid_: Counts, steps, position units
