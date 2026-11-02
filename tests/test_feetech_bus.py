from fake_servo import FakeServos
from joint_bus_contract import JointBusContract

from dros.servo import STS, FeetechBus, rad_to_ticks

SERVO_IDS = (1, 2, 3)


def fake_servos_at(start_angles):
    return FakeServos(dict(zip(SERVO_IDS, rad_to_ticks(start_angles).tolist())))


class TestFeetechBusFollowsTheContract(JointBusContract):
    def make_bus(self, geom, start_angles, **limits):
        return FeetechBus(fake_servos_at(start_angles), geom, model=STS, **limits)
