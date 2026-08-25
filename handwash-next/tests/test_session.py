import unittest

from bubbles.session import Observation, SessionEngine, Stage


class SessionTests(unittest.TestCase):
    def feed(self, engine, start, duration, hands=1.0, rubbing=1.0, soap=0.0):
        state = None
        steps = int(duration * 10) + 1
        for i in range(steps):
            state = engine.update(Observation(start + i / 10, hands, rubbing, soap if i == 0 else 0))
        return state

    def test_success_requires_soap_and_enough_rubbing(self):
        engine = SessionEngine(required_rub_seconds=3, confirmation_seconds=.5)
        state = self.feed(engine, 0, 4, soap=1)
        self.assertEqual(state.stage, Stage.COMPLETE)

    def test_no_soap_fails_at_time_limit(self):
        engine = SessionEngine(required_rub_seconds=2, confirmation_seconds=.5)
        state = self.feed(engine, 0, 3)
        self.assertEqual(state.stage, Stage.FAILED)

    def test_noise_does_not_confirm_rubbing(self):
        engine = SessionEngine(required_rub_seconds=2, confirmation_seconds=.5)
        for i in range(20):
            state = engine.update(Observation(i / 10, 1, .2 if i % 2 else .8, 1 if i == 0 else 0))
        self.assertEqual(state.rubbing_seconds, 0)

    def test_room_mode_completes_without_soap(self):
        engine = SessionEngine(required_rub_seconds=2, confirmation_seconds=.5,
                               require_soap=False)
        state = self.feed(engine, 0, 3, soap=0)
        self.assertEqual(state.stage, Stage.COMPLETE)


if __name__ == "__main__":
    unittest.main()
