import unittest

from operation_counter import counting_scope, increment, reset, snapshot


class OperationCounterTests(unittest.TestCase):
    def setUp(self) -> None:
        reset()

    def test_counters_are_disabled_by_default(self) -> None:
        increment("default.off")
        self.assertEqual(snapshot(), {})

    def test_counting_scope_enables_collection(self) -> None:
        with counting_scope():
            increment("scoped.counter")
            counters = snapshot()

        self.assertEqual(counters.get("scoped.counter"), 1.0)
        self.assertEqual(snapshot().get("scoped.counter"), 1.0)


if __name__ == "__main__":
    unittest.main()
