import unittest

from benchmark_true_cycles import parse_xctrace_process_cycles


class BenchmarkTrueCyclesTests(unittest.TestCase):
    def test_parse_xctrace_process_cycles_sums_precise_cycle_rows(self) -> None:
        xml_text = """<?xml version="1.0"?>
<trace-query-result>
  <node xpath="/demo">
    <schema name="MetricAggregationForProcess" />
    <row>
      <process id="1" fmt="python3 (123)">
        <pid id="2" fmt="123">123</pid>
      </process>
      <uint64 id="3" fmt="1000">1000</uint64>
      <string id="4" fmt="cycle">cycle</string>
      <boolean id="5" fmt="Yes">1</boolean>
    </row>
    <row>
      <process ref="1" />
      <uint64 id="6" fmt="2500">2500</uint64>
      <string ref="4" />
      <boolean ref="5" />
    </row>
    <row>
      <process ref="1" />
      <uint64 id="7" fmt="9999">9999</uint64>
      <string ref="4" />
      <boolean id="8" fmt="No">0</boolean>
    </row>
    <row>
      <process ref="1" />
      <uint64 id="9" fmt="88">88</uint64>
      <string id="10" fmt="useful">useful</string>
      <boolean ref="5" />
    </row>
  </node>
</trace-query-result>
"""
        self.assertEqual(
            parse_xctrace_process_cycles(xml_text, expected_pid=123),
            3500,
        )

    def test_parse_xctrace_process_cycles_filters_expected_pid(self) -> None:
        xml_text = """<?xml version="1.0"?>
<trace-query-result>
  <node xpath="/demo">
    <schema name="MetricAggregationForProcess" />
    <row>
      <process id="1" fmt="python3 (123)">
        <pid id="2" fmt="123">123</pid>
      </process>
      <uint64 id="3" fmt="1000">1000</uint64>
      <string id="4" fmt="cycle">cycle</string>
      <boolean id="5" fmt="Yes">1</boolean>
    </row>
    <row>
      <process id="6" fmt="xctrace (999)">
        <pid id="7" fmt="999">999</pid>
      </process>
      <uint64 id="8" fmt="9000">9000</uint64>
      <string ref="4" />
      <boolean ref="5" />
    </row>
  </node>
</trace-query-result>
"""
        self.assertEqual(parse_xctrace_process_cycles(xml_text, expected_pid=123), 1000)
        self.assertEqual(parse_xctrace_process_cycles(xml_text), 10000)


if __name__ == "__main__":
    unittest.main()
