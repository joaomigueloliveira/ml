## 2x2 on BFCL vs the 2B: rows where the 2B is right / wrong
B: right on 2B-right rows 737/827; right on 2B-wrong rows 54/173
D: right on 2B-right rows 765/827; right on 2B-wrong rows 60/173
A: right on 2B-right rows 749/827; right on 2B-wrong rows 96/173
Bf: right on 2B-right rows 731/827; right on 2B-wrong rows 77/173

## 'missing required arg' on BFCL: arg names and whether the spec description mentions a default
A: missing required args 51, spec mentions 'default' for 50; top names [('unit', 9), ('confidence_level', 4), ('include_influences', 4), ('alpha', 4), ('include_player_stats', 4), ('historical', 3)]
Bf: missing required args 39, spec mentions 'default' for 16; top names [('data', 16), ('include_influences', 4), ('include_player_stats', 4), ('state', 2), ('status', 2), ('to_timezone', 2)]
B: missing required args 13, spec mentions 'default' for 7; top names [('name', 4), ('view', 2), ('initial_velocity', 1), ('detail', 1), ('features', 1), ('parameters', 1)]
D: missing required args 0, spec mentions 'default' for 0; top names []

## Rows only D right / only B right, by category and loser's error
simple: only D 23 [('missing required arg', 4), ('number (gold not in query)', 4), ('content (gold in query)', 3), ('content (gold not in query)', 3), ('string format/case only', 2)] | only B 10 [('optional arg given a non-default value', 4), ('content (gold not in query)', 2), ('number (gold in query)', 1), ('number (gold not in query)', 1)]
multiple: only D 16 [('string format/case only', 3), ('missing required arg', 3), ('unparsed', 2), ('content (gold in query)', 2), ('content (gold not in query)', 2)] | only B 3 [('content (gold in query)', 1), ('optional arg given a non-default value', 1), ('content (gold not in query)', 1)]
parallel: only D 12 [('missing call', 5), ('extra call', 2), ('content (gold in query)', 1), ('unparsed', 1), ('optional arg given a non-default value', 1)] | only B 4 [('missing call', 3), ('list/dict value', 1)]
parallel_multiple: only D 11 [('content (gold in query)', 4), ('list/dict value', 2), ('string format/case only', 1), ('missing call', 1), ('optional arg given a non-default value', 1)] | only B 11 [('optional arg given a non-default value', 4), ('content (gold in query)', 3), ('wrong function', 2), ('extra unknown arg', 1)]

only-D-right rows where the 2B is also right: 48/62
only-B-right rows where the 2B is also right: 20/28

examples (B output | D output), truncated:
simple_13: string format/case only | B [{"name": "calculate_area_under_curve", "arguments": {"function": "y=x^2", "interval": [1, 3], "method": "trapezoidal"}} | D [{"name": "calculate_area_under_curve", "arguments": {"function": "x**2", "interval": [1, 3], "method": "trapezoidal"}}]
simple_32: missing required arg | B [{"name": "calculate_final_speed", "arguments": {"height": 100, "gravity": 9.8}}] | D [{"name": "calculate_final_speed", "arguments": {"initial_velocity": 0, "height": 100, "gravity": 9.8}}]
simple_42: number (gold not in query) | B [{"name": "calculate_resonant_frequency", "arguments": {"capacitance": 100, "inductance": 50}}] | D [{"name": "calculate_resonant_frequency", "arguments": {"inductance": 50e-3, "capacitance": 100e-6, "round_off": 2}}]
simple_55: content (gold in query) | B [{"name": "biology.get_cell_info", "arguments": {"cell_type": "human cell", "detailed": true}}] | D [{"name": "biology.get_cell_info", "arguments": {"cell_type": "human", "detailed": true}}]
simple_65: number (gold not in query) | B [{"name": "calculate_density", "arguments": {"country": "Brazil", "year": "2022", "population": 213, "land_area": 8500}} | D [{"name": "calculate_density", "arguments": {"country": "Brazil", "year": "2022", "population": 213000000, "land_area":
simple_94: unparsed | B [{"name": "update_user_info", "arguments": {"user_id": 43523, "update_info": {"name": "John Doe", "email": "johndoe@emai | D ```json
[
  {
    "name": "update_user_info",
    "arguments": {
      "user_id": 43523,
      "update_info": {

simple_151: number (gold not in query) | B [{"name": "highest_grossing_banks", "arguments": {"country": "U.S", "year": 2020, "top_n": 5}}] | D [{"name": "highest_grossing_banks", "arguments": {"country": "United States", "year": 2020, "top_n": 1}}]
simple_152: number (gold not in query) | B [{"name": "calculate_mutual_fund_balance", "arguments": {"investment_amount": 50000, "annual_yield": 5.0, "years": 3}}] | D [{"name": "calculate_mutual_fund_balance", "arguments": {"investment_amount": 50000, "annual_yield": 0.05, "years": 3}}]
