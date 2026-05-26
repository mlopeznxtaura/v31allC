import json
data = json.load(open('/mnt/d/NextAura/L2EVAL/OpenSourceWorldModel-headless/v2L2/origin/4563-fixed.json'))
print('records:', len(data))
print('keys:', list(data[0].keys()))
print('---rec0---')
print(json.dumps(data[0], indent=2)[:1000])
print('---rec1---')
print(json.dumps(data[1], indent=2)[:500])
