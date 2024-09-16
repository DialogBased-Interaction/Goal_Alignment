total_responses = 0
not_specified_responses = 0
rat = []

with open('filtered_cross_domain_baseline.txt', 'r') as file:
    for line in file:
        if line.startswith('identity') and total_responses != 0:
            rat.append(not_specified_responses / total_responses)
            total_responses = 0
            not_specified_responses = 0
        if line.startswith('user response'):
            total_responses += 1
        if 'user response -> Not specified' in line:
            not_specified_responses += 1

print(f'The ratio is {sum(rat) / len(rat)}')