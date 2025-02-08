import json


recs = []

f = open('/Users/lordkersting/Downloads/responses.jsonl', 'r')
counter = 0
for line in f.readlines():
    counter += 1
    curr_json = json.loads(line)
    recs.append([curr_json["prompt"].replace('\n', ' '), curr_json["response"].replace('\n', ' '), counter])

outfile = open('program_questions.txt', 'w')
    
for r in recs:
    outfile.write(f"{r[0]} \t {r[1]} \t {r[2]} \n")

outfile.close()
