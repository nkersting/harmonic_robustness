
import csv


questions = []
answers = []
sources = []
with open("TruthfulQA.csv", 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        questions.append(row['Question'])
        answers.append(row['Best Answer'])
        sources.append(row['Source'])

outfile = open('truthful.tsv','w')
for i in range(len(questions)):
    outfile.write(f"{questions[i]}\t {answers[i]}\t {sources[i]}\n")

outfile.close()
