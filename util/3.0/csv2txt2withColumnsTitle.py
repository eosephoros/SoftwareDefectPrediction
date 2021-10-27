with open('CM1.csv', 'r') as inp, open('CM1.txt', 'w') as out:
    for line in inp:
        line = line.replace(',', ' ')
        out.write(line)