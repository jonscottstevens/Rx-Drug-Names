drug_names = []

with open('drug_info.txt','r') as f:
	for line in f:
		drug_names.append(line.split('\t')[0])
        
drug_names = [name.split(' ')[0] for name in drug_names]
drug_names = set(drug_names)

with open('drug_names.txt','w') as f:
    f.write('#')
    for name in drug_names:
		f.write(name+'#')