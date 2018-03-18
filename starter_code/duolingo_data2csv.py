import pandas as pd

def create_csv(input_path):

    training = ("train" in input_path)    
    columns=['user','countries','days','client','session','format','time','instance_id','token','part_of_speech','dependency_label','dependency_edge_head','correctness']
    df=pd.DataFrame(columns=columns)
    num_exercises=0
    csv_path=input_path+".csv"
    records=[]
    
    lines=open(input_path).readlines()
    print("Total lines count is "+str(len(lines)))
    for count,line in enumerate(lines):
        line = line.strip()
        if count%100000==0:
            print("within line "+str(count))

        # If there's nothing in the line, then we're done with the exercise. Print if needed, otherwise continue
        if len(line) == 0:
            num_exercises += 1
            if num_exercises % 10000 == 0:
                print('Loaded ' + str(len(df.index)) + ' instances across ' + str(num_exercises) + ' exercises...')
                records_dict=dict()
                for ind,record in enumerate(records):
                    records_dict[ind]=record
                df.from_dict(records_dict,orient='index').to_csv(csv_path)

        # If the line starts with #, then we're beginning a new exercise
        elif line[0] == '#':
            list_of_exercise_parameters = line[2:].split()
            exercise_properties = dict()
            for exercise_parameter in list_of_exercise_parameters:
                [key, value] = exercise_parameter.split(':')
                if key=='user':
                    value=str(value)
                if key == 'countries':
                    value = value.split('|')[0]  # select the very first country that the user specified
#                     if (len(value)>1):
#                         print("This user has more than one country "+line)
                elif key == 'days':
                    value = float(value)
                elif key == 'client':
                    value = (1 if value=="web" else (2 if value=="ios" else 3))
                elif key=='session':
                    value=(1 if value=="lesson" else (2 if value=="practice" else 3))
                elif key=='format':
                    value=(1 if value=="reverse_translate" else (2 if value=="reverse_tap" else 3))
                elif key == 'time':
                    if value == 'null' or float(value)<=0:
                        value = None
                    else:
                        assert '.' not in value
                        value = int(value)
                if value!=None:
                    exercise_properties[key] = value

        # Otherwise we're parsing a new Instance for the current exercise
        else:
            instance_properties=dict(exercise_properties)
            line = line.split()
            if training:
                assert len(line) == 7
            else:
                assert len(line) == 6
            assert len(line[0]) == 12

            instance_properties['session_id'] = line[0][:8]
            instance_properties['exercise_id'] = int(line[0][8:10])
            instance_properties['token_id'] = int(line[0][10:12])

            instance_properties['token'] = line[1]
            instance_properties['part_of_speech'] = line[2]

            # TODO starts
            for l in line[3].split('|'):
                [key, value] = l.split('=')
                if key == 'Person':
                    value = int(value)
                instance_properties['morphological_features_'+key]=value
                
            # TODO ends from pandas though

            instance_properties['dependency_label'] = line[4]
            instance_properties['dependency_edge_head'] = int(line[5])
            if training:
                instance_properties['correctness'] = float(line[6])
#             df=df.append(instance_properties,ignore_index=True)
            records+=[instance_properties]
        
#         df.to_csv(csv_path)
                
                
# create_csv("../data/data_es_en/es_en.slam.20171218.train")

