import ujson as json


def add_speaker_for_medical_data(file_name):
    old_data = list()
    with open(file_name, 'r') as f:
        for line in f:
            old_data.append(json.loads(line))

    new_data = list()
    for i, tmp_example in enumerate(old_data):
        tmp_speakers = list()
        for s in tmp_example['sentences']:
            tmp_speakers_by_sentence = list()
            for w in s:
                tmp_speakers_by_sentence.append(str(i))
            tmp_speakers.append(tmp_speakers_by_sentence)
        new_example = tmp_example
        new_example['speakers'] = tmp_speakers
        new_data.append(new_example)
    with open(file_name, 'w') as f:
        for tmp_example in new_data:
            f.write(json.dumps(tmp_example))
            f.write('\n')


add_speaker_for_medical_data('medical_data/train.jsonlines')
add_speaker_for_medical_data('medical_data/test.jsonlines')
print('end')
